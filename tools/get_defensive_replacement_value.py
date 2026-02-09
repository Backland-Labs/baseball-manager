# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Evaluates the net value of a defensive substitution.

Loads fielder and batter data from sample_rosters.json and derives the
defensive upgrade (OAA difference), offensive downgrade (projected wOBA),
and net expected value scaled by estimated innings remaining. Provides a
textual recommendation (favorable, marginal, unfavorable).
"""

import json
from pathlib import Path

from anthropic import beta_tool

from tools.response import success_response, error_response

# ---------------------------------------------------------------------------
# Load roster data and build player lookup
# ---------------------------------------------------------------------------

_ROSTER_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_rosters.json"

_PLAYERS: dict[str, dict] = {}


def _load_players() -> None:
    """Load all players from the roster file into _PLAYERS keyed by player_id."""
    if _PLAYERS:
        return
    if not _ROSTER_PATH.exists():
        return
    with open(_ROSTER_PATH) as f:
        rosters = json.load(f)
    for team_key in ("home", "away"):
        team = rosters.get(team_key, {})
        for player in team.get("lineup", []):
            _PLAYERS[player["player_id"]] = player
        for player in team.get("bench", []):
            _PLAYERS[player["player_id"]] = player
        sp = team.get("starting_pitcher")
        if sp:
            _PLAYERS[sp["player_id"]] = sp
        for player in team.get("bullpen", []):
            _PLAYERS[player["player_id"]] = player


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Valid positions
# ---------------------------------------------------------------------------

VALID_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "P"}


# ---------------------------------------------------------------------------
# Derive OAA (Outs Above Average) from fielder attributes at a position
# ---------------------------------------------------------------------------


def _derive_oaa(fielder_attrs: dict, position: str) -> float:
    """Derive an OAA-like metric from fielder attributes at a given position.

    OAA measures how many outs a fielder saves (or costs) compared to average.
    MLB range: roughly -15 to +20 per season, scaled here per ~150 games.

    Factors:
    - Range: primary driver of OAA for outfielders and middle infielders
    - Arm strength: matters more for SS, 3B, RF, CF
    - Error rate: penalty for high error rate
    - Position fit: if the player can play the position (listed in their positions)

    Returns a float representing OAA (positive = above average, negative = below).
    """
    range_val = fielder_attrs.get("range", 50)
    arm = fielder_attrs.get("arm_strength", 50)
    error_rate = fielder_attrs.get("error_rate", 0.03)
    positions = fielder_attrs.get("positions", [])

    # Base OAA from range: range 50 = 0 OAA (average), range 85 = +10, range 30 = -6
    range_oaa = (range_val - 50) * 0.30  # -15 to +15

    # Arm contribution varies by position
    arm_weight = {
        "C": 0.20,   # Catcher: arm is critical
        "SS": 0.15,  # SS: arm matters for throws across diamond
        "3B": 0.15,  # 3B: long throws to first
        "RF": 0.12,  # RF: throwing runners out at bases
        "CF": 0.10,  # CF: moderate arm importance
        "LF": 0.06,  # LF: least arm-dependent outfield spot
        "2B": 0.08,  # 2B: short throws, less arm-dependent
        "1B": 0.04,  # 1B: minimal arm requirement
        "P": 0.02,   # Pitcher fielding: minimal
        "DH": 0.00,  # DH: no fielding
    }
    arm_w = arm_weight.get(position, 0.05)
    arm_oaa = (arm - 50) * arm_w  # Scaled contribution from arm

    # Error rate penalty: MLB average ~0.03, each 0.01 above costs ~1 OAA
    error_penalty = (error_rate - 0.03) * -100  # e.g., 0.05 -> -2.0 OAA

    # Position fit penalty: if the player doesn't list this position, they're
    # out of position and take a significant penalty
    out_of_position_penalty = 0.0
    if position != "DH" and position not in positions:
        out_of_position_penalty = -5.0  # Significant penalty for playing out of position

    total_oaa = range_oaa + arm_oaa + error_penalty + out_of_position_penalty

    # DH has no defensive value
    if position == "DH":
        return 0.0

    return round(_clamp(total_oaa, -15.0, 20.0), 1)


# ---------------------------------------------------------------------------
# Derive projected wOBA from batter attributes vs a generic pitcher
# ---------------------------------------------------------------------------


def _derive_projected_woba(batter_attrs: dict) -> float:
    """Derive projected wOBA from batter attributes.

    wOBA (weighted on-base average) measures offensive production.
    MLB average wOBA is ~.315-.320.
    Range: ~.250 (poor) to ~.420 (elite).

    Derived from:
    - Contact: contributes to getting on base
    - Power: contributes to extra-base hits
    - Eye (plate discipline): contributes to walks
    - Speed: minor contribution (infield hits, avoiding double plays)
    """
    contact = batter_attrs.get("contact", 50)
    power = batter_attrs.get("power", 50)
    eye = batter_attrs.get("eye", 50)
    speed = batter_attrs.get("speed", 50)

    # Base wOBA from contact: contact 50 = ~.300
    contact_contrib = 0.250 + (contact / 100) * 0.100  # .250 to .350

    # Power adds extra-base hit value
    power_contrib = (power / 100) * 0.060  # 0 to .060

    # Eye contributes walk value
    eye_contrib = (eye / 100) * 0.030  # 0 to .030

    # Speed minor contribution (infield hits, avoiding GIDPs)
    speed_contrib = (speed / 100) * 0.010  # 0 to .010

    woba = contact_contrib + power_contrib + eye_contrib + speed_contrib

    return round(_clamp(woba, 0.220, 0.430), 3)


# ---------------------------------------------------------------------------
# Derive wRC+ (Weighted Runs Created Plus) from wOBA
# ---------------------------------------------------------------------------


def _derive_wrc_plus(woba: float) -> int:
    """Derive wRC+ from wOBA.

    wRC+ is scaled so that 100 = league average.
    Approximate conversion: wRC+ ≈ (wOBA / league_avg_wOBA) * 100

    Using .315 as league average wOBA.
    """
    league_avg_woba = 0.315
    wrc_plus = (woba / league_avg_woba) * 100
    return int(round(_clamp(wrc_plus, 40, 200)))


# ---------------------------------------------------------------------------
# Estimate innings remaining from game state
# ---------------------------------------------------------------------------


def _estimate_innings_remaining(inning: int, half: str, outs: int) -> float:
    """Estimate innings remaining in the game from current state.

    A regulation game is 9 innings. Each team bats once per inning.
    From the managed team's defensive perspective, we care about
    how many more half-innings the replacement will be in the field.

    Args:
        inning: Current inning number (1-based).
        half: "top" or "bottom" of the inning.
        outs: Current outs (0, 1, 2).

    Returns:
        Estimated defensive innings remaining (float).
    """
    if inning > 9:
        # Extra innings: assume ~1.5 more innings on average
        remaining_in_current = (3 - outs) / 3.0
        return round(remaining_in_current + 1.0, 1)

    # Full innings remaining after current inning
    full_innings_after = 9 - inning

    # Remaining fraction of current inning
    remaining_in_current = (3 - outs) / 3.0

    # If we're in the top, the current half-inning plus remaining full innings
    # If bottom, the current half-inning is offense (doesn't count for defense)
    if half == "top":
        # Defending now, plus full remaining innings
        total = remaining_in_current + full_innings_after
    else:
        # Currently batting (bottom), defense starts next inning
        total = float(full_innings_after)

    return round(max(total, 0.0), 1)


# ---------------------------------------------------------------------------
# Compute net value of the substitution
# ---------------------------------------------------------------------------


def _compute_net_value(
    defensive_upgrade_oaa: float,
    offensive_downgrade_woba: float,
    innings_remaining: float,
) -> float:
    """Compute the net expected value of the defensive substitution.

    The value combines:
    - Defensive upgrade: OAA difference scaled to remaining innings
      (OAA is per ~150 games / ~1400 innings; scale to remaining)
    - Offensive downgrade: wOBA difference converted to run value
      (each .001 wOBA ≈ .0012 runs per PA, ~3.5 PA per game)

    Both are scaled to the estimated innings remaining.

    Returns expected run value (positive = favorable substitution).
    """
    # Defensive value: OAA per full season (~1400 defensive innings)
    # Scale to innings remaining
    season_innings = 1400.0
    defensive_value = (defensive_upgrade_oaa / season_innings) * innings_remaining

    # Offensive cost: wOBA difference to runs
    # ~3.5 PA per 9 innings, so PA remaining ≈ (innings_remaining / 9) * 3.5
    pa_remaining = (innings_remaining / 9.0) * 3.5
    # Each .001 wOBA ≈ .0012 runs per PA (wOBA scale factor)
    woba_run_factor = 1.2  # runs per point of wOBA per PA
    offensive_cost = offensive_downgrade_woba * woba_run_factor * pa_remaining

    net_value = defensive_value + offensive_cost  # offensive_cost is negative when downgrade

    return round(net_value, 3)


# ---------------------------------------------------------------------------
# Recommendation from net value
# ---------------------------------------------------------------------------


def _recommendation(net_value: float) -> str:
    """Return a textual recommendation based on net expected value.

    Thresholds:
    - favorable: net value > +0.010 runs (clear defensive gain outweighs offensive cost)
    - marginal: net value between -0.010 and +0.010 (close call)
    - unfavorable: net value < -0.010 (offensive cost outweighs defensive gain)
    """
    if net_value > 0.010:
        return "favorable"
    elif net_value < -0.010:
        return "unfavorable"
    else:
        return "marginal"


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def get_defensive_replacement_value(
    current_fielder_id: str,
    replacement_id: str,
    position: str,
    inning: int = 7,
    half: str = "top",
    outs: int = 0,
) -> str:
    """Evaluates the net value of a defensive substitution by comparing the
    defensive upgrade against the offensive downgrade, scaled by estimated
    innings remaining.

    Args:
        current_fielder_id: The unique identifier of the current fielder.
        replacement_id: The unique identifier of the potential replacement.
        position: The fielding position for the substitution (e.g., 'LF', 'SS', 'CF').
        inning: Current inning number (default 7).
        half: Current half of the inning, 'top' or 'bottom' (default 'top').
        outs: Current number of outs, 0-2 (default 0).
    Returns:
        JSON string with defensive replacement evaluation including OAA upgrade,
        offensive downgrade, innings remaining, net value, and recommendation.
    """
    _load_players()
    TOOL_NAME = "get_defensive_replacement_value"

    # --- Validate current_fielder_id ---
    if current_fielder_id not in _PLAYERS:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
            f"Player '{current_fielder_id}' not found in any roster.")

    # --- Validate replacement_id ---
    if replacement_id not in _PLAYERS:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
            f"Player '{replacement_id}' not found in any roster.")

    # --- Validate position ---
    position_upper = position.upper()
    if position_upper not in VALID_POSITIONS:
        return error_response(TOOL_NAME, "INVALID_POSITION",
            f"Invalid position '{position}'. Must be one of: {', '.join(sorted(VALID_POSITIONS))}.")

    # --- Validate inning ---
    if not isinstance(inning, int) or inning < 1:
        return error_response(TOOL_NAME, "INVALID_PARAMETER",
            f"Invalid inning value: {inning}. Must be 1 or greater.")

    # --- Validate half ---
    if half not in ("top", "bottom"):
        return error_response(TOOL_NAME, "INVALID_PARAMETER",
            f"Invalid half value: '{half}'. Must be 'top' or 'bottom'.")

    # --- Validate outs ---
    if not isinstance(outs, int) or outs < 0 or outs > 2:
        return error_response(TOOL_NAME, "INVALID_PARAMETER",
            f"Invalid outs value: {outs}. Must be 0, 1, or 2.")

    current_player = _PLAYERS[current_fielder_id]
    replacement_player = _PLAYERS[replacement_id]

    # --- Validate both have fielder attributes ---
    if "fielder" not in current_player:
        return error_response(TOOL_NAME, "NO_FIELDER_DATA",
            f"Player '{current_fielder_id}' ({current_player.get('name', 'unknown')}) has no fielder attributes.")
    if "fielder" not in replacement_player:
        return error_response(TOOL_NAME, "NO_FIELDER_DATA",
            f"Player '{replacement_id}' ({replacement_player.get('name', 'unknown')}) has no fielder attributes.")

    current_fielder = current_player["fielder"]
    replacement_fielder = replacement_player["fielder"]

    # --- Compute defensive OAA for each player at this position ---
    current_oaa = _derive_oaa(current_fielder, position_upper)
    replacement_oaa = _derive_oaa(replacement_fielder, position_upper)
    defensive_upgrade_oaa = round(replacement_oaa - current_oaa, 1)

    # --- Compute offensive wOBA for each player ---
    current_batter = current_player.get("batter")
    replacement_batter = replacement_player.get("batter")

    # If a player has no batter attributes (e.g., pitcher), assign minimal offense
    if current_batter:
        current_woba = _derive_projected_woba(current_batter)
        current_wrc_plus = _derive_wrc_plus(current_woba)
    else:
        current_woba = 0.220
        current_wrc_plus = _derive_wrc_plus(current_woba)

    if replacement_batter:
        replacement_woba = _derive_projected_woba(replacement_batter)
        replacement_wrc_plus = _derive_wrc_plus(replacement_woba)
    else:
        replacement_woba = 0.220
        replacement_wrc_plus = _derive_wrc_plus(replacement_woba)

    # Offensive downgrade: negative means the replacement is worse offensively
    offensive_downgrade_woba = round(replacement_woba - current_woba, 3)
    offensive_downgrade_wrc_plus = replacement_wrc_plus - current_wrc_plus

    # --- Estimate innings remaining ---
    innings_remaining = _estimate_innings_remaining(inning, half, outs)

    # --- Compute net expected value ---
    net_value = _compute_net_value(
        defensive_upgrade_oaa,
        offensive_downgrade_woba,
        innings_remaining,
    )

    # --- Recommendation ---
    rec = _recommendation(net_value)

    return success_response(TOOL_NAME, {
        "current_fielder": {
            "player_id": current_fielder_id,
            "name": current_player.get("name", "Unknown"),
            "oaa_at_position": current_oaa,
            "projected_woba": current_woba,
            "projected_wrc_plus": current_wrc_plus,
            "positions": current_fielder.get("positions", []),
        },
        "replacement": {
            "player_id": replacement_id,
            "name": replacement_player.get("name", "Unknown"),
            "oaa_at_position": replacement_oaa,
            "projected_woba": replacement_woba,
            "projected_wrc_plus": replacement_wrc_plus,
            "positions": replacement_fielder.get("positions", []),
        },
        "position": position_upper,
        "defensive_upgrade_oaa": defensive_upgrade_oaa,
        "offensive_downgrade_woba": offensive_downgrade_woba,
        "offensive_downgrade_wrc_plus": offensive_downgrade_wrc_plus,
        "estimated_innings_remaining": innings_remaining,
        "net_expected_value": net_value,
        "recommendation": rec,
    })
