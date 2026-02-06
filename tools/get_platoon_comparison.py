# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Compares a potential pinch hitter against the current batter.

Loads player data from sample_rosters.json and derives projected wOBA for each
batter against the current pitcher (factoring in platoon splits), the platoon
advantage delta, defensive cost of the substitution, and bench depth impact.
"""

import json
from pathlib import Path

from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Load roster data and build player + team lookups
# ---------------------------------------------------------------------------

_ROSTER_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_rosters.json"

_PLAYERS: dict[str, dict] = {}
# Maps player_id -> list of team bench player_ids (for bench depth calculation)
_TEAM_BENCH: dict[str, list[str]] = {}
# Maps player_id -> team key ("home" or "away")
_PLAYER_TEAM: dict[str, str] = {}


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
        bench_ids: list[str] = []
        all_team_ids: list[str] = []

        for player in team.get("lineup", []):
            _PLAYERS[player["player_id"]] = player
            _PLAYER_TEAM[player["player_id"]] = team_key
            all_team_ids.append(player["player_id"])
        for player in team.get("bench", []):
            _PLAYERS[player["player_id"]] = player
            _PLAYER_TEAM[player["player_id"]] = team_key
            bench_ids.append(player["player_id"])
            all_team_ids.append(player["player_id"])
        sp = team.get("starting_pitcher")
        if sp:
            _PLAYERS[sp["player_id"]] = sp
            _PLAYER_TEAM[sp["player_id"]] = team_key
            all_team_ids.append(sp["player_id"])
        for player in team.get("bullpen", []):
            _PLAYERS[player["player_id"]] = player
            _PLAYER_TEAM[player["player_id"]] = team_key
            all_team_ids.append(player["player_id"])

        # Store bench ids for each player on this team
        for pid in all_team_ids:
            _TEAM_BENCH[pid] = bench_ids


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Derive projected wOBA vs a specific pitcher handedness
# ---------------------------------------------------------------------------


def _derive_projected_woba_vs_pitcher(batter_attrs: dict, batter_hand: str,
                                       pitcher_hand: str) -> float:
    """Derive projected wOBA for a batter against a pitcher of given handedness.

    Uses the batter's attributes (contact, power, eye, speed) as the base,
    then applies a platoon adjustment based on batter vs pitcher handedness.

    Platoon splits in MLB:
    - Opposite-hand matchup (L vs R or R vs L): ~+0.015-0.020 wOBA advantage
    - Same-hand matchup (L vs L or R vs R): ~-0.015-0.020 wOBA disadvantage
    - Switch hitters: always bat from the opposite side, getting a smaller advantage

    Args:
        batter_attrs: The batter's attribute dict (contact, power, speed, eye, avg_vs_l, avg_vs_r).
        batter_hand: The batter's handedness ("L", "R", or "S").
        pitcher_hand: The pitcher's throwing hand ("L" or "R").

    Returns:
        Projected wOBA as a float.
    """
    contact = batter_attrs.get("contact", 50)
    power = batter_attrs.get("power", 50)
    eye = batter_attrs.get("eye", 50)
    speed = batter_attrs.get("speed", 50)

    # Base wOBA from attributes (same formula as get_defensive_replacement_value)
    contact_contrib = 0.250 + (contact / 100) * 0.100  # .250 to .350
    power_contrib = (power / 100) * 0.060  # 0 to .060
    eye_contrib = (eye / 100) * 0.030  # 0 to .030
    speed_contrib = (speed / 100) * 0.010  # 0 to .010
    base_woba = contact_contrib + power_contrib + eye_contrib + speed_contrib

    # Apply platoon adjustment
    # Use avg_vs_l and avg_vs_r to derive a more player-specific platoon effect
    avg_vs_l = batter_attrs.get("avg_vs_l", 0.250)
    avg_vs_r = batter_attrs.get("avg_vs_r", 0.250)

    # The difference between avg_vs_l and avg_vs_r tells us this batter's
    # personal platoon split magnitude
    if pitcher_hand == "L":
        # Batter facing LHP
        if batter_hand == "R":
            # Opposite-hand: favorable platoon
            platoon_adj = +0.018
            # Enhance with personal split: if batter hits LHP better, add more
            personal_adj = (avg_vs_l - avg_vs_r) * 0.15
        elif batter_hand == "L":
            # Same-hand: unfavorable platoon
            platoon_adj = -0.018
            personal_adj = (avg_vs_l - avg_vs_r) * 0.15
        else:
            # Switch hitter: bats right vs LHP, slight advantage
            platoon_adj = +0.008
            personal_adj = (avg_vs_l - avg_vs_r) * 0.10
    else:
        # Batter facing RHP
        if batter_hand == "L":
            # Opposite-hand: favorable platoon
            platoon_adj = +0.018
            personal_adj = (avg_vs_r - avg_vs_l) * 0.15
        elif batter_hand == "R":
            # Same-hand: unfavorable platoon
            platoon_adj = -0.018
            personal_adj = (avg_vs_r - avg_vs_l) * 0.15
        else:
            # Switch hitter: bats left vs RHP, slight advantage
            platoon_adj = +0.008
            personal_adj = (avg_vs_r - avg_vs_l) * 0.10

    woba = base_woba + platoon_adj + personal_adj

    return round(_clamp(woba, 0.220, 0.430), 3)


# ---------------------------------------------------------------------------
# Derive wRC+ from wOBA
# ---------------------------------------------------------------------------


def _derive_wrc_plus(woba: float) -> int:
    """Derive wRC+ from wOBA. 100 = league average."""
    league_avg_woba = 0.315
    wrc_plus = (woba / league_avg_woba) * 100
    return int(round(_clamp(wrc_plus, 40, 200)))


# ---------------------------------------------------------------------------
# Derive OAA (defensive value) from fielder attributes at a position
# ---------------------------------------------------------------------------


def _derive_oaa(fielder_attrs: dict, position: str) -> float:
    """Derive an OAA-like metric from fielder attributes at a given position.

    Mirrors the logic in get_defensive_replacement_value.
    """
    range_val = fielder_attrs.get("range", 50)
    arm = fielder_attrs.get("arm_strength", 50)
    error_rate = fielder_attrs.get("error_rate", 0.03)
    positions = fielder_attrs.get("positions", [])

    range_oaa = (range_val - 50) * 0.30

    arm_weight = {
        "C": 0.20, "SS": 0.15, "3B": 0.15, "RF": 0.12, "CF": 0.10,
        "LF": 0.06, "2B": 0.08, "1B": 0.04, "P": 0.02, "DH": 0.00,
    }
    arm_w = arm_weight.get(position, 0.05)
    arm_oaa = (arm - 50) * arm_w

    error_penalty = (error_rate - 0.03) * -100

    out_of_position_penalty = 0.0
    if position != "DH" and position not in positions:
        out_of_position_penalty = -5.0

    total_oaa = range_oaa + arm_oaa + error_penalty + out_of_position_penalty

    if position == "DH":
        return 0.0

    return round(_clamp(total_oaa, -15.0, 20.0), 1)


# ---------------------------------------------------------------------------
# Compute defensive cost of substitution
# ---------------------------------------------------------------------------


def _compute_defensive_cost(current_player: dict, pinch_hitter: dict,
                             position: str) -> dict:
    """Compute the defensive cost of replacing the current batter's position
    with the pinch hitter.

    Returns a dict with OAA values and the defensive cost in wOBA-equivalent.
    """
    current_fielder = current_player.get("fielder")
    pinch_fielder = pinch_hitter.get("fielder")

    # If current player is DH, no defensive cost
    if position == "DH":
        return {
            "current_oaa": 0.0,
            "replacement_oaa": 0.0,
            "oaa_difference": 0.0,
            "defensive_cost_woba": 0.0,
            "position": "DH",
            "notes": "No defensive cost for DH substitution.",
        }

    # If current player has no fielder data, their defense is minimal
    if current_fielder:
        current_oaa = _derive_oaa(current_fielder, position)
    else:
        current_oaa = -5.0  # No fielding data = poor defender

    if pinch_fielder:
        replacement_oaa = _derive_oaa(pinch_fielder, position)
    else:
        replacement_oaa = -5.0

    oaa_diff = round(replacement_oaa - current_oaa, 1)

    # Convert OAA difference to approximate wOBA-equivalent cost
    # 1 OAA over a season (~1400 innings) ≈ 0.5 runs
    # For the rest of a game (~3 innings), scale down
    # Rough conversion: 1 OAA difference ≈ 0.002 wOBA-equivalent per game remaining
    defensive_cost_woba = round(oaa_diff * 0.002, 3)

    # Check if pinch hitter can play the position
    can_play = position in pinch_fielder.get("positions", []) if pinch_fielder else False
    notes = ""
    if not can_play and position != "DH":
        notes = f"Pinch hitter is out of position at {position}."

    return {
        "current_oaa": current_oaa,
        "replacement_oaa": replacement_oaa,
        "oaa_difference": oaa_diff,
        "defensive_cost_woba": defensive_cost_woba,
        "position": position,
        "pinch_hitter_can_play_position": can_play,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Compute bench depth impact
# ---------------------------------------------------------------------------


def _compute_bench_depth_impact(pinch_hitter_id: str, team_key: str) -> dict:
    """Compute how using this pinch hitter affects remaining bench depth.

    Returns info about bench players remaining, including by handedness.
    """
    bench_ids = _TEAM_BENCH.get(pinch_hitter_id, [])

    # Remaining bench after using this pinch hitter
    remaining_ids = [pid for pid in bench_ids if pid != pinch_hitter_id]
    remaining_players = []
    left_batters = 0
    right_batters = 0
    switch_batters = 0

    for pid in remaining_ids:
        p = _PLAYERS.get(pid)
        if p:
            hand = p.get("bats", "R")
            if hand == "L":
                left_batters += 1
            elif hand == "R":
                right_batters += 1
            else:
                switch_batters += 1
            remaining_players.append({
                "player_id": pid,
                "name": p.get("name", "Unknown"),
                "bats": hand,
                "positions": p.get("fielder", {}).get("positions", []),
            })

    return {
        "bench_remaining": len(remaining_ids),
        "left_handed_remaining": left_batters,
        "right_handed_remaining": right_batters,
        "switch_remaining": switch_batters,
        "remaining_bench_players": remaining_players,
    }


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def get_platoon_comparison(
    current_batter_id: str,
    pinch_hitter_id: str,
    pitcher_id: str,
) -> str:
    """Compares a potential pinch hitter against the current batter for the
    active matchup. Returns projected wOBA for each, platoon advantage delta,
    defensive cost, and bench depth impact.

    Args:
        current_batter_id: The unique identifier of the current batter.
        pinch_hitter_id: The unique identifier of the potential pinch hitter.
        pitcher_id: The unique identifier of the current pitcher for matchup context.
    Returns:
        JSON string with platoon comparison.
    """
    _load_players()

    # --- Validate current_batter_id ---
    if current_batter_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{current_batter_id}' not found in any roster.",
        })

    # --- Validate pinch_hitter_id ---
    if pinch_hitter_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{pinch_hitter_id}' not found in any roster.",
        })

    # --- Validate pitcher_id ---
    if pitcher_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{pitcher_id}' not found in any roster.",
        })

    current_player = _PLAYERS[current_batter_id]
    pinch_hitter_player = _PLAYERS[pinch_hitter_id]
    pitcher_player = _PLAYERS[pitcher_id]

    # --- Validate pitcher has pitcher data to get throw hand ---
    pitcher_data = pitcher_player.get("pitcher")
    if not pitcher_data:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_PITCHER",
            "message": f"Player '{pitcher_id}' ({pitcher_player.get('name', 'unknown')}) does not have pitcher attributes.",
        })

    pitcher_hand = pitcher_player.get("throws", "R")

    # --- Get batter attributes ---
    current_batter = current_player.get("batter")
    pinch_batter = pinch_hitter_player.get("batter")

    # Handedness
    current_hand = current_player.get("bats", "R")
    pinch_hand = pinch_hitter_player.get("bats", "R")

    # --- Derive projected wOBA for each batter vs this pitcher ---
    if current_batter:
        current_woba = _derive_projected_woba_vs_pitcher(
            current_batter, current_hand, pitcher_hand,
        )
        current_wrc_plus = _derive_wrc_plus(current_woba)
    else:
        current_woba = 0.220
        current_wrc_plus = _derive_wrc_plus(current_woba)

    if pinch_batter:
        pinch_woba = _derive_projected_woba_vs_pitcher(
            pinch_batter, pinch_hand, pitcher_hand,
        )
        pinch_wrc_plus = _derive_wrc_plus(pinch_woba)
    else:
        pinch_woba = 0.220
        pinch_wrc_plus = _derive_wrc_plus(pinch_woba)

    # --- Platoon advantage delta ---
    platoon_delta = round(pinch_woba - current_woba, 3)

    # --- Defensive cost ---
    # Determine the current batter's fielding position
    current_position = current_player.get("primary_position", "DH")
    # Normalize position names
    if current_position in ("SP", "RP"):
        current_position = "DH"  # Pitcher batting = DH-like role
    defensive_cost = _compute_defensive_cost(
        current_player, pinch_hitter_player, current_position,
    )

    # --- Bench depth impact ---
    pinch_team = _PLAYER_TEAM.get(pinch_hitter_id, "home")
    bench_impact = _compute_bench_depth_impact(pinch_hitter_id, pinch_team)

    return json.dumps({
        "status": "ok",
        "current_batter": {
            "player_id": current_batter_id,
            "name": current_player.get("name", "Unknown"),
            "bats": current_hand,
            "projected_woba_vs_pitcher": current_woba,
            "projected_wrc_plus": current_wrc_plus,
        },
        "pinch_hitter": {
            "player_id": pinch_hitter_id,
            "name": pinch_hitter_player.get("name", "Unknown"),
            "bats": pinch_hand,
            "projected_woba_vs_pitcher": pinch_woba,
            "projected_wrc_plus": pinch_wrc_plus,
        },
        "pitcher": {
            "player_id": pitcher_id,
            "name": pitcher_player.get("name", "Unknown"),
            "throws": pitcher_hand,
        },
        "platoon_advantage_delta": platoon_delta,
        "defensive_cost": defensive_cost,
        "bench_depth_impact": bench_impact,
    })
