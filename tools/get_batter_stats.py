# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Retrieves batting statistics for a player."""

import json
from pathlib import Path
from typing import Optional

from anthropic import beta_tool

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
# Stat derivation from player attributes
# ---------------------------------------------------------------------------

def _derive_stats(player: dict, vs_hand: Optional[str], home_away: Optional[str],
                  recency_window: Optional[str]) -> dict:
    """Derive realistic batting statistics from player attributes.

    Player attributes used:
        contact (0-100): drives AVG, K%, whiff rate
        power (0-100): drives SLG, barrel rate, exit velocity, launch angle
        speed (0-100): drives sprint speed, infield hit contribution
        eye (0-100): drives BB%, chase rate, OBP premium
        avg_vs_l, avg_vs_r (0-1): batting averages vs L/R pitchers
    """
    batter = player.get("batter")
    if not batter:
        return {}

    contact = batter["contact"]
    power = batter["power"]
    speed = batter["speed"]
    eye = batter["eye"]
    avg_vs_l = batter["avg_vs_l"]
    avg_vs_r = batter["avg_vs_r"]

    # Determine base batting average
    if vs_hand == "L":
        base_avg = avg_vs_l
    elif vs_hand == "R":
        base_avg = avg_vs_r
    else:
        # Season avg is weighted ~60% vs RHP, ~40% vs LHP (typical MLB distribution)
        base_avg = avg_vs_r * 0.60 + avg_vs_l * 0.40

    # Home/away adjustment (home hitters get ~+.010 historically)
    if home_away == "home":
        base_avg = _clamp(base_avg + 0.010, 0.100, 0.400)
    elif home_away == "away":
        base_avg = _clamp(base_avg - 0.010, 0.100, 0.400)

    # Recency window adds slight variance to simulate hot/cold streaks
    recency_factor = 1.0
    if recency_window == "last_7":
        recency_factor = 1.05  # small hot streak bias for stub
    elif recency_window == "last_14":
        recency_factor = 1.02
    elif recency_window == "last_30":
        recency_factor = 1.01
    # "season" or None = no adjustment

    avg = _clamp(round(base_avg * recency_factor, 3), 0.100, 0.400)

    # BB% derived from eye (higher eye = more walks)
    # MLB avg BB% ~8.5%. Eye 50 = average, scale linearly.
    bb_pct = _clamp(round(0.04 + (eye / 100) * 0.12, 3), 0.02, 0.20)

    # OBP = AVG + walk contribution + HBP approximation
    obp = _clamp(round(avg + bb_pct * 0.9 + 0.010, 3), 0.200, 0.500)

    # SLG derived from AVG and power. Power increases ISO (SLG - AVG).
    # MLB avg ISO ~.150. Power 50 = avg, scale.
    iso = _clamp(round(0.050 + (power / 100) * 0.250, 3), 0.030, 0.350)
    slg = _clamp(round(avg + iso, 3), 0.250, 0.750)

    ops = round(obp + slg, 3)

    # Advanced metrics
    # wOBA approximation: weighted combination of OBP and SLG
    woba = _clamp(round(obp * 0.69 + slg * 0.21 + 0.030, 3), 0.200, 0.450)

    # wRC+ = 100 is league average. Derive from wOBA.
    # League avg wOBA ~.310. wRC+ scales roughly linearly.
    league_woba = 0.310
    wrc_plus = _clamp(round(100 * (woba / league_woba), 0), 40, 200)

    # Barrel rate: power-driven (MLB avg ~6.5%)
    barrel_rate = _clamp(round(0.02 + (power / 100) * 0.12, 3), 0.01, 0.18)

    # xwOBA: slightly regressed version of wOBA toward league average
    xwoba = _clamp(round(woba * 0.90 + league_woba * 0.10, 3), 0.200, 0.440)

    # Plate discipline
    # K% inversely related to contact (higher contact = lower K%)
    # MLB avg K% ~22.5%. Contact 50 = average.
    k_pct = _clamp(round(0.35 - (contact / 100) * 0.25, 3), 0.08, 0.40)

    # Chase rate: inversely related to eye (MLB avg ~28%)
    chase_rate = _clamp(round(0.42 - (eye / 100) * 0.28, 3), 0.15, 0.40)

    # Whiff rate: combines contact and eye (MLB avg ~25%)
    whiff_rate = _clamp(round(0.38 - (contact / 100) * 0.18 - (eye / 100) * 0.05, 3), 0.10, 0.40)

    # Batted ball profile
    # GB% inversely related to power/launch angle (MLB avg ~43%)
    gb_pct = _clamp(round(0.55 - (power / 100) * 0.25, 3), 0.25, 0.60)

    # Pull%: power hitters pull more (MLB avg ~40%)
    pull_pct = _clamp(round(0.30 + (power / 100) * 0.18, 3), 0.28, 0.55)

    # Exit velocity: power-driven (MLB avg ~88.5 mph)
    exit_velocity = _clamp(round(82.0 + (power / 100) * 14.0, 1), 82.0, 97.0)

    # Launch angle: power hitters have higher LA (MLB avg ~12 degrees)
    launch_angle = _clamp(round(6.0 + (power / 100) * 14.0, 1), 4.0, 22.0)

    # Sprint speed: speed-driven (MLB avg ~27.0 ft/s, range 23-31)
    sprint_speed = _clamp(round(23.0 + (speed / 100) * 8.0, 1), 23.0, 31.0)

    # Situational stats (derived with small adjustments from base avg)
    risp_avg = _clamp(round(avg + (eye - 50) / 1000, 3), 0.100, 0.400)
    high_leverage_ops = _clamp(round(ops + (contact - 50) / 500, 3), 0.400, 1.200)
    late_close_ops = _clamp(round(ops - 0.010, 3), 0.400, 1.200)

    return {
        "traditional": {
            "AVG": avg,
            "OBP": obp,
            "SLG": slg,
            "OPS": ops,
        },
        "advanced": {
            "wOBA": woba,
            "wRC_plus": int(wrc_plus),
            "barrel_rate": barrel_rate,
            "xwOBA": xwoba,
        },
        "plate_discipline": {
            "K_pct": k_pct,
            "BB_pct": bb_pct,
            "chase_rate": chase_rate,
            "whiff_rate": whiff_rate,
        },
        "batted_ball": {
            "GB_pct": gb_pct,
            "pull_pct": pull_pct,
            "exit_velocity": exit_velocity,
            "launch_angle": launch_angle,
        },
        "sprint_speed": sprint_speed,
        "situational": {
            "RISP_avg": risp_avg,
            "high_leverage_ops": high_leverage_ops,
            "late_and_close_ops": late_close_ops,
        },
    }


# ---------------------------------------------------------------------------
# Module-level game state for tracking today's performance
# ---------------------------------------------------------------------------

# Mutable dict keyed by player_id, tracking in-game batting lines.
# Populated externally by the simulation engine as events occur.
_today_lines: dict[str, dict] = {}


def set_today_line(player_id: str, line: dict) -> None:
    """Update a player's current-game batting line (called by simulation engine)."""
    _today_lines[player_id] = line


def reset_today_lines() -> None:
    """Reset all in-game batting lines (called at game start)."""
    _today_lines.clear()


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@beta_tool
def get_batter_stats(
    player_id: str,
    vs_hand: Optional[str] = None,
    home_away: Optional[str] = None,
    recency_window: Optional[str] = None,
) -> str:
    """Retrieves batting statistics for a player, including traditional stats,
    advanced metrics, plate discipline, batted ball profile, and sprint speed.
    Supports splits by handedness, home/away, and recency windows.

    Args:
        player_id: The unique identifier of the batter.
        vs_hand: Optional split by pitcher handedness ('L' or 'R').
        home_away: Optional split by venue ('home' or 'away').
        recency_window: Optional recency filter ('last_7', 'last_14', 'last_30', 'season').
    Returns:
        JSON string with batting statistics.
    """
    _load_players()

    # Validate player_id
    if player_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{player_id}' not found in any roster.",
        })

    player = _PLAYERS[player_id]

    # Validate optional parameters
    if vs_hand is not None and vs_hand not in ("L", "R"):
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid vs_hand value '{vs_hand}'. Must be 'L' or 'R'.",
        })

    if home_away is not None and home_away not in ("home", "away"):
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid home_away value '{home_away}'. Must be 'home' or 'away'.",
        })

    valid_recency = ("last_7", "last_14", "last_30", "season")
    if recency_window is not None and recency_window not in valid_recency:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid recency_window value '{recency_window}'. Must be one of {valid_recency}.",
        })

    # Check if the player has batter attributes
    if "batter" not in player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_BATTER",
            "message": f"Player '{player_id}' ({player.get('name', 'unknown')}) does not have batting attributes.",
        })

    # Derive stats from player attributes
    stats = _derive_stats(player, vs_hand, home_away, recency_window)

    # Get today's line (default to zeros if not tracked yet)
    today = _today_lines.get(player_id, {
        "AB": 0, "H": 0, "BB": 0, "K": 0, "RBI": 0,
    })

    return json.dumps({
        "status": "ok",
        "player_id": player_id,
        "player_name": player.get("name", "Unknown"),
        "bats": player.get("bats", "R"),
        "splits": {
            "vs_hand": vs_hand,
            "home_away": home_away,
            "recency_window": recency_window,
        },
        **stats,
        "today": today,
    })
