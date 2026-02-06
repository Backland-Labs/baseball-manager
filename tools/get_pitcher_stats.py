# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Retrieves pitching statistics for a pitcher."""

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
# Stat derivation from pitcher attributes
# ---------------------------------------------------------------------------

def _derive_stats(player: dict, vs_hand: Optional[str], home_away: Optional[str],
                  recency_window: Optional[str]) -> dict:
    """Derive realistic pitching statistics from player attributes.

    Player attributes used:
        stuff (0-100): drives K%, whiff rate, swinging strike rate, pitch quality
        control (0-100): drives BB%, FIP - ERA gap, SIERA
        stamina (0-100): affects times-through-order degradation
        velocity (float): base fastball velocity in mph
        era_vs_l, era_vs_r (float): ERA against left/right-handed batters
    """
    pitcher = player.get("pitcher")
    if not pitcher:
        return {}

    stuff = pitcher["stuff"]
    control = pitcher["control"]
    stamina = pitcher["stamina"]
    velocity = pitcher["velocity"]
    era_vs_l = pitcher["era_vs_l"]
    era_vs_r = pitcher["era_vs_r"]

    # Determine base ERA from split data
    if vs_hand == "L":
        base_era = era_vs_l
    elif vs_hand == "R":
        base_era = era_vs_r
    else:
        # Season ERA weighted ~55% vs RHB, ~45% vs LHB (typical MLB distribution)
        base_era = era_vs_r * 0.55 + era_vs_l * 0.45

    # Home/away adjustment (home pitchers get ~-0.20 ERA historically)
    if home_away == "home":
        base_era = _clamp(base_era - 0.20, 1.50, 7.00)
    elif home_away == "away":
        base_era = _clamp(base_era + 0.20, 1.50, 7.00)

    # Recency window adds slight variance
    recency_factor = 1.0
    if recency_window == "last_7":
        recency_factor = 0.95  # slightly better in recent outings (stub bias)
    elif recency_window == "last_14":
        recency_factor = 0.98
    elif recency_window == "last_30":
        recency_factor = 0.99
    # "season" or None = no adjustment

    era = _clamp(round(base_era * recency_factor, 2), 1.50, 7.00)

    # FIP: driven by stuff and control. Better stuff = more K, fewer HR.
    # Better control = fewer BB. FIP tends to be close to ERA but skill-based.
    # FIP = ((13*HR + 3*BB - 2*K) / IP) + cFIP (~3.10)
    # We approximate: high stuff lowers FIP, high control lowers FIP
    fip_base = 5.50 - (stuff / 100) * 2.50 - (control / 100) * 1.50
    fip = _clamp(round(fip_base * recency_factor, 2), 1.50, 6.50)

    # xFIP: same as FIP but normalizes HR/FB rate to league average
    # Tends to regress FIP toward the mean slightly
    xfip = _clamp(round(fip * 0.90 + 3.80 * 0.10, 2), 1.80, 6.00)

    # SIERA: Skill-Interactive ERA -- heavily weights K and BB rates plus GB%
    # Strong correlation with control and stuff
    siera = _clamp(round((fip + era) / 2 * 0.95 + 0.15, 2), 1.80, 6.00)

    # K% derived from stuff (higher stuff = more strikeouts)
    # MLB avg K% ~22%. Stuff 50 = average.
    k_pct = _clamp(round(0.10 + (stuff / 100) * 0.22, 3), 0.08, 0.40)

    # BB% derived from control (higher control = fewer walks)
    # MLB avg BB% ~8.5%. Control 50 = average.
    bb_pct = _clamp(round(0.15 - (control / 100) * 0.10, 3), 0.03, 0.15)

    # Ground ball rate: inversely related to stuff/velocity for power pitchers
    # Sinker/groundball pitchers have lower stuff but higher GB%.
    # We model moderate stuff as higher GB%, high stuff as more flyball/whiffs
    gb_pct = _clamp(round(0.55 - (stuff / 100) * 0.20 + (control / 100) * 0.10, 3), 0.25, 0.60)
    fb_pct = _clamp(round(1.0 - gb_pct - 0.20, 3), 0.20, 0.45)
    ld_pct = _clamp(round(1.0 - gb_pct - fb_pct, 3), 0.15, 0.25)

    # Pitch mix: derived from velocity and stuff
    # Higher velocity = more fastball reliance, higher stuff = better secondary pitches
    pitch_mix = _build_pitch_mix(velocity, stuff, control)

    # Times-through-order wOBA splits
    # Higher stamina = less degradation through the order
    # Base wOBA ~ derived from ERA: wOBA ~= 0.200 + ERA * 0.025
    base_woba = _clamp(round(0.200 + era * 0.025, 3), 0.250, 0.400)
    # TTO penalty decreases with stamina
    tto_penalty_2nd = _clamp(round(0.015 * (1.0 - stamina / 200), 3), 0.005, 0.020)
    tto_penalty_3rd = _clamp(round(0.035 * (1.0 - stamina / 200), 3), 0.015, 0.050)

    tto = {
        "1st": base_woba,
        "2nd": _clamp(round(base_woba + tto_penalty_2nd, 3), 0.250, 0.420),
        "3rd_plus": _clamp(round(base_woba + tto_penalty_3rd, 3), 0.260, 0.450),
    }

    return {
        "traditional": {
            "ERA": era,
            "FIP": fip,
            "xFIP": xfip,
            "SIERA": siera,
        },
        "rates": {
            "K_pct": k_pct,
            "BB_pct": bb_pct,
        },
        "batted_ball": {
            "GB_pct": gb_pct,
            "FB_pct": fb_pct,
            "LD_pct": ld_pct,
        },
        "pitch_mix": pitch_mix,
        "times_through_order": tto,
    }


def _build_pitch_mix(velocity: float, stuff: int, control: int) -> list[dict]:
    """Build a realistic pitch mix from pitcher attributes.

    Pitch selection logic:
    - All pitchers have a fastball (4-seam or sinker)
    - Higher stuff pitchers have better secondary pitch quality
    - Pitch count varies: most have 3-4 pitch types
    - Velocity scales for off-speed pitches relative to fastball
    """
    pitches = []

    # Fastball (4-seam): always present
    ff_usage = _clamp(round(0.65 - (stuff / 100) * 0.20, 2), 0.35, 0.65)
    ff_spin = round(2000 + (velocity - 88) * 40 + (stuff - 50) * 5, 0)
    ff_whiff = _clamp(round(0.12 + (stuff / 100) * 0.14, 3), 0.10, 0.30)
    pitches.append({
        "pitch_type": "FF",
        "usage": ff_usage,
        "velocity": round(velocity, 1),
        "spin_rate": int(ff_spin),
        "whiff_rate": ff_whiff,
    })

    remaining_usage = round(1.0 - ff_usage, 2)

    # Slider: present for most pitchers, especially high-stuff ones
    sl_usage = _clamp(round(remaining_usage * 0.45, 2), 0.10, 0.30)
    sl_velocity = round(velocity - 7.5 - (100 - stuff) * 0.03, 1)
    sl_spin = round(2300 + (stuff - 50) * 8, 0)
    sl_whiff = _clamp(round(0.20 + (stuff / 100) * 0.20, 3), 0.18, 0.45)
    pitches.append({
        "pitch_type": "SL",
        "usage": sl_usage,
        "velocity": sl_velocity,
        "spin_rate": int(sl_spin),
        "whiff_rate": sl_whiff,
    })

    remaining_usage = round(remaining_usage - sl_usage, 2)

    # Changeup: usage depends on remaining allocation
    ch_usage = _clamp(round(remaining_usage * 0.65, 2), 0.08, 0.25)
    ch_velocity = round(velocity - 9.0 - (100 - stuff) * 0.04, 1)
    ch_spin = round(1600 + (control - 50) * 4, 0)
    ch_whiff = _clamp(round(0.18 + (stuff / 100) * 0.16, 3), 0.15, 0.38)
    pitches.append({
        "pitch_type": "CH",
        "usage": ch_usage,
        "velocity": ch_velocity,
        "spin_rate": int(ch_spin),
        "whiff_rate": ch_whiff,
    })

    # Curveball: only if there's enough remaining usage and the pitcher has decent stuff
    remaining_usage = round(remaining_usage - ch_usage, 2)
    if remaining_usage >= 0.05 and stuff >= 60:
        cb_velocity = round(velocity - 14.0 - (100 - stuff) * 0.05, 1)
        cb_spin = round(2500 + (stuff - 50) * 10, 0)
        cb_whiff = _clamp(round(0.22 + (stuff / 100) * 0.15, 3), 0.18, 0.40)
        pitches.append({
            "pitch_type": "CU",
            "usage": remaining_usage,
            "velocity": cb_velocity,
            "spin_rate": int(cb_spin),
            "whiff_rate": cb_whiff,
        })

    return pitches


# ---------------------------------------------------------------------------
# Module-level game state for tracking today's performance
# ---------------------------------------------------------------------------

# Mutable dict keyed by player_id, tracking in-game pitching lines.
# Populated externally by the simulation engine as events occur.
_today_lines: dict[str, dict] = {}


def set_today_line(player_id: str, line: dict) -> None:
    """Update a pitcher's current-game pitching line (called by simulation engine)."""
    _today_lines[player_id] = line


def reset_today_lines() -> None:
    """Reset all in-game pitching lines (called at game start)."""
    _today_lines.clear()


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@beta_tool
def get_pitcher_stats(
    player_id: str,
    vs_hand: Optional[str] = None,
    home_away: Optional[str] = None,
    recency_window: Optional[str] = None,
) -> str:
    """Retrieves pitching statistics for a pitcher, including ERA/FIP/xFIP,
    strikeout and walk rates, ground ball rate, pitch mix with per-pitch metrics,
    and times-through-order splits.

    Args:
        player_id: The unique identifier of the pitcher.
        vs_hand: Optional split by batter handedness ('L' or 'R').
        home_away: Optional split by venue ('home' or 'away').
        recency_window: Optional recency filter ('last_7', 'last_14', 'last_30', 'season').
    Returns:
        JSON string with pitching statistics.
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

    # Check if the player has pitcher attributes
    if "pitcher" not in player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_PITCHER",
            "message": f"Player '{player_id}' ({player.get('name', 'unknown')}) does not have pitching attributes.",
        })

    # Derive stats from player attributes
    stats = _derive_stats(player, vs_hand, home_away, recency_window)

    # Get today's line (default to zeros if not tracked yet)
    today = _today_lines.get(player_id, {
        "IP": 0.0, "H": 0, "R": 0, "ER": 0, "BB": 0, "K": 0,
    })

    return json.dumps({
        "status": "ok",
        "player_id": player_id,
        "player_name": player.get("name", "Unknown"),
        "throws": player.get("throws", "R"),
        "splits": {
            "vs_hand": vs_hand,
            "home_away": home_away,
            "recency_window": recency_window,
        },
        **stats,
        "today": today,
    })
