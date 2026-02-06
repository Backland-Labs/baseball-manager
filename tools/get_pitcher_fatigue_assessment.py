# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Assesses the current pitcher's fatigue based on in-game trends.

Loads pitcher data from sample_rosters.json and derives fatigue indicators
from pitcher attributes, pitch count, times through order, and innings pitched.
Models velocity decline, spin rate decline, batted ball quality degradation,
and an overall fatigue rating (fresh, normal, fatigued, gassed).
"""

import json
from pathlib import Path
from typing import Optional

from anthropic import beta_tool

from tools.response import success_response, error_response, player_ref

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
# Derive in-game pitch count distribution by inning
# ---------------------------------------------------------------------------


def _derive_pitch_counts_by_inning(
    total_pitch_count: int,
    innings_pitched: float,
    stamina: float,
) -> list[int]:
    """Derive a plausible per-inning pitch distribution from total pitch count.

    Pitchers tend to throw more pitches in later innings as they tire and
    face more hitters. Stamina affects how evenly distributed pitches are.

    Args:
        total_pitch_count: Total pitches thrown so far in the game.
        innings_pitched: Innings completed (e.g., 5.0 for 5 full innings, 5.1 for 5 and 1 out).
        stamina: Pitcher's stamina attribute (0-100).

    Returns:
        List of pitch counts per inning (length = number of innings started).
    """
    if total_pitch_count <= 0:
        return []

    # Parse innings: 5.1 means 5 full + 1 out into 6th, so pitcher is in 6th inning
    full_innings = int(innings_pitched)
    partial_outs = round((innings_pitched - full_innings) * 10)

    # Number of innings started (including current partial inning)
    innings_started = full_innings + (1 if partial_outs > 0 else 0)
    if innings_started == 0:
        # Pitcher has entered but hasn't recorded an out yet; still in 1st inning of work
        innings_started = 1

    if innings_started == 1:
        return [total_pitch_count]

    # Distribute pitches with a rising trend (pitchers throw slightly more
    # per inning as they go deeper due to fatigue and longer at-bats).
    # High-stamina pitchers have flatter distributions.
    # Fatigue escalation: low stamina = 3% more per inning, high stamina = 1%
    escalation = 0.03 - (stamina / 100) * 0.02  # 0.01 to 0.03

    # Build weights: inning 1 = 1.0, inning 2 = 1 + escalation, etc.
    weights = []
    for i in range(innings_started):
        weights.append(1.0 + escalation * i)
    total_weight = sum(weights)

    # Distribute total pitches proportionally
    raw = [total_pitch_count * w / total_weight for w in weights]
    counts = [round(x) for x in raw]

    # Adjust rounding error
    diff = total_pitch_count - sum(counts)
    if diff != 0:
        # Add/subtract from the last inning
        counts[-1] += diff

    # Ensure no negative counts
    counts = [max(0, c) for c in counts]

    return counts


# ---------------------------------------------------------------------------
# Derive velocity decline from fatigue
# ---------------------------------------------------------------------------


def _derive_velocity_change(
    base_velocity: float,
    pitch_count: int,
    innings_pitched: float,
    stamina: float,
) -> float:
    """Derive velocity change from start of game to current point.

    MLB data shows that starting pitchers lose ~0.5-1.5 mph over the course
    of a game. The decline accelerates as pitch count increases and is
    worse for low-stamina pitchers.

    Returns:
        Velocity change in mph (negative = decline).
    """
    if pitch_count <= 0:
        return 0.0

    # Base decline rate per pitch: 0.005 to 0.015 mph per pitch depending on stamina
    # High stamina (100) = 0.005 mph per pitch
    # Low stamina (0) = 0.015 mph per pitch
    decline_per_pitch = 0.015 - (stamina / 100) * 0.010

    # Linear decline plus acceleration at higher pitch counts
    # After ~75 pitches, decline accelerates
    if pitch_count <= 75:
        decline = decline_per_pitch * pitch_count
    else:
        base_decline = decline_per_pitch * 75
        excess = pitch_count - 75
        # Acceleration factor: 1.5x decline rate for each pitch past 75
        decline = base_decline + decline_per_pitch * 1.5 * excess

    # Clamp to realistic range: 0 to -4.0 mph
    return round(_clamp(-decline, -4.0, 0.0), 1)


# ---------------------------------------------------------------------------
# Derive spin rate decline from fatigue
# ---------------------------------------------------------------------------


def _derive_spin_rate_change(
    base_velocity: float,
    stuff: float,
    pitch_count: int,
    stamina: float,
) -> int:
    """Derive spin rate change from start of game to current point.

    Spin rate tends to decline as pitchers fatigue, typically 50-200 RPM
    over a full outing. Higher-stuff pitchers maintain spin better.

    Returns:
        Spin rate change in RPM (negative = decline).
    """
    if pitch_count <= 0:
        return 0

    # Base spin rate: approximate from velocity and stuff
    # (Not returned directly, but used to scale the decline)
    # Decline rate per pitch: 0.3 to 1.0 RPM depending on stamina + stuff
    # High stamina+stuff = 0.3, low = 1.0
    combined_resilience = (stamina + stuff) / 200  # 0.0 to 1.0
    decline_per_pitch = 1.0 - combined_resilience * 0.7  # 0.3 to 1.0

    if pitch_count <= 75:
        decline = decline_per_pitch * pitch_count
    else:
        base_decline = decline_per_pitch * 75
        excess = pitch_count - 75
        decline = base_decline + decline_per_pitch * 1.3 * excess

    # Clamp to realistic range: 0 to -300 RPM
    return round(_clamp(-decline, -300, 0))


# ---------------------------------------------------------------------------
# Derive batted ball quality trend by inning
# ---------------------------------------------------------------------------


def _derive_batted_ball_trend(
    pitch_counts_by_inning: list[int],
    stuff: float,
    stamina: float,
    velocity: float,
) -> list[dict]:
    """Derive average exit velocity against per inning.

    As pitchers tire, batted ball quality against them increases. This models
    the trend using the pitcher's stuff (base quality) and stamina (degradation rate).

    Returns:
        List of dicts with inning number and average exit velocity against.
    """
    if not pitch_counts_by_inning:
        return []

    # Base exit velocity allowed: better stuff = lower exit velo
    # MLB avg exit velo ~88.5 mph. Stuff 50 = average.
    base_ev = 91.0 - (stuff / 100) * 5.0  # 86.0 to 91.0

    # Per-inning degradation: low stamina pitchers see exit velo climb faster
    ev_increase_per_inning = 0.8 - (stamina / 100) * 0.5  # 0.3 to 0.8

    trend = []
    cumulative_pitches = 0
    for i, pc in enumerate(pitch_counts_by_inning):
        inning_num = i + 1
        cumulative_pitches += pc

        # Exit velo increases with innings pitched
        base_increase = ev_increase_per_inning * i

        # Additional spike when cumulative pitch count is high
        pitch_count_penalty = 0.0
        if cumulative_pitches > 75:
            pitch_count_penalty = (cumulative_pitches - 75) * 0.02
        elif cumulative_pitches > 100:
            pitch_count_penalty = (cumulative_pitches - 75) * 0.02 + (cumulative_pitches - 100) * 0.03

        ev = round(base_ev + base_increase + pitch_count_penalty, 1)
        trend.append({
            "inning": inning_num,
            "avg_exit_velo": ev,
        })

    return trend


# ---------------------------------------------------------------------------
# Derive times-through-order wOBA
# ---------------------------------------------------------------------------


def _derive_tto_woba(
    era_vs_l: float,
    era_vs_r: float,
    stamina: float,
) -> dict:
    """Derive wOBA allowed per time through the order.

    The times-through-order penalty is well documented in baseball analytics.
    Batters improve ~15-20 points of wOBA on the 2nd time through, and
    ~30-40 points on the 3rd+. Higher stamina mitigates this effect.

    Returns:
        Dict mapping "1st", "2nd", "3rd_plus" to wOBA values.
    """
    # Average ERA for overall wOBA baseline
    base_era = era_vs_r * 0.55 + era_vs_l * 0.45
    base_woba = _clamp(round(0.200 + base_era * 0.025, 3), 0.250, 0.400)

    # TTO penalty: stamina reduces penalty
    # Stamina 100 -> 2nd: +0.008, 3rd: +0.018
    # Stamina 0 -> 2nd: +0.020, 3rd: +0.050
    stamina_factor = stamina / 100
    penalty_2nd = _clamp(round(0.020 - stamina_factor * 0.012, 3), 0.005, 0.025)
    penalty_3rd = _clamp(round(0.050 - stamina_factor * 0.032, 3), 0.015, 0.055)

    return {
        "1st": base_woba,
        "2nd": _clamp(round(base_woba + penalty_2nd, 3), 0.250, 0.420),
        "3rd_plus": _clamp(round(base_woba + penalty_3rd, 3), 0.260, 0.450),
    }


# ---------------------------------------------------------------------------
# Derive overall fatigue level
# ---------------------------------------------------------------------------


def _derive_fatigue_level(
    pitch_count: int,
    innings_pitched: float,
    stamina: float,
    times_through_order: int,
) -> str:
    """Derive an overall fatigue rating from game metrics and attributes.

    Fatigue levels:
    - fresh: Low pitch count, early in game, can go deeper
    - normal: Moderate usage, still effective but showing some wear
    - fatigued: High pitch count or deep in game, effectiveness declining
    - gassed: Extreme usage, should be removed ASAP

    The thresholds are stamina-dependent. A high-stamina workhorse can
    throw 100+ pitches and still be "normal", while a low-stamina reliever
    might be "fatigued" at 30 pitches.
    """
    if pitch_count <= 0:
        return "fresh"

    # Stamina-adjusted thresholds
    # High stamina (100): fresh < 40, normal < 90, fatigued < 115, gassed >= 115
    # Low stamina (0): fresh < 15, normal < 50, fatigued < 75, gassed >= 75
    s = stamina / 100  # 0.0 to 1.0

    fresh_max = 15 + s * 25         # 15-40
    normal_max = 50 + s * 40        # 50-90
    fatigued_max = 75 + s * 40      # 75-115

    # Times through order adds fatigue penalty
    # Each TTO past 1st is like extra 8-15 pitches (less for high stamina)
    tto_penalty = max(0, times_through_order - 1) * (15 - s * 7)  # 8-15 per TTO

    effective_load = pitch_count + tto_penalty

    if effective_load < fresh_max:
        return "fresh"
    elif effective_load < normal_max:
        return "normal"
    elif effective_load < fatigued_max:
        return "fatigued"
    else:
        return "gassed"


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def get_pitcher_fatigue_assessment(
    pitcher_id: str,
    pitch_count: int = 0,
    innings_pitched: float = 0.0,
    times_through_order: int = 1,
    runs_allowed: int = 0,
    in_current_game: bool = True,
) -> str:
    """Assesses the current pitcher's fatigue based on in-game trends: velocity
    changes, spin rate decline, batted ball quality trend, pitch count, times
    through order, and an overall fatigue rating.

    Args:
        pitcher_id: The unique identifier of the pitcher to assess.
        pitch_count: Total pitches thrown so far in this game. Defaults to 0 (start of game).
        innings_pitched: Innings pitched so far (e.g., 5.0 for 5 complete innings, 5.2 for 5 and 2 outs).
        times_through_order: Times through the batting order (1 = first time, 2 = second, etc.).
        runs_allowed: Runs allowed in the current game.
        in_current_game: Whether the pitcher is currently in the game. Set to false to get assessment for a pitcher not yet in the game.
    Returns:
        JSON string with fatigue assessment including velocity change, spin rate change,
        batted ball quality trend, pitch count by inning, TTO wOBA, and fatigue level.
    """
    _load_players()
    TOOL_NAME = "get_pitcher_fatigue_assessment"

    # --- Validate pitcher_id ---
    if pitcher_id not in _PLAYERS:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID", f"Player '{pitcher_id}' not found in any roster.")

    player = _PLAYERS[pitcher_id]

    # --- Validate pitcher has pitcher attributes ---
    if "pitcher" not in player:
        return error_response(TOOL_NAME, "NOT_A_PITCHER", f"Player '{pitcher_id}' ({player.get('name', 'unknown')}) does not have pitching attributes.")

    # --- Validate in_current_game ---
    if not in_current_game:
        return error_response(TOOL_NAME, "PITCHER_NOT_IN_GAME", f"Player '{pitcher_id}' ({player.get('name', 'unknown')}) is not in the current game.")

    # --- Validate pitch_count ---
    if pitch_count < 0:
        return error_response(TOOL_NAME, "INVALID_PARAMETER", f"Pitch count must be non-negative, got {pitch_count}.")

    # --- Validate innings_pitched ---
    if innings_pitched < 0:
        return error_response(TOOL_NAME, "INVALID_PARAMETER", f"Innings pitched must be non-negative, got {innings_pitched}.")

    # --- Validate times_through_order ---
    if times_through_order < 1:
        return error_response(TOOL_NAME, "INVALID_PARAMETER", f"Times through order must be at least 1, got {times_through_order}.")

    pitcher_attrs = player["pitcher"]
    stuff = pitcher_attrs["stuff"]
    control = pitcher_attrs["control"]
    stamina = pitcher_attrs["stamina"]
    velocity = pitcher_attrs["velocity"]
    era_vs_l = pitcher_attrs["era_vs_l"]
    era_vs_r = pitcher_attrs["era_vs_r"]

    # --- Derive pitch counts by inning ---
    pitch_counts_by_inning = _derive_pitch_counts_by_inning(
        pitch_count, innings_pitched, stamina,
    )

    # --- Derive velocity change ---
    velocity_change = _derive_velocity_change(
        velocity, pitch_count, innings_pitched, stamina,
    )

    # --- Derive spin rate change ---
    spin_rate_change = _derive_spin_rate_change(
        velocity, stuff, pitch_count, stamina,
    )

    # --- Derive batted ball quality trend ---
    batted_ball_trend = _derive_batted_ball_trend(
        pitch_counts_by_inning, stuff, stamina, velocity,
    )

    # --- Derive TTO wOBA ---
    tto_woba = _derive_tto_woba(era_vs_l, era_vs_r, stamina)

    # --- Derive fatigue level ---
    fatigue_level = _derive_fatigue_level(
        pitch_count, innings_pitched, stamina, times_through_order,
    )

    return success_response(TOOL_NAME, {
        "pitcher_id": pitcher_id,
        "pitcher_name": player.get("name", "Unknown"),
        "throws": player.get("throws", "R"),
        "base_velocity": velocity,
        "velocity_change": velocity_change,
        "spin_rate_change": spin_rate_change,
        "batted_ball_quality_trend": batted_ball_trend,
        "pitch_count": pitch_count,
        "pitch_count_by_inning": pitch_counts_by_inning,
        "innings_pitched": innings_pitched,
        "times_through_order": times_through_order,
        "wOBA_per_time_through": tto_woba,
        "runs_allowed": runs_allowed,
        "fatigue_level": fatigue_level,
        "stamina": stamina,
    })
