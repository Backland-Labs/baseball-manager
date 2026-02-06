"""Evaluates a potential stolen base attempt.

Uses runner speed attributes, pitcher hold time (derived from pitcher
attributes), catcher pop time, and the run expectancy matrix to produce
a success probability estimate, breakeven rate, expected RE changes, and
a textual recommendation.
"""

import json
from pathlib import Path
from typing import Optional

from anthropic import beta_tool

from tools.get_run_expectancy import RE_MATRIX, _runners_key, _get_re

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
# Derived metrics from player attributes
# ---------------------------------------------------------------------------


def _derive_sprint_speed(player: dict) -> float:
    """Derive sprint speed (ft/s) from batter speed attribute.

    MLB range: ~23 ft/s (slow) to ~31 ft/s (elite).
    Speed attribute 0-100 maps linearly into this range.
    """
    batter = player.get("batter", {})
    speed = batter.get("speed", 50)
    return _clamp(round(23.0 + (speed / 100) * 8.0, 1), 23.0, 31.0)


def _derive_sb_success_rate(player: dict) -> float:
    """Derive stolen base success rate from speed attribute.

    Faster runners have higher career SB%. MLB average is ~72%.
    Speed 50 = ~72%, speed 100 = ~88%, speed 0 = ~55%.
    """
    batter = player.get("batter", {})
    speed = batter.get("speed", 50)
    return _clamp(round(0.55 + (speed / 100) * 0.33, 3), 0.50, 0.92)


def _derive_pitcher_hold_time(player: dict) -> float:
    """Derive pitcher hold time (seconds to plate) from pitcher attributes.

    Hold time = time from set position to ball reaching the catcher.
    Good hold times: ~1.2s. Bad hold times: ~1.5s+.
    Control attribute inversely affects hold time (better control = quicker delivery).
    Velocity also helps (faster pitch = less flight time).

    Returns time in seconds.
    """
    pitcher = player.get("pitcher", {})
    control = pitcher.get("control", 50)
    velocity = pitcher.get("velocity", 90.0)

    # Base delivery time from wind-up/set mechanics: ~1.0-1.1s
    # Higher control = more repeatable, slightly quicker mechanics
    delivery_time = _clamp(1.35 - (control / 100) * 0.25, 1.05, 1.40)

    # Flight time based on velocity (60.5 feet to plate)
    # v in mph -> ft/s = v * 5280/3600 = v * 1.467
    velocity_fps = velocity * 1.467
    flight_time = 60.5 / velocity_fps

    return round(delivery_time + flight_time, 3)


def _derive_catcher_pop_time(player: dict) -> float:
    """Get catcher pop time from catcher attributes, or derive from arm strength.

    Pop time = time from ball hitting catcher's mitt to ball arriving at 2nd base.
    MLB average: ~2.0s. Elite: ~1.85s. Poor: ~2.15s.
    """
    catcher_attrs = player.get("catcher")
    if catcher_attrs and "pop_time" in catcher_attrs:
        return catcher_attrs["pop_time"]

    # Derive from arm strength if no explicit pop time
    fielder = player.get("fielder", {})
    arm = fielder.get("arm_strength", 50)
    return _clamp(round(2.20 - (arm / 100) * 0.40, 2), 1.80, 2.20)


# ---------------------------------------------------------------------------
# Success probability model
# ---------------------------------------------------------------------------


def _estimate_success_probability(
    sprint_speed: float,
    sb_rate: float,
    pitcher_hold_time: float,
    catcher_pop_time: float,
    target_base: int,
) -> float:
    """Estimate stolen base success probability.

    The model combines:
    1. Runner's career SB success rate as a base
    2. Adjustments for pitcher hold time (faster = harder to steal)
    3. Adjustments for catcher pop time (faster = harder to steal)
    4. Sprint speed as a secondary factor
    5. Target base penalty (stealing 3rd is harder due to shorter distance
       but offset by pitcher attention to batter)

    MLB averages used for calibration:
    - Average SB success rate: ~72%
    - Average pitcher hold time: ~1.35s (delivery + flight)
    - Average catcher pop time to 2nd: ~2.00s
    - Average sprint speed: ~27.0 ft/s
    """
    # Start with career SB rate as base probability
    base_prob = sb_rate

    # Pitcher hold time adjustment
    # Average hold time ~1.35s. Each 0.1s faster reduces success by ~4%
    avg_hold = 1.35
    hold_delta = pitcher_hold_time - avg_hold  # positive = slower pitcher (easier)
    hold_adj = hold_delta * 0.40  # 0.1s difference = ~4% change

    # Catcher pop time adjustment
    # Average pop time ~2.00s. Each 0.05s faster reduces success by ~3%
    avg_pop = 2.00
    pop_delta = catcher_pop_time - avg_pop  # positive = slower catcher (easier)
    pop_adj = pop_delta * 0.60  # 0.05s difference = ~3% change

    # Sprint speed adjustment (secondary - partially captured in SB rate already)
    # Average sprint speed ~27.0 ft/s
    avg_sprint = 27.0
    sprint_delta = sprint_speed - avg_sprint
    sprint_adj = sprint_delta * 0.008  # each 1 ft/s = ~0.8% change

    # Target base penalty: stealing 3rd has slightly different dynamics
    # Shorter lead from 2nd, but pitcher is focused on batter
    # Net effect: ~3% harder than stealing 2nd on average
    base_penalty = 0.0
    if target_base == 3:
        base_penalty = -0.03
    elif target_base == 4:
        # Steal of home is extremely rare and risky
        base_penalty = -0.35

    raw_prob = base_prob + hold_adj + pop_adj + sprint_adj + base_penalty
    return _clamp(round(raw_prob, 3), 0.10, 0.98)


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def evaluate_stolen_base(
    runner_id: str,
    target_base: int,
    pitcher_id: str,
    catcher_id: str,
    runner_on_first: bool = False,
    runner_on_second: bool = False,
    runner_on_third: bool = False,
    outs: int = 0,
) -> str:
    """Evaluates a potential stolen base attempt given runner speed, pitcher hold
    time, catcher arm, and current base-out state. Returns success probability,
    breakeven rate, expected run-expectancy change, and a recommendation.

    Args:
        runner_id: The unique identifier of the baserunner attempting the steal.
        target_base: The base being stolen (2 for 2nd, 3 for 3rd, 4 for home).
        pitcher_id: The unique identifier of the current pitcher.
        catcher_id: The unique identifier of the opposing catcher.
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        outs: Number of outs (0, 1, or 2).
    Returns:
        JSON string with stolen base evaluation.
    """
    _load_players()

    # --- Validate player IDs ---
    for pid, label in [
        (runner_id, "runner"),
        (pitcher_id, "pitcher"),
        (catcher_id, "catcher"),
    ]:
        if pid not in _PLAYERS:
            return json.dumps({
                "status": "error",
                "error_code": "INVALID_PLAYER_ID",
                "message": f"Player '{pid}' ({label}) not found in any roster.",
            })

    # --- Validate target base ---
    if target_base not in (2, 3, 4):
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid target_base value: {target_base}. Must be 2, 3, or 4.",
        })

    # --- Validate outs ---
    if not isinstance(outs, int) or outs < 0 or outs > 2:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid outs value: {outs}. Must be 0, 1, or 2.",
        })

    # --- Validate the steal attempt is logically consistent ---
    # Stealing 2nd requires runner on 1st and 2nd base open
    if target_base == 2:
        if not runner_on_first:
            return json.dumps({
                "status": "error",
                "error_code": "INVALID_SITUATION",
                "message": "Cannot steal 2nd: no runner on first base.",
            })
        if runner_on_second:
            return json.dumps({
                "status": "error",
                "error_code": "INVALID_SITUATION",
                "message": "Cannot steal 2nd: second base is already occupied.",
            })

    # Stealing 3rd requires runner on 2nd and 3rd base open
    if target_base == 3:
        if not runner_on_second:
            return json.dumps({
                "status": "error",
                "error_code": "INVALID_SITUATION",
                "message": "Cannot steal 3rd: no runner on second base.",
            })
        if runner_on_third:
            return json.dumps({
                "status": "error",
                "error_code": "INVALID_SITUATION",
                "message": "Cannot steal 3rd: third base is already occupied.",
            })

    # Stealing home requires runner on 3rd
    if target_base == 4:
        if not runner_on_third:
            return json.dumps({
                "status": "error",
                "error_code": "INVALID_SITUATION",
                "message": "Cannot steal home: no runner on third base.",
            })

    # --- Get player data ---
    runner = _PLAYERS[runner_id]
    pitcher = _PLAYERS[pitcher_id]
    catcher = _PLAYERS[catcher_id]

    # Derive metrics
    sprint_speed = _derive_sprint_speed(runner)
    sb_rate = _derive_sb_success_rate(runner)
    pitcher_hold_time = _derive_pitcher_hold_time(pitcher)
    catcher_pop_time = _derive_catcher_pop_time(catcher)

    # --- Compute success probability ---
    success_prob = _estimate_success_probability(
        sprint_speed, sb_rate, pitcher_hold_time, catcher_pop_time, target_base,
    )

    # --- Compute RE-based breakeven and expected value ---
    current_key = _runners_key(runner_on_first, runner_on_second, runner_on_third)
    current_re = _get_re(current_key, outs)

    if target_base == 2:
        # Success: runner moves 1st -> 2nd
        success_key = _runners_key(False, True, runner_on_third)
        success_re = _get_re(success_key, outs)
        # Caught: runner removed from 1st, +1 out
        caught_key = _runners_key(False, False, runner_on_third)
        caught_re = _get_re(caught_key, outs + 1)
    elif target_base == 3:
        # Success: runner moves 2nd -> 3rd
        success_key = _runners_key(runner_on_first, False, True)
        success_re = _get_re(success_key, outs)
        # Caught: runner removed from 2nd, +1 out
        caught_key = _runners_key(runner_on_first, False, False)
        caught_re = _get_re(caught_key, outs + 1)
    else:
        # target_base == 4: stealing home
        # Success: runner scores from 3rd (1 run + remaining state)
        success_key = _runners_key(runner_on_first, runner_on_second, False)
        success_re = 1.0 + _get_re(success_key, outs)
        # Caught: runner out at home, +1 out
        caught_key = _runners_key(runner_on_first, runner_on_second, False)
        caught_re = _get_re(caught_key, outs + 1)

    re_if_success = round(success_re - current_re, 3)
    re_if_caught = round(caught_re - current_re, 3)

    # Breakeven rate: at what success rate is the steal EV-neutral?
    re_gain = success_re - current_re
    re_loss = current_re - caught_re  # positive value
    if re_gain + re_loss > 0:
        breakeven = round(re_loss / (re_gain + re_loss), 3)
    else:
        breakeven = 1.0

    # Net expected RE change
    net_re = round(success_prob * re_if_success + (1 - success_prob) * re_if_caught, 3)

    # --- Recommendation ---
    margin = success_prob - breakeven
    if margin >= 0.05:
        recommendation = "favorable"
    elif margin >= 0.0:
        recommendation = "marginal"
    else:
        recommendation = "unfavorable"

    # --- Build detailed context ---
    context = {
        "runner_sprint_speed_ft_per_s": sprint_speed,
        "runner_career_sb_rate": sb_rate,
        "pitcher_hold_time_s": pitcher_hold_time,
        "catcher_pop_time_s": catcher_pop_time,
    }

    return json.dumps({
        "status": "ok",
        "runner_id": runner_id,
        "runner_name": runner.get("name", "Unknown"),
        "target_base": target_base,
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher.get("name", "Unknown"),
        "catcher_id": catcher_id,
        "catcher_name": catcher.get("name", "Unknown"),
        "base_out_state": {
            "first": runner_on_first,
            "second": runner_on_second,
            "third": runner_on_third,
            "outs": outs,
        },
        "success_probability": success_prob,
        "breakeven_rate": breakeven,
        "re_change_if_successful": re_if_success,
        "re_change_if_caught": re_if_caught,
        "net_expected_re_change": net_re,
        "recommendation": recommendation,
        "context": context,
    })
