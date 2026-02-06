"""Evaluates whether a sacrifice bunt is optimal.

Uses batter attributes (contact, speed, power), the run expectancy matrix,
and scoring probability tables to compare bunting vs swinging away. Returns
expected runs, scoring probabilities, bunt proficiency, net expected value,
and a contextual recommendation.
"""

import json
from pathlib import Path

from anthropic import beta_tool

from tools.get_run_expectancy import (
    RE_MATRIX,
    PROB_AT_LEAST_ONE,
    _runners_key,
    _get_re,
)

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
# Bunt proficiency derivation
# ---------------------------------------------------------------------------


def _derive_bunt_proficiency(player: dict) -> float:
    """Derive bunt proficiency rating (0.0 - 1.0) from player attributes.

    Bunt proficiency depends on:
    - Contact (primary): high-contact hitters can place bunts better
    - Speed (secondary): faster players beat out bunts / put pressure on defense
    - Power (negative factor): high-power hitters rarely practice bunting

    MLB average successful sac bunt rate is ~70-80%.
    Scale: 0.0 = terrible bunter, 1.0 = elite bunter.
    """
    batter = player.get("batter", {})
    contact = batter.get("contact", 50)
    speed = batter.get("speed", 50)
    power = batter.get("power", 50)

    # Contact is the primary driver (weight 0.50)
    # Speed helps (weight 0.30)
    # Power hurts (weight -0.20) -- sluggers don't practice bunting
    raw = (contact * 0.50 + speed * 0.30 + (100 - power) * 0.20) / 100
    return _clamp(round(raw, 3), 0.10, 0.95)


# ---------------------------------------------------------------------------
# Bunt outcome modeling
# ---------------------------------------------------------------------------


def _compute_bunt_expected_runs(
    first: bool,
    second: bool,
    third: bool,
    outs: int,
    bunt_proficiency: float,
) -> tuple[float, float]:
    """Compute expected runs and P(scoring at least one) when bunting.

    A sacrifice bunt has several possible outcomes:
    1. Successful sacrifice (batter out, runners advance) -- most common
    2. Bunt hit (batter reaches, runners advance) -- rare but possible
    3. Failed bunt (popup, foul out, fielder's choice that gets lead runner)

    Returns (expected_runs_bunt, prob_score_at_least_one_bunt).
    """
    current_key = _runners_key(first, second, third)
    current_re = _get_re(current_key, outs)
    current_prob = PROB_AT_LEAST_ONE[current_key][outs]

    # Probability weights for bunt outcomes based on proficiency
    # Good bunters: higher sac success rate, lower failure rate
    # Sac success: runners advance, batter out (+1 out)
    # Bunt hit: runners advance, batter reaches 1st (no out added)
    # Failed bunt: popup/foul out = just an out, no advancement
    #              fielder's choice = lead runner out, batter on 1st

    p_sac_success = _clamp(0.50 + bunt_proficiency * 0.35, 0.40, 0.85)
    p_bunt_hit = _clamp(0.03 + bunt_proficiency * 0.07, 0.03, 0.12)
    p_fielders_choice = _clamp(0.15 - bunt_proficiency * 0.08, 0.02, 0.15)
    p_failed = 1.0 - p_sac_success - p_bunt_hit - p_fielders_choice
    p_failed = max(0.0, p_failed)

    # Normalize to sum to 1.0
    total = p_sac_success + p_bunt_hit + p_fielders_choice + p_failed
    p_sac_success /= total
    p_bunt_hit /= total
    p_fielders_choice /= total
    p_failed /= total

    weighted_re = 0.0
    weighted_prob = 0.0

    # --- Outcome 1: Successful sacrifice ---
    # Runners advance one base, batter is out
    if first and not second and not third:
        # R1 -> R2, batter out
        sac_key = _runners_key(False, True, False)
    elif second and not first and not third:
        # R2 -> R3, batter out
        sac_key = _runners_key(False, False, True)
    elif first and second and not third:
        # R1->R2, R2->R3, batter out
        sac_key = _runners_key(False, True, True)
    elif first and not second and third:
        # R1->R2, R3 stays (can't advance home on sac bunt safely usually)
        # Actually R3 can score on a squeeze -- but standard sac bunt holds R3
        sac_key = _runners_key(False, True, True)
    elif second and third and not first:
        # R2->R3 is moot (already on 3rd), this is unusual
        # R3 stays, R2->R3: both on 3rd? No -- R2 stays, R3 stays
        # Actually with runners on 2nd and 3rd, a sac bunt could score R3
        # Squeeze play: R3 scores, R2->R3, batter out
        # But standard sac bunt doesn't try to score R3 from 3rd here
        # We'll model as: runners hold, batter just bunts for out (bad idea)
        sac_key = _runners_key(False, second, third)
    elif first and second and third:
        # Bases loaded bunt: R3 scores (force play), R1->R2, R2->R3, batter out
        # This is effectively a squeeze with bases loaded
        sac_key = _runners_key(False, True, True)
    else:
        # No runners -- sacrifice bunt makes no sense, but handle gracefully
        sac_key = _runners_key(False, False, False)

    sac_outs = min(outs + 1, 3)

    # For bases-loaded bunt, runner on 3rd scores
    sac_run_bonus = 0.0
    if first and second and third:
        sac_run_bonus = 1.0  # R3 scores on force

    sac_re = _get_re(sac_key, sac_outs) + sac_run_bonus
    sac_prob = PROB_AT_LEAST_ONE.get(sac_key, [0, 0, 0])[sac_outs] if sac_outs <= 2 else 0.0
    if sac_run_bonus > 0:
        sac_prob = 1.0  # Already scored

    weighted_re += p_sac_success * sac_re
    weighted_prob += p_sac_success * sac_prob

    # --- Outcome 2: Bunt hit (batter reaches 1st) ---
    # Runners advance, batter reaches 1st, no out
    if first and not second and not third:
        # R1->R2, batter on 1st
        hit_key = _runners_key(True, True, False)
    elif second and not first and not third:
        # R2->R3, batter on 1st
        hit_key = _runners_key(True, False, True)
    elif first and second and not third:
        # R1->R2, R2->R3, batter on 1st
        hit_key = _runners_key(True, True, True)
    elif first and not second and third:
        # R1->R2, R3 stays, batter on 1st
        hit_key = _runners_key(True, True, True)
    elif second and third and not first:
        # R2->R3 (already there), batter on 1st
        hit_key = _runners_key(True, False, True)
    elif first and second and third:
        # Bases loaded, bunt hit: R3 scores, all advance, batter on 1st
        hit_key = _runners_key(True, True, True)
    else:
        hit_key = _runners_key(True, False, False)

    hit_run_bonus = 0.0
    if first and second and third:
        hit_run_bonus = 1.0  # R3 scores on force

    hit_re = _get_re(hit_key, outs) + hit_run_bonus
    hit_prob = PROB_AT_LEAST_ONE[hit_key][outs]
    if hit_run_bonus > 0:
        hit_prob = 1.0

    weighted_re += p_bunt_hit * hit_re
    weighted_prob += p_bunt_hit * hit_prob

    # --- Outcome 3: Fielder's choice (lead runner out, batter on 1st) ---
    # The defense throws out the lead runner instead of the batter
    if first and not second and not third:
        # R1 out at 2nd, batter on 1st
        fc_key = _runners_key(True, False, False)
    elif second and not first and not third:
        # R2 out at 3rd, batter on 1st
        fc_key = _runners_key(True, False, False)
    elif first and second and not third:
        # Lead runner (R2) out at 3rd, R1 advances to 2nd, batter on 1st
        fc_key = _runners_key(True, True, False)
    elif first and not second and third:
        # R3 out at home (or R1 out at 2nd), batter on 1st
        fc_key = _runners_key(True, False, third)
    else:
        # Generic: one out gained, batter reaches
        fc_key = _runners_key(True, False, False)

    fc_outs = min(outs + 1, 3)
    fc_re = _get_re(fc_key, fc_outs)
    fc_prob = PROB_AT_LEAST_ONE.get(fc_key, [0, 0, 0])[fc_outs] if fc_outs <= 2 else 0.0

    weighted_re += p_fielders_choice * fc_re
    weighted_prob += p_fielders_choice * fc_prob

    # --- Outcome 4: Failed bunt (popup, batter out, no advancement) ---
    failed_re = _get_re(current_key, min(outs + 1, 3))
    failed_prob = PROB_AT_LEAST_ONE.get(current_key, [0, 0, 0])[min(outs + 1, 3)] if outs + 1 <= 2 else 0.0

    weighted_re += p_failed * failed_re
    weighted_prob += p_failed * failed_prob

    return round(weighted_re, 3), round(weighted_prob, 3)


def _compute_swing_expected_runs(
    first: bool,
    second: bool,
    third: bool,
    outs: int,
) -> tuple[float, float]:
    """Return expected runs and P(scoring at least one) when swinging away.

    This is simply the current state's run expectancy and scoring probability
    from the RE matrix -- since swinging away is the default action, the
    matrix already represents the average outcome of all plate appearances.
    """
    key = _runners_key(first, second, third)
    re = _get_re(key, outs)
    prob = PROB_AT_LEAST_ONE[key][outs]
    return round(re, 3), round(prob, 3)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def _make_recommendation(
    re_bunt: float,
    re_swing: float,
    prob_bunt: float,
    prob_swing: float,
    bunt_proficiency: float,
    score_differential: int,
    inning: int,
    outs: int,
    third: bool,
) -> str:
    """Generate a contextual recommendation string.

    The recommendation considers:
    - Expected runs comparison (bunt vs swing)
    - Probability of scoring at least one run (important in close games)
    - Game context (score differential, inning, late-game situations)
    - Batter's bunt proficiency
    """
    re_advantage = re_bunt - re_swing
    prob_advantage = prob_bunt - prob_swing

    # Late and close: inning 7+ and within 1 run
    late_and_close = inning >= 7 and abs(score_differential) <= 1

    # Tie or trailing by 1: maximizing P(score 1 run) matters more than RE
    need_one_run = score_differential in (-1, 0)

    # Runner on 3rd with less than 2 outs: bunt could be squeeze
    squeeze_situation = third and outs < 2

    if bunt_proficiency < 0.30:
        return "swing away is preferred; batter has poor bunt proficiency"

    if outs == 2:
        return "swing away is preferred; bunting with 2 outs is almost never correct"

    # If bunt is better on both RE and P(score), it's clearly favorable
    if re_advantage > 0.01 and prob_advantage > 0.01:
        return "bunt is favorable on both expected runs and scoring probability"

    # Late and close game where we need exactly 1 run
    if late_and_close and need_one_run:
        if prob_advantage >= -0.02:
            return "bunt is favorable in a close, late game to maximize scoring probability"
        elif prob_advantage >= -0.06:
            return "bunt is marginal in a close, late game; slight scoring probability cost"
        else:
            return "swing away is preferred even in a close, late game; scoring probability cost is too high"

    # Squeeze situation (runner on 3rd, less than 2 outs)
    if squeeze_situation and need_one_run:
        if bunt_proficiency >= 0.50:
            return "squeeze bunt is viable with runner on third and proficient bunter"
        else:
            return "squeeze bunt is risky; batter lacks bunt proficiency"

    # General case: compare expected runs
    if re_advantage > 0.02:
        return "bunt is favorable on expected runs"
    elif re_advantage >= -0.02:
        if prob_advantage >= 0.0:
            return "bunt is marginal; slight scoring probability edge"
        else:
            return "bunt is marginal; swing away has slight edge on both metrics"
    else:
        if re_advantage < -0.10:
            return "swing away is strongly preferred; significant expected run cost to bunting"
        else:
            return "swing away is preferred on expected runs"


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def evaluate_sacrifice_bunt(
    batter_id: str,
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
    outs: int,
    score_differential: int,
    inning: int,
) -> str:
    """Evaluates whether a sacrifice bunt is optimal given the batter, base-out
    state, score differential, and inning. Compares bunt vs swing-away expected
    runs and probability of scoring at least one run.

    Args:
        batter_id: The unique identifier of the current batter.
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        outs: Number of outs (0 or 1; bunting with 2 outs is rare).
        score_differential: Score difference from managed team perspective (positive = leading).
        inning: Current inning number (1+).
    Returns:
        JSON string with bunt evaluation.
    """
    _load_players()

    # --- Validate batter_id ---
    if batter_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{batter_id}' not found in any roster.",
        })

    player = _PLAYERS[batter_id]

    # --- Validate player has batter attributes ---
    if "batter" not in player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_BATTER",
            "message": f"Player '{batter_id}' ({player.get('name', 'unknown')}) does not have batting attributes.",
        })

    # --- Validate outs ---
    if not isinstance(outs, int) or outs < 0 or outs > 2:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid outs value: {outs}. Must be 0, 1, or 2.",
        })

    # --- Validate inning ---
    if not isinstance(inning, int) or inning < 1:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid inning value: {inning}. Must be 1 or greater.",
        })

    # --- Validate there are runners on base ---
    if not runner_on_first and not runner_on_second and not runner_on_third:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_SITUATION",
            "message": "Sacrifice bunt requires at least one runner on base.",
        })

    # --- Derive bunt proficiency ---
    bunt_proficiency = _derive_bunt_proficiency(player)

    # --- Compute bunt expected value ---
    re_bunt, prob_bunt = _compute_bunt_expected_runs(
        runner_on_first, runner_on_second, runner_on_third, outs, bunt_proficiency,
    )

    # --- Compute swing-away expected value ---
    re_swing, prob_swing = _compute_swing_expected_runs(
        runner_on_first, runner_on_second, runner_on_third, outs,
    )

    # --- Net expected value ---
    net_ev = round(re_bunt - re_swing, 3)

    # --- Recommendation ---
    recommendation = _make_recommendation(
        re_bunt, re_swing, prob_bunt, prob_swing, bunt_proficiency,
        score_differential, inning, outs, runner_on_third,
    )

    return json.dumps({
        "status": "ok",
        "batter_id": batter_id,
        "batter_name": player.get("name", "Unknown"),
        "base_out_state": {
            "first": runner_on_first,
            "second": runner_on_second,
            "third": runner_on_third,
            "outs": outs,
        },
        "score_differential": score_differential,
        "inning": inning,
        "expected_runs_bunt": re_bunt,
        "expected_runs_swing": re_swing,
        "prob_score_bunt": prob_bunt,
        "prob_score_swing": prob_swing,
        "bunt_proficiency": bunt_proficiency,
        "net_expected_value": net_ev,
        "recommendation": recommendation,
    })
