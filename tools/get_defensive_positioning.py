"""Returns recommended defensive positioning for a given matchup and situation.

Loads batter and pitcher data from sample_rosters.json and derives spray chart
tendencies, infield/outfield positioning recommendations, infield-in cost/benefit
analysis, and shift recommendations within current MLB rule constraints.
"""

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
# Spray chart derivation
# ---------------------------------------------------------------------------


def _derive_spray_chart(batter: dict, bats: str, pitcher_throws: str) -> dict:
    """Derive spray chart tendencies from batter attributes and matchup handedness.

    MLB tendencies:
    - LHB pull to right field, RHB pull to left field
    - High-power hitters pull more frequently
    - High-contact hitters spray the ball more evenly
    - Same-side matchups (LHP vs LHB) tend to produce more pull-side contact
    - Ground balls are pulled more than fly balls

    Returns:
        Dict with groundball and flyball sub-dicts, each containing pull_pct,
        center_pct, and oppo_pct (summing to 1.0).
    """
    power = batter.get("power", 50)
    contact = batter.get("contact", 50)
    speed = batter.get("speed", 50)

    # Base pull tendency: higher power -> more pull
    # MLB average: ~40% pull, 34% center, 26% oppo for groundballs
    # MLB average: ~37% pull, 34% center, 29% oppo for flyballs
    power_factor = power / 100  # 0.0 to 1.0
    contact_factor = contact / 100  # High contact = more spread

    # Ground ball pull tendency: power hitters pull GBs heavily
    gb_pull_base = 0.38 + power_factor * 0.12  # 0.38 to 0.50
    gb_oppo_base = 0.24 - power_factor * 0.06  # 0.18 to 0.24
    # High contact hitters spread it around more
    gb_pull_base -= contact_factor * 0.04  # reduce pull for contact hitters
    gb_oppo_base += contact_factor * 0.03  # increase oppo for contact hitters

    # Fly ball pull tendency: similar but less extreme
    fb_pull_base = 0.35 + power_factor * 0.10  # 0.35 to 0.45
    fb_oppo_base = 0.27 - power_factor * 0.05  # 0.22 to 0.27
    fb_pull_base -= contact_factor * 0.03
    fb_oppo_base += contact_factor * 0.02

    # Same-side matchup effect: LHB vs LHP or RHB vs RHP increases pull
    # because same-side pitching tends to run the ball in on the hands
    effective_bats = bats
    if bats == "S":
        # Switch hitters bat opposite the pitcher
        effective_bats = "R" if pitcher_throws == "L" else "L"

    same_side = (effective_bats == "L" and pitcher_throws == "L") or \
                (effective_bats == "R" and pitcher_throws == "R")

    if same_side:
        gb_pull_base += 0.03
        gb_oppo_base -= 0.02
        fb_pull_base += 0.02
        fb_oppo_base -= 0.01

    # Opposite-side matchup: slightly less pull
    opposite_side = (effective_bats == "L" and pitcher_throws == "R") or \
                    (effective_bats == "R" and pitcher_throws == "L")
    if opposite_side:
        gb_pull_base -= 0.01
        gb_oppo_base += 0.01

    # Clamp and compute center as remainder
    gb_pull = _clamp(round(gb_pull_base, 2), 0.25, 0.55)
    gb_oppo = _clamp(round(gb_oppo_base, 2), 0.12, 0.35)
    gb_center = round(1.0 - gb_pull - gb_oppo, 2)
    # Ensure center is positive
    if gb_center < 0.15:
        excess = 0.15 - gb_center
        gb_pull -= excess / 2
        gb_oppo -= excess / 2
        gb_center = round(1.0 - gb_pull - gb_oppo, 2)

    fb_pull = _clamp(round(fb_pull_base, 2), 0.22, 0.50)
    fb_oppo = _clamp(round(fb_oppo_base, 2), 0.15, 0.35)
    fb_center = round(1.0 - fb_pull - fb_oppo, 2)
    if fb_center < 0.15:
        excess = 0.15 - fb_center
        fb_pull -= excess / 2
        fb_oppo -= excess / 2
        fb_center = round(1.0 - fb_pull - fb_oppo, 2)

    return {
        "groundball": {
            "pull_pct": gb_pull,
            "center_pct": gb_center,
            "oppo_pct": gb_oppo,
        },
        "flyball": {
            "pull_pct": fb_pull,
            "center_pct": fb_center,
            "oppo_pct": fb_oppo,
        },
    }


# ---------------------------------------------------------------------------
# Ground ball rate derivation
# ---------------------------------------------------------------------------


def _derive_gb_rate(batter: dict, pitcher: dict) -> float:
    """Derive the expected ground ball rate for this matchup.

    Ground ball rate depends on:
    - Batter power (high power = more fly balls, lower GB%)
    - Pitcher stuff and pitch type tendencies (high stuff sinkerballers = high GB%)
    - Pitcher control (better control = more induced contact)

    MLB average GB% is ~43%.
    Returns a float between 0.30 and 0.60.
    """
    power = batter.get("power", 50)
    stuff = pitcher.get("stuff", 50)
    control = pitcher.get("control", 50)

    # High power batters hit fewer grounders
    batter_gb = 0.48 - (power / 100) * 0.12  # 0.36 to 0.48

    # High stuff pitchers can induce more ground balls (sinkers, cutters)
    pitcher_gb = 0.40 + (stuff / 100) * 0.08  # 0.40 to 0.48

    # Blend batter and pitcher tendencies (batter 55%, pitcher 45%)
    gb_rate = batter_gb * 0.55 + pitcher_gb * 0.45

    return _clamp(round(gb_rate, 3), 0.30, 0.60)


# ---------------------------------------------------------------------------
# Infield positioning recommendation
# ---------------------------------------------------------------------------


def _recommend_infield(
    spray_chart: dict,
    gb_rate: float,
    outs: int,
    runner_1st: bool,
    runner_2nd: bool,
    runner_3rd: bool,
    score_diff: int,
    inning: int,
) -> tuple[str, dict]:
    """Recommend infield positioning and provide details.

    Possible recommendations:
    - "standard": Default depth and positioning
    - "double_play_depth": Back to turn two, with runner on 1st and less than 2 outs
    - "infield_in": Bring infield in to cut off runs at home
    - "halfway": Infield halfway (compromise between in and back)
    - "guard_lines": Late innings, protect against extra-base hits down the lines

    Also returns the infield-in cost/benefit analysis.

    Returns:
        (recommendation_string, infield_in_analysis_dict)
    """
    # Infield-in analysis: always computed regardless of recommendation
    # Runs saved at home: depends on runner on 3rd, GB rate, and pull tendency
    # Extra hits allowed: infield in allows more ground balls through

    # Base probability of a run scoring from 3rd on a GB with infield back: ~60%
    # Infield in reduces that to ~25%
    # But infield in allows ~8-12% more hits through (BABIP increase)

    gb_pull = spray_chart["groundball"]["pull_pct"]

    # Runs saved at home: only relevant with runner on 3rd
    if runner_3rd and outs < 2:
        # Higher GB rate = more chances for the run to score on GB
        base_save = 0.10 + gb_rate * 0.15  # 0.13 to 0.19
        # Adjust for situation: fewer outs = more valuable to cut off run
        if outs == 0:
            base_save *= 1.15
        runs_saved = round(base_save, 3)
    else:
        runs_saved = 0.0

    # Extra hits allowed: infield in increases BABIP on ground balls
    # MLB data: infield in adds ~0.050-0.100 to BABIP
    extra_hits = round(0.06 + gb_rate * 0.08, 3)  # 0.084 to 0.108

    infield_in_analysis = {
        "runs_saved_at_home": runs_saved,
        "extra_hits_allowed": extra_hits,
    }

    # --- Determine recommendation ---

    # Priority 1: Infield in with runner on 3rd, less than 2 outs
    if runner_3rd and outs < 2:
        # In a tie or trailing game, especially late, bring infield in
        late_game = inning >= 7
        close_game = abs(score_diff) <= 2

        if late_game and (score_diff <= 0):
            # Late and tied/trailing: definitely infield in
            return "infield_in", infield_in_analysis
        elif score_diff <= 0 and outs == 0:
            # Early but tied/trailing with 0 outs: halfway to compromise
            return "halfway", infield_in_analysis
        elif score_diff <= 0:
            # Tied/trailing with 1 out: infield in to prevent the run
            return "infield_in", infield_in_analysis
        elif score_diff >= 3:
            # Leading by 3+: no need to play in, concede the run
            return "standard", infield_in_analysis
        else:
            # Leading by 1-2: halfway
            return "halfway", infield_in_analysis

    # Priority 2: Double play depth with runner on 1st, less than 2 outs
    if runner_1st and outs < 2:
        # Standard double play positioning
        if gb_rate >= 0.40:
            return "double_play_depth", infield_in_analysis
        else:
            # Low GB rate batter: still play DP depth but it's less effective
            return "double_play_depth", infield_in_analysis

    # Priority 3: Guard the lines in late, close games
    if inning >= 8 and abs(score_diff) <= 1 and outs == 2:
        return "guard_lines", infield_in_analysis

    # Priority 4: Standard positioning
    return "standard", infield_in_analysis


# ---------------------------------------------------------------------------
# Outfield positioning recommendation
# ---------------------------------------------------------------------------


def _recommend_outfield(
    spray_chart: dict,
    batter: dict,
    outs: int,
    runner_1st: bool,
    runner_2nd: bool,
    runner_3rd: bool,
    score_diff: int,
    inning: int,
) -> str:
    """Recommend outfield positioning.

    Possible recommendations:
    - "standard": Default depth
    - "shallow": Play shallow for weak hitters or to throw out runners at home
    - "deep": Play deep for power hitters
    - "no_doubles": Deep and toward the lines to prevent extra-base hits
    - "shaded_pull": Shade toward the pull side for heavy pull hitters
    - "shaded_oppo": Shade toward the opposite field for oppo hitters

    Returns:
        Recommendation string with directional detail.
    """
    power = batter.get("power", 50)
    speed = batter.get("speed", 50)
    contact = batter.get("contact", 50)

    fb_pull = spray_chart["flyball"]["pull_pct"]
    fb_oppo = spray_chart["flyball"]["oppo_pct"]

    # No-doubles defense: late, close game with runner in scoring position
    late_close = inning >= 8 and abs(score_diff) <= 1
    if late_close and (runner_2nd or runner_1st) and outs == 2:
        return "no_doubles"

    # Deep for power hitters (power >= 75)
    if power >= 75:
        # Also shade toward pull side if heavy pull hitter
        if fb_pull >= 0.42:
            return "deep, shaded_pull"
        elif fb_oppo >= 0.32:
            return "deep, shaded_oppo"
        else:
            return "deep"

    # Shallow for low-power, high-contact hitters
    if power <= 40 and contact >= 70:
        return "shallow"

    # Shade toward pull for heavy pull hitters
    if fb_pull >= 0.42:
        return "shaded_pull"

    # Shade toward oppo for unusual oppo tendencies
    if fb_oppo >= 0.32:
        return "shaded_oppo"

    return "standard"


# ---------------------------------------------------------------------------
# Shift recommendation (MLB rule compliant)
# ---------------------------------------------------------------------------


def _recommend_shift(spray_chart: dict, bats: str) -> str:
    """Recommend positioning shift within MLB rule constraints.

    Since 2023, MLB requires:
    - All four infielders must have both feet on the infield dirt
    - Exactly 2 infielders on each side of 2nd base
    - Outright shifts (3 infielders on one side) are banned

    Remaining legal adjustments:
    - Infielders can shade toward the pull or oppo side within their zone
    - SS and 2B can adjust depth and lateral position
    - 3B and 1B can guard the line or play off it

    Returns:
        Description of recommended positioning adjustments within rules.
    """
    gb_pull = spray_chart["groundball"]["pull_pct"]
    gb_oppo = spray_chart["groundball"]["oppo_pct"]

    # Heavy pull hitter: shade infielders toward pull side within 2-and-2 constraint
    if gb_pull >= 0.45:
        if bats in ("L", "S"):
            return ("shade infielders toward first-base side within 2-and-2 rule; "
                    "SS and 2B shift a few steps toward 1B-2B hole")
        else:
            return ("shade infielders toward third-base side within 2-and-2 rule; "
                    "SS and 2B shift a few steps toward SS-3B hole")

    # Heavy oppo hitter: shade the other way
    if gb_oppo >= 0.30:
        if bats in ("L", "S"):
            return ("shade infielders toward third-base side within 2-and-2 rule; "
                    "SS and 2B shift a few steps toward SS-3B hole")
        else:
            return ("shade infielders toward first-base side within 2-and-2 rule; "
                    "SS and 2B shift a few steps toward 1B-2B hole")

    # Balanced spray chart: no shift needed
    return "no shift recommended; standard 2-and-2 positioning"


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def get_defensive_positioning(
    batter_id: str,
    pitcher_id: str,
    outs: int,
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
    score_differential: int,
    inning: int,
) -> str:
    """Returns recommended infield and outfield positioning for a given
    batter-pitcher matchup and game situation. Includes spray chart summary,
    infield-in cost/benefit, and shift recommendations.

    Args:
        batter_id: The unique identifier of the batter.
        pitcher_id: The unique identifier of the pitcher.
        outs: Number of outs (0, 1, or 2).
        runner_on_first: Whether there is a runner on first base.
        runner_on_second: Whether there is a runner on second base.
        runner_on_third: Whether there is a runner on third base.
        score_differential: Score difference from managed team perspective.
        inning: Current inning number.
    Returns:
        JSON string with defensive positioning recommendations.
    """
    _load_players()

    # --- Validate batter_id ---
    if batter_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{batter_id}' not found in any roster.",
        })

    # --- Validate pitcher_id ---
    if pitcher_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Player '{pitcher_id}' not found in any roster.",
        })

    batter_player = _PLAYERS[batter_id]
    pitcher_player = _PLAYERS[pitcher_id]

    # --- Validate batter has batter attributes ---
    if "batter" not in batter_player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_BATTER",
            "message": f"Player '{batter_id}' ({batter_player.get('name', 'unknown')}) does not have batting attributes.",
        })

    # --- Validate pitcher has pitcher attributes ---
    if "pitcher" not in pitcher_player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_PITCHER",
            "message": f"Player '{pitcher_id}' ({pitcher_player.get('name', 'unknown')}) does not have pitching attributes.",
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

    batter_attrs = batter_player["batter"]
    pitcher_attrs = pitcher_player["pitcher"]
    bats = batter_player.get("bats", "R")
    pitcher_throws = pitcher_player.get("throws", "R")

    # --- Derive spray chart ---
    spray_chart = _derive_spray_chart(batter_attrs, bats, pitcher_throws)

    # --- Derive ground ball rate for this matchup ---
    gb_rate = _derive_gb_rate(batter_attrs, pitcher_attrs)

    # --- Recommend infield positioning ---
    infield_rec, infield_in_analysis = _recommend_infield(
        spray_chart, gb_rate, outs,
        runner_on_first, runner_on_second, runner_on_third,
        score_differential, inning,
    )

    # --- Recommend outfield positioning ---
    outfield_rec = _recommend_outfield(
        spray_chart, batter_attrs, outs,
        runner_on_first, runner_on_second, runner_on_third,
        score_differential, inning,
    )

    # --- Recommend shift ---
    shift_rec = _recommend_shift(spray_chart, bats)

    return json.dumps({
        "status": "ok",
        "batter_id": batter_id,
        "batter_name": batter_player.get("name", "Unknown"),
        "bats": bats,
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_player.get("name", "Unknown"),
        "throws": pitcher_throws,
        "spray_chart": spray_chart,
        "groundball_rate": gb_rate,
        "infield_recommendation": infield_rec,
        "outfield_recommendation": outfield_rec,
        "infield_in_analysis": infield_in_analysis,
        "shift_recommendation": shift_rec,
    })
