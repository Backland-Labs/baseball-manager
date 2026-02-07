# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Decision quality scoring via Win Probability Added (WPA).

Measures the quality of each managerial decision by computing the win
probability delta caused by the decision.  WPA is the standard sabermetric
measure of how much a single play (or decision) changed the game's win
probability.

For each decision the agent makes, we compute:
    WPA = WP_after - WP_before

where WP_before is the win probability at the moment of the decision, and
WP_after is the win probability after the decision's outcome is resolved.

Active decisions (pitching changes, stolen bases, pinch hits, etc.) are
scored directly.  No-action decisions carry an implicit WPA of 0.0 (the
game state did not change as a result of the decision itself).

The module also computes game-level aggregate metrics:
    - Total WPA (sum of all decision WPAs)
    - Average WPA per active decision
    - Best and worst decisions by WPA
    - Decision quality distribution
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# WP lookup -- reuses the same pre-computed tables as get_win_probability.py
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"

_WP_TABLE: dict = {}
_LI_TABLE: dict = {}


def _ensure_tables_loaded() -> None:
    """Lazy-load the WP and LI tables on first use."""
    global _WP_TABLE, _LI_TABLE
    if not _WP_TABLE:
        wp_path = _DATA_DIR / "win_probability.json"
        if wp_path.exists():
            _WP_TABLE = json.loads(wp_path.read_text())["wp_table"]
    if not _LI_TABLE:
        li_path = _DATA_DIR / "leverage_index.json"
        if li_path.exists():
            _LI_TABLE = json.loads(li_path.read_text())["li_table"]


def _runners_key(first: bool, second: bool, third: bool) -> str:
    return f"{'1' if first else '0'}{'1' if second else '0'}{'1' if third else '0'}"


def lookup_wp(
    inning: int,
    half: str,
    outs: int,
    runner_key: str,
    score_diff: int,
    is_home: bool = True,
) -> float:
    """Look up win probability for the managed team.

    Args:
        inning: Current inning (1+).
        half: 'TOP' or 'BOTTOM'.
        outs: Number of outs (0-2).
        runner_key: Base state key like '000', '110', etc.
        score_diff: Score differential from the managed team's perspective
            (positive = leading).
        is_home: Whether the managed team is the home team.

    Returns:
        Win probability for the managed team (0.01-0.99).
    """
    _ensure_tables_loaded()

    clamped_inning = min(inning, 12)
    inning_key = f"{clamped_inning}_{half}"
    clamped_diff = max(-10, min(10, score_diff))

    # Tables store WP from the away team's perspective.
    if is_home:
        away_diff = -clamped_diff
    else:
        away_diff = clamped_diff

    diff_key = str(away_diff)
    outs_key = str(outs)

    away_wp = 0.50  # fallback

    if inning_key in _WP_TABLE and outs_key in _WP_TABLE[inning_key]:
        outs_data = _WP_TABLE[inning_key][outs_key]
        if runner_key in outs_data and diff_key in outs_data[runner_key]:
            away_wp = outs_data[runner_key][diff_key]
        elif "000" in outs_data and diff_key in outs_data["000"]:
            away_wp = outs_data["000"][diff_key]

    if is_home:
        managed_wp = 1.0 - away_wp
    else:
        managed_wp = away_wp

    return round(max(0.01, min(0.99, managed_wp)), 4)


def lookup_li(
    inning: int,
    half: str,
    outs: int,
    runner_key: str,
    score_diff: int,
    is_home: bool = True,
) -> float:
    """Look up leverage index for the game state.

    Args:
        inning: Current inning (1+).
        half: 'TOP' or 'BOTTOM'.
        outs: Number of outs (0-2).
        runner_key: Base state key like '000', '110', etc.
        score_diff: Score differential from the managed team's perspective.
        is_home: Whether the managed team is the home team.

    Returns:
        Leverage index (0.1 to ~10.0, where 1.0 is average).
    """
    _ensure_tables_loaded()

    clamped_inning = min(inning, 12)
    inning_key = f"{clamped_inning}_{half}"

    if is_home:
        away_diff = -max(-10, min(10, score_diff))
    else:
        away_diff = max(-10, min(10, score_diff))

    diff_key = str(away_diff)
    outs_key = str(outs)

    if inning_key in _LI_TABLE and outs_key in _LI_TABLE[inning_key]:
        outs_data = _LI_TABLE[inning_key][outs_key]
        if runner_key in outs_data and diff_key in outs_data[runner_key]:
            return outs_data[runner_key][diff_key]

    return 1.0


# ---------------------------------------------------------------------------
# WPA result data classes
# ---------------------------------------------------------------------------

NO_ACTION_TYPES = frozenset({
    "NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "LET_BATTER_HIT",
    "NO_CHANGE", "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER",
    "KEEP_CURRENT",
})


@dataclass
class DecisionWPA:
    """WPA score for a single decision."""
    turn: int
    decision_type: str
    is_active: bool
    wp_before: float
    wp_after: float
    wpa: float  # wp_after - wp_before
    leverage_index: float
    # Game context at decision time
    inning: int = 0
    half: str = ""
    outs: int = 0
    score_diff: int = 0
    runners: str = "000"  # base state key
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "decision_type": self.decision_type,
            "is_active": self.is_active,
            "wp_before": round(self.wp_before, 4),
            "wp_after": round(self.wp_after, 4),
            "wpa": round(self.wpa, 4),
            "leverage_index": round(self.leverage_index, 3),
            "inning": self.inning,
            "half": self.half,
            "outs": self.outs,
            "score_diff": self.score_diff,
            "runners": self.runners,
            "description": self.description,
        }


@dataclass
class GameWPAReport:
    """Aggregate WPA metrics for an entire game."""
    total_decisions: int = 0
    active_decisions: int = 0
    total_wpa: float = 0.0
    active_wpa: float = 0.0
    avg_wpa_per_active: float = 0.0
    best_decision: Optional[DecisionWPA] = None
    worst_decision: Optional[DecisionWPA] = None
    high_leverage_decisions: int = 0  # LI >= 1.5
    positive_wpa_count: int = 0
    negative_wpa_count: int = 0
    neutral_wpa_count: int = 0
    decision_scores: list[DecisionWPA] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "total_decisions": self.total_decisions,
            "active_decisions": self.active_decisions,
            "total_wpa": round(self.total_wpa, 4),
            "active_wpa": round(self.active_wpa, 4),
            "avg_wpa_per_active": round(self.avg_wpa_per_active, 4),
            "high_leverage_decisions": self.high_leverage_decisions,
            "positive_wpa_count": self.positive_wpa_count,
            "negative_wpa_count": self.negative_wpa_count,
            "neutral_wpa_count": self.neutral_wpa_count,
            "best_decision": self.best_decision.to_dict() if self.best_decision else None,
            "worst_decision": self.worst_decision.to_dict() if self.worst_decision else None,
            "decision_scores": [d.to_dict() for d in self.decision_scores],
        }
        return result


# ---------------------------------------------------------------------------
# WPA computation from game state objects (simulation integration)
# ---------------------------------------------------------------------------

def compute_wp_from_game_state(game_state, managed_team: str = "home") -> float:
    """Compute WP for the managed team from a GameState object.

    Args:
        game_state: A simulation.GameState instance.
        managed_team: "home" or "away".

    Returns:
        Win probability for the managed team (0.01-0.99).
    """
    is_home = managed_team == "home"

    if is_home:
        score_diff = game_state.score_home - game_state.score_away
    else:
        score_diff = game_state.score_away - game_state.score_home

    runner_key = game_state.bases_string()
    outs = min(game_state.outs, 2)

    return lookup_wp(
        inning=game_state.inning,
        half=game_state.half,
        outs=outs,
        runner_key=runner_key,
        score_diff=score_diff,
        is_home=is_home,
    )


def compute_li_from_game_state(game_state, managed_team: str = "home") -> float:
    """Compute leverage index from a GameState object.

    Args:
        game_state: A simulation.GameState instance.
        managed_team: "home" or "away".

    Returns:
        Leverage index for the current situation.
    """
    is_home = managed_team == "home"

    if is_home:
        score_diff = game_state.score_home - game_state.score_away
    else:
        score_diff = game_state.score_away - game_state.score_home

    runner_key = game_state.bases_string()
    outs = min(game_state.outs, 2)

    return lookup_li(
        inning=game_state.inning,
        half=game_state.half,
        outs=outs,
        runner_key=runner_key,
        score_diff=score_diff,
        is_home=is_home,
    )


def score_decision(
    wp_before: float,
    wp_after: float,
    decision_dict: dict,
    leverage_index: float,
    turn: int = 0,
    inning: int = 0,
    half: str = "",
    outs: int = 0,
    score_diff: int = 0,
    runners: str = "000",
) -> DecisionWPA:
    """Score a single decision by computing its WPA.

    Args:
        wp_before: Win probability before the decision.
        wp_after: Win probability after the decision outcome is resolved.
        decision_dict: The agent's decision dict.
        leverage_index: Leverage index at decision time.
        turn: Decision turn number.
        inning: Inning number.
        half: "TOP" or "BOTTOM".
        outs: Outs at decision time.
        score_diff: Score differential from managed team's perspective.
        runners: Base state key like "000", "110".

    Returns:
        DecisionWPA with the computed WPA score.
    """
    decision_type = decision_dict.get("decision", "").upper().strip()
    is_active = decision_type not in NO_ACTION_TYPES and bool(decision_type)

    wpa = wp_after - wp_before
    description = decision_dict.get("action_details", "")

    return DecisionWPA(
        turn=turn,
        decision_type=decision_type,
        is_active=is_active,
        wp_before=wp_before,
        wp_after=wp_after,
        wpa=wpa,
        leverage_index=leverage_index,
        inning=inning,
        half=half,
        outs=outs,
        score_diff=score_diff,
        runners=runners,
        description=description,
    )


# ---------------------------------------------------------------------------
# Game-level WPA report
# ---------------------------------------------------------------------------

def generate_game_wpa_report(decision_scores: list[DecisionWPA]) -> GameWPAReport:
    """Generate aggregate WPA metrics from a list of decision scores.

    Args:
        decision_scores: List of DecisionWPA objects from score_decision().

    Returns:
        GameWPAReport with aggregate metrics.
    """
    if not decision_scores:
        return GameWPAReport()

    total_decisions = len(decision_scores)
    active = [d for d in decision_scores if d.is_active]
    active_decisions = len(active)

    total_wpa = sum(d.wpa for d in decision_scores)
    active_wpa = sum(d.wpa for d in active)
    avg_wpa_per_active = active_wpa / active_decisions if active_decisions > 0 else 0.0

    best_decision = None
    worst_decision = None
    if active:
        best_decision = max(active, key=lambda d: d.wpa)
        worst_decision = min(active, key=lambda d: d.wpa)

    high_leverage = sum(1 for d in decision_scores if d.leverage_index >= 1.5)

    # Count positive, negative, and neutral WPA among active decisions
    positive = sum(1 for d in active if d.wpa > 0.001)
    negative = sum(1 for d in active if d.wpa < -0.001)
    neutral = active_decisions - positive - negative

    return GameWPAReport(
        total_decisions=total_decisions,
        active_decisions=active_decisions,
        total_wpa=total_wpa,
        active_wpa=active_wpa,
        avg_wpa_per_active=avg_wpa_per_active,
        best_decision=best_decision,
        worst_decision=worst_decision,
        high_leverage_decisions=high_leverage,
        positive_wpa_count=positive,
        negative_wpa_count=negative,
        neutral_wpa_count=neutral,
        decision_scores=decision_scores,
    )


# ---------------------------------------------------------------------------
# Score decisions from a game log file
# ---------------------------------------------------------------------------

def score_game_log(game_log: dict, managed_team: str = "home") -> GameWPAReport:
    """Score all decisions in a saved game log file.

    The game log must contain decision entries with game_state context
    (inning, half, outs, score, runners). WP is computed from the game
    state context at each decision point.

    For decisions within a game log (post-hoc analysis), we approximate
    WP_after by looking at the *next* decision's WP_before. The last
    decision's WP_after is set to 1.0 or 0.0 based on the game outcome.

    Args:
        game_log: Parsed game log dict (from data/game_logs/game_*.json).
        managed_team: "home" or "away" (should match the log).

    Returns:
        GameWPAReport with computed WPA scores for all decisions.
    """
    decisions = game_log.get("decisions", [])
    if not decisions:
        return GameWPAReport()

    is_home = (game_log.get("game_info", {}).get("managed_team", managed_team) == "home")

    # Compute WP_before for each decision from its game state
    wp_values: list[float] = []
    for entry in decisions:
        gs = entry.get("game_state", {})
        inning = gs.get("inning", 1)
        half = gs.get("half", "TOP")
        outs = gs.get("outs", 0)
        score = gs.get("score", {})
        home_score = score.get("home", 0)
        away_score = score.get("away", 0)

        if is_home:
            score_diff = home_score - away_score
        else:
            score_diff = away_score - home_score

        # Build runner key from runners dict
        runners_dict = gs.get("runners", {})
        first = "1" in runners_dict
        second = "2" in runners_dict
        third = "3" in runners_dict
        runner_key = _runners_key(first, second, third)

        wp = lookup_wp(inning, half, min(outs, 2), runner_key, score_diff, is_home)
        wp_values.append(wp)

    # Determine final WP from game outcome
    game_info = game_log.get("game_info", {})
    final_score = game_info.get("final_score", {})
    winner = game_info.get("winner", "")
    home_team = game_info.get("home_team", "")
    away_team = game_info.get("away_team", "")

    if is_home:
        managed_won = (final_score.get("home", 0) > final_score.get("away", 0))
    else:
        managed_won = (final_score.get("away", 0) > final_score.get("home", 0))

    final_wp = 1.0 if managed_won else 0.0

    # Score each decision: WP_after = next decision's WP_before (or final WP)
    decision_scores: list[DecisionWPA] = []
    for i, entry in enumerate(decisions):
        wp_before = wp_values[i]
        wp_after = wp_values[i + 1] if i + 1 < len(wp_values) else final_wp

        gs = entry.get("game_state", {})
        inning = gs.get("inning", 1)
        half = gs.get("half", "TOP")
        outs = gs.get("outs", 0)
        score = gs.get("score", {})
        home_score = score.get("home", 0)
        away_score = score.get("away", 0)

        if is_home:
            score_diff = home_score - away_score
        else:
            score_diff = away_score - home_score

        runners_dict = gs.get("runners", {})
        first = "1" in runners_dict
        second = "2" in runners_dict
        third = "3" in runners_dict
        runner_key = _runners_key(first, second, third)

        li = lookup_li(inning, half, min(outs, 2), runner_key, score_diff, is_home)

        decision_dict = entry.get("decision", {})
        dwpa = score_decision(
            wp_before=wp_before,
            wp_after=wp_after,
            decision_dict=decision_dict,
            leverage_index=li,
            turn=entry.get("turn", i + 1),
            inning=inning,
            half=half,
            outs=outs,
            score_diff=score_diff,
            runners=runner_key,
        )
        decision_scores.append(dwpa)

    return generate_game_wpa_report(decision_scores)


# ---------------------------------------------------------------------------
# Score decisions from a game log file path
# ---------------------------------------------------------------------------

def score_game_log_file(log_path: str | Path) -> GameWPAReport:
    """Load and score a game log JSON file.

    Args:
        log_path: Path to a game log JSON file.

    Returns:
        GameWPAReport with computed WPA scores.

    Raises:
        FileNotFoundError: If the log file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(log_path)
    game_log = json.loads(path.read_text())
    managed_team = game_log.get("game_info", {}).get("managed_team", "home")
    return score_game_log(game_log, managed_team)


# ---------------------------------------------------------------------------
# Format WPA report as human-readable text
# ---------------------------------------------------------------------------

def format_wpa_report(report: GameWPAReport) -> str:
    """Format a GameWPAReport as human-readable text.

    Args:
        report: The WPA report to format.

    Returns:
        Multi-line string summarizing the report.
    """
    lines = [
        "DECISION QUALITY REPORT (WPA)",
        "=" * 40,
        f"Total decisions:        {report.total_decisions}",
        f"Active decisions:       {report.active_decisions}",
        f"Total WPA:              {report.total_wpa:+.4f}",
        f"Active WPA:             {report.active_wpa:+.4f}",
        f"Avg WPA per active:     {report.avg_wpa_per_active:+.4f}",
        f"High leverage (LI>=1.5):{report.high_leverage_decisions}",
        f"Positive WPA decisions: {report.positive_wpa_count}",
        f"Negative WPA decisions: {report.negative_wpa_count}",
        f"Neutral WPA decisions:  {report.neutral_wpa_count}",
    ]

    if report.best_decision:
        bd = report.best_decision
        lines.append("")
        lines.append(f"Best decision:  Turn {bd.turn} ({bd.decision_type})")
        lines.append(f"  WPA: {bd.wpa:+.4f} | LI: {bd.leverage_index:.2f}")
        lines.append(f"  Inning: {bd.half} {bd.inning}, {bd.outs} out")
        if bd.description:
            lines.append(f"  {bd.description[:80]}")

    if report.worst_decision:
        wd = report.worst_decision
        lines.append("")
        lines.append(f"Worst decision: Turn {wd.turn} ({wd.decision_type})")
        lines.append(f"  WPA: {wd.wpa:+.4f} | LI: {wd.leverage_index:.2f}")
        lines.append(f"  Inning: {wd.half} {wd.inning}, {wd.outs} out")
        if wd.description:
            lines.append(f"  {wd.description[:80]}")

    return "\n".join(lines)
