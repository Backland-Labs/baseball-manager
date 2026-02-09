# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0", "pydantic>=2.0"]
# ///
"""Backtest orchestration: build scenarios, run agent, compare decisions.

Ties together the extractor (decision points from game feeds), tools
(backtest-specific stat lookups), and the agent (_call_agent from game.py)
to produce a comparison report of agent decisions vs. real manager actions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from backtest.extractor import (
    DecisionPoint,
    ActionType,
    walk_game_feed,
    resolve_team_side,
)
from backtest.tools import (
    BACKTEST_TOOLS,
    set_backtest_context,
    clear_backtest_context,
)
from decision_quality_wpa import (
    lookup_wp,
    lookup_li,
    _runners_key,
    NO_ACTION_TYPES,
)
from data.mlb_api import get_live_game_feed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison model
# ---------------------------------------------------------------------------

class ComparisonEntry(BaseModel):
    play_index: int
    inning: int
    half: str
    outs: int
    score_home: int
    score_away: int
    batter_name: str
    pitcher_name: str
    leverage_index: float
    real_outcome: str
    agent_decision_type: str = ""
    agent_action_details: str = ""
    agent_reasoning: str = ""
    agent_confidence: float = 0.0
    real_manager_action_type: str = "NO_ACTION"
    real_manager_action_details: str = ""
    category_match: bool = False
    action_match: bool = False
    disagreement_type: str | None = None
    tool_calls: list[dict] = []
    token_usage: dict = {}


# ---------------------------------------------------------------------------
# build_scenario
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_scenario(dp: DecisionPoint) -> dict:
    """Convert a DecisionPoint to the agent scenario format.

    Produces a dict with matchup_state, roster_state, opponent_roster_state,
    and decision_prompt -- matching the structure from game_state_to_scenario().
    """
    is_home = dp.managed_team_side == "home"
    is_top = dp.half == "TOP"

    # Determine if managed team is batting
    batting_side = "away" if is_top else "home"
    we_are_batting = (batting_side == dp.managed_team_side)

    # Runner info
    runners_dict = {}
    for base in ("first", "second", "third"):
        r = dp.runners.get(base)
        if r:
            runners_dict[base] = {
                "player_id": r.get("id", ""),
                "name": r.get("name", ""),
                "sprint_speed": None,
                "sb_success_rate": None,
            }
        else:
            runners_dict[base] = None

    # Pitcher TTO estimate
    tto = max(1, (dp.pitcher_batters_faced // 9) + 1) if dp.pitcher_batters_faced > 0 else 1

    matchup_state = {
        "inning": dp.inning,
        "half": dp.half,
        "outs": dp.outs,
        "count": {"balls": 0, "strikes": 0},
        "runners": runners_dict,
        "score": {"home": dp.score_home, "away": dp.score_away},
        "batting_team": "AWAY" if is_top else "HOME",
        "batter": {
            "player_id": dp.batter_id,
            "name": dp.batter_name,
            "bats": dp.batter_bats,
            "lineup_position": 0,
        },
        "pitcher": {
            "player_id": dp.pitcher_id,
            "name": dp.pitcher_name,
            "throws": dp.pitcher_throws,
            "pitch_count_today": dp.pitcher_pitch_count,
            "batters_faced_today": dp.pitcher_batters_faced,
            "times_through_order": tto,
            "innings_pitched_today": dp.pitcher_innings_pitched,
            "runs_allowed_today": dp.pitcher_runs_allowed,
            "today_line": {
                "IP": dp.pitcher_innings_pitched,
                "H": 0,
                "R": dp.pitcher_runs_allowed,
                "ER": dp.pitcher_runs_allowed,
                "BB": 0,
                "K": 0,
            },
        },
        "on_deck_batter": {
            "player_id": dp.on_deck_batter_id or "",
            "name": dp.on_deck_batter_name or "",
            "bats": dp.on_deck_batter_bats,
        } if dp.on_deck_batter_id else None,
    }

    # Our lineup
    our_lineup = []
    for p in dp.current_lineup:
        our_lineup.append({
            "player_id": p.player_id,
            "name": p.name,
            "position": p.position,
            "bats": p.bats,
            "in_game": True,
        })

    our_bench = []
    for p in dp.bench:
        our_bench.append({
            "player_id": p.player_id,
            "name": p.name,
            "bats": p.bats,
            "positions": [p.position],
            "available": True,
        })

    our_bullpen = []
    for bp in dp.bullpen:
        our_bullpen.append({
            "player_id": bp.player_id,
            "name": bp.name,
            "throws": bp.throws,
            "role": "MIDDLE",
            "available": bp.available,
            "freshness": "FRESH",
            "pitches_last_3_days": [0, 0, 0],
            "days_since_last_appearance": 5,
            "is_warming_up": False,
        })

    roster_state = {
        "our_lineup": our_lineup,
        "our_lineup_position": 0,
        "bench": our_bench,
        "bullpen": our_bullpen,
        "mound_visits_remaining": 5,
        "challenge_available": True,
    }

    # Opponent
    opp_lineup = []
    for p in dp.opp_lineup:
        opp_lineup.append({
            "player_id": p.player_id,
            "name": p.name,
            "position": p.position,
            "bats": p.bats,
            "in_game": True,
        })

    opp_bench = []
    for p in dp.opp_bench:
        opp_bench.append({
            "player_id": p.player_id,
            "name": p.name,
            "bats": p.bats,
            "available": True,
        })

    opp_bullpen = []
    for bp in dp.opp_bullpen:
        opp_bullpen.append({
            "player_id": bp.player_id,
            "name": bp.name,
            "throws": bp.throws,
            "role": "MIDDLE",
            "available": bp.available,
            "freshness": "FRESH",
        })

    opponent_roster_state = {
        "their_lineup": opp_lineup,
        "their_lineup_position": 0,
        "their_bench": opp_bench,
        "their_bullpen": opp_bullpen,
    }

    # Decision prompt
    score_diff = (dp.score_home - dp.score_away) if is_home else (dp.score_away - dp.score_home)
    if score_diff > 0:
        score_desc = f"Your team leads {score_diff}"
    elif score_diff < 0:
        score_desc = f"Your team trails {abs(score_diff)}"
    else:
        score_desc = "Tied game"

    half_str = "Top" if dp.half == "TOP" else "Bottom"
    runner_descs = []
    for base, label in [("first", "1st"), ("second", "2nd"), ("third", "3rd")]:
        r = dp.runners.get(base)
        if r:
            runner_descs.append(f"{r.get('name', 'runner')} on {label}")
    runners_text = ", ".join(runner_descs) if runner_descs else "bases empty"

    if we_are_batting:
        decision_prompt = (
            f"{half_str} of the {_ordinal(dp.inning)}, {dp.outs} out, {runners_text}. "
            f"Score: Away {dp.score_away}, Home {dp.score_home}. {score_desc}. "
            f"{dp.batter_name} ({dp.batter_bats}) is batting against {dp.pitcher_name} "
            f"({dp.pitcher_throws}HP, {dp.pitcher_pitch_count} pitches). "
            f"Your team is BATTING. Consider: pinch-hit, stolen base, sacrifice bunt, "
            f"hit-and-run, or let the batter swing away. Gather relevant data with tools before deciding."
        )
    else:
        decision_prompt = (
            f"{half_str} of the {_ordinal(dp.inning)}, {dp.outs} out, {runners_text}. "
            f"Score: Away {dp.score_away}, Home {dp.score_home}. {score_desc}. "
            f"{dp.batter_name} ({dp.batter_bats}) is batting against your pitcher {dp.pitcher_name} "
            f"({dp.pitcher_throws}HP, {dp.pitcher_pitch_count} pitches, "
            f"{dp.pitcher_innings_pitched:.1f} IP, {dp.pitcher_runs_allowed} R). "
            f"Your team is FIELDING. Consider: pitching change, defensive positioning, intentional walk, "
            f"mound visit, or no action needed. Gather relevant data with tools before deciding."
        )

    if dp.on_deck_batter_name:
        decision_prompt += f" On deck: {dp.on_deck_batter_name} ({dp.on_deck_batter_bats})."

    return {
        "matchup_state": matchup_state,
        "roster_state": roster_state,
        "opponent_roster_state": opponent_roster_state,
        "decision_prompt": decision_prompt,
    }


# ---------------------------------------------------------------------------
# compare_decisions
# ---------------------------------------------------------------------------

# Canonical category mappings
_PITCHING_CHANGE_TYPES = frozenset({
    "PITCHING_CHANGE", "PULL_STARTER", "BRING_IN_RELIEVER",
    "PITCHING_SUBSTITUTION", "CALL_BULLPEN",
})
_PINCH_HIT_TYPES = frozenset({
    "PINCH_HIT", "PINCH_HITTER", "PINCH-HIT",
})
_PINCH_RUN_TYPES = frozenset({
    "PINCH_RUN", "PINCH_RUNNER", "PINCH-RUN",
})


def _normalize_decision_type(raw: str) -> str:
    """Normalize a decision type string to a canonical category."""
    upper = raw.upper().strip().replace("-", "_").replace(" ", "_")
    if upper in NO_ACTION_TYPES:
        return "NO_ACTION"
    if upper in _PITCHING_CHANGE_TYPES:
        return "PITCHING_CHANGE"
    if upper in _PINCH_HIT_TYPES:
        return "PINCH_HIT"
    if upper in _PINCH_RUN_TYPES:
        return "PINCH_RUN"
    if "STOLEN" in upper or "STEAL" in upper:
        return "STOLEN_BASE"
    if "BUNT" in upper:
        return "BUNT"
    if "IBB" in upper or "INTENTIONAL" in upper:
        return "IBB"
    return upper


def compare_decisions(
    agent_decision: dict,
    real_action: Any,
) -> dict:
    """Compare agent decision to real manager action.

    Args:
        agent_decision: The agent's decision dict (from ManagerDecision).
        real_action: RealManagerAction or None.

    Returns:
        Dict with category_match, action_match, agent_type, real_type,
        disagreement_type.
    """
    agent_raw = agent_decision.get("decision", "NO_ACTION")
    agent_type = _normalize_decision_type(agent_raw)

    if real_action is None:
        real_type = "NO_ACTION"
    else:
        real_type = real_action.action_type.value

    category_match = (agent_type == real_type)

    # Action match: same specific player involved
    action_match = False
    if category_match:
        if real_type == "NO_ACTION":
            action_match = True
        elif real_action is not None:
            # Check if agent mentioned the same player
            agent_details = agent_decision.get("action_details", "").lower()
            if real_action.player_in_name:
                action_match = real_action.player_in_name.lower() in agent_details
            else:
                action_match = True

    disagreement_type = None
    if not category_match:
        disagreement_type = f"Agent: {agent_type}, Real: {real_type}"

    return {
        "category_match": category_match,
        "action_match": action_match,
        "agent_type": agent_type,
        "real_type": real_type,
        "disagreement_type": disagreement_type,
    }


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def run_backtest(
    game_pk: int,
    team: str,
    *,
    dry_run: bool = False,
    verbose: bool = True,
    sample: int | None = None,
) -> list[ComparisonEntry]:
    """Run a full backtest on a completed game.

    Args:
        game_pk: MLB gamePk identifier.
        team: Team name, abbreviation, or "home"/"away".
        dry_run: If True, extract decision points only without calling the agent.
        verbose: Print progress.
        sample: If set, run only N decision points (always includes real
                manager actions, fills remaining slots with highest-LI picks).

    Returns:
        List of ComparisonEntry objects.
    """
    # Fetch game feed
    if verbose:
        print(f"Fetching game feed for gamePk={game_pk}...")
    feed = get_live_game_feed(game_pk)

    # Validate game is Final
    status = feed.get("gameData", {}).get("status", {}).get("detailedState", "")
    if status != "Final":
        raise ValueError(f"Game {game_pk} is not final (status: {status})")

    # Get game info
    game_data = feed.get("gameData", {})
    game_date = game_data.get("datetime", {}).get("officialDate", "")
    teams = game_data.get("teams", {})
    home_name = teams.get("home", {}).get("name", "?")
    away_name = teams.get("away", {}).get("name", "?")

    # Resolve team
    managed_side = resolve_team_side(feed, team)

    if verbose:
        print(f"Game: {away_name} @ {home_name}, {game_date}")
        print(f"Managing: {'Home' if managed_side == 'home' else 'Away'} team")

    # Walk game feed to extract decision points
    if verbose:
        print("Extracting decision points...")
    decision_points = list(walk_game_feed(feed, team))

    if verbose:
        active_actions = sum(1 for dp in decision_points if dp.real_manager_action is not None)
        print(f"Found {len(decision_points)} decision points ({active_actions} with manager actions)")

    # Sample decision points if requested
    if sample is not None and not dry_run and sample < len(decision_points):
        import random
        # Always include real manager actions
        must_include = [dp for dp in decision_points if dp.real_manager_action is not None]
        rest = [dp for dp in decision_points if dp.real_manager_action is None]
        # Fill remaining slots with highest-leverage picks
        rest.sort(key=lambda dp: dp.leverage_index, reverse=True)
        remaining_slots = max(0, sample - len(must_include))
        sampled = must_include + rest[:remaining_slots]
        # Re-sort by play_index to preserve game order
        sampled.sort(key=lambda dp: dp.play_index)
        if verbose:
            print(f"Sampled {len(sampled)} of {len(decision_points)} decision points "
                  f"({len(must_include)} manager actions + {remaining_slots} highest-LI)")
        decision_points = sampled

    if dry_run:
        # Print summary and return empty
        print("\n--- DRY RUN SUMMARY ---")
        print(f"Decision points: {len(decision_points)}")
        for i, dp in enumerate(decision_points):
            action_str = "NO_ACTION"
            if dp.real_manager_action:
                action_str = f"{dp.real_manager_action.action_type.value}: {dp.real_manager_action.details[:60]}"
            if dp.real_manager_action or i < 5 or i == len(decision_points) - 1:
                print(f"  [{i:3d}] Inn {dp.inning} {dp.half[:3]} | {dp.outs} out | "
                      f"{dp.score_away}-{dp.score_home} | "
                      f"{dp.batter_name[:15]:15s} vs {dp.pitcher_name[:15]:15s} | "
                      f"LI={dp.leverage_index:.2f} | {action_str}")
        return []

    # Full backtest: call the agent for each decision point
    from config import create_anthropic_client
    from game import _call_agent, load_system_prompt

    client = create_anthropic_client()
    system_prompt = load_system_prompt()

    entries: list[ComparisonEntry] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, dp in enumerate(decision_points):
        if verbose:
            print(f"\n[{i+1}/{len(decision_points)}] Inn {dp.inning} {dp.half}, "
                  f"{dp.outs} out | {dp.batter_name} vs {dp.pitcher_name} "
                  f"(LI={dp.leverage_index:.2f})")

        # Set backtest context for tools
        set_backtest_context(game_date, feed, dp)

        # Build scenario
        scenario = build_scenario(dp)

        # Format as user message (matching game.py format)
        user_message = (
            "Here is the current game scenario:\n\n"
            f"**Matchup State:**\n```json\n{json.dumps(scenario['matchup_state'], indent=2)}\n```\n\n"
            f"**Roster State:**\n```json\n{json.dumps(scenario['roster_state'], indent=2)}\n```\n\n"
            f"**Opponent Roster State:**\n```json\n{json.dumps(scenario['opponent_roster_state'], indent=2)}\n```\n\n"
            f"**Decision Needed:** {scenario['decision_prompt']}"
        )

        messages = [{"role": "user", "content": user_message}]

        # Call agent
        try:
            decision_dict, final_message, call_meta = _call_agent(
                client, messages,
                tools=BACKTEST_TOOLS,
                system=system_prompt,
                verbose=verbose,
            )
        except Exception as e:
            logger.error("Agent call failed at play %d: %s", dp.play_index, e)
            decision_dict = {"decision": "ERROR", "action_details": str(e)}
            call_meta = {"tool_calls": [], "token_usage": {}}

        clear_backtest_context()

        # Compare
        comparison = compare_decisions(decision_dict, dp.real_manager_action)

        # Track tokens
        usage = call_meta.get("token_usage", {})
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

        entry = ComparisonEntry(
            play_index=dp.play_index,
            inning=dp.inning,
            half=dp.half,
            outs=dp.outs,
            score_home=dp.score_home,
            score_away=dp.score_away,
            batter_name=dp.batter_name,
            pitcher_name=dp.pitcher_name,
            leverage_index=dp.leverage_index,
            real_outcome=dp.real_outcome,
            agent_decision_type=comparison["agent_type"],
            agent_action_details=decision_dict.get("action_details", ""),
            agent_reasoning=decision_dict.get("reasoning", ""),
            agent_confidence=decision_dict.get("confidence", 0.0),
            real_manager_action_type=comparison["real_type"],
            real_manager_action_details=(
                dp.real_manager_action.details if dp.real_manager_action else ""
            ),
            category_match=comparison["category_match"],
            action_match=comparison["action_match"],
            disagreement_type=comparison["disagreement_type"],
            tool_calls=call_meta.get("tool_calls", []),
            token_usage=usage,
        )
        entries.append(entry)

        if verbose:
            match_str = "MATCH" if comparison["category_match"] else "DISAGREE"
            print(f"  Agent: {comparison['agent_type']:20s} | "
                  f"Real: {comparison['real_type']:20s} | {match_str}")

    return entries


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

def format_report(entries: list[ComparisonEntry], game_info: dict) -> str:
    """Format comparison entries into a human-readable report.

    Args:
        entries: List of ComparisonEntry objects.
        game_info: Dict with game metadata (home, away, date, final_score, etc.).

    Returns:
        Multi-line report string.
    """
    if not entries:
        return "No entries to report."

    total = len(entries)
    category_matches = sum(1 for e in entries if e.category_match)
    action_matches = sum(1 for e in entries if e.action_match)
    active_real = [e for e in entries if e.real_manager_action_type != "NO_ACTION"]
    active_agent = [e for e in entries if e.agent_decision_type != "NO_ACTION"]
    active_matches = sum(1 for e in active_real if e.category_match)

    disagreements = [e for e in entries if not e.category_match]
    disagreements.sort(key=lambda e: e.leverage_index, reverse=True)

    # Token totals
    total_input = sum(e.token_usage.get("input_tokens", 0) for e in entries)
    total_output = sum(e.token_usage.get("output_tokens", 0) for e in entries)
    total_tokens = total_input + total_output

    lines = [
        "=== BACKTEST REPORT ===",
        f"Game: {game_info.get('away', '?')} @ {game_info.get('home', '?')}, {game_info.get('date', '?')}",
        f"Final: {game_info.get('away', '?')} {game_info.get('away_score', '?')}, "
        f"{game_info.get('home', '?')} {game_info.get('home_score', '?')}",
        f"Managed team: {game_info.get('managed_side', '?').title()}",
        f"Decision points: {total} ({len(active_real)} active manager actions, "
        f"{len(active_agent)} active agent actions)",
        "",
        "--- Agreement Summary ---",
        f"Category agreement: {category_matches}/{total} ({100*category_matches/total:.1f}%)",
        f"Active action matches: {active_matches}/{len(active_real)} active decisions matched"
        if active_real else "No active manager decisions in this game",
        "",
    ]

    if disagreements:
        lines.append("--- Disagreements (by leverage index) ---")
        lines.append(f"{'#':>2s}  {'Inn':>4s}  {'Situation':<25s}  {'LI':>5s}  {'Agent':20s}  {'Real':20s}")
        for i, e in enumerate(disagreements[:10], 1):
            half_str = "T" if e.half == "TOP" else "B"
            situation = f"{half_str}{e.inning} {e.outs}out {e.score_away}-{e.score_home}"
            lines.append(
                f"{i:2d}  {half_str}{e.inning:>3d}  {situation:<25s}  {e.leverage_index:5.2f}  "
                f"{e.agent_decision_type:20s}  {e.real_manager_action_type:20s}"
            )
            if e.agent_action_details:
                lines.append(f"     Agent: {e.agent_action_details[:80]}")
            if e.real_manager_action_details:
                lines.append(f"     Real:  {e.real_manager_action_details[:80]}")
        lines.append("")

    # High-leverage highlights
    hi_li = [e for e in entries if e.leverage_index >= 1.5]
    if hi_li:
        lines.append(f"--- High-Leverage Decisions (LI >= 1.5): {len(hi_li)} ---")
        for e in sorted(hi_li, key=lambda x: x.leverage_index, reverse=True)[:5]:
            match_str = "MATCH" if e.category_match else "DISAGREE"
            half_str = "T" if e.half == "TOP" else "B"
            lines.append(
                f"  {half_str}{e.inning} {e.outs}out LI={e.leverage_index:.2f}: "
                f"Agent={e.agent_decision_type}, Real={e.real_manager_action_type} [{match_str}]"
            )
        lines.append("")

    lines.append("--- Cost ---")
    lines.append(
        f"API calls: {total} | Tokens: {total_tokens:,} "
        f"(input: {total_input:,}, output: {total_output:,})"
    )

    return "\n".join(lines)
