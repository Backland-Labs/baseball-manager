# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.78.0",
#     "pydantic>=2.0",
# ]
# ///
"""Baseball Manager AI Agent -- main entry point.

Run with:  uv run game.py           # full agent game (requires ANTHROPIC_API_KEY)
           uv run game.py --dry-run  # validate setup without API calls
           uv run game.py --sim      # run automated sim (no agent, no API key needed)
           uv run game.py --seed 42  # set random seed
           uv run game.py --away     # agent manages away team (default: home)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

from anthropic import Anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Claude API rate limit constants
# ---------------------------------------------------------------------------

CLAUDE_MAX_RETRIES = 5
CLAUDE_BACKOFF_BASE = 2.0  # seconds; actual delay = base * 2^attempt + jitter

from models import (
    BatterInfo,
    BattingTeam,
    BullpenPitcher,
    BullpenRole,
    Count,
    Freshness,
    Half,
    Hand,
    LineupPlayer,
    ManagerDecision,
    MatchupState,
    OnDeckBatter,
    OpponentBenchPlayer,
    OpponentBullpenPitcher,
    OpponentRosterState,
    PitcherInfo,
    RosterState,
    Runners,
    Score,
    ThrowHand,
)
from tools import ALL_TOOLS
from simulation import (
    SimulationEngine,
    GameState,
    PlayEvent,
    load_rosters,
    game_state_to_scenario,
    validate_and_apply_decision,
    DecisionResult,
    game_state_to_dict,
)
from decision_quality_wpa import (
    compute_wp_from_game_state,
    compute_li_from_game_state,
    score_decision,
    generate_game_wpa_report,
    format_wpa_report,
    DecisionWPA,
    GameWPAReport,
)


# ---------------------------------------------------------------------------
# System prompt -- loaded from AGENT_PROMPT.md at runtime
# ---------------------------------------------------------------------------

_AGENT_PROMPT_PATH = Path(__file__).parent / "AGENT_PROMPT.md"


def load_system_prompt(path: Path | None = None) -> str:
    """Load the agent system prompt from AGENT_PROMPT.md.

    Args:
        path: Override path to the prompt file. Defaults to AGENT_PROMPT.md
              in the project root (same directory as game.py).

    Returns:
        The contents of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    prompt_path = path or _AGENT_PROMPT_PATH
    return prompt_path.read_text(encoding="utf-8")


# Load at module level so it's available throughout the module.
# This is the prompt that gets sent to the Claude agent as the system message.
SYSTEM_PROMPT = load_system_prompt()


# ---------------------------------------------------------------------------
# Validation-only check (no game state mutation)
# ---------------------------------------------------------------------------

def _peek_validate(game_state: GameState, decision: dict,
                   managed_team: str) -> DecisionResult:
    """Check if a decision would be valid WITHOUT applying it to game state.

    This performs the same validation checks as validate_and_apply_decision
    but never modifies the game state. Used by the retry loop to decide
    whether to re-prompt the agent before the actual apply step.

    Returns:
        DecisionResult with valid/error fields set. The events list is always
        empty since nothing is applied.
    """
    is_home = managed_team == "home"
    our_team = game_state.home if is_home else game_state.away
    bt = game_state.batting_team()
    ft = game_state.fielding_team()
    we_are_batting = (bt == our_team)

    decision_type = decision.get("decision", "").upper().strip()

    # No-action decisions are always valid
    no_action_types = {
        "NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
        "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER",
    }
    if decision_type in no_action_types or not decision_type:
        return DecisionResult(valid=True)

    # PITCHING_CHANGE / PULL_STARTER
    if decision_type in ("PITCHING_CHANGE", "PULL_STARTER", "BRING_IN_RELIEVER"):
        if we_are_batting:
            return DecisionResult(valid=False, error="Cannot make pitching change while batting")
        if ft.current_pitcher_batters_faced_this_stint < 3 and game_state.outs < 3:
            return DecisionResult(
                valid=False,
                error=f"3-batter minimum not met ({ft.current_pitcher_batters_faced_this_stint} faced). "
                      f"Pitcher must face at least 3 batters before being removed."
            )
        from simulation import _extract_player_id
        new_pitcher_id = _extract_player_id(decision.get("action_details", ""), our_team)
        if not new_pitcher_id:
            available = [p for p in our_team.bullpen if p.player_id not in our_team.used_pitchers]
            if not available:
                return DecisionResult(valid=False, error="No available relievers in bullpen")
        else:
            new_pitcher = our_team.get_player_by_id(new_pitcher_id)
            if not new_pitcher:
                return DecisionResult(valid=False, error=f"Player {new_pitcher_id} not found on roster")
            if new_pitcher_id in our_team.used_pitchers:
                return DecisionResult(valid=False, error=f"Pitcher {new_pitcher.name} has already been used")
        return DecisionResult(valid=True)

    # PINCH_HIT
    if decision_type in ("PINCH_HIT", "PINCH_HITTER"):
        if not we_are_batting:
            return DecisionResult(valid=False, error="Cannot pinch hit while fielding")
        from simulation import _extract_player_id, _extract_player_id_from_bench
        pinch_hitter_id = _extract_player_id_from_bench(decision.get("action_details", ""), our_team)
        if not pinch_hitter_id:
            pinch_hitter_id = _extract_player_id(decision.get("action_details", ""), our_team)
        if not pinch_hitter_id:
            return DecisionResult(valid=False, error="Could not identify pinch hitter from action details")
        pinch_hitter = our_team.get_player_by_id(pinch_hitter_id)
        if not pinch_hitter:
            return DecisionResult(valid=False, error=f"Player {pinch_hitter_id} not found on roster")
        if pinch_hitter_id in our_team.removed_players:
            return DecisionResult(valid=False, error=f"{pinch_hitter.name} has already been removed from game")
        return DecisionResult(valid=True)

    # STOLEN_BASE
    if decision_type in ("STOLEN_BASE", "STEAL"):
        if not we_are_batting:
            return DecisionResult(valid=False, error="Cannot attempt steal while fielding")
        r1 = game_state.runner_on(1)
        r2 = game_state.runner_on(2)
        runner = None
        if r1 and not game_state.runner_on(2):
            runner = r1
        elif r2 and not game_state.runner_on(3):
            runner = r2
        elif r1:
            runner = r1
        if not runner:
            return DecisionResult(valid=False, error="No eligible runner for stolen base attempt")
        return DecisionResult(valid=True)

    # INTENTIONAL_WALK
    if decision_type in ("INTENTIONAL_WALK", "IBB"):
        if we_are_batting:
            return DecisionResult(valid=False, error="Cannot issue intentional walk while batting")
        return DecisionResult(valid=True)

    # DEFENSIVE_POSITIONING
    if decision_type in ("DEFENSIVE_POSITIONING", "SHIFT", "INFIELD_IN", "POSITION_CHANGE"):
        if we_are_batting:
            return DecisionResult(valid=False, error="Cannot change defensive positioning while batting")
        return DecisionResult(valid=True)

    # MOUND_VISIT
    if decision_type in ("MOUND_VISIT",):
        if we_are_batting:
            return DecisionResult(valid=False, error="Cannot make mound visit while batting")
        if our_team.mound_visits_remaining <= 0:
            return DecisionResult(valid=False, error="No mound visits remaining")
        return DecisionResult(valid=True)

    # SACRIFICE_BUNT / BUNT
    if decision_type in ("SACRIFICE_BUNT", "BUNT", "SQUEEZE"):
        if not we_are_batting:
            return DecisionResult(valid=False, error="Cannot bunt while fielding")
        return DecisionResult(valid=True)

    # PINCH_RUN
    if decision_type in ("PINCH_RUN", "PINCH_RUNNER"):
        if not we_are_batting:
            return DecisionResult(valid=False, error="Cannot pinch run while fielding")
        return DecisionResult(valid=True)

    # REPLAY_CHALLENGE
    if decision_type in ("REPLAY_CHALLENGE", "CHALLENGE"):
        if not our_team.challenge_available:
            return DecisionResult(valid=False, error="No challenge available")
        return DecisionResult(valid=True)

    # Unknown decision types are treated as no-action (valid)
    return DecisionResult(valid=True)


# ---------------------------------------------------------------------------
# Agent decision loop
# ---------------------------------------------------------------------------

def _claude_backoff_sleep(attempt: int, retry_after: float | None = None) -> None:
    """Sleep with exponential backoff and jitter for Claude API retries.

    Args:
        attempt: Zero-based retry attempt number.
        retry_after: Optional server-requested delay (from Retry-After
            or ``x-retry-after`` headers).
    """
    base_delay = CLAUDE_BACKOFF_BASE * (2 ** attempt)
    jitter = random.random() * base_delay
    delay = base_delay + jitter
    if retry_after is not None and retry_after > delay:
        delay = retry_after
    time.sleep(delay)


def _extract_retry_after(exc: Exception) -> float | None:
    """Try to extract a Retry-After value from an Anthropic API error.

    Anthropic rate limit errors often carry a ``response`` attribute
    with headers.  We look for ``retry-after`` or ``x-retry-after``.

    Args:
        exc: The exception to inspect.

    Returns:
        Seconds to wait, or ``None`` if no value can be extracted.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for key in ("retry-after", "x-retry-after"):
        val = headers.get(key)
        if val is not None:
            try:
                return max(0.0, float(val))
            except (TypeError, ValueError):
                pass
    return None


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check whether an Anthropic exception is a 429 rate limit error."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    # Some SDK versions use 'status' instead of 'status_code'
    status = getattr(exc, "status", None)
    if status == 429:
        return True
    # Check the response object
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status == 429:
            return True
    return False


def _call_agent(client: Anthropic, messages: list[dict],
                verbose: bool = True) -> tuple[dict, object | None, dict]:
    """Send messages to the Claude agent and extract a ManagerDecision.

    Includes retry logic for Claude API rate limits (429) with
    exponential backoff and jitter.

    Args:
        client: Anthropic API client.
        messages: Full conversation history including the latest user message.
        verbose: Print agent activity.

    Returns:
        Tuple of (decision_dict, final_message, call_metadata).
        call_metadata contains tool_calls, token_usage, agent_turns,
        and rate_limit_retries.

    Raises:
        Exception: Re-raised after exhausting rate-limit retries.
    """
    tool_calls: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    rate_limit_retries = 0

    # Retry loop for rate-limit (429) errors on the initial API call
    runner = None
    for rl_attempt in range(CLAUDE_MAX_RETRIES):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*output_format.*deprecated.*")
                runner = client.beta.messages.tool_runner(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=ALL_TOOLS,
                    output_format=ManagerDecision,
                    messages=messages,
                )
            break  # Success -- exit retry loop
        except Exception as exc:
            if _is_rate_limit_error(exc):
                rate_limit_retries += 1
                retry_after = _extract_retry_after(exc)
                logger.warning(
                    "Claude API rate limit (429) on attempt %d/%d "
                    "(Retry-After: %s)",
                    rl_attempt + 1, CLAUDE_MAX_RETRIES,
                    retry_after if retry_after is not None else "not set",
                )
                if verbose:
                    print(f"    [Rate Limit] Claude API 429, retry {rl_attempt + 1}/{CLAUDE_MAX_RETRIES}")
                if rl_attempt < CLAUDE_MAX_RETRIES - 1:
                    _claude_backoff_sleep(rl_attempt, retry_after=retry_after)
                    continue
            raise  # Non-rate-limit error or retries exhausted

    if runner is None:
        raise RuntimeError("Failed to create agent runner after rate limit retries")

    turn = 0
    final_message = None
    for message in runner:
        turn += 1
        # Accumulate token usage from each message
        if hasattr(message, "usage") and message.usage:
            total_input_tokens += getattr(message.usage, "input_tokens", 0)
            total_output_tokens += getattr(message.usage, "output_tokens", 0)

        for block in message.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "tool_name": block.name,
                    "tool_input": block.input,
                })
                if verbose:
                    args_str = json.dumps(block.input, separators=(",", ":"))
                    if len(args_str) > 80:
                        args_str = args_str[:80] + "..."
                    print(f"    [Agent] Tool: {block.name}({args_str})")
            elif block.type == "text" and block.text.strip() and verbose:
                text = block.text.strip()
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"    [Agent] {text}")
        final_message = message

    # Extract decision
    decision_dict = None
    if final_message and hasattr(final_message, "parsed") and final_message.parsed:
        decision: ManagerDecision = final_message.parsed
        decision_dict = decision.model_dump()
        if verbose:
            print(f"    [Decision] {decision.decision}: {decision.action_details}")
    else:
        # Fallback: no structured output
        if verbose:
            print("    [Decision] No structured output received, defaulting to NO_ACTION")
        decision_dict = {
            "decision": "NO_ACTION",
            "action_details": "No valid decision received, proceeding with default",
            "confidence": 0.0,
            "reasoning": "Agent did not return structured output",
            "key_factors": [],
            "risks": [],
        }

    call_metadata = {
        "tool_calls": tool_calls,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        "agent_turns": turn,
        "rate_limit_retries": rate_limit_retries,
    }

    return decision_dict, final_message, call_metadata


def run_agent_decision(client: Anthropic, game_state: GameState,
                       managed_team: str, messages: list[dict],
                       verbose: bool = True,
                       max_retries: int = 5) -> tuple[dict, list[dict], dict]:
    """Present the current game state to the agent and get a ManagerDecision.

    If the agent's decision is invalid, it is re-prompted with the error
    message so it can correct its decision. After ``max_retries`` consecutive
    invalid decisions, a forced NO_ACTION is returned.

    Args:
        client: Anthropic API client.
        game_state: Current authoritative game state.
        managed_team: "home" or "away".
        messages: Conversation history (mutated in place with new messages).
        verbose: Print agent activity.
        max_retries: Maximum number of retry attempts for invalid decisions.

    Returns:
        Tuple of (decision_dict, updated_messages, decision_metadata).
        decision_metadata contains tool_calls, token_usage, latency_ms,
        and retries.
    """
    from simulation import validate_and_apply_decision as _validate

    start_time = time.time()
    all_tool_calls: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_agent_turns = 0

    scenario = game_state_to_scenario(game_state, managed_team)

    user_message = (
        "Here is the current game scenario:\n\n"
        f"**Matchup State:**\n```json\n{json.dumps(scenario['matchup_state'], indent=2)}\n```\n\n"
        f"**Roster State:**\n```json\n{json.dumps(scenario['roster_state'], indent=2)}\n```\n\n"
        f"**Opponent Roster State:**\n```json\n{json.dumps(scenario['opponent_roster_state'], indent=2)}\n```\n\n"
        f"**Decision Needed:** {scenario['decision_prompt']}"
    )

    messages.append({"role": "user", "content": user_message})

    # First attempt
    try:
        decision_dict, final_message, call_meta = _call_agent(client, messages, verbose=verbose)
        all_tool_calls.extend(call_meta["tool_calls"])
        total_input_tokens += call_meta["token_usage"]["input_tokens"]
        total_output_tokens += call_meta["token_usage"]["output_tokens"]
        total_agent_turns += call_meta["agent_turns"]
    except Exception as e:
        if verbose:
            print(f"    [Error] Agent call failed: {e}")
        decision_dict = {
            "decision": "NO_ACTION",
            "action_details": f"Agent error: {e}",
            "confidence": 0.0,
            "reasoning": "Agent call failed",
            "key_factors": [],
            "risks": [],
        }
        final_message = None

    # Add assistant response to messages for context continuity
    if final_message:
        messages.append({"role": "assistant", "content": final_message.content})

    # Validate (dry -- we only check, we do NOT apply yet)
    result = _peek_validate(game_state, decision_dict, managed_team)

    retries = 0
    while not result.valid and retries < max_retries:
        retries += 1
        if verbose:
            print(f"    [Invalid] {result.error} (retry {retries}/{max_retries})")

        # Send the error back to the agent so it can correct its decision
        error_msg = (
            f"Your previous decision was INVALID: {result.error}\n\n"
            f"Please reconsider and provide a valid ManagerDecision. "
            f"Remember: if no strategic move is needed, use NO_ACTION."
        )
        messages.append({"role": "user", "content": error_msg})

        try:
            decision_dict, final_message, call_meta = _call_agent(client, messages, verbose=verbose)
            all_tool_calls.extend(call_meta["tool_calls"])
            total_input_tokens += call_meta["token_usage"]["input_tokens"]
            total_output_tokens += call_meta["token_usage"]["output_tokens"]
            total_agent_turns += call_meta["agent_turns"]
        except Exception as e:
            if verbose:
                print(f"    [Error] Agent retry call failed: {e}")
            decision_dict = {
                "decision": "NO_ACTION",
                "action_details": f"Agent retry error: {e}",
                "confidence": 0.0,
                "reasoning": "Agent call failed on retry",
                "key_factors": [],
                "risks": [],
            }
            final_message = None

        if final_message:
            messages.append({"role": "assistant", "content": final_message.content})

        result = _peek_validate(game_state, decision_dict, managed_team)

    # If still invalid after all retries, force NO_ACTION
    if not result.valid:
        if verbose:
            print(f"    [Forced] {max_retries} consecutive invalid decisions, forcing NO_ACTION")
        decision_dict = {
            "decision": "NO_ACTION",
            "action_details": "Forced no-action after repeated invalid decisions",
            "confidence": 0.0,
            "reasoning": f"Agent failed to produce a valid decision after {max_retries} retries. Last error: {result.error}",
            "key_factors": [],
            "risks": [],
        }

    elapsed_ms = round((time.time() - start_time) * 1000)

    decision_metadata = {
        "tool_calls": all_tool_calls,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        "latency_ms": elapsed_ms,
        "agent_turns": total_agent_turns,
        "retries": retries,
    }

    return decision_dict, messages, decision_metadata


NO_ACTION_TYPES = {
    "NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
    "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER",
}


def build_decision_log_entry(
    turn: int,
    game_state: GameState,
    managed_team: str,
    decision_dict: dict,
    decision_metadata: dict,
    timestamp: float,
) -> dict:
    """Build a comprehensive decision log entry with full game context.

    Each entry captures everything needed to replay and analyze a decision:
    game state, tool calls with parameters, the full decision response,
    token usage, and latency.

    Args:
        turn: Agent invocation counter (1-indexed).
        game_state: Current game state at decision time.
        managed_team: "home" or "away".
        decision_dict: The agent's decision dict.
        decision_metadata: Metadata from run_agent_decision (tool_calls,
            token_usage, latency_ms, retries).
        timestamp: Wall-clock time of the decision.

    Returns:
        Structured dict suitable for JSON serialization.
    """
    bt = game_state.batting_team()
    ft = game_state.fielding_team()
    pitcher = game_state.current_pitcher()
    batter = bt.current_batter()

    # Runner state
    runners = {}
    for base in (1, 2, 3):
        r = game_state.runner_on(base)
        if r:
            runners[str(base)] = {"player_id": r.player.player_id, "name": r.player.name}

    # Determine if active decision
    decision_type = decision_dict.get("decision", "").upper().strip()
    is_active = decision_type not in NO_ACTION_TYPES and bool(decision_type)

    return {
        "turn": turn,
        "timestamp": timestamp,
        # Full game state context
        "game_state": {
            "inning": game_state.inning,
            "half": game_state.half,
            "outs": game_state.outs,
            "score": {"home": game_state.score_home, "away": game_state.score_away},
            "runners": runners,
            "batter": {"player_id": batter.player_id, "name": batter.name},
            "pitcher": {"player_id": pitcher.player_id, "name": pitcher.name},
            "situation": game_state.situation_display(),
        },
        "managed_team": managed_team,
        # Tool calls the agent made (with parameters)
        "tool_calls": decision_metadata.get("tool_calls", []),
        # The agent's full decision response
        "decision": decision_dict,
        "is_active_decision": is_active,
        # WPA scoring -- wp_before and leverage_index are known at decision
        # time; wp_after and wpa are filled in after the play resolves.
        "wp_before": decision_metadata.get("wp_before"),
        "leverage_index": decision_metadata.get("leverage_index"),
        "wp_after": None,
        "wpa": None,
        # Token usage and performance
        "token_usage": decision_metadata.get("token_usage", {}),
        "latency_ms": decision_metadata.get("latency_ms", 0),
        "agent_turns": decision_metadata.get("agent_turns", 0),
        "retries": decision_metadata.get("retries", 0),
    }


def write_game_log(
    game_state: GameState,
    decision_log: list[dict],
    error_log: list[dict],
    seed: int,
    managed_team: str,
    log_dir: Path | None = None,
    wpa_report: GameWPAReport | None = None,
) -> Path:
    """Write the complete game decision log to a structured JSON file.

    Log files are organized by game in data/game_logs/, with one file per
    game identified by seed and timestamp.

    Args:
        game_state: Final game state.
        decision_log: List of decision log entries from build_decision_log_entry.
        error_log: List of error entries.
        seed: Game seed for identification.
        managed_team: Which team the agent managed.
        log_dir: Override directory for logs (default: data/game_logs/).
        wpa_report: Optional WPA report for decision quality scoring.

    Returns:
        Path to the written log file.
    """
    if log_dir is None:
        log_dir = Path(__file__).parent / "data" / "game_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Count active vs no-action decisions
    active_count = sum(1 for d in decision_log if d.get("is_active_decision"))
    no_action_count = len(decision_log) - active_count

    # Aggregate token usage
    total_input = sum(d.get("token_usage", {}).get("input_tokens", 0) for d in decision_log)
    total_output = sum(d.get("token_usage", {}).get("output_tokens", 0) for d in decision_log)
    total_latency = sum(d.get("latency_ms", 0) for d in decision_log)

    game_log = {
        "game_info": {
            "seed": seed,
            "home_team": game_state.home.name,
            "away_team": game_state.away.name,
            "managed_team": managed_team,
            "final_score": {
                "home": game_state.score_home,
                "away": game_state.score_away,
            },
            "winner": game_state.winning_team,
            "innings": game_state.inning,
        },
        "summary": {
            "total_decisions": len(decision_log),
            "active_decisions": active_count,
            "no_action_decisions": no_action_count,
            "total_errors": len(error_log),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_latency_ms": total_latency,
        },
        "decisions": decision_log,
        "errors": error_log,
    }

    # Include WPA report if available
    if wpa_report is not None:
        game_log["wpa_report"] = wpa_report.to_dict()

    log_path = log_dir / f"game_{seed}.json"
    with open(log_path, "w") as f:
        json.dump(game_log, f, indent=2, default=str)

    return log_path


def _update_last_decision_wp_after(
    decision_log: list[dict], game_state: GameState, managed_team: str,
) -> None:
    """Update the most recent decision log entry with wp_after and wpa.

    Called after the game state changes (PA resolves, IBB, CS, etc.)
    to fill in the post-decision win probability.
    """
    if not decision_log:
        return
    entry = decision_log[-1]
    if entry.get("wp_after") is not None:
        return  # already set

    if game_state.game_over:
        # Game is over: WP is 1.0 or 0.0 based on outcome
        is_home = managed_team == "home"
        if is_home:
            wp_after = 1.0 if game_state.score_home > game_state.score_away else 0.0
        else:
            wp_after = 1.0 if game_state.score_away > game_state.score_home else 0.0
    else:
        wp_after = compute_wp_from_game_state(game_state, managed_team)

    entry["wp_after"] = round(wp_after, 4)
    wp_before = entry.get("wp_before")
    if wp_before is not None:
        entry["wpa"] = round(wp_after - wp_before, 4)


def run_agent_game(seed: int | None = None, managed_team: str = "home",
                   verbose: bool = True, max_innings: int = 15,
                   max_consecutive_failures: int = 5) -> GameState:
    """Run a full game with the Claude agent managing one team.

    The agent is consulted at each decision point. If its decision is invalid,
    run_agent_decision re-prompts the agent with the error message, giving it
    up to ``max_consecutive_failures`` retries before forcing NO_ACTION. The
    opposing team uses automated management.

    Args:
        seed: Random seed for deterministic replay.
        managed_team: "home" or "away".
        verbose: Print play-by-play and agent activity.
        max_innings: Safety limit for extra innings.
        max_consecutive_failures: Max invalid-decision retries per decision
            point (forwarded to run_agent_decision).

    Returns:
        Final GameState.
    """
    client = Anthropic()
    rosters = load_rosters()
    engine = SimulationEngine(seed=seed)

    game = engine.initialize_game(rosters)
    messages: list[dict] = []
    decision_log: list[dict] = []
    error_log: list[dict] = []  # Errors logged separately from play-by-play
    total_agent_calls = 0

    if verbose:
        print("=" * 72)
        print("BASEBALL MANAGER AI AGENT -- Full Game")
        print("=" * 72)
        print(f"  {rosters['away']['team_name']} at {rosters['home']['team_name']}")
        print(f"  Agent manages: {'Home' if managed_team == 'home' else 'Away'} team")
        print(f"  Seed: {engine.seed}")
        print("=" * 72)
        print()

    # Print initial event
    if verbose and game.play_log:
        print(game.play_log[0].description)

    while not game.game_over:
        if game.inning > max_innings:
            game.game_over = True
            if game.score_home == game.score_away:
                game.winning_team = "TIE (innings limit)"
            else:
                game.winning_team = (game.home.name if game.score_home > game.score_away
                                     else game.away.name)
            break

        bt = game.batting_team()
        ft = game.fielding_team()
        is_home = managed_team == "home"
        our_team = game.home if is_home else game.away

        # Determine if agent needs to make a decision
        we_are_batting = (bt == our_team)
        we_are_fielding = (ft == our_team)

        agent_decides = we_are_batting or we_are_fielding

        if agent_decides:
            # Trim messages to manage context window
            if len(messages) > 20:
                # Keep system context fresh -- only retain last few exchanges
                messages = messages[-10:]

            # Get agent decision (with built-in retry on invalid decisions)
            total_agent_calls += 1

            if verbose:
                situation = game.situation_display()
                batter = bt.current_batter()
                pitcher = game.current_pitcher()
                print(f"\n  [{situation}] {batter.name} vs {pitcher.name}")

            # Compute WP and LI before the decision for WPA scoring
            wp_before = compute_wp_from_game_state(game, managed_team)
            li_before = compute_li_from_game_state(game, managed_team)

            decision_metadata = {}
            try:
                decision_dict, messages, decision_metadata = run_agent_decision(
                    client, game, managed_team, messages,
                    verbose=verbose, max_retries=max_consecutive_failures,
                )
            except Exception as e:
                if verbose:
                    print(f"    [Error] Agent call failed: {e}")
                error_log.append({
                    "turn": total_agent_calls,
                    "inning": game.inning,
                    "half": game.half,
                    "error_type": "agent_call_failure",
                    "error": str(e),
                    "timestamp": time.time(),
                })
                decision_dict = {
                    "decision": "NO_ACTION",
                    "action_details": f"Agent error: {e}",
                    "confidence": 0.0,
                    "reasoning": "Agent call failed",
                    "key_factors": [],
                    "risks": [],
                }

            # Attach WP_before and LI to decision metadata for logging
            decision_metadata["wp_before"] = wp_before
            decision_metadata["leverage_index"] = li_before

            # Build comprehensive decision log entry
            decision_entry = build_decision_log_entry(
                turn=total_agent_calls,
                game_state=game,
                managed_team=managed_team,
                decision_dict=decision_dict,
                decision_metadata=decision_metadata,
                timestamp=time.time(),
            )
            decision_log.append(decision_entry)

            # Apply the decision (should be valid after retry loop, but
            # handle any edge cases defensively)
            result = validate_and_apply_decision(game, decision_dict, managed_team, engine)

            if not result.valid:
                # This should be rare since run_agent_decision pre-validates.
                # Log the error (not in play-by-play) and proceed as no-action.
                error_log.append({
                    "turn": total_agent_calls,
                    "inning": game.inning,
                    "half": game.half,
                    "error_type": "invalid_decision_at_apply",
                    "decision": decision_dict.get("decision", ""),
                    "error": result.error,
                    "timestamp": time.time(),
                })
                if verbose:
                    print(f"    [Invalid] {result.error}")
            else:
                if result.description and verbose:
                    for event in result.events:
                        if event.event_type == "decision":
                            print(f"    >> {event.description}")

                # If the decision consumed the plate appearance (e.g., IBB, stolen base out),
                # skip the normal PA simulation
                if decision_dict.get("decision", "").upper() in ("INTENTIONAL_WALK", "IBB"):
                    if verbose:
                        for event in result.events:
                            print(f"  {event.description}")
                    # Compute wp_after for this decision (IBB consumed the PA)
                    _update_last_decision_wp_after(decision_log, game, managed_team)
                    # Check if game ended on walk-off IBB
                    if game.game_over:
                        break
                    continue

                # Stolen base: the PA still happens unless caught stealing caused 3rd out
                if decision_dict.get("decision", "").upper() in ("STOLEN_BASE", "STEAL"):
                    if verbose:
                        for event in result.events:
                            if event.event_type != "inning_change":
                                print(f"  {event.description}")
                            else:
                                print(f"\n{event.description}")
                    if game.game_over or game.outs >= 3:
                        # Compute wp_after for this decision (CS ended the inning)
                        _update_last_decision_wp_after(decision_log, game, managed_team)
                        continue
        else:
            # Opponent's turn: use automated management
            engine._auto_manage_pitcher(game)

        # Simulate plate appearance
        pa_result = engine.simulate_plate_appearance(game)

        if verbose:
            print(f"  {pa_result['description']}")
            for d in pa_result.get("detail_descriptions", []):
                if d:
                    print(f"    {d}")

        # Apply result
        events = engine.apply_pa_result(game, pa_result)

        if verbose:
            for e in events:
                if e.event_type == "inning_change":
                    print(f"\n{e.description}")
                elif e.event_type == "game_end":
                    print(f"\n{e.description}")
                elif e.runs_scored > 0:
                    print(f"  Score: Away {game.score_away} - Home {game.score_home}")

        # Compute wp_after for the most recent agent decision (after PA resolved)
        if decision_log and decision_log[-1].get("wp_after") is None:
            _update_last_decision_wp_after(decision_log, game, managed_team)

    # Ensure last decision has wp_after filled in
    if decision_log and decision_log[-1].get("wp_after") is None:
        _update_last_decision_wp_after(decision_log, game, managed_team)

    # Generate WPA report from decision log
    wpa_scores: list[DecisionWPA] = []
    for entry in decision_log:
        wp_b = entry.get("wp_before")
        wp_a = entry.get("wp_after")
        if wp_b is not None and wp_a is not None:
            gs = entry.get("game_state", {})
            score = gs.get("score", {})
            is_home = managed_team == "home"
            if is_home:
                sd = score.get("home", 0) - score.get("away", 0)
            else:
                sd = score.get("away", 0) - score.get("home", 0)

            runners_dict = gs.get("runners", {})
            from decision_quality_wpa import _runners_key
            runner_key = _runners_key("1" in runners_dict, "2" in runners_dict, "3" in runners_dict)

            dwpa = score_decision(
                wp_before=wp_b,
                wp_after=wp_a,
                decision_dict=entry.get("decision", {}),
                leverage_index=entry.get("leverage_index", 1.0),
                turn=entry.get("turn", 0),
                inning=gs.get("inning", 0),
                half=gs.get("half", ""),
                outs=gs.get("outs", 0),
                score_diff=sd,
                runners=runner_key,
            )
            wpa_scores.append(dwpa)

    wpa_report = generate_game_wpa_report(wpa_scores)

    # Game over -- print summary
    if verbose:
        print()
        print(engine.print_box_score(game))
        decisions_summary = engine.generate_decisions_summary(game)
        if decisions_summary and "No managerial decisions" not in decisions_summary:
            print()
            print(decisions_summary)
        print(f"\nAgent decisions: {total_agent_calls}")
        print(f"Decision log entries: {len(decision_log)}")
        active_count = sum(1 for d in decision_log if d.get("is_active_decision"))
        print(f"Active decisions: {active_count}")
        if error_log:
            print(f"Error log entries: {len(error_log)}")

        # Print WPA report
        if wpa_scores:
            print()
            print(format_wpa_report(wpa_report))

    # Save structured game decision log
    try:
        log_path = write_game_log(
            game_state=game,
            decision_log=decision_log,
            error_log=error_log,
            seed=engine.seed,
            managed_team=managed_team,
            wpa_report=wpa_report,
        )
        if verbose:
            print(f"Decision log saved to: {log_path}")
    except Exception:
        pass  # Non-critical

    return game


# ---------------------------------------------------------------------------
# Sample scenario (for dry-run and single-turn tests)
# ---------------------------------------------------------------------------

def build_sample_scenario() -> dict:
    """Build a minimal sample scenario for testing the agent."""
    matchup = MatchupState(
        inning=7,
        half=Half.BOTTOM,
        outs=1,
        count=Count(balls=1, strikes=1),
        runners=Runners(),
        score=Score(home=3, away=4),
        batting_team=BattingTeam.HOME,
        batter=BatterInfo(
            player_id="h_003",
            name="Rafael Ortiz",
            bats=Hand.L,
            lineup_position=3,
        ),
        pitcher=PitcherInfo(
            player_id="a_sp1",
            name="Matt Henderson",
            throws=ThrowHand.L,
            pitch_count_today=88,
            batters_faced_today=24,
            times_through_order=3,
            innings_pitched_today=6.1,
            runs_allowed_today=3,
            today_line={"IP": 6.1, "H": 7, "R": 3, "ER": 3, "BB": 2, "K": 6},
        ),
        on_deck_batter=OnDeckBatter(
            player_id="h_004",
            name="Tyrone Jackson",
            bats=Hand.R,
        ),
    )

    roster = RosterState(
        our_lineup=[
            LineupPlayer(player_id="h_001", name="Marcus Chen", position="CF", bats=Hand.L),
            LineupPlayer(player_id="h_002", name="Derek Williams", position="SS", bats=Hand.R),
            LineupPlayer(player_id="h_003", name="Rafael Ortiz", position="1B", bats=Hand.L),
            LineupPlayer(player_id="h_004", name="Tyrone Jackson", position="RF", bats=Hand.R),
            LineupPlayer(player_id="h_005", name="Jake Morrison", position="3B", bats=Hand.R),
            LineupPlayer(player_id="h_006", name="Shin-Soo Park", position="DH", bats=Hand.L),
            LineupPlayer(player_id="h_007", name="Carlos Ramirez", position="LF", bats=Hand.S),
            LineupPlayer(player_id="h_008", name="Tommy Sullivan", position="C", bats=Hand.R),
            LineupPlayer(player_id="h_009", name="Andre Davis", position="2B", bats=Hand.L),
        ],
        our_lineup_position=2,
        bench=[
            {"player_id": "h_012", "name": "Darnell Washington", "bats": "L", "positions": ["LF", "RF"], "available": True},
            {"player_id": "h_013", "name": "Eduardo Reyes", "bats": "L", "positions": ["1B", "DH"], "available": True},
            {"player_id": "h_014", "name": "Kenji Tanaka", "bats": "R", "positions": ["CF", "LF", "RF"], "available": True},
            {"player_id": "h_010", "name": "Victor Nguyen", "bats": "R", "positions": ["C", "1B"], "available": True},
            {"player_id": "h_011", "name": "Ryan O'Brien", "bats": "R", "positions": ["2B", "SS", "3B"], "available": True},
        ],
        bullpen=[
            BullpenPitcher(player_id="h_bp1", name="Greg Foster", throws=ThrowHand.R, role=BullpenRole.CLOSER, freshness=Freshness.FRESH, days_since_last_appearance=2),
            BullpenPitcher(player_id="h_bp2", name="Luis Herrera", throws=ThrowHand.L, role=BullpenRole.SETUP, freshness=Freshness.MODERATE, days_since_last_appearance=1),
            BullpenPitcher(player_id="h_bp3", name="Marcus Webb", throws=ThrowHand.R, role=BullpenRole.SETUP, freshness=Freshness.FRESH, days_since_last_appearance=3),
            BullpenPitcher(player_id="h_bp4", name="Danny Kim", throws=ThrowHand.R, role=BullpenRole.MIDDLE, freshness=Freshness.FRESH, days_since_last_appearance=4),
            BullpenPitcher(player_id="h_bp5", name="Alex Turner", throws=ThrowHand.L, role=BullpenRole.MIDDLE, freshness=Freshness.TIRED, days_since_last_appearance=0, pitches_last_3_days=[25, 18, 0]),
            BullpenPitcher(player_id="h_bp6", name="Jason Blake", throws=ThrowHand.R, role=BullpenRole.LONG, freshness=Freshness.FRESH, days_since_last_appearance=5),
            BullpenPitcher(player_id="h_bp7", name="Chris Evans", throws=ThrowHand.R, role=BullpenRole.MOPUP, freshness=Freshness.FRESH, days_since_last_appearance=3),
            BullpenPitcher(player_id="h_bp8", name="Sam Rodriguez", throws=ThrowHand.R, role=BullpenRole.MOPUP, freshness=Freshness.FRESH, days_since_last_appearance=4),
        ],
        mound_visits_remaining=4,
        challenge_available=True,
    )

    opponent_roster = OpponentRosterState(
        their_lineup=[
            LineupPlayer(player_id="a_001", name="Jordan Bell", position="2B", bats=Hand.R),
            LineupPlayer(player_id="a_002", name="Liam O'Connor", position="CF", bats=Hand.L),
            LineupPlayer(player_id="a_003", name="Anthony Russo", position="DH", bats=Hand.R),
            LineupPlayer(player_id="a_004", name="Malik Thompson", position="LF", bats=Hand.L),
            LineupPlayer(player_id="a_005", name="Kevin Park", position="1B", bats=Hand.L),
            LineupPlayer(player_id="a_006", name="Trey Anderson", position="RF", bats=Hand.R),
            LineupPlayer(player_id="a_007", name="Nathan Cruz", position="SS", bats=Hand.R),
            LineupPlayer(player_id="a_008", name="Ben Harper", position="3B", bats=Hand.L),
            LineupPlayer(player_id="a_009", name="Diego Santos", position="C", bats=Hand.R),
        ],
        their_lineup_position=4,
        their_bench=[
            OpponentBenchPlayer(player_id="a_010", name="James Wright", bats=Hand.L),
            OpponentBenchPlayer(player_id="a_011", name="Tyler Brooks", bats=Hand.S),
            OpponentBenchPlayer(player_id="a_012", name="Marcus Green", bats=Hand.R),
            OpponentBenchPlayer(player_id="a_013", name="Pete Lawson", bats=Hand.R),
            OpponentBenchPlayer(player_id="a_014", name="Isaiah Carter", bats=Hand.L),
        ],
        their_bullpen=[
            OpponentBullpenPitcher(player_id="a_bp1", name="Zach Miller", throws=ThrowHand.R, role=BullpenRole.CLOSER),
            OpponentBullpenPitcher(player_id="a_bp2", name="Omar Hassan", throws=ThrowHand.R, role=BullpenRole.SETUP),
            OpponentBullpenPitcher(player_id="a_bp3", name="Trevor Fox", throws=ThrowHand.L, role=BullpenRole.SETUP),
            OpponentBullpenPitcher(player_id="a_bp4", name="Rick Simmons", throws=ThrowHand.R, role=BullpenRole.MIDDLE),
            OpponentBullpenPitcher(player_id="a_bp5", name="Will Chang", throws=ThrowHand.R, role=BullpenRole.MIDDLE),
            OpponentBullpenPitcher(player_id="a_bp6", name="Brian Kelly", throws=ThrowHand.R, role=BullpenRole.LONG),
        ],
    )

    decision_prompt = (
        "Bottom of the 7th, 1 out, bases empty. Your team trails 3-4. "
        "Rafael Ortiz (L) is batting against Matt Henderson (LHP) who is in his "
        "3rd time through the order with 88 pitches. Ortiz is a lefty facing a lefty. "
        "On deck is Tyrone Jackson (R). "
        "Should you pinch-hit for Ortiz, let him bat, or consider another strategy? "
        "Gather relevant data with tools before deciding."
    )

    return {
        "matchup_state": matchup.model_dump(),
        "roster_state": roster.model_dump(),
        "opponent_roster_state": opponent_roster.model_dump(),
        "decision_prompt": decision_prompt,
    }


# ---------------------------------------------------------------------------
# Dry-run validation
# ---------------------------------------------------------------------------

def run_dry_run() -> None:
    """Validate all models, tools, and scenario construction without API calls."""
    print("=" * 72)
    print("BASEBALL MANAGER AI AGENT -- Dry Run Validation")
    print("=" * 72)

    # 1. Validate models import and construction
    print("\n[1/6] Validating Pydantic models...")
    scenario = build_sample_scenario()
    # Round-trip the scenario through JSON to verify serialization
    scenario_json = json.dumps(scenario, default=str)
    assert len(scenario_json) > 100, "Scenario JSON is too small"
    print(f"  MatchupState:        OK ({len(json.dumps(scenario['matchup_state']))} bytes)")
    print(f"  RosterState:         OK ({len(json.dumps(scenario['roster_state']))} bytes)")
    print(f"  OpponentRosterState: OK ({len(json.dumps(scenario['opponent_roster_state']))} bytes)")

    # Validate ManagerDecision can be constructed
    test_decision = ManagerDecision(
        decision="PINCH_HIT",
        action_details="Send Kenji Tanaka (R) to pinch-hit for Rafael Ortiz (L)",
        confidence=0.75,
        reasoning="Ortiz has a same-hand disadvantage vs LHP Henderson",
        key_factors=["L-L matchup disadvantage", "Henderson 3rd time through order"],
        risks=["Lose Ortiz's bat for rest of game"],
    )
    assert test_decision.decision == "PINCH_HIT"
    print(f"  ManagerDecision:     OK")

    # 2. Validate all 12 tools are registered
    print("\n[2/6] Validating tools...")
    assert len(ALL_TOOLS) == 12, f"Expected 12 tools, got {len(ALL_TOOLS)}"
    tool_names = []
    for tool in ALL_TOOLS:
        name = tool.name
        tool_names.append(name)
        print(f"  {name}: OK")
    expected_tools = [
        "get_batter_stats", "get_pitcher_stats", "get_matchup_data",
        "get_run_expectancy", "get_win_probability", "evaluate_stolen_base",
        "evaluate_sacrifice_bunt", "get_bullpen_status", "get_pitcher_fatigue_assessment",
        "get_defensive_positioning", "get_defensive_replacement_value", "get_platoon_comparison",
    ]
    for expected in expected_tools:
        assert expected in tool_names, f"Missing tool: {expected}"

    # 3. Validate tool stubs return valid JSON
    print("\n[3/6] Validating tool stub outputs...")
    # Call each tool stub and verify it returns parseable JSON
    results = {
        "get_batter_stats": ALL_TOOLS[0]("h_003"),
        "get_pitcher_stats": ALL_TOOLS[1]("a_sp1"),
        "get_matchup_data": ALL_TOOLS[2]("h_003", "a_sp1"),
        "get_run_expectancy": ALL_TOOLS[3](True, False, False, 1),
        "get_win_probability": ALL_TOOLS[4](7, "BOTTOM", 1, False, False, False, -1),
        "evaluate_stolen_base": ALL_TOOLS[5]("h_001", 2, "a_sp1", "a_009"),
        "evaluate_sacrifice_bunt": ALL_TOOLS[6]("h_009", True, False, False, 0, -1, 7),
        "get_bullpen_status": ALL_TOOLS[7](),
        "get_pitcher_fatigue_assessment": ALL_TOOLS[8]("a_sp1"),
        "get_defensive_positioning": ALL_TOOLS[9]("h_003", "a_sp1", 1, False, False, False, -1, 7),
        "get_defensive_replacement_value": ALL_TOOLS[10]("h_007", "h_012", "LF"),
        "get_platoon_comparison": ALL_TOOLS[11]("h_003", "h_014", "a_sp1"),
    }
    for name, result in results.items():
        data = json.loads(result)
        assert "status" in data, f"Tool {name} missing 'status' field"
        print(f"  {name}: returns valid JSON ({data['status']})")

    # 4. Validate sample rosters file exists and is valid
    print("\n[4/6] Validating sample rosters...")
    roster_path = Path(__file__).parent / "data" / "sample_rosters.json"
    assert roster_path.exists(), f"Missing {roster_path}"
    with open(roster_path) as f:
        rosters = json.load(f)
    for team_key in ("home", "away"):
        team = rosters[team_key]
        lineup_count = len(team["lineup"])
        bench_count = len(team["bench"])
        bp_count = len(team["bullpen"])
        total = lineup_count + bench_count + bp_count + 1  # +1 for starting pitcher
        print(f"  {team['team_name']}: {lineup_count} lineup + {bench_count} bench + 1 SP + {bp_count} bullpen = {total} players")
        assert lineup_count == 9, f"{team_key} lineup should have 9 players"
        assert bench_count >= 5, f"{team_key} bench should have at least 5 players"
        assert bp_count >= 8, f"{team_key} bullpen should have at least 8 pitchers"

    # 5. Validate game state to scenario conversion
    print("\n[5/6] Validating game state conversion...")
    engine = SimulationEngine(seed=42)
    game = engine.initialize_game(rosters)
    scenario = game_state_to_scenario(game, "home")
    assert "matchup_state" in scenario
    assert "roster_state" in scenario
    assert "opponent_roster_state" in scenario
    assert "decision_prompt" in scenario
    assert scenario["matchup_state"]["inning"] == 1
    assert scenario["matchup_state"]["half"] == "TOP"
    print(f"  game_state_to_scenario: OK")

    # Validate decision application
    test_result = validate_and_apply_decision(
        game,
        {"decision": "NO_ACTION", "action_details": "test"},
        "home",
        engine,
    )
    assert test_result.valid
    print(f"  validate_and_apply_decision (NO_ACTION): OK")

    test_result = validate_and_apply_decision(
        game,
        {"decision": "PITCHING_CHANGE", "action_details": "bring in someone"},
        "home",
        engine,
    )
    # May be invalid (3-batter minimum) - that's expected
    print(f"  validate_and_apply_decision (PITCHING_CHANGE): OK (valid={test_result.valid})")

    # 6. Validate system prompt and agent setup
    print("\n[6/6] Validating agent configuration...")
    assert len(SYSTEM_PROMPT) > 100, "System prompt is too short"
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  Tools registered: {len(ALL_TOOLS)}")
    print(f"  Output schema: ManagerDecision")
    print(f"  Agent decision loop: integrated")

    print("\n" + "=" * 72)
    print("ALL VALIDATIONS PASSED")
    print("=" * 72)
    print("\nTo run a full agent-managed game:")
    print("  ANTHROPIC_API_KEY=<key> uv run game.py")
    print("\nTo run an automated simulation (no API key):")
    print("  uv run game.py --sim")


# ---------------------------------------------------------------------------
# Single-turn test (legacy)
# ---------------------------------------------------------------------------

def run_single_turn() -> None:
    """Run a single-turn test: send a scenario, allow tool calls, get a ManagerDecision."""
    client = Anthropic()
    scenario = build_sample_scenario()

    user_message = (
        "Here is the current game scenario:\n\n"
        f"**Matchup State:**\n```json\n{json.dumps(scenario['matchup_state'], indent=2)}\n```\n\n"
        f"**Roster State:**\n```json\n{json.dumps(scenario['roster_state'], indent=2)}\n```\n\n"
        f"**Opponent Roster State:**\n```json\n{json.dumps(scenario['opponent_roster_state'], indent=2)}\n```\n\n"
        f"**Decision Needed:** {scenario['decision_prompt']}"
    )

    print("=" * 72)
    print("BASEBALL MANAGER AI AGENT -- Single-Turn Test")
    print("=" * 72)
    print(f"\nScenario: {scenario['decision_prompt']}\n")
    print("Sending to agent with 12 tools registered...")
    print("-" * 72)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*output_format.*deprecated.*")
        runner = client.beta.messages.tool_runner(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            output_format=ManagerDecision,
            messages=[{"role": "user", "content": user_message}],
        )

    turn = 0
    final_message = None
    for message in runner:
        turn += 1
        for block in message.content:
            if block.type == "tool_use":
                print(f"  [Turn {turn}] Tool call: {block.name}({json.dumps(block.input, separators=(',', ':'))})")
            elif block.type == "text" and block.text.strip():
                print(f"  [Turn {turn}] Agent: {block.text[:200]}...")
        final_message = message

    print("-" * 72)

    if final_message and hasattr(final_message, "parsed") and final_message.parsed:
        decision: ManagerDecision = final_message.parsed
        print("\nMANAGER DECISION:")
        print(f"  Decision:    {decision.decision}")
        print(f"  Details:     {decision.action_details}")
        print(f"  Confidence:  {decision.confidence}")
        print(f"  Reasoning:   {decision.reasoning[:200]}...")
        print(f"  Key Factors: {decision.key_factors}")
        print(f"  Risks:       {decision.risks}")
        if decision.alternatives_considered:
            print(f"  Alternatives: {len(decision.alternatives_considered)} considered")
        print("\nTest PASSED: Agent received scenario, called tools, and returned a valid ManagerDecision.")
    else:
        print("\nTest FAILED: No valid ManagerDecision received.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Automated simulation mode (no agent, no API key)
# ---------------------------------------------------------------------------

def run_sim(seed: int | None = None, verbose: bool = True) -> GameState:
    """Run a fully automated game simulation (no agent)."""
    rosters = load_rosters()
    engine = SimulationEngine(seed=seed)

    print(f"Simulating game with seed {engine.seed}...")
    print(f"{rosters['away']['team_name']} at {rosters['home']['team_name']}")
    print("=" * 72)

    if verbose:
        print(f"\n--- Top of the 1st ---")

    game = engine.simulate_game(rosters, verbose=verbose)

    print()
    print(engine.print_box_score(game))

    decisions_summary = engine.generate_decisions_summary(game)
    if decisions_summary and "No managerial decisions" not in decisions_summary:
        print()
        print(decisions_summary)

    print(f"\nTotal plate appearances: {len([e for e in game.play_log if e.event_type not in ('inning_change', 'game_end', 'decision')])}")
    print(f"Total play events: {len(game.play_log)}")

    return game


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse arguments
    seed = None
    managed_team = "home"
    verbose = True

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--seed" and i < len(sys.argv) - 1:
            try:
                seed = int(sys.argv[i + 1])
            except (ValueError, IndexError):
                pass
        elif arg == "--away":
            managed_team = "away"
        elif arg == "--quiet":
            verbose = False

    if "--dry-run" in sys.argv:
        run_dry_run()
    elif "--sim" in sys.argv:
        run_sim(seed=seed, verbose=verbose)
    elif "--single-turn" in sys.argv:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY required for single-turn test.")
            sys.exit(1)
        run_single_turn()
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY set. Running dry-run validation instead.")
        print("Set ANTHROPIC_API_KEY to run the full agent game.\n")
        run_dry_run()
    else:
        run_agent_game(seed=seed, managed_team=managed_team, verbose=verbose)
