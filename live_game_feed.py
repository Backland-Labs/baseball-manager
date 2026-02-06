# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.78.0",
#     "pydantic>=2.0",
# ]
# ///
"""Live game feed service for the baseball manager AI agent.

Polls the MLB Stats API live game feed, detects new at-bats, and invokes
the agent for each one.  This is the orchestrator that drives the agent
during a live game.

Usage::

    # Run against a live game (requires ANTHROPIC_API_KEY)
    uv run live_game_feed.py --game-pk 716463 --team "Red Sox"

    # Run with custom poll interval
    uv run live_game_feed.py --game-pk 716463 --team BOS --interval 15

    # Dry-run mode (log decisions without posting)
    uv run live_game_feed.py --game-pk 716463 --team BOS --dry-run
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL = 10  # seconds
MIN_POLL_INTERVAL = 5
MAX_POLL_INTERVAL = 120
BACKOFF_POLL_INTERVAL = 30  # seconds, used when feed is temporarily unavailable
MAX_CONSECUTIVE_ERRORS = 10  # stop after this many consecutive poll failures

# Game states from the MLB Stats API
GAME_STATE_FINAL = "Final"
GAME_STATE_LIVE = "Live"
GAME_STATE_PREVIEW = "Preview"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("live_game_feed")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AtBatKey:
    """Identifies a unique at-bat in a game.

    The MLB Stats API play index is a monotonically increasing integer
    that uniquely identifies each play (at-bat) within a game.  Combined
    with the at-bat index, this gives us a reliable way to detect when a
    new at-bat begins.
    """
    play_index: int
    at_bat_index: int
    inning: int
    half: str  # "top" or "bottom"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtBatKey):
            return NotImplemented
        return (self.play_index == other.play_index
                and self.at_bat_index == other.at_bat_index)

    def __hash__(self) -> int:
        return hash((self.play_index, self.at_bat_index))


@dataclass
class GameFeedState:
    """Tracks the polling state for a live game.

    Maintained across poll iterations so the service can detect new
    at-bats and track game progress.
    """
    game_pk: int
    team: str
    managed_team_side: str  # "home" or "away"
    poll_interval: float = DEFAULT_POLL_INTERVAL
    last_play_index: int = -1
    last_at_bat_index: int = -1
    game_status: str = ""
    inning: int = 0
    half: str = ""
    consecutive_errors: int = 0
    total_polls: int = 0
    total_agent_calls: int = 0
    active_decisions: int = 0
    decision_log: list = field(default_factory=list)
    error_log: list = field(default_factory=list)
    messages: list = field(default_factory=list)


@dataclass
class PollResult:
    """Result of a single poll of the live game feed.

    Attributes:
        new_at_bat: True if a new at-bat was detected.
        game_over: True if the game has ended.
        feed: The raw feed data (None on error).
        error: Error message if the poll failed.
        at_bat_key: Key of the detected at-bat (None if no new at-bat).
        game_status: Current game status string.
    """
    new_at_bat: bool = False
    game_over: bool = False
    feed: dict | None = None
    error: str | None = None
    at_bat_key: AtBatKey | None = None
    game_status: str = ""


# ---------------------------------------------------------------------------
# Feed polling and at-bat detection
# ---------------------------------------------------------------------------

def extract_current_play_index(feed: dict[str, Any]) -> tuple[int, int]:
    """Extract the current play index and at-bat index from a live feed.

    The play index is the position of the current play in the allPlays
    array.  The at-bat index is the atBatIndex field on the current play.

    Args:
        feed: MLB Stats API live game feed dict.

    Returns:
        Tuple of (play_index, at_bat_index).  Returns (-1, -1) if the
        current play cannot be determined.
    """
    live_data = feed.get("liveData", {})
    plays = live_data.get("plays", {})
    current_play = plays.get("currentPlay", {})
    all_plays = plays.get("allPlays", [])

    at_bat_index = current_play.get("atBatIndex", -1)

    # The play index is the index in allPlays of the current play.
    # The currentPlayIndex field tells us directly in many API versions.
    play_index = plays.get("currentPlayIndex", -1)
    if play_index < 0 and all_plays:
        play_index = len(all_plays) - 1

    return play_index, at_bat_index


def extract_game_status(feed: dict[str, Any]) -> str:
    """Extract the abstract game state from a live feed.

    Returns one of: "Preview", "Live", "Final", or "Unknown".
    """
    game_data = feed.get("gameData", {})
    status = game_data.get("status", {})
    return status.get("abstractGameState", "Unknown")


def extract_current_inning_half(feed: dict[str, Any]) -> tuple[int, str]:
    """Extract the current inning number and half from a live feed.

    Returns:
        Tuple of (inning, half_str) where half_str is "top" or "bottom".
    """
    live_data = feed.get("liveData", {})
    linescore = live_data.get("linescore", {})
    inning = linescore.get("currentInning", 0)
    is_top = linescore.get("isTopInning", True)
    return inning, "top" if is_top else "bottom"


def determine_managed_team_side(
    feed: dict[str, Any],
    team: str,
) -> str:
    """Determine if the managed team is home or away from the feed.

    Args:
        feed: MLB Stats API live game feed dict.
        team: Team name, abbreviation, or ID string.

    Returns:
        "home" or "away".

    Raises:
        ValueError: If the team cannot be matched to either side.
    """
    from data.mlb_api import lookup_team_id, TEAM_NAMES

    try:
        team_id = lookup_team_id(team)
    except ValueError:
        # Fall back to name matching
        team_id = None

    game_data = feed.get("gameData", {})
    teams = game_data.get("teams", {})

    home_team = teams.get("home", {})
    away_team = teams.get("away", {})

    home_id = home_team.get("id")
    away_id = away_team.get("id")

    if team_id is not None:
        if team_id == home_id:
            return "home"
        if team_id == away_id:
            return "away"

    # Try name matching as fallback
    team_lower = str(team).lower().strip()
    home_name = home_team.get("name", "").lower()
    away_name = away_team.get("name", "").lower()
    home_abbrev = home_team.get("abbreviation", "").lower()
    away_abbrev = away_team.get("abbreviation", "").lower()

    if team_lower in home_name or team_lower == home_abbrev:
        return "home"
    if team_lower in away_name or team_lower == away_abbrev:
        return "away"

    raise ValueError(
        f"Team '{team}' not found in this game. "
        f"Home: {home_team.get('name', '?')} ({home_id}), "
        f"Away: {away_team.get('name', '?')} ({away_id})"
    )


def is_new_at_bat(
    feed: dict[str, Any],
    state: GameFeedState,
) -> tuple[bool, AtBatKey | None]:
    """Check if a new at-bat has started since the last poll.

    Compares the current play index and at-bat index against the last
    known values.  A new at-bat is detected when either index has
    increased.

    Args:
        feed: Current live game feed.
        state: Current polling state.

    Returns:
        Tuple of (is_new, at_bat_key).  is_new is True when a new
        at-bat has been detected.
    """
    play_index, at_bat_index = extract_current_play_index(feed)
    inning, half = extract_current_inning_half(feed)

    at_bat_key = AtBatKey(
        play_index=play_index,
        at_bat_index=at_bat_index,
        inning=inning,
        half=half,
    )

    if play_index < 0:
        return False, None

    # First poll -- always treat as a new at-bat
    if state.last_play_index < 0:
        return True, at_bat_key

    # Detect new at-bat by index change
    if (at_bat_index > state.last_at_bat_index
            or play_index > state.last_play_index):
        return True, at_bat_key

    return False, at_bat_key


def poll_game_feed(
    game_pk: int,
    state: GameFeedState,
    fetch_fn: Any | None = None,
) -> PollResult:
    """Poll the live game feed and check for new at-bats.

    Args:
        game_pk: MLB game identifier.
        state: Current polling state (updated in-place on success).
        fetch_fn: Optional override for the feed fetch function
            (for testing).  Signature: ``fetch_fn(game_pk) -> dict``.

    Returns:
        PollResult describing what happened.
    """
    if fetch_fn is None:
        from data.mlb_api import get_live_game_feed
        fetch_fn = get_live_game_feed

    try:
        feed = fetch_fn(game_pk)
    except Exception as exc:
        state.consecutive_errors += 1
        # Detect rate limit errors from the MLB API
        is_rate_limited = False
        try:
            from data.mlb_api import MLBApiRateLimitError
            if isinstance(exc, MLBApiRateLimitError):
                is_rate_limited = True
        except ImportError:
            pass

        error_type = "rate_limit_error" if is_rate_limited else "feed_fetch_error"
        error_msg = f"Failed to fetch game feed: {exc}"
        logger.warning(error_msg)
        state.error_log.append({
            "timestamp": time.time(),
            "error_type": error_type,
            "error": error_msg,
            "consecutive_errors": state.consecutive_errors,
        })
        return PollResult(error=error_msg)

    # Reset consecutive errors on successful fetch
    state.consecutive_errors = 0
    state.total_polls += 1

    game_status = extract_game_status(feed)
    state.game_status = game_status

    # Game not yet started
    if game_status == GAME_STATE_PREVIEW:
        return PollResult(game_status=game_status, feed=feed)

    # Game finished
    if game_status == GAME_STATE_FINAL:
        return PollResult(game_over=True, game_status=game_status, feed=feed)

    # Game is live -- check for new at-bat
    new, at_bat_key = is_new_at_bat(feed, state)

    # Update tracking state
    if at_bat_key:
        state.last_play_index = at_bat_key.play_index
        state.last_at_bat_index = at_bat_key.at_bat_index
        state.inning = at_bat_key.inning
        state.half = at_bat_key.half

    return PollResult(
        new_at_bat=new,
        game_status=game_status,
        feed=feed,
        at_bat_key=at_bat_key,
    )


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

def invoke_agent(
    feed: dict[str, Any],
    state: GameFeedState,
    client: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Invoke the agent for the current game state.

    Parses the live feed into agent input models, calls the agent, and
    returns the decision with metadata.

    Args:
        feed: Current MLB Stats API live game feed.
        state: Current game feed state.
        client: Anthropic client (created if None and not dry_run).
        dry_run: If True, skip agent invocation and return a mock decision.

    Returns:
        Dict with keys: ``decision``, ``decision_output``, ``metadata``,
        ``timestamp``.
    """
    from game_state_ingestion import ingest_game_state
    from decision_output import format_decision_output, NO_ACTION_TYPES

    timestamp = time.time()

    # Parse the live feed into agent input models
    try:
        ingested = ingest_game_state(
            feed,
            managed_team_side=state.managed_team_side,
        )
    except Exception as exc:
        logger.error("Failed to ingest game state: %s", exc)
        state.error_log.append({
            "timestamp": timestamp,
            "error_type": "ingestion_error",
            "error": str(exc),
        })
        return {
            "decision": {"decision": "NO_ACTION", "action_details": f"Ingestion error: {exc}"},
            "decision_output": None,
            "metadata": {"error": str(exc)},
            "timestamp": timestamp,
        }

    matchup = ingested["matchup_state"]
    roster = ingested["roster_state"]
    opponent = ingested["opponent_roster_state"]

    if dry_run:
        decision_dict = {
            "decision": "NO_ACTION",
            "action_details": "Dry-run mode, no agent invocation",
            "confidence": 0.0,
            "reasoning": "Dry-run mode",
            "key_factors": [],
            "risks": [],
        }
        metadata = {"dry_run": True, "tool_calls": [], "token_usage": {},
                     "latency_ms": 0}
    else:
        # Import agent functions
        from game import run_agent_decision, SYSTEM_PROMPT
        from anthropic import Anthropic

        if client is None:
            client = Anthropic()

        # Build scenario from ingested models
        scenario = {
            "matchup_state": matchup.model_dump(),
            "roster_state": roster.model_dump(),
            "opponent_roster_state": opponent.model_dump(),
            "decision_prompt": (
                f"{'Top' if matchup.half == 'TOP' else 'Bottom'} of the "
                f"{matchup.inning}, {matchup.outs} out(s). "
                f"{matchup.batter.name} ({matchup.batter.bats.value}) vs "
                f"{matchup.pitcher.name} ({matchup.pitcher.throws.value}HP). "
                f"Score: Away {matchup.score.away}, Home {matchup.score.home}. "
                f"Assess the situation and decide."
            ),
        }

        # Trim messages to prevent context overflow
        if len(state.messages) > 20:
            state.messages = state.messages[-10:]

        user_message = (
            "Here is the current game scenario:\n\n"
            f"**Matchup State:**\n```json\n{json.dumps(scenario['matchup_state'], indent=2, default=str)}\n```\n\n"
            f"**Roster State:**\n```json\n{json.dumps(scenario['roster_state'], indent=2, default=str)}\n```\n\n"
            f"**Opponent Roster State:**\n```json\n{json.dumps(scenario['opponent_roster_state'], indent=2, default=str)}\n```\n\n"
            f"**Decision Needed:** {scenario['decision_prompt']}"
        )

        state.messages.append({"role": "user", "content": user_message})

        try:
            from game import _call_agent
            decision_dict, final_message, call_meta = _call_agent(
                client, state.messages, verbose=True,
            )
            metadata = call_meta

            if final_message:
                state.messages.append({"role": "assistant", "content": final_message.content})
        except Exception as exc:
            logger.error("Agent call failed: %s", exc)
            decision_dict = {
                "decision": "NO_ACTION",
                "action_details": f"Agent error: {exc}",
                "confidence": 0.0,
                "reasoning": "Agent call failed",
                "key_factors": [],
                "risks": [],
            }
            metadata = {"error": str(exc), "tool_calls": [],
                         "token_usage": {}, "latency_ms": 0}
            state.error_log.append({
                "timestamp": timestamp,
                "error_type": "agent_call_error",
                "error": str(exc),
            })

    state.total_agent_calls += 1

    # Format decision output
    runners_dict = {}
    runners_model = matchup.runners
    if runners_model.first:
        runners_dict["first"] = runners_model.first.model_dump()
    if runners_model.second:
        runners_dict["second"] = runners_model.second.model_dump()
    if runners_model.third:
        runners_dict["third"] = runners_model.third.model_dump()

    decision_output = format_decision_output(
        decision_dict=decision_dict,
        inning=matchup.inning,
        half=matchup.half.value,
        outs=matchup.outs,
        score_home=matchup.score.home,
        score_away=matchup.score.away,
        runners=runners_dict if runners_dict else None,
    )

    if decision_output.is_active:
        state.active_decisions += 1

    # Build log entry
    log_entry = {
        "turn": state.total_agent_calls,
        "timestamp": timestamp,
        "game_state": {
            "inning": matchup.inning,
            "half": matchup.half.value,
            "outs": matchup.outs,
            "score": {"home": matchup.score.home, "away": matchup.score.away},
            "batter": {"player_id": matchup.batter.player_id, "name": matchup.batter.name},
            "pitcher": {"player_id": matchup.pitcher.player_id, "name": matchup.pitcher.name},
        },
        "decision": decision_dict,
        "is_active_decision": decision_output.is_active,
        "tweet_text": decision_output.tweet_text,
        "log_entry": decision_output.log_entry,
        "tool_calls": metadata.get("tool_calls", []),
        "token_usage": metadata.get("token_usage", {}),
        "latency_ms": metadata.get("latency_ms", 0),
    }
    state.decision_log.append(log_entry)

    return {
        "decision": decision_dict,
        "decision_output": decision_output,
        "metadata": metadata,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# Game log persistence
# ---------------------------------------------------------------------------

def write_live_game_log(
    state: GameFeedState,
    feed: dict[str, Any] | None = None,
    log_dir: Path | None = None,
) -> Path:
    """Write the live game decision log to a structured JSON file.

    Args:
        state: The game feed state containing decision and error logs.
        feed: Final game feed (for extracting final score/teams).
        log_dir: Directory for log files. Defaults to data/game_logs/.

    Returns:
        Path to the written log file.
    """
    if log_dir is None:
        log_dir = Path(__file__).parent / "data" / "game_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Extract final game info from feed
    home_team = ""
    away_team = ""
    final_score = {"home": 0, "away": 0}
    if feed:
        game_data = feed.get("gameData", {})
        teams = game_data.get("teams", {})
        home_team = teams.get("home", {}).get("name", "")
        away_team = teams.get("away", {}).get("name", "")
        linescore = feed.get("liveData", {}).get("linescore", {})
        teams_score = linescore.get("teams", {})
        final_score = {
            "home": teams_score.get("home", {}).get("runs", 0),
            "away": teams_score.get("away", {}).get("runs", 0),
        }

    # Aggregate token usage
    total_input = sum(
        d.get("token_usage", {}).get("input_tokens", 0)
        for d in state.decision_log
    )
    total_output = sum(
        d.get("token_usage", {}).get("output_tokens", 0)
        for d in state.decision_log
    )
    total_latency = sum(d.get("latency_ms", 0) for d in state.decision_log)

    game_log = {
        "game_info": {
            "game_pk": state.game_pk,
            "home_team": home_team,
            "away_team": away_team,
            "managed_team": state.team,
            "managed_team_side": state.managed_team_side,
            "final_score": final_score,
        },
        "summary": {
            "total_polls": state.total_polls,
            "total_agent_calls": state.total_agent_calls,
            "active_decisions": state.active_decisions,
            "no_action_decisions": state.total_agent_calls - state.active_decisions,
            "total_errors": len(state.error_log),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_latency_ms": total_latency,
        },
        "decisions": state.decision_log,
        "errors": state.error_log,
    }

    log_path = log_dir / f"live_game_{state.game_pk}.json"
    with open(log_path, "w") as f:
        json.dump(game_log, f, indent=2, default=str)

    return log_path


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def run_live_game(
    game_pk: int,
    team: str,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    dry_run: bool = False,
    verbose: bool = True,
    max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
    fetch_fn: Any | None = None,
    client: Any | None = None,
    log_dir: Path | None = None,
) -> GameFeedState:
    """Run the live game feed service for a single game.

    Polls the MLB Stats API live game feed at the specified interval,
    detects new at-bats, and invokes the agent for each one.

    Args:
        game_pk: MLB game identifier (gamePk).
        team: Team the agent manages (name, abbreviation, or ID).
        poll_interval: Seconds between polls (default 10).
        dry_run: If True, log decisions without invoking the agent.
        verbose: Print progress to stdout.
        max_consecutive_errors: Stop after this many consecutive errors.
        fetch_fn: Override for the feed fetch function (for testing).
        client: Anthropic client instance (created if needed).
        log_dir: Override for log file directory.

    Returns:
        Final GameFeedState with all logs.
    """
    # Validate poll interval
    interval = max(MIN_POLL_INTERVAL, min(MAX_POLL_INTERVAL, poll_interval))

    state = GameFeedState(
        game_pk=game_pk,
        team=team,
        managed_team_side="",  # Determined from first successful feed
        poll_interval=interval,
    )

    if verbose:
        print("=" * 72)
        print("LIVE GAME FEED SERVICE")
        print("=" * 72)
        print(f"  Game PK: {game_pk}")
        print(f"  Team:    {team}")
        print(f"  Poll interval: {interval}s")
        print(f"  Dry run: {dry_run}")
        print("=" * 72)

    last_feed = None

    while True:
        result = poll_game_feed(game_pk, state, fetch_fn=fetch_fn)

        if result.error:
            if verbose:
                print(f"  [Error] {result.error} "
                      f"(consecutive: {state.consecutive_errors})")

            if state.consecutive_errors >= max_consecutive_errors:
                if verbose:
                    print(f"\n  Stopping: {max_consecutive_errors} "
                          f"consecutive errors reached.")
                break

            # Back off on repeated errors
            time.sleep(BACKOFF_POLL_INTERVAL)
            continue

        last_feed = result.feed

        # Determine managed team side on first successful feed
        if not state.managed_team_side and result.feed:
            try:
                state.managed_team_side = determine_managed_team_side(
                    result.feed, team
                )
                if verbose:
                    game_data = result.feed.get("gameData", {})
                    teams = game_data.get("teams", {})
                    home_name = teams.get("home", {}).get("name", "?")
                    away_name = teams.get("away", {}).get("name", "?")
                    print(f"\n  {away_name} at {home_name}")
                    print(f"  Managing: {state.managed_team_side.upper()} team")
            except ValueError as exc:
                if verbose:
                    print(f"  [Error] {exc}")
                state.error_log.append({
                    "timestamp": time.time(),
                    "error_type": "team_resolution_error",
                    "error": str(exc),
                })
                break

        # Handle game states
        if result.game_status == GAME_STATE_PREVIEW:
            if verbose:
                print(f"  Game not started yet. Waiting...")
            time.sleep(interval)
            continue

        if result.game_over:
            if verbose:
                print(f"\n  Game is FINAL.")
                if result.feed:
                    linescore = result.feed.get("liveData", {}).get("linescore", {})
                    teams_score = linescore.get("teams", {})
                    home_runs = teams_score.get("home", {}).get("runs", 0)
                    away_runs = teams_score.get("away", {}).get("runs", 0)
                    print(f"  Final score: Away {away_runs}, Home {home_runs}")
            break

        # New at-bat detected
        if result.new_at_bat and result.feed and state.managed_team_side:
            inning, half = extract_current_inning_half(result.feed)
            if verbose:
                linescore = result.feed.get("liveData", {}).get("linescore", {})
                plays = result.feed.get("liveData", {}).get("plays", {})
                current_play = plays.get("currentPlay", {})
                matchup = current_play.get("matchup", {})
                batter_name = matchup.get("batter", {}).get("fullName", "?")
                pitcher_name = matchup.get("pitcher", {}).get("fullName", "?")
                print(f"\n  [{'Top' if half == 'top' else 'Bot'} {inning}] "
                      f"New at-bat: {batter_name} vs {pitcher_name}")

            # Invoke agent
            agent_result = invoke_agent(
                feed=result.feed,
                state=state,
                client=client,
                dry_run=dry_run,
            )

            output = agent_result.get("decision_output")
            if output and verbose:
                if output.is_active:
                    print(f"    >> ACTIVE: {output.tweet_text}")
                else:
                    print(f"    >> {output.log_entry}")

        # Wait before next poll
        time.sleep(interval)

    # Write game log
    if verbose:
        print(f"\n  Total polls: {state.total_polls}")
        print(f"  Total agent calls: {state.total_agent_calls}")
        print(f"  Active decisions: {state.active_decisions}")
        if state.error_log:
            print(f"  Errors: {len(state.error_log)}")

    try:
        log_path = write_live_game_log(state, last_feed, log_dir=log_dir)
        if verbose:
            print(f"  Game log saved to: {log_path}")
    except Exception as exc:
        logger.error("Failed to write game log: %s", exc)

    return state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live game feed service for the baseball manager AI agent."
    )
    parser.add_argument(
        "--game-pk", type=int, required=True,
        help="MLB game identifier (gamePk)",
    )
    parser.add_argument(
        "--team", type=str, required=True,
        help="Team to manage (name, abbreviation, or ID)",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_POLL_INTERVAL,
        help=f"Poll interval in seconds (default {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log decisions without invoking the agent",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required (use --dry-run to skip agent calls)")
        sys.exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_live_game(
        game_pk=args.game_pk,
        team=args.team,
        poll_interval=args.interval,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
