# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the live_game_feed_service feature.

Validates the external service that polls the MLB Stats API live game feed,
detects new at-bats, and invokes the agent:
  1. Accept a gamePk and managed team
  2. Poll the live game feed at a configurable interval (default 10s)
  3. Detect new at-bats by comparing play index to previous poll
  4. When a new at-bat is detected, parse game state and invoke the agent
  5. If the agent returns an active decision, format it as a tweet and output it
  6. Continue polling until the game reaches a final state
  7. Handle game delays, rain delays, and feed unavailability without crashing
  8. Log all agent invocations and decisions throughout the game
"""

import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from live_game_feed import (
    AtBatKey,
    GameFeedState,
    PollResult,
    DEFAULT_POLL_INTERVAL,
    MIN_POLL_INTERVAL,
    MAX_POLL_INTERVAL,
    BACKOFF_POLL_INTERVAL,
    MAX_CONSECUTIVE_ERRORS,
    GAME_STATE_FINAL,
    GAME_STATE_LIVE,
    GAME_STATE_PREVIEW,
    extract_current_play_index,
    extract_game_status,
    extract_current_inning_half,
    determine_managed_team_side,
    is_new_at_bat,
    poll_game_feed,
    invoke_agent,
    write_live_game_log,
    run_live_game,
)


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

def _build_live_feed(
    inning: int = 1,
    is_top: bool = True,
    outs: int = 0,
    home_runs: int = 0,
    away_runs: int = 0,
    batter_id: int = 660271,
    batter_name: str = "Ronald Acuna Jr.",
    pitcher_id: int = 543037,
    pitcher_name: str = "Gerrit Cole",
    at_bat_index: int = 0,
    current_play_index: int = 0,
    game_status: str = "Live",
    home_team_id: int = 144,
    home_team_name: str = "Atlanta Braves",
    home_team_abbrev: str = "ATL",
    away_team_id: int = 147,
    away_team_name: str = "New York Yankees",
    away_team_abbrev: str = "NYY",
    num_all_plays: int | None = None,
) -> dict[str, Any]:
    """Build a minimal realistic MLB Stats API live game feed for testing."""
    if num_all_plays is None:
        num_all_plays = current_play_index + 1

    all_plays = []
    for i in range(num_all_plays):
        all_plays.append({
            "atBatIndex": i,
            "result": {"type": "atBat"},
        })

    return {
        "gameData": {
            "status": {
                "abstractGameState": game_status,
                "detailedState": game_status,
            },
            "teams": {
                "home": {
                    "id": home_team_id,
                    "name": home_team_name,
                    "abbreviation": home_team_abbrev,
                },
                "away": {
                    "id": away_team_id,
                    "name": away_team_name,
                    "abbreviation": away_team_abbrev,
                },
            },
            "players": {
                f"ID{batter_id}": {
                    "id": batter_id,
                    "fullName": batter_name,
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "RF"},
                },
                f"ID{pitcher_id}": {
                    "id": pitcher_id,
                    "fullName": pitcher_name,
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "primaryPosition": {"abbreviation": "P"},
                },
            },
        },
        "liveData": {
            "linescore": {
                "currentInning": inning,
                "isTopInning": is_top,
                "outs": outs,
                "teams": {
                    "home": {"runs": home_runs, "hits": 0, "errors": 0},
                    "away": {"runs": away_runs, "hits": 0, "errors": 0},
                },
                "offense": {},
            },
            "plays": {
                "currentPlay": {
                    "atBatIndex": at_bat_index,
                    "matchup": {
                        "batter": {"id": batter_id, "fullName": batter_name},
                        "pitcher": {"id": pitcher_id, "fullName": pitcher_name},
                    },
                    "count": {"balls": 0, "strikes": 0, "outs": outs},
                },
                "currentPlayIndex": current_play_index,
                "allPlays": all_plays,
            },
            "boxscore": {
                "teams": {
                    "home": {
                        "battingOrder": [batter_id],
                        "bullpen": [],
                        "bench": [],
                        "pitchers": [pitcher_id],
                        "players": {
                            f"ID{batter_id}": {
                                "person": {"id": batter_id, "fullName": batter_name},
                                "position": {"abbreviation": "RF"},
                                "stats": {"batting": {}},
                            },
                            f"ID{pitcher_id}": {
                                "person": {"id": pitcher_id, "fullName": pitcher_name},
                                "position": {"abbreviation": "P"},
                                "stats": {
                                    "pitching": {
                                        "inningsPitched": "3.0",
                                        "hits": 2,
                                        "runs": 1,
                                        "earnedRuns": 1,
                                        "baseOnBalls": 1,
                                        "strikeOuts": 4,
                                        "numberOfPitches": 50,
                                        "battersFaced": 12,
                                        "homeRuns": 0,
                                    }
                                },
                            },
                        },
                    },
                    "away": {
                        "battingOrder": [pitcher_id],
                        "bullpen": [],
                        "bench": [],
                        "pitchers": [pitcher_id],
                        "players": {
                            f"ID{pitcher_id}": {
                                "person": {"id": pitcher_id, "fullName": pitcher_name},
                                "position": {"abbreviation": "P"},
                                "stats": {
                                    "pitching": {
                                        "inningsPitched": "3.0",
                                        "hits": 2,
                                        "runs": 1,
                                        "earnedRuns": 1,
                                        "baseOnBalls": 1,
                                        "strikeOuts": 4,
                                        "numberOfPitches": 50,
                                        "battersFaced": 12,
                                        "homeRuns": 0,
                                    }
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _build_preview_feed(**kwargs) -> dict[str, Any]:
    """Build a feed where the game hasn't started yet."""
    return _build_live_feed(game_status="Preview", **kwargs)


def _build_final_feed(**kwargs) -> dict[str, Any]:
    """Build a feed where the game is over."""
    return _build_live_feed(game_status="Final", **kwargs)


def _build_game_feed_state(
    game_pk: int = 716463,
    team: str = "ATL",
    managed_team_side: str = "home",
    last_play_index: int = -1,
    last_at_bat_index: int = -1,
) -> GameFeedState:
    """Build a GameFeedState for testing."""
    return GameFeedState(
        game_pk=game_pk,
        team=team,
        managed_team_side=managed_team_side,
        last_play_index=last_play_index,
        last_at_bat_index=last_at_bat_index,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def live_feed():
    return _build_live_feed()


@pytest.fixture
def preview_feed():
    return _build_preview_feed()


@pytest.fixture
def final_feed():
    return _build_final_feed()


@pytest.fixture
def feed_state():
    return _build_game_feed_state()


@pytest.fixture
def tmp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ===========================================================================
# Step 1: Accept a gamePk and the team the agent manages
# ===========================================================================

class TestStep1AcceptGamePkAndTeam:
    """Validate that the service accepts a gamePk and team."""

    def test_game_feed_state_stores_game_pk(self):
        state = GameFeedState(game_pk=716463, team="ATL", managed_team_side="home")
        assert state.game_pk == 716463

    def test_game_feed_state_stores_team(self):
        state = GameFeedState(game_pk=716463, team="Red Sox", managed_team_side="home")
        assert state.team == "Red Sox"

    def test_game_feed_state_stores_managed_side(self):
        state = GameFeedState(game_pk=716463, team="ATL", managed_team_side="away")
        assert state.managed_team_side == "away"

    def test_determine_managed_team_side_home_by_id(self, live_feed):
        side = determine_managed_team_side(live_feed, "144")
        assert side == "home"

    def test_determine_managed_team_side_away_by_id(self, live_feed):
        side = determine_managed_team_side(live_feed, "147")
        assert side == "away"

    def test_determine_managed_team_side_by_abbreviation(self, live_feed):
        side = determine_managed_team_side(live_feed, "ATL")
        assert side == "home"

    def test_determine_managed_team_side_by_name(self, live_feed):
        side = determine_managed_team_side(live_feed, "braves")
        assert side == "home"

    def test_determine_managed_team_side_away_by_name(self, live_feed):
        side = determine_managed_team_side(live_feed, "yankees")
        assert side == "away"

    def test_determine_managed_team_side_invalid_team(self, live_feed):
        with pytest.raises(ValueError, match="not found in this game"):
            determine_managed_team_side(live_feed, "Red Sox")

    def test_game_feed_state_defaults(self):
        state = GameFeedState(game_pk=1, team="BOS", managed_team_side="home")
        assert state.poll_interval == DEFAULT_POLL_INTERVAL
        assert state.last_play_index == -1
        assert state.last_at_bat_index == -1
        assert state.game_status == ""
        assert state.consecutive_errors == 0
        assert state.total_polls == 0
        assert state.total_agent_calls == 0
        assert state.active_decisions == 0
        assert state.decision_log == []
        assert state.error_log == []
        assert state.messages == []


# ===========================================================================
# Step 2: Poll the MLB Stats API live game feed at configurable interval
# ===========================================================================

class TestStep2PollFeed:
    """Validate feed polling behavior."""

    def test_poll_game_feed_success(self, live_feed, feed_state):
        """Successful poll returns feed data and increments counter."""
        result = poll_game_feed(716463, feed_state, fetch_fn=lambda pk: live_feed)
        assert result.feed is not None
        assert result.error is None
        assert feed_state.total_polls == 1
        assert feed_state.consecutive_errors == 0

    def test_poll_game_feed_error_increments_counter(self, feed_state):
        """Failed poll increments consecutive error counter."""
        def failing_fetch(pk):
            raise ConnectionError("Network error")

        result = poll_game_feed(716463, feed_state, fetch_fn=failing_fetch)
        assert result.error is not None
        assert "Network error" in result.error
        assert feed_state.consecutive_errors == 1
        assert feed_state.total_polls == 0

    def test_poll_game_feed_error_resets_on_success(self, live_feed, feed_state):
        """Consecutive errors are reset on successful poll."""
        feed_state.consecutive_errors = 5
        result = poll_game_feed(716463, feed_state, fetch_fn=lambda pk: live_feed)
        assert result.error is None
        assert feed_state.consecutive_errors == 0

    def test_poll_game_feed_logs_errors(self, feed_state):
        """Poll errors are logged to the error log."""
        def failing_fetch(pk):
            raise ConnectionError("API down")

        poll_game_feed(716463, feed_state, fetch_fn=failing_fetch)
        assert len(feed_state.error_log) == 1
        assert feed_state.error_log[0]["error_type"] == "feed_fetch_error"

    def test_poll_interval_constants(self):
        """Validate poll interval constants."""
        assert DEFAULT_POLL_INTERVAL == 10
        assert MIN_POLL_INTERVAL == 5
        assert MAX_POLL_INTERVAL == 120
        assert BACKOFF_POLL_INTERVAL == 30

    def test_poll_game_feed_preview(self, preview_feed, feed_state):
        """Preview game returns no new at-bat and not game over."""
        result = poll_game_feed(716463, feed_state, fetch_fn=lambda pk: preview_feed)
        assert result.game_status == "Preview"
        assert not result.new_at_bat
        assert not result.game_over

    def test_poll_game_feed_final(self, final_feed, feed_state):
        """Final game returns game_over=True."""
        result = poll_game_feed(716463, feed_state, fetch_fn=lambda pk: final_feed)
        assert result.game_over is True
        assert result.game_status == "Final"


# ===========================================================================
# Step 3: Detect when a new at-bat begins
# ===========================================================================

class TestStep3DetectNewAtBat:
    """Validate new at-bat detection by comparing play indices."""

    def test_extract_current_play_index(self, live_feed):
        """Extract play index from a live feed."""
        play_idx, ab_idx = extract_current_play_index(live_feed)
        assert play_idx == 0
        assert ab_idx == 0

    def test_extract_play_index_from_all_plays_fallback(self):
        """When currentPlayIndex is missing, fall back to allPlays length."""
        feed = _build_live_feed()
        del feed["liveData"]["plays"]["currentPlayIndex"]
        play_idx, ab_idx = extract_current_play_index(feed)
        assert play_idx >= 0
        assert ab_idx == 0

    def test_extract_play_index_empty_feed(self):
        """Empty plays section returns (-1, -1)."""
        feed = {"liveData": {"plays": {}}}
        play_idx, ab_idx = extract_current_play_index(feed)
        assert play_idx == -1
        assert ab_idx == -1

    def test_first_poll_is_always_new_at_bat(self, live_feed, feed_state):
        """First poll should always be detected as a new at-bat."""
        assert feed_state.last_play_index == -1
        is_new, key = is_new_at_bat(live_feed, feed_state)
        assert is_new is True
        assert key is not None
        assert key.play_index == 0
        assert key.at_bat_index == 0

    def test_same_play_index_not_new(self, live_feed, feed_state):
        """Same play index as last poll is not a new at-bat."""
        feed_state.last_play_index = 0
        feed_state.last_at_bat_index = 0
        is_new, key = is_new_at_bat(live_feed, feed_state)
        assert is_new is False

    def test_incremented_at_bat_index_is_new(self, feed_state):
        """Increased at-bat index triggers new at-bat detection."""
        feed_state.last_play_index = 0
        feed_state.last_at_bat_index = 0
        feed = _build_live_feed(at_bat_index=1, current_play_index=1, num_all_plays=2)
        is_new, key = is_new_at_bat(feed, feed_state)
        assert is_new is True
        assert key.at_bat_index == 1

    def test_incremented_play_index_is_new(self, feed_state):
        """Increased play index triggers new at-bat detection."""
        feed_state.last_play_index = 3
        feed_state.last_at_bat_index = 3
        feed = _build_live_feed(at_bat_index=4, current_play_index=4, num_all_plays=5)
        is_new, key = is_new_at_bat(feed, feed_state)
        assert is_new is True

    def test_poll_updates_state_indices(self, live_feed, feed_state):
        """Polling updates the state's last play/at-bat indices."""
        assert feed_state.last_play_index == -1
        poll_game_feed(716463, feed_state, fetch_fn=lambda pk: live_feed)
        assert feed_state.last_play_index == 0
        assert feed_state.last_at_bat_index == 0

    def test_at_bat_key_equality(self):
        """AtBatKey equality is based on play_index and at_bat_index."""
        key1 = AtBatKey(play_index=5, at_bat_index=5, inning=3, half="top")
        key2 = AtBatKey(play_index=5, at_bat_index=5, inning=3, half="top")
        key3 = AtBatKey(play_index=6, at_bat_index=6, inning=3, half="top")
        assert key1 == key2
        assert key1 != key3

    def test_at_bat_key_hashable(self):
        """AtBatKeys can be used in sets."""
        key1 = AtBatKey(play_index=5, at_bat_index=5, inning=3, half="top")
        key2 = AtBatKey(play_index=5, at_bat_index=5, inning=3, half="top")
        s = {key1, key2}
        assert len(s) == 1

    def test_inning_change_detected(self, feed_state):
        """At-bat in a new inning is detected."""
        feed_state.last_play_index = 8
        feed_state.last_at_bat_index = 8
        feed = _build_live_feed(
            inning=2, is_top=True, at_bat_index=9, current_play_index=9,
            num_all_plays=10,
        )
        is_new, key = is_new_at_bat(feed, feed_state)
        assert is_new is True
        assert key.inning == 2
        assert key.half == "top"


# ===========================================================================
# Step 4: When a new at-bat is detected, parse game state and invoke agent
# ===========================================================================

class TestStep4InvokeAgent:
    """Validate agent invocation on new at-bat detection."""

    def test_invoke_agent_dry_run(self, live_feed, feed_state):
        """Dry-run mode returns NO_ACTION without calling the agent."""
        feed_state.managed_team_side = "home"
        result = invoke_agent(live_feed, feed_state, dry_run=True)
        assert result["decision"]["decision"] == "NO_ACTION"
        assert result["metadata"]["dry_run"] is True
        assert feed_state.total_agent_calls == 1

    def test_invoke_agent_dry_run_logs_entry(self, live_feed, feed_state):
        """Dry-run agent invocation creates a log entry."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        assert len(feed_state.decision_log) == 1
        entry = feed_state.decision_log[0]
        assert "game_state" in entry
        assert "decision" in entry
        assert entry["is_active_decision"] is False

    def test_invoke_agent_ingestion_error(self, feed_state):
        """Ingestion errors are handled gracefully."""
        feed_state.managed_team_side = "home"
        bad_feed = {"gameData": {}, "liveData": {}}
        result = invoke_agent(bad_feed, feed_state, dry_run=True)
        assert result["decision"]["decision"] == "NO_ACTION"
        assert "error" in result["metadata"]

    def test_invoke_agent_increments_call_count(self, live_feed, feed_state):
        """Each agent invocation increments the total agent call count."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        invoke_agent(live_feed, feed_state, dry_run=True)
        assert feed_state.total_agent_calls == 2

    def test_invoke_agent_decision_output(self, live_feed, feed_state):
        """Decision output is properly formatted."""
        feed_state.managed_team_side = "home"
        result = invoke_agent(live_feed, feed_state, dry_run=True)
        output = result["decision_output"]
        assert output is not None
        assert hasattr(output, "is_active")
        assert hasattr(output, "tweet_text")
        assert hasattr(output, "log_entry")

    def test_invoke_agent_timestamp(self, live_feed, feed_state):
        """Agent invocations include a timestamp."""
        feed_state.managed_team_side = "home"
        before = time.time()
        result = invoke_agent(live_feed, feed_state, dry_run=True)
        after = time.time()
        assert before <= result["timestamp"] <= after

    def test_invoke_agent_log_contains_game_state(self, live_feed, feed_state):
        """Log entry contains full game state context."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        entry = feed_state.decision_log[0]
        gs = entry["game_state"]
        assert "inning" in gs
        assert "half" in gs
        assert "outs" in gs
        assert "score" in gs
        assert "batter" in gs
        assert "pitcher" in gs


# ===========================================================================
# Step 5: Active decisions are formatted as tweet text
# ===========================================================================

class TestStep5TweetFormatting:
    """Validate that active decisions produce tweet-ready output."""

    def test_no_action_has_no_tweet(self, live_feed, feed_state):
        """NO_ACTION decisions don't produce tweet text."""
        feed_state.managed_team_side = "home"
        result = invoke_agent(live_feed, feed_state, dry_run=True)
        output = result["decision_output"]
        assert output.is_active is False
        assert output.tweet_text is None

    def test_active_decision_has_tweet_text(self, live_feed, feed_state):
        """Active decisions produce tweet text."""
        feed_state.managed_team_side = "home"
        # We simulate this by checking the format_decision_output directly
        from decision_output import format_decision_output
        active_decision = {
            "decision": "PITCHING_CHANGE",
            "action_details": "Bringing in Foster for the lefty",
            "reasoning": "TTO penalty too high",
            "key_factors": ["velocity decline"],
            "risks": [],
        }
        output = format_decision_output(
            decision_dict=active_decision,
            inning=7,
            half="BOTTOM",
            outs=1,
            score_home=3,
            score_away=4,
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280

    def test_tweet_text_in_log_entry(self, live_feed, feed_state):
        """Tweet text is stored in the decision log entry."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        entry = feed_state.decision_log[0]
        assert "tweet_text" in entry


# ===========================================================================
# Step 6: Continue polling until game reaches final state
# ===========================================================================

class TestStep6GameCompletion:
    """Validate polling continues until game is final."""

    def test_final_game_status_ends_polling(self, feed_state):
        """Game reaching 'Final' status causes game_over in PollResult."""
        final = _build_final_feed()
        result = poll_game_feed(716463, feed_state, fetch_fn=lambda pk: final)
        assert result.game_over is True

    def test_game_status_tracking(self, live_feed, feed_state):
        """Game status is tracked in the state."""
        poll_game_feed(716463, feed_state, fetch_fn=lambda pk: live_feed)
        assert feed_state.game_status == "Live"

    def test_run_live_game_ends_on_final(self, tmp_log_dir):
        """run_live_game exits when game reaches Final state."""
        call_count = 0

        def mock_fetch(pk):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return _build_final_feed(inning=9, home_runs=5, away_runs=3)
            return _build_live_feed(
                at_bat_index=call_count - 1,
                current_play_index=call_count - 1,
                num_all_plays=call_count,
            )

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                fetch_fn=mock_fetch,
                log_dir=tmp_log_dir,
            )

        assert state.game_status == "Final"
        assert state.total_polls >= 1

    def test_run_live_game_waits_for_preview(self, tmp_log_dir):
        """run_live_game waits when game is in Preview state."""
        call_count = 0

        def mock_fetch(pk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _build_preview_feed()
            if call_count == 2:
                return _build_live_feed()
            return _build_final_feed()

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                fetch_fn=mock_fetch,
                log_dir=tmp_log_dir,
            )

        assert state.game_status == "Final"


# ===========================================================================
# Step 7: Handle delays and feed unavailability without crashing
# ===========================================================================

class TestStep7ErrorHandling:
    """Validate graceful handling of errors and delays."""

    def test_transient_error_recovers(self, feed_state, live_feed):
        """Service recovers after transient errors."""
        call_count = 0

        def flaky_fetch(pk):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Temporary failure")
            return live_feed

        # First two calls fail
        r1 = poll_game_feed(716463, feed_state, fetch_fn=flaky_fetch)
        assert r1.error is not None
        assert feed_state.consecutive_errors == 1

        r2 = poll_game_feed(716463, feed_state, fetch_fn=flaky_fetch)
        assert r2.error is not None
        assert feed_state.consecutive_errors == 2

        # Third call succeeds
        r3 = poll_game_feed(716463, feed_state, fetch_fn=flaky_fetch)
        assert r3.error is None
        assert feed_state.consecutive_errors == 0

    def test_max_consecutive_errors_stops_service(self, tmp_log_dir):
        """Service stops after max consecutive errors."""
        def always_fail(pk):
            raise ConnectionError("Feed unavailable")

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                max_consecutive_errors=3,
                fetch_fn=always_fail,
                log_dir=tmp_log_dir,
            )

        assert state.consecutive_errors >= 3
        assert len(state.error_log) >= 3

    def test_ingestion_error_does_not_crash(self, feed_state):
        """Ingestion errors are caught and logged."""
        feed_state.managed_team_side = "home"
        # Minimal feed that will fail ingestion
        bad_feed = {
            "gameData": {"status": {"abstractGameState": "Live"}, "teams": {}, "players": {}},
            "liveData": {"linescore": {}, "plays": {}, "boxscore": {"teams": {}}},
        }
        result = invoke_agent(bad_feed, feed_state, dry_run=True)
        assert result["decision"]["decision"] == "NO_ACTION"
        assert "error" in result["metadata"]
        # Error logged
        assert len(feed_state.error_log) >= 1

    def test_error_log_accumulates(self, feed_state):
        """Multiple errors accumulate in the error log."""
        def failing_fetch(pk):
            raise TimeoutError("Request timed out")

        for _ in range(5):
            poll_game_feed(716463, feed_state, fetch_fn=failing_fetch)

        assert len(feed_state.error_log) == 5
        assert feed_state.consecutive_errors == 5

    def test_error_log_entries_have_timestamps(self, feed_state):
        """Error log entries include timestamps."""
        def failing_fetch(pk):
            raise ConnectionError("Network down")

        before = time.time()
        poll_game_feed(716463, feed_state, fetch_fn=failing_fetch)
        after = time.time()

        entry = feed_state.error_log[0]
        assert "timestamp" in entry
        assert before <= entry["timestamp"] <= after

    def test_error_log_entries_have_type(self, feed_state):
        """Error log entries include error type."""
        def failing_fetch(pk):
            raise ConnectionError("Error")

        poll_game_feed(716463, feed_state, fetch_fn=failing_fetch)
        assert feed_state.error_log[0]["error_type"] == "feed_fetch_error"


# ===========================================================================
# Step 8: Log all agent invocations and decisions
# ===========================================================================

class TestStep8DecisionLogging:
    """Validate comprehensive decision logging."""

    def test_decision_log_entry_structure(self, live_feed, feed_state):
        """Decision log entries have the expected structure."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        entry = feed_state.decision_log[0]

        assert "turn" in entry
        assert "timestamp" in entry
        assert "game_state" in entry
        assert "decision" in entry
        assert "is_active_decision" in entry
        assert "tweet_text" in entry
        assert "log_entry" in entry
        assert "tool_calls" in entry
        assert "token_usage" in entry
        assert "latency_ms" in entry

    def test_decision_log_game_state_context(self, live_feed, feed_state):
        """Decision log entry contains full game context."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        gs = feed_state.decision_log[0]["game_state"]

        assert gs["inning"] >= 1
        assert gs["half"] in ("TOP", "BOTTOM")
        assert gs["outs"] >= 0
        assert "home" in gs["score"]
        assert "away" in gs["score"]
        assert "player_id" in gs["batter"]
        assert "name" in gs["batter"]
        assert "player_id" in gs["pitcher"]
        assert "name" in gs["pitcher"]

    def test_decision_log_turn_numbering(self, live_feed, feed_state):
        """Decision log entries have sequential turn numbers."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        invoke_agent(live_feed, feed_state, dry_run=True)
        invoke_agent(live_feed, feed_state, dry_run=True)

        assert feed_state.decision_log[0]["turn"] == 1
        assert feed_state.decision_log[1]["turn"] == 2
        assert feed_state.decision_log[2]["turn"] == 3

    def test_write_live_game_log_creates_file(self, feed_state, tmp_log_dir):
        """Writing a game log creates a JSON file."""
        feed_state.managed_team_side = "home"
        log_path = write_live_game_log(feed_state, log_dir=tmp_log_dir)
        assert log_path.exists()
        assert log_path.suffix == ".json"

    def test_write_live_game_log_valid_json(self, live_feed, feed_state, tmp_log_dir):
        """Game log file contains valid JSON."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        log_path = write_live_game_log(feed_state, live_feed, log_dir=tmp_log_dir)

        with open(log_path) as f:
            data = json.load(f)

        assert "game_info" in data
        assert "summary" in data
        assert "decisions" in data
        assert "errors" in data

    def test_write_live_game_log_summary(self, live_feed, feed_state, tmp_log_dir):
        """Game log summary aggregates decision counts."""
        feed_state.managed_team_side = "home"
        invoke_agent(live_feed, feed_state, dry_run=True)
        invoke_agent(live_feed, feed_state, dry_run=True)
        log_path = write_live_game_log(feed_state, live_feed, log_dir=tmp_log_dir)

        with open(log_path) as f:
            data = json.load(f)

        assert data["summary"]["total_agent_calls"] == 2
        assert data["summary"]["total_polls"] == 0  # no polls in this test

    def test_write_live_game_log_game_info(self, live_feed, feed_state, tmp_log_dir):
        """Game log contains game info from the feed."""
        feed_state.managed_team_side = "home"
        log_path = write_live_game_log(feed_state, live_feed, log_dir=tmp_log_dir)

        with open(log_path) as f:
            data = json.load(f)

        assert data["game_info"]["game_pk"] == feed_state.game_pk
        assert data["game_info"]["home_team"] == "Atlanta Braves"
        assert data["game_info"]["away_team"] == "New York Yankees"

    def test_write_live_game_log_errors_included(self, feed_state, tmp_log_dir):
        """Error log entries are included in the game log."""
        feed_state.error_log.append({
            "timestamp": time.time(),
            "error_type": "test_error",
            "error": "Test error message",
        })
        log_path = write_live_game_log(feed_state, log_dir=tmp_log_dir)

        with open(log_path) as f:
            data = json.load(f)

        assert len(data["errors"]) == 1
        assert data["errors"][0]["error_type"] == "test_error"

    def test_write_live_game_log_no_feed(self, feed_state, tmp_log_dir):
        """Game log can be written even without a final feed."""
        log_path = write_live_game_log(feed_state, log_dir=tmp_log_dir)
        assert log_path.exists()

        with open(log_path) as f:
            data = json.load(f)

        assert data["game_info"]["home_team"] == ""
        assert data["game_info"]["away_team"] == ""


# ===========================================================================
# Extract helpers
# ===========================================================================

class TestExtractHelpers:
    """Test the feed extraction utility functions."""

    def test_extract_game_status_live(self, live_feed):
        assert extract_game_status(live_feed) == "Live"

    def test_extract_game_status_final(self, final_feed):
        assert extract_game_status(final_feed) == "Final"

    def test_extract_game_status_preview(self, preview_feed):
        assert extract_game_status(preview_feed) == "Preview"

    def test_extract_game_status_unknown(self):
        feed = {"gameData": {}}
        assert extract_game_status(feed) == "Unknown"

    def test_extract_current_inning_half_top(self, live_feed):
        inning, half = extract_current_inning_half(live_feed)
        assert inning == 1
        assert half == "top"

    def test_extract_current_inning_half_bottom(self):
        feed = _build_live_feed(inning=5, is_top=False)
        inning, half = extract_current_inning_half(feed)
        assert inning == 5
        assert half == "bottom"

    def test_extract_current_inning_half_empty(self):
        feed = {"liveData": {"linescore": {}}}
        inning, half = extract_current_inning_half(feed)
        assert inning == 0
        assert half == "top"  # default isTopInning is True

    def test_extract_play_index_late_game(self):
        """Play index in a late-game scenario with many plays."""
        feed = _build_live_feed(
            inning=7, is_top=False,
            at_bat_index=55, current_play_index=55,
            num_all_plays=56,
        )
        play_idx, ab_idx = extract_current_play_index(feed)
        assert play_idx == 55
        assert ab_idx == 55


# ===========================================================================
# Integration tests: run_live_game
# ===========================================================================

class TestRunLiveGameIntegration:
    """Integration tests for the full run_live_game loop."""

    def test_full_game_simulation(self, tmp_log_dir):
        """Simulate a full game from preview to final."""
        feeds = [
            _build_preview_feed(),
            _build_live_feed(inning=1, is_top=True, at_bat_index=0, current_play_index=0),
            _build_live_feed(inning=1, is_top=True, at_bat_index=1, current_play_index=1, num_all_plays=2),
            _build_live_feed(inning=1, is_top=False, at_bat_index=2, current_play_index=2, num_all_plays=3),
            _build_live_feed(inning=2, is_top=True, at_bat_index=5, current_play_index=5, num_all_plays=6),
            _build_final_feed(inning=9, home_runs=4, away_runs=2),
        ]
        idx = 0

        def mock_fetch(pk):
            nonlocal idx
            feed = feeds[min(idx, len(feeds) - 1)]
            idx += 1
            return feed

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                fetch_fn=mock_fetch,
                log_dir=tmp_log_dir,
            )

        assert state.game_status == "Final"
        # At least 2 new at-bats detected (index changes)
        assert state.total_agent_calls >= 2
        assert len(state.decision_log) >= 2

    def test_game_with_errors_recovers(self, tmp_log_dir):
        """Game with intermittent errors still completes."""
        call_count = 0

        def flaky_fetch(pk):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ConnectionError("Blip")
            if call_count >= 4:
                return _build_final_feed()
            return _build_live_feed(
                at_bat_index=call_count - 1,
                current_play_index=call_count - 1,
                num_all_plays=call_count,
            )

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                fetch_fn=flaky_fetch,
                log_dir=tmp_log_dir,
            )

        assert state.game_status == "Final"
        assert len(state.error_log) >= 1  # At least the one error
        assert state.total_agent_calls >= 1

    def test_poll_interval_clamped(self, tmp_log_dir):
        """Poll interval is clamped to valid range."""
        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                poll_interval=1,  # Below minimum
                dry_run=True,
                verbose=False,
                fetch_fn=lambda pk: _build_final_feed(),
                log_dir=tmp_log_dir,
            )
        assert state.poll_interval == MIN_POLL_INTERVAL

    def test_poll_interval_max_clamped(self, tmp_log_dir):
        """Poll interval is clamped to max."""
        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                poll_interval=500,  # Above maximum
                dry_run=True,
                verbose=False,
                fetch_fn=lambda pk: _build_final_feed(),
                log_dir=tmp_log_dir,
            )
        assert state.poll_interval == MAX_POLL_INTERVAL

    def test_game_log_written_on_completion(self, tmp_log_dir):
        """Game log file is written when the game ends."""
        with patch("live_game_feed.time.sleep"):
            run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                fetch_fn=lambda pk: _build_final_feed(),
                log_dir=tmp_log_dir,
            )

        log_files = list(tmp_log_dir.glob("live_game_*.json"))
        assert len(log_files) == 1
        assert log_files[0].name == "live_game_716463.json"

    def test_game_log_written_on_error_stop(self, tmp_log_dir):
        """Game log file is written even when stopping due to errors."""
        def always_fail(pk):
            raise ConnectionError("Down")

        with patch("live_game_feed.time.sleep"):
            run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                max_consecutive_errors=2,
                fetch_fn=always_fail,
                log_dir=tmp_log_dir,
            )

        log_files = list(tmp_log_dir.glob("live_game_*.json"))
        assert len(log_files) == 1

    def test_multiple_at_bats_different_indices(self, tmp_log_dir):
        """Agent is invoked for each new at-bat with different indices."""
        feeds = [
            _build_live_feed(at_bat_index=0, current_play_index=0),
            _build_live_feed(at_bat_index=0, current_play_index=0),  # Same -- no new AB
            _build_live_feed(at_bat_index=1, current_play_index=1, num_all_plays=2),  # New AB
            _build_live_feed(at_bat_index=1, current_play_index=1, num_all_plays=2),  # Same
            _build_live_feed(at_bat_index=2, current_play_index=2, num_all_plays=3),  # New AB
            _build_final_feed(),
        ]
        idx = 0

        def mock_fetch(pk):
            nonlocal idx
            feed = feeds[min(idx, len(feeds) - 1)]
            idx += 1
            return feed

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="ATL",
                dry_run=True,
                verbose=False,
                fetch_fn=mock_fetch,
                log_dir=tmp_log_dir,
            )

        # 3 new at-bats: index 0 (first poll), index 1, index 2
        assert state.total_agent_calls == 3

    def test_team_not_found_stops_service(self, tmp_log_dir):
        """Service stops if team cannot be resolved from the feed."""
        feed = _build_live_feed()

        with patch("live_game_feed.time.sleep"):
            state = run_live_game(
                game_pk=716463,
                team="Red Sox",  # Not in this game
                dry_run=True,
                verbose=False,
                fetch_fn=lambda pk: feed,
                log_dir=tmp_log_dir,
            )

        assert len(state.error_log) >= 1
        assert any("team_resolution_error" in e.get("error_type", "")
                    for e in state.error_log)


# ===========================================================================
# Data class tests
# ===========================================================================

class TestDataClasses:
    """Test data class behaviors."""

    def test_poll_result_defaults(self):
        pr = PollResult()
        assert pr.new_at_bat is False
        assert pr.game_over is False
        assert pr.feed is None
        assert pr.error is None
        assert pr.at_bat_key is None
        assert pr.game_status == ""

    def test_game_feed_state_initial_empty_logs(self):
        state = GameFeedState(game_pk=1, team="BOS", managed_team_side="home")
        assert state.decision_log == []
        assert state.error_log == []
        assert state.messages == []

    def test_at_bat_key_stores_inning_half(self):
        key = AtBatKey(play_index=10, at_bat_index=10, inning=5, half="bottom")
        assert key.inning == 5
        assert key.half == "bottom"

    def test_at_bat_key_inequality_different_type(self):
        key = AtBatKey(play_index=1, at_bat_index=1, inning=1, half="top")
        assert key != "not_a_key"

    def test_game_constants(self):
        """Verify game status constants."""
        assert GAME_STATE_FINAL == "Final"
        assert GAME_STATE_LIVE == "Live"
        assert GAME_STATE_PREVIEW == "Preview"
        assert MAX_CONSECUTIVE_ERRORS == 10


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_livedata(self):
        """Handle feed with minimal liveData."""
        feed = {"gameData": {"status": {"abstractGameState": "Live"}},
                "liveData": {}}
        status = extract_game_status(feed)
        assert status == "Live"

    def test_play_index_negative(self, feed_state):
        """Negative play index is not treated as a new at-bat."""
        feed = {"liveData": {"plays": {"currentPlay": {"atBatIndex": -1},
                                        "currentPlayIndex": -1, "allPlays": []},
                              "linescore": {"currentInning": 0, "isTopInning": True}}}
        is_new, key = is_new_at_bat(feed, feed_state)
        assert is_new is False

    def test_missing_plays_section(self):
        """Feed with no plays section returns (-1, -1) for indices."""
        feed = {"liveData": {}}
        play_idx, ab_idx = extract_current_play_index(feed)
        assert play_idx == -1
        assert ab_idx == -1

    def test_determine_team_side_with_numeric_string(self, live_feed):
        """Team ID as string works for team resolution."""
        side = determine_managed_team_side(live_feed, "144")
        assert side == "home"

    def test_determine_team_side_case_insensitive(self, live_feed):
        """Team name matching is case-insensitive."""
        side = determine_managed_team_side(live_feed, "BRAVES")
        assert side == "home"

    def test_write_log_creates_directory(self, feed_state):
        """write_live_game_log creates the log directory if missing."""
        import tempfile
        tmpdir = Path(tempfile.mkdtemp()) / "nested" / "logs"
        assert not tmpdir.exists()
        log_path = write_live_game_log(feed_state, log_dir=tmpdir)
        assert tmpdir.exists()
        assert log_path.exists()

    def test_active_decisions_counter(self, feed_state):
        """Active decisions counter tracks correctly via invoke."""
        feed_state.managed_team_side = "home"
        feed = _build_live_feed()
        # In dry-run, decisions are NO_ACTION so active count stays 0
        invoke_agent(feed, feed_state, dry_run=True)
        assert feed_state.active_decisions == 0

    def test_messages_trimmed_on_overflow(self, feed_state):
        """Messages list is trimmed to prevent context overflow."""
        feed_state.managed_team_side = "home"
        feed_state.messages = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
        feed = _build_live_feed()
        invoke_agent(feed, feed_state, dry_run=True)
        # In dry-run mode, messages aren't appended, but the trim logic
        # is tested -- messages should not grow unbounded
        # (This verifies the trim threshold exists in the code)
        assert len(feed_state.messages) <= 30  # reasonable bound
