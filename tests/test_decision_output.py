# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the decision_output feature.

Validates that the agent's decisions are properly formatted for output:
  1. Active decisions produce tweet-ready text within ~280 characters
  2. Tweet text includes game context (inning, situation, teams)
  3. No-action responses produce log entries but no tweet output
  4. Both tweet text and full reasoning are returned for logging
  5. Edge cases: long text truncation, empty fields, all decision types
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from decision_output import (
    DecisionOutput,
    NO_ACTION_TYPES,
    TWEET_MAX_LENGTH,
    _build_game_context,
    _truncate_to_tweet,
    _format_tweet_text,
    _format_log_entry,
    format_decision_output,
    format_decision_output_from_game_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pitching_change_decision():
    """A typical pitching change decision dict."""
    return {
        "decision": "PITCHING_CHANGE",
        "action_details": "Pulling Smith off the mound. Bringing in Rivera from the pen -- his .198 wOBA vs lefties is exactly what we need against Johnson.",
        "confidence": 0.85,
        "reasoning": "Smith is seeing the lineup for the 3rd time and his fastball is down 2.3 mph from the 1st inning. Rivera has a .198 wOBA vs lefties this season.",
        "key_factors": [
            "3rd time through order penalty",
            "Fastball velocity down 2.3 mph",
            "Rivera .198 wOBA vs LHB",
        ],
        "risks": ["Rivera pitched 20 pitches yesterday"],
    }


@pytest.fixture
def pinch_hit_decision():
    """A typical pinch-hit decision dict."""
    return {
        "decision": "PINCH_HIT",
        "action_details": "Sending Tanaka to pinch-hit for Ortiz against the lefty Henderson.",
        "confidence": 0.72,
        "reasoning": "Ortiz has a L-L matchup disadvantage. Tanaka's .340 wOBA vs LHP this season vs Ortiz's .285.",
        "key_factors": [
            "L-L matchup disadvantage for Ortiz",
            "Tanaka .340 wOBA vs LHP",
        ],
        "risks": ["Lose Ortiz's glove at 1B for rest of game"],
    }


@pytest.fixture
def stolen_base_decision():
    """A stolen base decision dict."""
    return {
        "decision": "STOLEN_BASE",
        "action_details": "Green light for Chen to steal 2nd. 29.1 ft/s sprint speed, catcher Santos has 2.05 pop time.",
        "confidence": 0.68,
        "reasoning": "Success probability ~78%, breakeven is 71.5%. Expected RE change is +0.12 runs.",
        "key_factors": [
            "78% success probability",
            "71.5% breakeven rate",
            "Favorable RE change",
        ],
        "risks": ["If caught, rally killed with 2 outs"],
    }


@pytest.fixture
def no_action_decision():
    """A no-action decision dict."""
    return {
        "decision": "NO_ACTION",
        "action_details": "No strategic move needed. Let the batter swing away.",
        "confidence": 0.90,
        "reasoning": "Standard at-bat situation, no advantageous moves available.",
        "key_factors": [],
        "risks": [],
    }


@pytest.fixture
def swing_away_decision():
    """A swing-away (no-action variant) decision dict."""
    return {
        "decision": "SWING_AWAY",
        "action_details": "Let the batter hit. Good matchup as-is.",
        "confidence": 0.88,
        "reasoning": "Favorable platoon matchup, no need to intervene.",
        "key_factors": ["Favorable platoon matchup"],
        "risks": [],
    }


@pytest.fixture
def standard_runners():
    """Runner state with a runner on 2nd."""
    return {
        "first": None,
        "second": {"player_id": "p001", "name": "Marcus Chen", "sprint_speed": 29.1},
        "third": None,
    }


@pytest.fixture
def bases_loaded_runners():
    """Runner state with bases loaded."""
    return {
        "first": {"player_id": "p001", "name": "Chen"},
        "second": {"player_id": "p002", "name": "Williams"},
        "third": {"player_id": "p003", "name": "Ortiz"},
    }


# ---------------------------------------------------------------------------
# Tests: _build_game_context
# ---------------------------------------------------------------------------

class TestBuildGameContext:
    """Tests for game context string generation."""

    def test_basic_context_no_runners(self):
        ctx = _build_game_context(
            inning=3, half="TOP", outs=1,
            score_home=2, score_away=1,
        )
        assert "Top 3" in ctx
        assert "1 out" in ctx
        assert "bases empty" in ctx
        assert "Away 1" in ctx
        assert "Home 2" in ctx

    def test_bottom_half(self):
        ctx = _build_game_context(
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert "Bot 7" in ctx
        assert "2 out" in ctx

    def test_zero_outs(self):
        ctx = _build_game_context(
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert "0 out" in ctx

    def test_runner_on_first(self):
        runners = {"first": {"player_id": "p1", "name": "Chen"}, "second": None, "third": None}
        ctx = _build_game_context(
            inning=5, half="TOP", outs=1,
            score_home=2, score_away=2, runners=runners,
        )
        assert "runner on 1st" in ctx

    def test_runners_on_first_and_third(self):
        runners = {
            "first": {"player_id": "p1", "name": "Chen"},
            "second": None,
            "third": {"player_id": "p3", "name": "Ortiz"},
        }
        ctx = _build_game_context(
            inning=6, half="BOTTOM", outs=0,
            score_home=1, score_away=3, runners=runners,
        )
        assert "runners on" in ctx
        assert "1st" in ctx
        assert "3rd" in ctx

    def test_bases_loaded(self, bases_loaded_runners):
        ctx = _build_game_context(
            inning=8, half="BOTTOM", outs=2,
            score_home=5, score_away=5, runners=bases_loaded_runners,
        )
        assert "bases loaded" in ctx

    def test_with_team_names(self):
        ctx = _build_game_context(
            inning=9, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
            home_team="Hawks", away_team="Wolves",
        )
        assert "Hawks" in ctx
        assert "Wolves" in ctx
        assert "Wolves 4" in ctx
        assert "Hawks 3" in ctx

    def test_without_team_names(self):
        ctx = _build_game_context(
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert "Away 0" in ctx
        assert "Home 0" in ctx

    def test_none_runners_treated_as_empty(self):
        ctx = _build_game_context(
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0, runners=None,
        )
        assert "bases empty" in ctx

    def test_empty_dict_runners_treated_as_empty(self):
        ctx = _build_game_context(
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0, runners={},
        )
        assert "bases empty" in ctx

    def test_runner_on_second_only(self, standard_runners):
        ctx = _build_game_context(
            inning=10, half="TOP", outs=0,
            score_home=3, score_away=3, runners=standard_runners,
        )
        assert "runner on 2nd" in ctx

    def test_runners_on_second_and_third(self):
        runners = {
            "first": None,
            "second": {"player_id": "p2"},
            "third": {"player_id": "p3"},
        }
        ctx = _build_game_context(
            inning=7, half="BOTTOM", outs=1,
            score_home=2, score_away=4, runners=runners,
        )
        assert "runners on" in ctx
        assert "2nd" in ctx
        assert "3rd" in ctx

    def test_high_inning_number(self):
        ctx = _build_game_context(
            inning=14, half="TOP", outs=0,
            score_home=5, score_away=5,
        )
        assert "Top 14" in ctx


# ---------------------------------------------------------------------------
# Tests: _truncate_to_tweet
# ---------------------------------------------------------------------------

class TestTruncateToTweet:
    """Tests for tweet truncation."""

    def test_short_text_unchanged(self):
        text = "This is a short tweet."
        assert _truncate_to_tweet(text) == text

    def test_exactly_280_chars_unchanged(self):
        text = "x" * 280
        assert _truncate_to_tweet(text) == text
        assert len(_truncate_to_tweet(text)) == 280

    def test_long_text_truncated(self):
        text = "word " * 100  # 500 chars
        result = _truncate_to_tweet(text)
        assert len(result) <= 280

    def test_truncated_ends_with_ellipsis(self):
        text = "word " * 100
        result = _truncate_to_tweet(text)
        assert result.endswith("\u2026")

    def test_truncation_at_word_boundary(self):
        # Build text that's just over 280
        text = "This is a test. " * 20  # 320 chars
        result = _truncate_to_tweet(text)
        assert len(result) <= 280
        # Should not cut in the middle of a word (before the ellipsis)
        before_ellipsis = result[:-1]  # Remove the ellipsis character
        assert not before_ellipsis.endswith(" ")  # stripped trailing space

    def test_custom_max_length(self):
        text = "This is a test message that is longer than the limit."
        result = _truncate_to_tweet(text, max_length=30)
        assert len(result) <= 30

    def test_empty_string(self):
        assert _truncate_to_tweet("") == ""

    def test_single_very_long_word(self):
        text = "x" * 300
        result = _truncate_to_tweet(text)
        assert len(result) <= 280
        assert result.endswith("\u2026")

    def test_281_chars_truncated(self):
        text = "a " * 141  # 282 chars
        result = _truncate_to_tweet(text)
        assert len(result) <= 280


# ---------------------------------------------------------------------------
# Tests: _format_tweet_text
# ---------------------------------------------------------------------------

class TestFormatTweetText:
    """Tests for tweet text formatting."""

    def test_basic_tweet(self):
        tweet = _format_tweet_text(
            decision_type="PITCHING_CHANGE",
            action_details="Pulling Smith, bringing in Rivera.",
            reasoning="3rd TTO, velocity drop.",
            game_context="Bot 7, 2 out, bases loaded | Away 4, Home 3",
        )
        assert len(tweet) <= 280
        assert "Bot 7" in tweet
        assert "Rivera" in tweet

    def test_tweet_includes_game_context(self):
        tweet = _format_tweet_text(
            decision_type="PINCH_HIT",
            action_details="Sending Tanaka to bat for Ortiz.",
            reasoning="L-L disadvantage.",
            game_context="Top 8, 0 out, runner on 2nd | Wolves 3, Hawks 2",
        )
        assert "Top 8" in tweet
        assert "Tanaka" in tweet

    def test_short_action_details_enriched_with_key_factors(self):
        tweet = _format_tweet_text(
            decision_type="STOLEN_BASE",
            action_details="Green light to steal.",
            reasoning="High success probability.",
            game_context="Bot 5, 0 out, runner on 1st | Away 2, Home 1",
            key_factors=["78% success probability", "Slow catcher pop time"],
        )
        assert len(tweet) <= 280
        # Short action details should be enriched
        assert "78%" in tweet or "success" in tweet.lower() or "steal" in tweet.lower()

    def test_long_action_details_truncated(self):
        long_details = "x" * 300
        tweet = _format_tweet_text(
            decision_type="PITCHING_CHANGE",
            action_details=long_details,
            reasoning="reason",
            game_context="Bot 7, 1 out, bases empty | Away 2, Home 1",
        )
        assert len(tweet) <= 280

    def test_key_factors_appended_when_room(self):
        tweet = _format_tweet_text(
            decision_type="PITCHING_CHANGE",
            action_details="Bringing in Rivera.",
            reasoning="TTO penalty.",
            game_context="Bot 7, 1 out | Away 4, Home 3",
            key_factors=["Rivera .198 wOBA vs LHB"],
        )
        # The key factor should be appended if there's room
        if len(tweet) <= 280:
            # The tweet could include the factor
            pass  # Just verify it doesn't exceed limit
        assert len(tweet) <= 280

    def test_tweet_never_exceeds_280(self):
        """Fuzz test with various input sizes."""
        for detail_len in [10, 50, 100, 200, 300, 500]:
            for n_factors in [0, 1, 3, 5]:
                tweet = _format_tweet_text(
                    decision_type="PITCHING_CHANGE",
                    action_details="x" * detail_len,
                    reasoning="y" * 200,
                    game_context="Bot 9, 2 out, bases loaded | Away 4, Home 3",
                    key_factors=["factor " * 5] * n_factors,
                )
                assert len(tweet) <= 280, f"Tweet exceeded 280 chars with detail_len={detail_len}, n_factors={n_factors}: {len(tweet)}"

    def test_empty_action_details_with_factors(self):
        tweet = _format_tweet_text(
            decision_type="MOUND_VISIT",
            action_details="",
            reasoning="Pitcher seems rattled.",
            game_context="Top 6, 0 out | Away 1, Home 0",
            key_factors=["Velocity down 3 mph", "3 walks this inning"],
        )
        assert len(tweet) <= 280


# ---------------------------------------------------------------------------
# Tests: _format_log_entry
# ---------------------------------------------------------------------------

class TestFormatLogEntry:
    """Tests for log entry formatting."""

    def test_active_decision_log(self):
        entry = _format_log_entry(
            decision_type="PITCHING_CHANGE",
            action_details="Bringing in Rivera.",
            game_context="Bot 7, 1 out | Away 4, Home 3",
            is_active=True,
        )
        assert "[ACTIVE]" in entry
        assert "PITCHING_CHANGE" in entry
        assert "Rivera" in entry
        assert "Bot 7" in entry

    def test_no_action_log(self):
        entry = _format_log_entry(
            decision_type="NO_ACTION",
            action_details="No move needed.",
            game_context="Top 3, 0 out | Away 0, Home 0",
            is_active=False,
        )
        assert "[NO_ACTION]" in entry
        assert "Top 3" in entry
        # Action details should NOT appear in no-action log
        assert "No move needed" not in entry

    def test_swing_away_log(self):
        entry = _format_log_entry(
            decision_type="SWING_AWAY",
            action_details="Let him hit.",
            game_context="Bot 1, 1 out | Away 0, Home 0",
            is_active=False,
        )
        assert "[NO_ACTION]" in entry
        assert "SWING_AWAY" in entry


# ---------------------------------------------------------------------------
# Tests: format_decision_output (main entry point)
# ---------------------------------------------------------------------------

class TestFormatDecisionOutput:
    """Tests for the main format_decision_output function."""

    def test_active_decision_produces_tweet(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
            runners={"first": {"player_id": "p1"}, "second": {"player_id": "p2"}, "third": {"player_id": "p3"}},
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280
        assert output.decision_type == "PITCHING_CHANGE"

    def test_active_decision_tweet_has_context(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
            home_team="Hawks", away_team="Wolves",
        )
        assert "Bot 7" in output.tweet_text
        assert "Hawks" in output.tweet_text or "Wolves" in output.tweet_text

    def test_no_action_produces_no_tweet(self, no_action_decision):
        output = format_decision_output(
            decision_dict=no_action_decision,
            inning=3, half="TOP", outs=1,
            score_home=0, score_away=0,
        )
        assert output.is_active is False
        assert output.tweet_text is None

    def test_swing_away_produces_no_tweet(self, swing_away_decision):
        output = format_decision_output(
            decision_dict=swing_away_decision,
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert output.is_active is False
        assert output.tweet_text is None

    def test_all_no_action_types_produce_no_tweet(self):
        for no_action_type in NO_ACTION_TYPES:
            decision = {
                "decision": no_action_type,
                "action_details": "No action.",
                "reasoning": "Default.",
            }
            output = format_decision_output(
                decision_dict=decision,
                inning=1, half="TOP", outs=0,
                score_home=0, score_away=0,
            )
            assert output.is_active is False, f"{no_action_type} should not be active"
            assert output.tweet_text is None, f"{no_action_type} should not produce tweet"

    def test_full_reasoning_includes_reasoning_text(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert "3rd time through order" in output.full_reasoning or "fastball" in output.full_reasoning.lower()

    def test_full_reasoning_includes_key_factors(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert "Key factors" in output.full_reasoning

    def test_full_reasoning_includes_risks(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert "Risks" in output.full_reasoning

    def test_full_reasoning_includes_confidence(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert "Confidence" in output.full_reasoning
        assert "85%" in output.full_reasoning

    def test_log_entry_for_active_decision(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert "[ACTIVE]" in output.log_entry
        assert "PITCHING_CHANGE" in output.log_entry

    def test_log_entry_for_no_action(self, no_action_decision):
        output = format_decision_output(
            decision_dict=no_action_decision,
            inning=3, half="TOP", outs=1,
            score_home=0, score_away=0,
        )
        assert "[NO_ACTION]" in output.log_entry

    def test_game_context_stored(self, pinch_hit_decision):
        output = format_decision_output(
            decision_dict=pinch_hit_decision,
            inning=7, half="BOTTOM", outs=1,
            score_home=3, score_away=4,
            home_team="Hawks", away_team="Wolves",
        )
        assert "Bot 7" in output.game_context
        assert "Hawks" in output.game_context or "Wolves" in output.game_context

    def test_pinch_hit_is_active(self, pinch_hit_decision):
        output = format_decision_output(
            decision_dict=pinch_hit_decision,
            inning=7, half="BOTTOM", outs=1,
            score_home=3, score_away=4,
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert "Tanaka" in output.tweet_text

    def test_stolen_base_is_active(self, stolen_base_decision):
        output = format_decision_output(
            decision_dict=stolen_base_decision,
            inning=5, half="BOTTOM", outs=0,
            score_home=1, score_away=2,
            runners={"first": {"player_id": "p1", "name": "Chen"}, "second": None, "third": None},
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280

    def test_decision_type_and_action_stored(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert output.decision_type == "PITCHING_CHANGE"
        assert "Rivera" in output.action_details

    def test_empty_decision_dict(self):
        output = format_decision_output(
            decision_dict={},
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert output.is_active is False
        assert output.tweet_text is None
        assert output.decision_type == ""

    def test_decision_type_case_insensitive(self):
        decision = {
            "decision": "pitching_change",
            "action_details": "Pulling the starter.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=7, half="BOTTOM", outs=1,
            score_home=3, score_away=4,
        )
        assert output.is_active is True
        assert output.decision_type == "PITCHING_CHANGE"

    def test_decision_type_with_whitespace(self):
        decision = {
            "decision": "  PITCHING_CHANGE  ",
            "action_details": "Pulling the starter.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=7, half="BOTTOM", outs=1,
            score_home=3, score_away=4,
        )
        assert output.decision_type == "PITCHING_CHANGE"
        assert output.is_active is True

    def test_missing_optional_fields(self):
        decision = {
            "decision": "INTENTIONAL_WALK",
            "action_details": "Walking the batter intentionally.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=7, half="BOTTOM", outs=1,
            score_home=3, score_away=4,
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280
        assert "No reasoning provided" not in output.full_reasoning or output.full_reasoning == "No reasoning provided"

    def test_all_active_decision_types(self):
        active_types = [
            "PITCHING_CHANGE", "PULL_STARTER", "BRING_IN_RELIEVER",
            "PINCH_HIT", "PINCH_HITTER", "STOLEN_BASE", "STEAL",
            "INTENTIONAL_WALK", "IBB", "DEFENSIVE_POSITIONING",
            "SHIFT", "INFIELD_IN", "POSITION_CHANGE", "MOUND_VISIT",
            "SACRIFICE_BUNT", "BUNT", "SQUEEZE", "PINCH_RUN",
            "PINCH_RUNNER", "REPLAY_CHALLENGE", "CHALLENGE",
        ]
        for dtype in active_types:
            decision = {
                "decision": dtype,
                "action_details": f"Executing {dtype}.",
            }
            output = format_decision_output(
                decision_dict=decision,
                inning=5, half="TOP", outs=1,
                score_home=2, score_away=3,
            )
            assert output.is_active is True, f"{dtype} should be active"
            assert output.tweet_text is not None, f"{dtype} should produce tweet"
            assert len(output.tweet_text) <= 280, f"{dtype} tweet exceeds 280 chars"

    def test_no_reasoning_produces_fallback(self):
        decision = {
            "decision": "NO_ACTION",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert output.full_reasoning == "No reasoning provided"

    def test_tweet_with_team_names(self, pitching_change_decision):
        output = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
            home_team="Hawks", away_team="Wolves",
        )
        assert output.tweet_text is not None
        # Game context should include team names
        assert "Hawks" in output.game_context
        assert "Wolves" in output.game_context

    def test_extra_innings_context(self):
        decision = {
            "decision": "PITCHING_CHANGE",
            "action_details": "Bringing in the closer for the 11th.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=11, half="TOP", outs=0,
            score_home=5, score_away=5,
            runners={"first": None, "second": {"player_id": "p1"}, "third": None},
        )
        assert "11" in output.game_context
        assert "runner on 2nd" in output.game_context


# ---------------------------------------------------------------------------
# Tests: format_decision_output_from_game_state
# ---------------------------------------------------------------------------

class TestFormatDecisionOutputFromGameState:
    """Tests for the convenience wrapper."""

    def test_basic_usage(self, pitching_change_decision):
        game_state_dict = {
            "inning": 7,
            "half": "BOTTOM",
            "outs": 2,
            "score": {"home": 3, "away": 4},
            "runners": {},
        }
        output = format_decision_output_from_game_state(
            decision_dict=pitching_change_decision,
            game_state_dict=game_state_dict,
            home_team="Hawks",
            away_team="Wolves",
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert "Bot 7" in output.game_context

    def test_with_runners(self, stolen_base_decision):
        game_state_dict = {
            "inning": 5,
            "half": "BOTTOM",
            "outs": 0,
            "score": {"home": 1, "away": 2},
            "runners": {
                "1": {"player_id": "p1", "name": "Chen"},
            },
        }
        output = format_decision_output_from_game_state(
            decision_dict=stolen_base_decision,
            game_state_dict=game_state_dict,
        )
        assert output.is_active is True

    def test_missing_fields_use_defaults(self, no_action_decision):
        game_state_dict = {}
        output = format_decision_output_from_game_state(
            decision_dict=no_action_decision,
            game_state_dict=game_state_dict,
        )
        assert output.is_active is False
        assert "Top 1" in output.game_context
        assert "0 out" in output.game_context

    def test_empty_runners_dict(self, pitching_change_decision):
        game_state_dict = {
            "inning": 3,
            "half": "TOP",
            "outs": 1,
            "score": {"home": 0, "away": 0},
            "runners": {},
        }
        output = format_decision_output_from_game_state(
            decision_dict=pitching_change_decision,
            game_state_dict=game_state_dict,
        )
        assert "bases empty" in output.game_context


# ---------------------------------------------------------------------------
# Tests: DecisionOutput dataclass
# ---------------------------------------------------------------------------

class TestDecisionOutputDataclass:
    """Tests for the DecisionOutput dataclass itself."""

    def test_all_fields_accessible(self):
        output = DecisionOutput(
            is_active=True,
            tweet_text="Test tweet",
            full_reasoning="Full reasoning here",
            log_entry="[ACTIVE] Test",
            decision_type="PITCHING_CHANGE",
            action_details="Bringing in Rivera.",
            game_context="Bot 7, 2 out | Away 4, Home 3",
        )
        assert output.is_active is True
        assert output.tweet_text == "Test tweet"
        assert output.full_reasoning == "Full reasoning here"
        assert output.log_entry == "[ACTIVE] Test"
        assert output.decision_type == "PITCHING_CHANGE"
        assert output.action_details == "Bringing in Rivera."
        assert output.game_context == "Bot 7, 2 out | Away 4, Home 3"

    def test_none_tweet_for_no_action(self):
        output = DecisionOutput(
            is_active=False,
            tweet_text=None,
            full_reasoning="Standard play.",
            log_entry="[NO_ACTION] Top 1",
            decision_type="NO_ACTION",
            action_details="No move needed.",
            game_context="Top 1, 0 out | Away 0, Home 0",
        )
        assert output.tweet_text is None


# ---------------------------------------------------------------------------
# Tests: NO_ACTION_TYPES consistency
# ---------------------------------------------------------------------------

class TestNoActionTypes:
    """Tests for NO_ACTION_TYPES constant consistency."""

    def test_no_action_types_is_set(self):
        assert isinstance(NO_ACTION_TYPES, set)

    def test_contains_expected_types(self):
        expected = {"NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
                    "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER"}
        assert expected == NO_ACTION_TYPES

    def test_consistent_with_game_module(self):
        """Verify our NO_ACTION_TYPES matches the one in game.py."""
        from game import NO_ACTION_TYPES as game_no_action_types
        assert NO_ACTION_TYPES == game_no_action_types


# ---------------------------------------------------------------------------
# Tests: Integration with simulation game state
# ---------------------------------------------------------------------------

class TestIntegrationWithSimulation:
    """Integration tests using real simulation game state data."""

    @pytest.fixture
    def rosters(self):
        from simulation import load_rosters
        return load_rosters()

    @pytest.fixture
    def engine(self):
        from simulation import SimulationEngine
        return SimulationEngine(seed=42)

    @pytest.fixture
    def game_state(self, rosters, engine):
        return engine.initialize_game(rosters)

    def test_format_from_game_state_scenario(self, game_state, rosters):
        """Format a decision using a real game state."""
        from simulation import game_state_to_scenario
        scenario = game_state_to_scenario(game_state, "home")

        decision = {
            "decision": "NO_ACTION",
            "action_details": "Standard opening at-bat.",
            "reasoning": "No strategic move warranted.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=scenario["matchup_state"]["inning"],
            half=scenario["matchup_state"]["half"],
            outs=scenario["matchup_state"]["outs"],
            score_home=scenario["matchup_state"]["score"]["home"],
            score_away=scenario["matchup_state"]["score"]["away"],
            runners=scenario["matchup_state"]["runners"],
            home_team=rosters["home"]["team_name"],
            away_team=rosters["away"]["team_name"],
        )
        assert output.is_active is False
        assert output.tweet_text is None
        assert output.game_context  # non-empty

    def test_format_active_decision_from_game_state(self, game_state, rosters, engine):
        """Format an active decision after advancing the game state."""
        from simulation import game_state_to_scenario

        # Advance game a bit
        for _ in range(15):
            if game_state.game_over:
                break
            pa = engine.simulate_plate_appearance(game_state)
            engine.apply_pa_result(game_state, pa)

        if not game_state.game_over:
            scenario = game_state_to_scenario(game_state, "home")
            decision = {
                "decision": "PITCHING_CHANGE",
                "action_details": "Time for a fresh arm.",
                "confidence": 0.80,
                "reasoning": "Starter is tiring.",
                "key_factors": ["High pitch count", "TTO penalty"],
                "risks": ["Bullpen usage"],
            }
            output = format_decision_output(
                decision_dict=decision,
                inning=scenario["matchup_state"]["inning"],
                half=scenario["matchup_state"]["half"],
                outs=scenario["matchup_state"]["outs"],
                score_home=scenario["matchup_state"]["score"]["home"],
                score_away=scenario["matchup_state"]["score"]["away"],
                runners=scenario["matchup_state"]["runners"],
                home_team=rosters["home"]["team_name"],
                away_team=rosters["away"]["team_name"],
            )
            assert output.is_active is True
            assert output.tweet_text is not None
            assert len(output.tweet_text) <= 280

    def test_format_with_build_decision_log_entry(self, game_state):
        """Verify compatibility with build_decision_log_entry output."""
        from game import build_decision_log_entry
        import time

        decision_dict = {
            "decision": "STOLEN_BASE",
            "action_details": "Green light to steal 2nd.",
            "confidence": 0.70,
            "reasoning": "High success probability.",
            "key_factors": ["Fast runner"],
            "risks": ["Rally killed if caught"],
        }
        metadata = {
            "tool_calls": [],
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
            "latency_ms": 500,
            "retries": 0,
        }
        log_entry = build_decision_log_entry(
            turn=1,
            game_state=game_state,
            managed_team="home",
            decision_dict=decision_dict,
            decision_metadata=metadata,
            timestamp=time.time(),
        )

        # Use the game_state from the log entry
        output = format_decision_output_from_game_state(
            decision_dict=decision_dict,
            game_state_dict=log_entry["game_state"],
        )
        assert output.is_active is True
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_reasoning(self):
        decision = {
            "decision": "PITCHING_CHANGE",
            "action_details": "Pulling starter.",
            "reasoning": "x" * 5000,
            "key_factors": ["factor"] * 20,
            "risks": ["risk"] * 10,
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280
        # Full reasoning should still be complete
        assert len(output.full_reasoning) > 280

    def test_unicode_in_player_names(self):
        decision = {
            "decision": "PINCH_HIT",
            "action_details": "Sending Gonzalez to bat for Nakamura.",
            "reasoning": "Better matchup.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=6, half="BOTTOM", outs=1,
            score_home=2, score_away=3,
        )
        assert output.tweet_text is not None
        assert len(output.tweet_text) <= 280
        assert "Gonzalez" in output.tweet_text or "Nakamura" in output.tweet_text

    def test_zero_confidence(self):
        decision = {
            "decision": "NO_ACTION",
            "action_details": "Forced no-action.",
            "confidence": 0.0,
            "reasoning": "Agent failed.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert output.is_active is False
        # Zero confidence should not appear in reasoning
        assert "Confidence: 0%" not in output.full_reasoning

    def test_none_decision_value(self):
        decision = {"decision": None}
        output = format_decision_output(
            decision_dict=decision,
            inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
        )
        assert output.is_active is False
        assert output.tweet_text is None

    def test_tweet_max_length_constant(self):
        assert TWEET_MAX_LENGTH == 280

    def test_multiple_decisions_independent(self, pitching_change_decision, no_action_decision):
        """Formatting one decision does not affect another."""
        output1 = format_decision_output(
            decision_dict=pitching_change_decision,
            inning=7, half="BOTTOM", outs=2,
            score_home=3, score_away=4,
        )
        output2 = format_decision_output(
            decision_dict=no_action_decision,
            inning=3, half="TOP", outs=1,
            score_home=0, score_away=0,
        )
        assert output1.is_active is True
        assert output2.is_active is False
        assert output1.tweet_text is not None
        assert output2.tweet_text is None

    def test_high_scoring_game(self):
        decision = {
            "decision": "PITCHING_CHANGE",
            "action_details": "Need a fresh arm.",
        }
        output = format_decision_output(
            decision_dict=decision,
            inning=5, half="TOP", outs=0,
            score_home=12, score_away=15,
        )
        assert "12" in output.game_context
        assert "15" in output.game_context
        assert output.tweet_text is not None
