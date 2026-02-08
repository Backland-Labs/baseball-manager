# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for backtest.tools module.

Validates:
  1. BACKTEST_TOOLS contains 12 tools
  2. set_backtest_context / clear_backtest_context lifecycle
  3. get_batter_stats returns structured response with today's line
  4. get_pitcher_stats returns structured response
  5. get_bullpen_status returns bullpen entries
  6. get_pitcher_fatigue_assessment derives fatigue levels
  7. get_matchup_data handles real BvP lookup (mocked)
  8. Error handling for invalid player IDs
  9. Context-not-set raises RuntimeError
  10. Pass-through tools are callable
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backtest.tools import (
    BACKTEST_TOOLS,
    clear_backtest_context,
    get_batter_stats,
    get_bullpen_status,
    get_matchup_data,
    get_pitcher_fatigue_assessment,
    get_pitcher_stats,
    set_backtest_context,
)
from backtest.extractor import BullpenEntry, DecisionPoint, PlayerState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "game_feed_746865.json"


@pytest.fixture(scope="module")
def game_feed():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def ctx_set(game_feed):
    """Set and clear backtest context around each test."""
    set_backtest_context("2024-06-15", game_feed)
    yield
    clear_backtest_context()


@pytest.fixture
def ctx_with_dp(game_feed):
    """Set context with a DecisionPoint for pitcher fatigue tests."""
    dp = DecisionPoint(
        play_index=0,
        inning=5,
        half="TOP",
        outs=1,
        runners={"first": None, "second": None, "third": None},
        score_home=3,
        score_away=1,
        batter_id="665742",
        batter_name="Masyn Winn",
        batter_bats="R",
        pitcher_id="684007",
        pitcher_name="Shota Imanaga",
        pitcher_throws="L",
        pitcher_pitch_count=75,
        pitcher_batters_faced=20,
        pitcher_innings_pitched=5.0,
        pitcher_runs_allowed=1,
        current_lineup=[],
        bench=[],
        bullpen=[
            BullpenEntry(player_id="670174", name="Tyson Miller", throws="R"),
            BullpenEntry(player_id="650633", name="Adbert Alzolay", throws="R"),
        ],
        opp_lineup=[],
        opp_bench=[],
        opp_bullpen=[],
        managed_team_side="home",
    )
    set_backtest_context("2024-06-15", game_feed, dp)
    yield dp
    clear_backtest_context()


def _parse(result: str) -> dict:
    """Parse a tool result JSON string, handling BetaFunctionTool wrapping."""
    # Tools decorated with @beta_tool return the raw string
    if isinstance(result, str):
        return json.loads(result)
    return result


# ---------------------------------------------------------------------------
# Tool list
# ---------------------------------------------------------------------------

class TestToolList:
    def test_backtest_tools_count(self):
        assert len(BACKTEST_TOOLS) == 12

    def test_all_tools_are_callable_or_beta(self):
        for t in BACKTEST_TOOLS:
            # BetaFunctionTool objects aren't directly callable but have a name
            assert hasattr(t, "name") or callable(t)


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

class TestContextManagement:
    def test_context_not_set_raises(self):
        clear_backtest_context()
        with pytest.raises(RuntimeError, match="Backtest context not set"):
            # Call a tool function directly (not via BetaFunctionTool)
            from backtest.tools import _get_context
            _get_context()

    def test_set_and_clear(self, game_feed):
        set_backtest_context("2024-06-15", game_feed)
        from backtest.tools import _get_context
        ctx = _get_context()
        assert ctx["game_date"] == "2024-06-15"
        clear_backtest_context()
        with pytest.raises(RuntimeError):
            _get_context()


# ---------------------------------------------------------------------------
# get_batter_stats
# ---------------------------------------------------------------------------

class TestGetBatterStats:
    def test_valid_batter(self, ctx_set, game_feed):
        # Find a real batter from the game feed
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        batter_id = str(first_play["matchup"]["batter"]["id"])

        result = _parse(get_batter_stats(batter_id))
        assert result["status"] == "ok"
        assert result["tool"] == "get_batter_stats"
        assert "data" in result
        assert result["data"]["player_id"] == batter_id
        assert "today" in result["data"]

    def test_invalid_batter(self, ctx_set):
        result = _parse(get_batter_stats("9999999"))
        assert result["status"] == "error"
        assert result["error_code"] == "INVALID_PLAYER_ID"

    def test_splits_passed_through(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        batter_id = str(first_play["matchup"]["batter"]["id"])

        result = _parse(get_batter_stats(batter_id, vs_hand="L"))
        assert result["data"]["splits"]["vs_hand"] == "L"

    def test_today_line_has_keys(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        batter_id = str(first_play["matchup"]["batter"]["id"])

        result = _parse(get_batter_stats(batter_id))
        today = result["data"]["today"]
        for key in ("AB", "H", "BB", "K", "RBI"):
            assert key in today


# ---------------------------------------------------------------------------
# get_pitcher_stats
# ---------------------------------------------------------------------------

class TestGetPitcherStats:
    def test_valid_pitcher(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        result = _parse(get_pitcher_stats(pitcher_id))
        assert result["status"] == "ok"
        assert result["tool"] == "get_pitcher_stats"
        assert "today" in result["data"]

    def test_invalid_pitcher(self, ctx_set):
        result = _parse(get_pitcher_stats("9999999"))
        assert result["status"] == "error"

    def test_today_line_has_keys(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        result = _parse(get_pitcher_stats(pitcher_id))
        today = result["data"]["today"]
        for key in ("IP", "H", "R", "ER", "BB", "K"):
            assert key in today


# ---------------------------------------------------------------------------
# get_bullpen_status
# ---------------------------------------------------------------------------

class TestGetBullpenStatus:
    def test_home_bullpen(self, ctx_set):
        result = _parse(get_bullpen_status("home"))
        assert result["status"] == "ok"
        assert result["data"]["team"] == "home"
        assert result["data"]["bullpen_count"] > 0
        assert len(result["data"]["bullpen"]) > 0

    def test_away_bullpen(self, ctx_set):
        result = _parse(get_bullpen_status("away"))
        assert result["status"] == "ok"
        assert result["data"]["team"] == "away"

    def test_invalid_team(self, ctx_set):
        result = _parse(get_bullpen_status("invalid"))
        assert result["status"] == "error"

    def test_with_decision_point(self, ctx_with_dp):
        result = _parse(get_bullpen_status("home"))
        assert result["status"] == "ok"
        # Should use the DP's bullpen
        assert result["data"]["bullpen_count"] == 2

    def test_used_pitcher_excluded(self, ctx_with_dp):
        result = _parse(get_bullpen_status("home", used_pitcher_ids="670174"))
        assert result["data"]["bullpen_count"] == 1
        pitcher_ids = [p["player_id"] for p in result["data"]["bullpen"]]
        assert "670174" not in pitcher_ids


# ---------------------------------------------------------------------------
# get_pitcher_fatigue_assessment
# ---------------------------------------------------------------------------

class TestGetPitcherFatigueAssessment:
    def test_fresh_pitcher(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        result = _parse(get_pitcher_fatigue_assessment(
            pitcher_id, pitch_count=10, innings_pitched=1.0,
        ))
        assert result["status"] == "ok"
        assert result["data"]["fatigue_level"] == "fresh"

    def test_fatigued_pitcher(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        result = _parse(get_pitcher_fatigue_assessment(
            pitcher_id, pitch_count=85, innings_pitched=6.0,
        ))
        assert result["status"] == "ok"
        assert result["data"]["fatigue_level"] in ("fatigued", "gassed")

    def test_gassed_pitcher(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        result = _parse(get_pitcher_fatigue_assessment(
            pitcher_id, pitch_count=110, innings_pitched=7.0,
        ))
        assert result["status"] == "ok"
        assert result["data"]["fatigue_level"] == "gassed"

    def test_uses_decision_point_data(self, ctx_with_dp):
        """When DP is set and pitcher matches, use DP's pitch count."""
        result = _parse(get_pitcher_fatigue_assessment("684007"))
        assert result["status"] == "ok"
        assert result["data"]["pitch_count"] == 75  # From DP

    def test_invalid_pitcher(self, ctx_set):
        result = _parse(get_pitcher_fatigue_assessment("9999999"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# get_matchup_data
# ---------------------------------------------------------------------------

class TestGetMatchupData:
    @patch("backtest.tools.get_batter_vs_pitcher")
    @patch("backtest.tools.extract_bvp_stats")
    def test_matchup_with_history(self, mock_extract, mock_bvp, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        batter_id = str(first_play["matchup"]["batter"]["id"])
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        mock_bvp.return_value = {}
        mock_extract.return_value = {
            "plate_appearances": 15,
            "at_bats": 12,
            "hits": 4,
            "doubles": 1,
            "triples": 0,
            "home_runs": 1,
            "walks": 2,
            "strikeouts": 3,
            "avg": 0.333,
            "obp": 0.400,
            "slg": 0.583,
            "ops": 0.983,
            "small_sample": False,
            "batter_name": "Test Batter",
            "pitcher_name": "Test Pitcher",
        }

        result = _parse(get_matchup_data(batter_id, pitcher_id))
        assert result["status"] == "ok"
        assert result["data"]["career_pa"] == 15
        assert result["data"]["sample_size_reliability"] == "medium"
        assert "matchup_stats" in result["data"]
        assert result["data"]["matchup_stats"]["AVG"] == 0.333

    @patch("backtest.tools.get_batter_vs_pitcher")
    @patch("backtest.tools.extract_bvp_stats")
    def test_matchup_no_history(self, mock_extract, mock_bvp, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        batter_id = str(first_play["matchup"]["batter"]["id"])
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        mock_bvp.return_value = {}
        mock_extract.return_value = {
            "plate_appearances": 0,
            "at_bats": 0,
            "hits": 0,
            "doubles": 0,
            "triples": 0,
            "home_runs": 0,
            "walks": 0,
            "strikeouts": 0,
            "avg": None,
            "obp": None,
            "slg": None,
            "ops": None,
            "small_sample": True,
            "batter_name": "",
            "pitcher_name": "",
        }

        result = _parse(get_matchup_data(batter_id, pitcher_id))
        assert result["status"] == "ok"
        assert result["data"]["career_pa"] == 0
        assert result["data"]["sample_size_reliability"] == "none"

    def test_invalid_batter(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        pitcher_id = str(first_play["matchup"]["pitcher"]["id"])

        result = _parse(get_matchup_data("9999999", pitcher_id))
        assert result["status"] == "error"

    def test_invalid_pitcher(self, ctx_set, game_feed):
        first_play = game_feed["liveData"]["plays"]["allPlays"][0]
        batter_id = str(first_play["matchup"]["batter"]["id"])

        result = _parse(get_matchup_data(batter_id, "9999999"))
        assert result["status"] == "error"
