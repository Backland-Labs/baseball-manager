# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for backtest.runner module.

Validates:
  1. build_scenario produces correct dict structure
  2. Scenario has matchup_state, roster_state, opponent_roster_state, decision_prompt
  3. compare_decisions handles category match/disagree cases
  4. _normalize_decision_type maps variations correctly
  5. format_report produces readable output
  6. ComparisonEntry model serialization
  7. run_backtest dry-run mode returns empty list
  8. Decision prompt text varies by batting/fielding
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backtest.extractor import (
    ActionType,
    BullpenEntry,
    DecisionPoint,
    PlayerState,
    RealManagerAction,
    walk_game_feed,
)
from backtest.runner import (
    ComparisonEntry,
    _normalize_decision_type,
    build_scenario,
    compare_decisions,
    format_report,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "game_feed_746865.json"


@pytest.fixture(scope="module")
def game_feed():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def sample_dp():
    """A minimal DecisionPoint for testing build_scenario."""
    return DecisionPoint(
        play_index=10,
        inning=3,
        half="TOP",
        outs=1,
        runners={
            "first": {"id": "111", "name": "Speed Guy"},
            "second": None,
            "third": None,
        },
        score_home=2,
        score_away=1,
        batter_id="222",
        batter_name="Test Batter",
        batter_bats="L",
        pitcher_id="333",
        pitcher_name="Test Pitcher",
        pitcher_throws="R",
        pitcher_pitch_count=45,
        pitcher_batters_faced=12,
        pitcher_innings_pitched=3.0,
        pitcher_runs_allowed=1,
        current_lineup=[
            PlayerState(player_id=str(i), name=f"Player {i}", position=pos, bats="R")
            for i, pos in enumerate(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"], 1)
        ],
        bench=[
            PlayerState(player_id="10", name="Bench Guy", position="OF", bats="L"),
        ],
        bullpen=[
            BullpenEntry(player_id="20", name="Reliever A", throws="R"),
            BullpenEntry(player_id="21", name="Reliever B", throws="L"),
        ],
        opp_lineup=[
            PlayerState(player_id=str(i+100), name=f"Opp {i}", position=pos, bats="R")
            for i, pos in enumerate(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"], 1)
        ],
        opp_bench=[
            PlayerState(player_id="110", name="Opp Bench", position="OF", bats="L"),
        ],
        opp_bullpen=[
            BullpenEntry(player_id="120", name="Opp Reliever", throws="R"),
        ],
        on_deck_batter_id="222",
        on_deck_batter_name="Next Batter",
        on_deck_batter_bats="R",
        managed_team_side="home",
    )


@pytest.fixture
def sample_dp_batting():
    """DP where managed team is batting (home team, bottom inning)."""
    return DecisionPoint(
        play_index=5,
        inning=2,
        half="BOTTOM",
        outs=0,
        runners={"first": None, "second": None, "third": None},
        score_home=0,
        score_away=0,
        batter_id="444",
        batter_name="Our Batter",
        batter_bats="R",
        pitcher_id="555",
        pitcher_name="Opp Pitcher",
        pitcher_throws="L",
        pitcher_pitch_count=30,
        pitcher_batters_faced=8,
        pitcher_innings_pitched=2.0,
        pitcher_runs_allowed=0,
        current_lineup=[
            PlayerState(player_id=str(i), name=f"P{i}", position="SS", bats="R")
            for i in range(1, 10)
        ],
        bench=[],
        bullpen=[],
        opp_lineup=[
            PlayerState(player_id=str(i+100), name=f"O{i}", position="SS", bats="R")
            for i in range(1, 10)
        ],
        opp_bench=[],
        opp_bullpen=[],
        managed_team_side="home",
    )


# ---------------------------------------------------------------------------
# build_scenario
# ---------------------------------------------------------------------------

class TestBuildScenario:
    def test_returns_required_keys(self, sample_dp):
        scenario = build_scenario(sample_dp)
        assert "matchup_state" in scenario
        assert "roster_state" in scenario
        assert "opponent_roster_state" in scenario
        assert "decision_prompt" in scenario

    def test_matchup_state_fields(self, sample_dp):
        scenario = build_scenario(sample_dp)
        ms = scenario["matchup_state"]
        assert ms["inning"] == 3
        assert ms["half"] == "TOP"
        assert ms["outs"] == 1
        assert ms["score"]["home"] == 2
        assert ms["score"]["away"] == 1
        assert ms["batter"]["player_id"] == "222"
        assert ms["batter"]["name"] == "Test Batter"
        assert ms["pitcher"]["player_id"] == "333"
        assert ms["pitcher"]["pitch_count_today"] == 45

    def test_runners_in_matchup(self, sample_dp):
        scenario = build_scenario(sample_dp)
        runners = scenario["matchup_state"]["runners"]
        assert runners["first"] is not None
        assert runners["first"]["name"] == "Speed Guy"
        assert runners["second"] is None
        assert runners["third"] is None

    def test_roster_state_lineup(self, sample_dp):
        scenario = build_scenario(sample_dp)
        rs = scenario["roster_state"]
        assert len(rs["our_lineup"]) == 9
        assert len(rs["bench"]) == 1
        assert len(rs["bullpen"]) == 2

    def test_opponent_roster_state(self, sample_dp):
        scenario = build_scenario(sample_dp)
        ors = scenario["opponent_roster_state"]
        assert len(ors["their_lineup"]) == 9
        assert len(ors["their_bench"]) == 1
        assert len(ors["their_bullpen"]) == 1

    def test_on_deck_batter_included(self, sample_dp):
        scenario = build_scenario(sample_dp)
        assert scenario["matchup_state"]["on_deck_batter"] is not None
        assert scenario["matchup_state"]["on_deck_batter"]["name"] == "Next Batter"

    def test_fielding_decision_prompt(self, sample_dp):
        """Home team in top of inning = fielding."""
        scenario = build_scenario(sample_dp)
        prompt = scenario["decision_prompt"]
        assert "FIELDING" in prompt
        assert "pitching change" in prompt.lower() or "no action" in prompt.lower()

    def test_batting_decision_prompt(self, sample_dp_batting):
        """Home team in bottom of inning = batting."""
        scenario = build_scenario(sample_dp_batting)
        prompt = scenario["decision_prompt"]
        assert "BATTING" in prompt
        assert "pinch-hit" in prompt.lower() or "stolen base" in prompt.lower()

    def test_score_description_leading(self, sample_dp):
        """Home leads 2-1, so 'Your team leads'."""
        scenario = build_scenario(sample_dp)
        assert "leads" in scenario["decision_prompt"].lower()

    def test_score_description_trailing(self):
        dp = DecisionPoint(
            play_index=0, inning=1, half="TOP", outs=0,
            runners={"first": None, "second": None, "third": None},
            score_home=0, score_away=3,
            batter_id="1", batter_name="B", batter_bats="R",
            pitcher_id="2", pitcher_name="P", pitcher_throws="R",
            pitcher_pitch_count=0, pitcher_batters_faced=0,
            pitcher_innings_pitched=0.0, pitcher_runs_allowed=0,
            current_lineup=[], bench=[], bullpen=[],
            opp_lineup=[], opp_bench=[], opp_bullpen=[],
            managed_team_side="home",
        )
        scenario = build_scenario(dp)
        assert "trails" in scenario["decision_prompt"].lower()

    def test_score_description_tied(self):
        dp = DecisionPoint(
            play_index=0, inning=1, half="TOP", outs=0,
            runners={"first": None, "second": None, "third": None},
            score_home=0, score_away=0,
            batter_id="1", batter_name="B", batter_bats="R",
            pitcher_id="2", pitcher_name="P", pitcher_throws="R",
            pitcher_pitch_count=0, pitcher_batters_faced=0,
            pitcher_innings_pitched=0.0, pitcher_runs_allowed=0,
            current_lineup=[], bench=[], bullpen=[],
            opp_lineup=[], opp_bench=[], opp_bullpen=[],
            managed_team_side="home",
        )
        scenario = build_scenario(dp)
        assert "tied" in scenario["decision_prompt"].lower()

    def test_on_deck_in_prompt(self, sample_dp):
        scenario = build_scenario(sample_dp)
        assert "Next Batter" in scenario["decision_prompt"]


# ---------------------------------------------------------------------------
# _normalize_decision_type
# ---------------------------------------------------------------------------

class TestNormalizeDecisionType:
    def test_no_action_variations(self):
        assert _normalize_decision_type("NO_ACTION") == "NO_ACTION"
        assert _normalize_decision_type("no action") == "NO_ACTION"
        assert _normalize_decision_type("NO_CHANGE") == "NO_ACTION"
        assert _normalize_decision_type("CONTINUE") == "NO_ACTION"
        assert _normalize_decision_type("SWING_AWAY") == "NO_ACTION"

    def test_pitching_change_variations(self):
        assert _normalize_decision_type("PITCHING_CHANGE") == "PITCHING_CHANGE"
        assert _normalize_decision_type("PULL_STARTER") == "PITCHING_CHANGE"
        assert _normalize_decision_type("BRING_IN_RELIEVER") == "PITCHING_CHANGE"
        assert _normalize_decision_type("CALL_BULLPEN") == "PITCHING_CHANGE"

    def test_pinch_hit_variations(self):
        assert _normalize_decision_type("PINCH_HIT") == "PINCH_HIT"
        assert _normalize_decision_type("PINCH_HITTER") == "PINCH_HIT"
        assert _normalize_decision_type("PINCH-HIT") == "PINCH_HIT"

    def test_pinch_run_variations(self):
        assert _normalize_decision_type("PINCH_RUN") == "PINCH_RUN"
        assert _normalize_decision_type("PINCH_RUNNER") == "PINCH_RUN"
        assert _normalize_decision_type("PINCH-RUN") == "PINCH_RUN"

    def test_stolen_base(self):
        assert _normalize_decision_type("STOLEN_BASE") == "STOLEN_BASE"
        assert _normalize_decision_type("STEAL") == "STOLEN_BASE"
        assert _normalize_decision_type("ATTEMPT_STEAL") == "STOLEN_BASE"

    def test_bunt(self):
        assert _normalize_decision_type("BUNT") == "BUNT"
        assert _normalize_decision_type("SAC_BUNT") == "BUNT"
        assert _normalize_decision_type("SACRIFICE_BUNT") == "BUNT"

    def test_ibb(self):
        assert _normalize_decision_type("IBB") == "IBB"
        assert _normalize_decision_type("INTENTIONAL_WALK") == "IBB"

    def test_unknown_passed_through(self):
        assert _normalize_decision_type("SOME_CUSTOM") == "SOME_CUSTOM"


# ---------------------------------------------------------------------------
# compare_decisions
# ---------------------------------------------------------------------------

class TestCompareDecisions:
    def test_both_no_action(self):
        result = compare_decisions(
            {"decision": "NO_ACTION"},
            None,
        )
        assert result["category_match"] is True
        assert result["action_match"] is True
        assert result["disagreement_type"] is None

    def test_category_disagree(self):
        action = RealManagerAction(
            action_type=ActionType.PITCHING_CHANGE,
            player_in="100",
            player_in_name="Rivera",
            details="Bring in Rivera",
        )
        result = compare_decisions(
            {"decision": "NO_ACTION"},
            action,
        )
        assert result["category_match"] is False
        assert result["disagreement_type"] is not None

    def test_category_match_pitching_change(self):
        action = RealManagerAction(
            action_type=ActionType.PITCHING_CHANGE,
            player_in="100",
            player_in_name="Rivera",
            details="Bring in Rivera",
        )
        result = compare_decisions(
            {"decision": "PITCHING_CHANGE", "action_details": "Bring in Rivera"},
            action,
        )
        assert result["category_match"] is True
        assert result["action_match"] is True

    def test_category_match_wrong_player(self):
        action = RealManagerAction(
            action_type=ActionType.PITCHING_CHANGE,
            player_in="100",
            player_in_name="Rivera",
            details="Bring in Rivera",
        )
        result = compare_decisions(
            {"decision": "PITCHING_CHANGE", "action_details": "Bring in Smith"},
            action,
        )
        assert result["category_match"] is True
        assert result["action_match"] is False

    def test_agent_pitching_variant_matches(self):
        action = RealManagerAction(
            action_type=ActionType.PITCHING_CHANGE,
            player_in="100",
            player_in_name="Rivera",
            details="",
        )
        result = compare_decisions(
            {"decision": "PULL_STARTER", "action_details": "Rivera should come in"},
            action,
        )
        assert result["category_match"] is True
        assert result["agent_type"] == "PITCHING_CHANGE"

    def test_real_none_agent_active(self):
        result = compare_decisions(
            {"decision": "PITCHING_CHANGE", "action_details": "Pull starter"},
            None,
        )
        assert result["category_match"] is False
        assert result["agent_type"] == "PITCHING_CHANGE"
        assert result["real_type"] == "NO_ACTION"


# ---------------------------------------------------------------------------
# ComparisonEntry
# ---------------------------------------------------------------------------

class TestComparisonEntry:
    def test_creation(self):
        entry = ComparisonEntry(
            play_index=0,
            inning=1,
            half="TOP",
            outs=0,
            score_home=0,
            score_away=0,
            batter_name="Batter",
            pitcher_name="Pitcher",
            leverage_index=1.0,
            real_outcome="Strikeout",
        )
        assert entry.agent_decision_type == ""
        assert entry.category_match is False
        assert entry.tool_calls == []
        assert entry.token_usage == {}

    def test_serialization(self):
        entry = ComparisonEntry(
            play_index=0, inning=1, half="TOP", outs=0,
            score_home=0, score_away=0,
            batter_name="B", pitcher_name="P",
            leverage_index=1.0, real_outcome="Single",
            category_match=True, action_match=True,
            agent_decision_type="NO_ACTION",
            real_manager_action_type="NO_ACTION",
        )
        d = entry.model_dump()
        assert d["category_match"] is True
        assert d["agent_decision_type"] == "NO_ACTION"


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_empty_entries(self):
        assert format_report([], {}) == "No entries to report."

    def test_report_has_header(self):
        entries = [
            ComparisonEntry(
                play_index=0, inning=1, half="TOP", outs=0,
                score_home=0, score_away=0,
                batter_name="B", pitcher_name="P",
                leverage_index=1.0, real_outcome="K",
                category_match=True,
                agent_decision_type="NO_ACTION",
                real_manager_action_type="NO_ACTION",
            ),
        ]
        game_info = {
            "home": "Cubs", "away": "Cardinals",
            "date": "2024-06-15",
            "home_score": 5, "away_score": 1,
            "managed_side": "home",
        }
        report = format_report(entries, game_info)
        assert "BACKTEST REPORT" in report
        assert "Cubs" in report
        assert "Cardinals" in report
        assert "Agreement Summary" in report

    def test_report_shows_disagreements(self):
        entries = [
            ComparisonEntry(
                play_index=0, inning=5, half="TOP", outs=1,
                score_home=2, score_away=1,
                batter_name="Hitter", pitcher_name="Thrower",
                leverage_index=2.5, real_outcome="HR",
                category_match=False,
                agent_decision_type="NO_ACTION",
                real_manager_action_type="PITCHING_CHANGE",
                real_manager_action_details="Pull the starter",
                disagreement_type="Agent: NO_ACTION, Real: PITCHING_CHANGE",
            ),
        ]
        game_info = {
            "home": "Cubs", "away": "Cards",
            "date": "2024-06-15",
            "home_score": 5, "away_score": 1,
            "managed_side": "home",
        }
        report = format_report(entries, game_info)
        assert "Disagreements" in report
        assert "NO_ACTION" in report
        assert "PITCHING_CHANGE" in report

    def test_report_shows_high_leverage(self):
        entries = [
            ComparisonEntry(
                play_index=0, inning=7, half="BOTTOM", outs=2,
                score_home=3, score_away=3,
                batter_name="Clutch", pitcher_name="Closer",
                leverage_index=3.0, real_outcome="Walk",
                category_match=True,
                agent_decision_type="IBB",
                real_manager_action_type="IBB",
            ),
        ]
        game_info = {"home": "A", "away": "B", "date": "x",
                     "home_score": 3, "away_score": 3, "managed_side": "home"}
        report = format_report(entries, game_info)
        assert "High-Leverage" in report

    def test_report_shows_cost(self):
        entries = [
            ComparisonEntry(
                play_index=0, inning=1, half="TOP", outs=0,
                score_home=0, score_away=0,
                batter_name="B", pitcher_name="P",
                leverage_index=1.0, real_outcome="K",
                category_match=True,
                agent_decision_type="NO_ACTION",
                real_manager_action_type="NO_ACTION",
                token_usage={"input_tokens": 1000, "output_tokens": 200},
            ),
        ]
        game_info = {"home": "A", "away": "B", "date": "x",
                     "home_score": 0, "away_score": 0, "managed_side": "home"}
        report = format_report(entries, game_info)
        assert "Cost" in report
        assert "1,200" in report


# ---------------------------------------------------------------------------
# run_backtest (dry-run, using fixture)
# ---------------------------------------------------------------------------

class TestRunBacktestDryRun:
    @patch("backtest.runner.get_live_game_feed")
    def test_dry_run_returns_empty(self, mock_feed, game_feed, capsys):
        mock_feed.return_value = game_feed
        entries = run_backtest(746865, "CHC", dry_run=True, verbose=False)
        assert entries == []

    @patch("backtest.runner.get_live_game_feed")
    def test_dry_run_prints_summary(self, mock_feed, game_feed, capsys):
        mock_feed.return_value = game_feed
        run_backtest(746865, "CHC", dry_run=True, verbose=True)
        captured = capsys.readouterr()
        assert "DRY RUN SUMMARY" in captured.out
        assert "Decision points:" in captured.out

    @patch("backtest.runner.get_live_game_feed")
    def test_dry_run_away_team(self, mock_feed, game_feed, capsys):
        mock_feed.return_value = game_feed
        entries = run_backtest(746865, "STL", dry_run=True, verbose=False)
        assert entries == []

    @patch("backtest.runner.get_live_game_feed")
    def test_not_final_raises(self, mock_feed, game_feed):
        feed_copy = json.loads(json.dumps(game_feed))
        feed_copy["gameData"]["status"]["detailedState"] = "In Progress"
        mock_feed.return_value = feed_copy
        with pytest.raises(ValueError, match="not final"):
            run_backtest(746865, "CHC", dry_run=True, verbose=False)


# ---------------------------------------------------------------------------
# Integration: build_scenario from real fixture DPs
# ---------------------------------------------------------------------------

class TestBuildScenarioFromFixture:
    def test_all_dps_produce_valid_scenarios(self, game_feed):
        dps = list(walk_game_feed(game_feed, "CHC"))
        for dp in dps:
            scenario = build_scenario(dp)
            assert "matchup_state" in scenario
            assert "roster_state" in scenario
            assert "opponent_roster_state" in scenario
            assert "decision_prompt" in scenario
            assert len(scenario["decision_prompt"]) > 20

    def test_first_dp_scenario_fields(self, game_feed):
        dps = list(walk_game_feed(game_feed, "CHC"))
        scenario = build_scenario(dps[0])
        ms = scenario["matchup_state"]
        assert ms["inning"] == 1
        assert ms["half"] == "TOP"
        assert ms["outs"] == 0
        assert ms["score"]["home"] == 0
        assert ms["score"]["away"] == 0
