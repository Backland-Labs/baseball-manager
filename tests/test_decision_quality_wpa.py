# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the decision_quality_wpa feature.

Validates:
  1. WP lookup returns correct values for known game states
  2. LI lookup returns correct values for known game states
  3. WPA computation: WP_after - WP_before
  4. Decision scoring classifies active vs no-action decisions
  5. Game-level WPA report aggregates correctly
  6. Game log scoring computes WPA from saved logs
  7. Game log file scoring loads and processes files
  8. WPA report formatting produces human-readable output
  9. Integration with game.py: decision log entries contain WPA fields
  10. Edge cases: game-over states, extreme scores, empty logs
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from decision_quality_wpa import (
    DecisionWPA,
    GameWPAReport,
    NO_ACTION_TYPES,
    _runners_key,
    compute_li_from_game_state,
    compute_wp_from_game_state,
    format_wpa_report,
    generate_game_wpa_report,
    lookup_li,
    lookup_wp,
    score_decision,
    score_game_log,
    score_game_log_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_active_decision():
    return {
        "decision": "PITCHING_CHANGE",
        "action_details": "Bring in Rivera from the pen",
        "confidence": 0.85,
        "reasoning": "Starter is fatigued",
        "key_factors": ["velocity drop", "TTO penalty"],
        "risks": ["bullpen usage"],
    }


@pytest.fixture
def sample_no_action_decision():
    return {
        "decision": "NO_ACTION",
        "action_details": "No strategic move needed",
        "confidence": 0.90,
        "reasoning": "Standard at-bat situation",
        "key_factors": [],
        "risks": [],
    }


@pytest.fixture
def sample_game_log():
    """A minimal game log with a few decisions for testing score_game_log."""
    return {
        "game_info": {
            "seed": 42,
            "home_team": "Cardinals",
            "away_team": "Cubs",
            "managed_team": "home",
            "final_score": {"home": 5, "away": 3},
            "winner": "Cardinals",
            "innings": 9,
        },
        "summary": {
            "total_decisions": 3,
            "active_decisions": 1,
            "no_action_decisions": 2,
        },
        "decisions": [
            {
                "turn": 1,
                "timestamp": 1000000,
                "game_state": {
                    "inning": 1,
                    "half": "TOP",
                    "outs": 0,
                    "score": {"home": 0, "away": 0},
                    "runners": {},
                },
                "managed_team": "home",
                "decision": {
                    "decision": "NO_ACTION",
                    "action_details": "Standard play",
                },
                "is_active_decision": False,
            },
            {
                "turn": 2,
                "timestamp": 1000010,
                "game_state": {
                    "inning": 7,
                    "half": "BOTTOM",
                    "outs": 1,
                    "score": {"home": 3, "away": 4},
                    "runners": {"1": {"player_id": "p1", "name": "Runner"}},
                },
                "managed_team": "home",
                "decision": {
                    "decision": "PITCHING_CHANGE",
                    "action_details": "Bring in closer",
                },
                "is_active_decision": True,
            },
            {
                "turn": 3,
                "timestamp": 1000020,
                "game_state": {
                    "inning": 9,
                    "half": "BOTTOM",
                    "outs": 2,
                    "score": {"home": 5, "away": 3},
                    "runners": {},
                },
                "managed_team": "home",
                "decision": {
                    "decision": "NO_ACTION",
                    "action_details": "Let him pitch",
                },
                "is_active_decision": False,
            },
        ],
        "errors": [],
    }


# ---------------------------------------------------------------------------
# 1. WP lookup tests
# ---------------------------------------------------------------------------

class TestWPLookup:
    """Win probability lookup from pre-computed tables."""

    def test_home_team_leading_late(self):
        """Home team leading in the 9th should have high WP."""
        wp = lookup_wp(inning=9, half="BOTTOM", outs=0, runner_key="000",
                       score_diff=3, is_home=True)
        assert wp > 0.90, f"Home leading by 3 in Bot 9 should have high WP, got {wp}"

    def test_away_team_leading_late(self):
        """Away team leading in the 9th should have high WP."""
        wp = lookup_wp(inning=9, half="TOP", outs=0, runner_key="000",
                       score_diff=3, is_home=False)
        assert wp > 0.85, f"Away leading by 3 in Top 9 should have high WP, got {wp}"

    def test_tied_game_start(self):
        """Tied game at start should be close to 0.50 for either team."""
        wp_home = lookup_wp(inning=1, half="TOP", outs=0, runner_key="000",
                            score_diff=0, is_home=True)
        assert 0.40 < wp_home < 0.65, f"Home team at start should be near 0.50, got {wp_home}"

    def test_trailing_team_lower_wp(self):
        """Trailing team should have lower WP than leading team."""
        wp_leading = lookup_wp(inning=5, half="TOP", outs=1, runner_key="000",
                               score_diff=2, is_home=True)
        wp_trailing = lookup_wp(inning=5, half="TOP", outs=1, runner_key="000",
                                score_diff=-2, is_home=True)
        assert wp_leading > wp_trailing

    def test_runners_on_base_affect_wp(self):
        """Runners on base should affect WP (higher for batting team)."""
        # With runners, the batting team (away, top) has more scoring potential
        wp_empty = lookup_wp(inning=5, half="TOP", outs=0, runner_key="000",
                             score_diff=0, is_home=True)
        wp_loaded = lookup_wp(inning=5, half="TOP", outs=0, runner_key="111",
                              score_diff=0, is_home=True)
        # Away is batting in top, so runners help away = lower WP for home
        assert wp_loaded != wp_empty, "Runners should affect WP"

    def test_wp_clamped_to_range(self):
        """WP should always be between 0.01 and 0.99."""
        wp = lookup_wp(inning=9, half="BOTTOM", outs=2, runner_key="000",
                       score_diff=10, is_home=True)
        assert 0.01 <= wp <= 0.99

        wp2 = lookup_wp(inning=9, half="BOTTOM", outs=2, runner_key="000",
                        score_diff=-10, is_home=True)
        assert 0.01 <= wp2 <= 0.99

    def test_extreme_score_diff_clamped(self):
        """Score differentials beyond +/-10 should be clamped."""
        wp_10 = lookup_wp(inning=5, half="TOP", outs=0, runner_key="000",
                          score_diff=10, is_home=True)
        wp_20 = lookup_wp(inning=5, half="TOP", outs=0, runner_key="000",
                          score_diff=20, is_home=True)
        assert wp_10 == wp_20, "Scores beyond 10 should clamp to 10"

    def test_extreme_inning_clamped(self):
        """Innings beyond 12 should use inning 12 data."""
        wp_12 = lookup_wp(inning=12, half="TOP", outs=0, runner_key="000",
                          score_diff=0, is_home=True)
        wp_15 = lookup_wp(inning=15, half="TOP", outs=0, runner_key="000",
                          score_diff=0, is_home=True)
        assert wp_12 == wp_15

    def test_home_vs_away_perspective(self):
        """Same state viewed from home vs away should sum to ~1.0."""
        wp_home = lookup_wp(inning=5, half="TOP", outs=1, runner_key="100",
                            score_diff=1, is_home=True)
        wp_away = lookup_wp(inning=5, half="TOP", outs=1, runner_key="100",
                            score_diff=-1, is_home=False)
        # They should be near complements
        assert abs(wp_home + wp_away - 1.0) < 0.02, \
            f"Home {wp_home} + Away {wp_away} should be ~1.0"

    def test_more_outs_reduces_threat(self):
        """More outs should generally change WP."""
        wp_0out = lookup_wp(inning=5, half="TOP", outs=0, runner_key="100",
                            score_diff=0, is_home=True)
        wp_2out = lookup_wp(inning=5, half="TOP", outs=2, runner_key="100",
                            score_diff=0, is_home=True)
        # With runners on for the away team (top), more outs = less threat = higher home WP
        assert wp_2out > wp_0out or abs(wp_2out - wp_0out) < 0.05


# ---------------------------------------------------------------------------
# 2. LI lookup tests
# ---------------------------------------------------------------------------

class TestLILookup:
    """Leverage index lookup from pre-computed tables."""

    def test_li_default_fallback(self):
        """Fallback LI should be 1.0 for invalid half value."""
        li = lookup_li(inning=5, half="INVALID", outs=0, runner_key="000",
                       score_diff=0, is_home=True)
        assert li == 1.0

    def test_li_positive(self):
        """LI should always be positive."""
        li = lookup_li(inning=7, half="BOTTOM", outs=1, runner_key="110",
                       score_diff=-1, is_home=True)
        assert li > 0

    def test_close_game_late_high_leverage(self):
        """Close game in late innings should have higher leverage."""
        li_early = lookup_li(inning=1, half="TOP", outs=0, runner_key="000",
                             score_diff=0, is_home=True)
        li_late = lookup_li(inning=9, half="BOTTOM", outs=2, runner_key="110",
                            score_diff=-1, is_home=True)
        assert li_late > li_early, "Late close game should have higher LI"

    def test_blowout_low_leverage(self):
        """Blowout should have low leverage."""
        li = lookup_li(inning=5, half="TOP", outs=0, runner_key="000",
                       score_diff=8, is_home=True)
        assert li < 1.0, f"Blowout should have low LI, got {li}"


# ---------------------------------------------------------------------------
# 3. runners_key helper
# ---------------------------------------------------------------------------

class TestRunnersKey:
    """Base state key generation."""

    def test_bases_empty(self):
        assert _runners_key(False, False, False) == "000"

    def test_runner_on_first(self):
        assert _runners_key(True, False, False) == "100"

    def test_runner_on_second(self):
        assert _runners_key(False, True, False) == "010"

    def test_runner_on_third(self):
        assert _runners_key(False, False, True) == "001"

    def test_runners_on_first_and_second(self):
        assert _runners_key(True, True, False) == "110"

    def test_bases_loaded(self):
        assert _runners_key(True, True, True) == "111"

    def test_corners(self):
        assert _runners_key(True, False, True) == "101"

    def test_second_and_third(self):
        assert _runners_key(False, True, True) == "011"


# ---------------------------------------------------------------------------
# 4. Decision scoring
# ---------------------------------------------------------------------------

class TestScoreDecision:
    """score_decision() tests."""

    def test_active_decision_positive_wpa(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.45,
            wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.5,
            turn=1,
            inning=7,
            half="BOTTOM",
            outs=1,
            score_diff=-1,
            runners="100",
        )
        assert dwpa.is_active is True
        assert dwpa.wpa == pytest.approx(0.10)
        assert dwpa.wp_before == 0.45
        assert dwpa.wp_after == 0.55
        assert dwpa.decision_type == "PITCHING_CHANGE"
        assert dwpa.leverage_index == 1.5
        assert dwpa.inning == 7
        assert dwpa.half == "BOTTOM"
        assert dwpa.outs == 1
        assert dwpa.score_diff == -1
        assert dwpa.runners == "100"

    def test_no_action_decision_zero_wpa(self, sample_no_action_decision):
        dwpa = score_decision(
            wp_before=0.50,
            wp_after=0.50,
            decision_dict=sample_no_action_decision,
            leverage_index=1.0,
        )
        assert dwpa.is_active is False
        assert dwpa.wpa == 0.0

    def test_negative_wpa(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.60,
            wp_after=0.40,
            decision_dict=sample_active_decision,
            leverage_index=2.0,
        )
        assert dwpa.wpa == pytest.approx(-0.20)

    def test_all_no_action_types_classified_correctly(self):
        for action_type in NO_ACTION_TYPES:
            decision = {"decision": action_type, "action_details": "test"}
            dwpa = score_decision(
                wp_before=0.50, wp_after=0.50,
                decision_dict=decision, leverage_index=1.0,
            )
            assert dwpa.is_active is False, f"{action_type} should be inactive"

    def test_active_types_classified_correctly(self):
        active_types = [
            "PITCHING_CHANGE", "STOLEN_BASE", "PINCH_HIT",
            "INTENTIONAL_WALK", "SACRIFICE_BUNT", "DEFENSIVE_POSITIONING",
        ]
        for action_type in active_types:
            decision = {"decision": action_type, "action_details": "test"}
            dwpa = score_decision(
                wp_before=0.50, wp_after=0.50,
                decision_dict=decision, leverage_index=1.0,
            )
            assert dwpa.is_active is True, f"{action_type} should be active"

    def test_empty_decision_is_inactive(self):
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.50,
            decision_dict={}, leverage_index=1.0,
        )
        assert dwpa.is_active is False

    def test_description_from_action_details(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.0,
        )
        assert dwpa.description == "Bring in Rivera from the pen"

    def test_case_insensitive_decision_type(self):
        decision = {"decision": "pitching_change", "action_details": "test"}
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.55,
            decision_dict=decision, leverage_index=1.0,
        )
        assert dwpa.is_active is True
        assert dwpa.decision_type == "PITCHING_CHANGE"

    def test_whitespace_stripped_from_decision_type(self):
        decision = {"decision": "  NO_ACTION  ", "action_details": "test"}
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.50,
            decision_dict=decision, leverage_index=1.0,
        )
        assert dwpa.is_active is False
        assert dwpa.decision_type == "NO_ACTION"


# ---------------------------------------------------------------------------
# 5. DecisionWPA dataclass
# ---------------------------------------------------------------------------

class TestDecisionWPA:
    """DecisionWPA to_dict() serialization."""

    def test_to_dict_has_all_fields(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.45, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.5,
            turn=3,
            inning=7,
            half="BOTTOM",
            outs=1,
            score_diff=-1,
            runners="100",
        )
        d = dwpa.to_dict()
        assert d["turn"] == 3
        assert d["decision_type"] == "PITCHING_CHANGE"
        assert d["is_active"] is True
        assert d["wp_before"] == pytest.approx(0.45)
        assert d["wp_after"] == pytest.approx(0.55)
        assert d["wpa"] == pytest.approx(0.10)
        assert d["leverage_index"] == pytest.approx(1.5)
        assert d["inning"] == 7
        assert d["half"] == "BOTTOM"
        assert d["outs"] == 1
        assert d["score_diff"] == -1
        assert d["runners"] == "100"
        assert d["description"] == "Bring in Rivera from the pen"

    def test_to_dict_rounds_values(self):
        dwpa = DecisionWPA(
            turn=1, decision_type="TEST", is_active=True,
            wp_before=0.123456789, wp_after=0.234567891,
            wpa=0.111111102, leverage_index=1.23456789,
        )
        d = dwpa.to_dict()
        assert d["wp_before"] == 0.1235
        assert d["wp_after"] == 0.2346
        assert d["wpa"] == 0.1111
        assert d["leverage_index"] == 1.235


# ---------------------------------------------------------------------------
# 6. Game-level WPA report
# ---------------------------------------------------------------------------

class TestGameWPAReport:
    """generate_game_wpa_report() tests."""

    def test_empty_scores_list(self):
        report = generate_game_wpa_report([])
        assert report.total_decisions == 0
        assert report.active_decisions == 0
        assert report.total_wpa == 0.0
        assert report.best_decision is None
        assert report.worst_decision is None

    def test_single_active_decision(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.45, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.5,
            turn=1,
        )
        report = generate_game_wpa_report([dwpa])
        assert report.total_decisions == 1
        assert report.active_decisions == 1
        assert report.total_wpa == pytest.approx(0.10)
        assert report.active_wpa == pytest.approx(0.10)
        assert report.avg_wpa_per_active == pytest.approx(0.10)
        assert report.best_decision is dwpa
        assert report.worst_decision is dwpa
        assert report.positive_wpa_count == 1
        assert report.negative_wpa_count == 0
        assert report.neutral_wpa_count == 0

    def test_single_no_action_decision(self, sample_no_action_decision):
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.50,
            decision_dict=sample_no_action_decision,
            leverage_index=1.0,
        )
        report = generate_game_wpa_report([dwpa])
        assert report.total_decisions == 1
        assert report.active_decisions == 0
        assert report.avg_wpa_per_active == 0.0
        assert report.best_decision is None
        assert report.worst_decision is None

    def test_mixed_decisions(self, sample_active_decision, sample_no_action_decision):
        active = score_decision(
            wp_before=0.45, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=2.0,
            turn=1,
        )
        inactive = score_decision(
            wp_before=0.55, wp_after=0.55,
            decision_dict=sample_no_action_decision,
            leverage_index=0.8,
            turn=2,
        )
        bad_active = score_decision(
            wp_before=0.55, wp_after=0.40,
            decision_dict={"decision": "STOLEN_BASE", "action_details": "steal 2nd"},
            leverage_index=1.8,
            turn=3,
        )
        report = generate_game_wpa_report([active, inactive, bad_active])
        assert report.total_decisions == 3
        assert report.active_decisions == 2
        assert report.total_wpa == pytest.approx(0.10 + 0.0 + (-0.15))
        assert report.active_wpa == pytest.approx(0.10 + (-0.15))
        assert report.best_decision is active
        assert report.worst_decision is bad_active
        assert report.positive_wpa_count == 1
        assert report.negative_wpa_count == 1
        assert report.neutral_wpa_count == 0
        assert report.high_leverage_decisions == 2  # active (2.0) and bad_active (1.8) both >= 1.5

    def test_high_leverage_count(self):
        scores = [
            DecisionWPA(turn=1, decision_type="PITCHING_CHANGE", is_active=True,
                        wp_before=0.50, wp_after=0.55, wpa=0.05, leverage_index=0.5),
            DecisionWPA(turn=2, decision_type="PINCH_HIT", is_active=True,
                        wp_before=0.55, wp_after=0.60, wpa=0.05, leverage_index=1.5),
            DecisionWPA(turn=3, decision_type="STOLEN_BASE", is_active=True,
                        wp_before=0.60, wp_after=0.70, wpa=0.10, leverage_index=2.5),
        ]
        report = generate_game_wpa_report(scores)
        assert report.high_leverage_decisions == 2

    def test_neutral_wpa_threshold(self):
        """WPA within +-0.001 of zero is considered neutral."""
        scores = [
            DecisionWPA(turn=1, decision_type="SHIFT", is_active=True,
                        wp_before=0.50, wp_after=0.5005, wpa=0.0005, leverage_index=1.0),
            DecisionWPA(turn=2, decision_type="SHIFT", is_active=True,
                        wp_before=0.50, wp_after=0.4995, wpa=-0.0005, leverage_index=1.0),
        ]
        report = generate_game_wpa_report(scores)
        assert report.neutral_wpa_count == 2
        assert report.positive_wpa_count == 0
        assert report.negative_wpa_count == 0

    def test_to_dict_serializable(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.45, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.5,
            turn=1,
        )
        report = generate_game_wpa_report([dwpa])
        d = report.to_dict()
        # Verify it's JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["total_decisions"] == 1
        assert parsed["active_decisions"] == 1
        assert len(parsed["decision_scores"]) == 1
        assert parsed["best_decision"]["turn"] == 1

    def test_to_dict_empty_report(self):
        report = GameWPAReport()
        d = report.to_dict()
        assert d["total_decisions"] == 0
        assert d["best_decision"] is None
        assert d["worst_decision"] is None
        assert d["decision_scores"] == []


# ---------------------------------------------------------------------------
# 7. Score game log
# ---------------------------------------------------------------------------

class TestScoreGameLog:
    """score_game_log() tests."""

    def test_score_game_log_produces_report(self, sample_game_log):
        report = score_game_log(sample_game_log, "home")
        assert report.total_decisions == 3
        assert report.active_decisions >= 1
        assert len(report.decision_scores) == 3

    def test_score_game_log_wp_values(self, sample_game_log):
        report = score_game_log(sample_game_log, "home")
        for ds in report.decision_scores:
            assert 0.01 <= ds.wp_before <= 0.99
            # Last decision wp_after can be 0.0 or 1.0 (game outcome)
            assert 0.0 <= ds.wp_after <= 1.0

    def test_score_game_log_final_wp_reflects_winner(self, sample_game_log):
        """Last decision's wp_after should be 1.0 since home (managed) won."""
        report = score_game_log(sample_game_log, "home")
        last = report.decision_scores[-1]
        assert last.wp_after == 1.0, "Managed team won, so final WP should be 1.0"

    def test_score_game_log_losing_team(self, sample_game_log):
        """If managed team lost, final WP should be 0.0."""
        # Modify to make managed team lose
        log = json.loads(json.dumps(sample_game_log))
        log["game_info"]["final_score"] = {"home": 2, "away": 5}
        log["game_info"]["winner"] = "Cubs"
        report = score_game_log(log, "home")
        last = report.decision_scores[-1]
        assert last.wp_after == 0.0

    def test_score_game_log_empty_decisions(self):
        log = {
            "game_info": {"managed_team": "home", "final_score": {"home": 3, "away": 2}},
            "decisions": [],
        }
        report = score_game_log(log, "home")
        assert report.total_decisions == 0

    def test_score_game_log_detects_managed_team_from_log(self, sample_game_log):
        """Should use managed_team from game_info if present."""
        report = score_game_log(sample_game_log)
        assert report.total_decisions == 3

    def test_score_game_log_away_managed(self):
        log = {
            "game_info": {
                "managed_team": "away",
                "home_team": "Cardinals",
                "away_team": "Cubs",
                "final_score": {"home": 3, "away": 5},
                "winner": "Cubs",
            },
            "decisions": [
                {
                    "turn": 1,
                    "game_state": {
                        "inning": 5,
                        "half": "TOP",
                        "outs": 1,
                        "score": {"home": 2, "away": 3},
                        "runners": {},
                    },
                    "decision": {"decision": "PINCH_HIT", "action_details": "Send in PH"},
                    "is_active_decision": True,
                },
            ],
        }
        report = score_game_log(log, "away")
        assert report.total_decisions == 1
        # Away team won, so final WP should be 1.0
        assert report.decision_scores[0].wp_after == 1.0

    def test_score_game_log_with_runners(self, sample_game_log):
        """Decisions with runners should produce valid runner keys."""
        report = score_game_log(sample_game_log, "home")
        # Decision 2 has runner on first
        d2 = report.decision_scores[1]
        assert d2.runners == "100"

    def test_score_game_log_consecutive_wp_delta(self, sample_game_log):
        """Each decision's wp_after should equal the next decision's wp_before."""
        report = score_game_log(sample_game_log, "home")
        scores = report.decision_scores
        for i in range(len(scores) - 1):
            assert scores[i].wp_after == pytest.approx(scores[i + 1].wp_before, abs=0.001), \
                f"Decision {i} wp_after should match decision {i+1} wp_before"


# ---------------------------------------------------------------------------
# 8. Score game log file
# ---------------------------------------------------------------------------

class TestScoreGameLogFile:
    """score_game_log_file() tests."""

    def test_score_game_log_file(self, sample_game_log):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_game_log, f)
            f.flush()
            report = score_game_log_file(f.name)
        assert report.total_decisions == 3
        assert len(report.decision_scores) == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            score_game_log_file("/nonexistent/path/game_99999.json")

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            with pytest.raises(json.JSONDecodeError):
                score_game_log_file(f.name)


# ---------------------------------------------------------------------------
# 9. Format WPA report
# ---------------------------------------------------------------------------

class TestFormatWPAReport:
    """format_wpa_report() text output tests."""

    def test_empty_report(self):
        report = GameWPAReport()
        text = format_wpa_report(report)
        assert "DECISION QUALITY REPORT (WPA)" in text
        assert "Total decisions:        0" in text

    def test_report_with_decisions(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.45, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.5,
            turn=3,
            inning=7,
            half="BOTTOM",
            outs=1,
        )
        report = generate_game_wpa_report([dwpa])
        text = format_wpa_report(report)
        assert "Total decisions:        1" in text
        assert "Active decisions:       1" in text
        assert "+0.1000" in text
        assert "Best decision:" in text
        assert "Turn 3" in text
        assert "PITCHING_CHANGE" in text

    def test_report_includes_worst_decision(self):
        scores = [
            DecisionWPA(turn=1, decision_type="STOLEN_BASE", is_active=True,
                        wp_before=0.60, wp_after=0.40, wpa=-0.20,
                        leverage_index=2.0, inning=8, half="TOP", outs=0),
        ]
        report = generate_game_wpa_report(scores)
        text = format_wpa_report(report)
        assert "Worst decision:" in text
        assert "-0.2000" in text

    def test_report_is_multiline_string(self, sample_active_decision):
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.55,
            decision_dict=sample_active_decision,
            leverage_index=1.0, turn=1,
        )
        report = generate_game_wpa_report([dwpa])
        text = format_wpa_report(report)
        lines = text.strip().split("\n")
        assert len(lines) >= 10, "Report should have multiple lines"


# ---------------------------------------------------------------------------
# 10. compute_wp_from_game_state / compute_li_from_game_state
# ---------------------------------------------------------------------------

class TestComputeFromGameState:
    """Tests using simulation GameState objects."""

    @pytest.fixture
    def game_state(self):
        """Create a GameState-like object using the simulation engine."""
        from simulation import SimulationEngine, load_rosters
        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        return engine.initialize_game(rosters)

    def test_compute_wp_at_game_start(self, game_state):
        wp = compute_wp_from_game_state(game_state, "home")
        assert 0.40 < wp < 0.65, f"Home WP at game start should be near 0.55, got {wp}"

    def test_compute_wp_away_at_game_start(self, game_state):
        wp = compute_wp_from_game_state(game_state, "away")
        assert 0.35 < wp < 0.60, f"Away WP at game start should be near 0.45, got {wp}"

    def test_compute_wp_home_away_complement(self, game_state):
        wp_home = compute_wp_from_game_state(game_state, "home")
        wp_away = compute_wp_from_game_state(game_state, "away")
        assert abs(wp_home + wp_away - 1.0) < 0.02

    def test_compute_li_at_game_start(self, game_state):
        li = compute_li_from_game_state(game_state, "home")
        assert li > 0, "LI should be positive"

    def test_wp_changes_with_score(self, game_state):
        wp_before = compute_wp_from_game_state(game_state, "home")
        game_state.score_home = 5
        game_state.score_away = 0
        wp_after = compute_wp_from_game_state(game_state, "home")
        assert wp_after > wp_before, "Leading team should have higher WP"


# ---------------------------------------------------------------------------
# 11. Integration: decision log entry WPA fields
# ---------------------------------------------------------------------------

class TestDecisionLogWPAFields:
    """Verify build_decision_log_entry includes WPA fields."""

    def test_log_entry_has_wpa_fields(self):
        from simulation import SimulationEngine, load_rosters
        from game import build_decision_log_entry

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)

        decision_dict = {
            "decision": "PITCHING_CHANGE",
            "action_details": "Bring in closer",
            "confidence": 0.9,
            "reasoning": "Starter fatigued",
            "key_factors": ["velocity drop"],
            "risks": ["bullpen usage"],
        }
        metadata = {
            "tool_calls": [],
            "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "latency_ms": 500,
            "agent_turns": 2,
            "retries": 0,
            "wp_before": 0.55,
            "leverage_index": 1.8,
        }
        entry = build_decision_log_entry(
            turn=1,
            game_state=game,
            managed_team="home",
            decision_dict=decision_dict,
            decision_metadata=metadata,
            timestamp=1000000.0,
        )

        assert "wp_before" in entry
        assert entry["wp_before"] == 0.55
        assert "leverage_index" in entry
        assert entry["leverage_index"] == 1.8
        assert "wp_after" in entry
        assert entry["wp_after"] is None  # filled in later
        assert "wpa" in entry
        assert entry["wpa"] is None  # filled in later

    def test_log_entry_without_wpa_metadata(self):
        """When metadata doesn't include WPA fields, they should be None."""
        from simulation import SimulationEngine, load_rosters
        from game import build_decision_log_entry

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)

        decision_dict = {"decision": "NO_ACTION", "action_details": "test"}
        metadata = {
            "tool_calls": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        entry = build_decision_log_entry(
            turn=1, game_state=game, managed_team="home",
            decision_dict=decision_dict, decision_metadata=metadata,
            timestamp=1000000.0,
        )
        assert entry["wp_before"] is None
        assert entry["leverage_index"] is None
        assert entry["wp_after"] is None
        assert entry["wpa"] is None


# ---------------------------------------------------------------------------
# 12. Integration: write_game_log includes WPA report
# ---------------------------------------------------------------------------

class TestWriteGameLogWPA:
    """Verify write_game_log includes WPA report when provided."""

    def test_game_log_includes_wpa_report(self):
        from simulation import SimulationEngine, load_rosters
        from game import write_game_log

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)
        # Simulate game ending
        game.game_over = True
        game.winning_team = game.home.name
        game.score_home = 5
        game.score_away = 3

        wpa_report = GameWPAReport(
            total_decisions=3,
            active_decisions=1,
            total_wpa=0.05,
            active_wpa=0.05,
            avg_wpa_per_active=0.05,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_game_log(
                game_state=game,
                decision_log=[],
                error_log=[],
                seed=42,
                managed_team="home",
                log_dir=Path(tmpdir),
                wpa_report=wpa_report,
            )
            with open(log_path) as f:
                log_data = json.load(f)

        assert "wpa_report" in log_data
        assert log_data["wpa_report"]["total_decisions"] == 3
        assert log_data["wpa_report"]["active_decisions"] == 1
        assert log_data["wpa_report"]["total_wpa"] == 0.05

    def test_game_log_without_wpa_report(self):
        from simulation import SimulationEngine, load_rosters
        from game import write_game_log

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)
        game.game_over = True
        game.winning_team = game.home.name

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_game_log(
                game_state=game,
                decision_log=[],
                error_log=[],
                seed=42,
                managed_team="home",
                log_dir=Path(tmpdir),
            )
            with open(log_path) as f:
                log_data = json.load(f)

        assert "wpa_report" not in log_data


# ---------------------------------------------------------------------------
# 13. Integration: _update_last_decision_wp_after
# ---------------------------------------------------------------------------

class TestUpdateLastDecisionWPAfter:
    """Verify _update_last_decision_wp_after fills in wp_after and wpa."""

    def test_updates_wp_after_during_game(self):
        from simulation import SimulationEngine, load_rosters
        from game import _update_last_decision_wp_after

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)

        decision_log = [
            {"wp_before": 0.55, "wp_after": None, "wpa": None}
        ]

        _update_last_decision_wp_after(decision_log, game, "home")

        assert decision_log[0]["wp_after"] is not None
        assert decision_log[0]["wpa"] is not None
        assert isinstance(decision_log[0]["wp_after"], float)
        assert isinstance(decision_log[0]["wpa"], float)

    def test_updates_wp_after_game_over_win(self):
        from simulation import SimulationEngine, load_rosters
        from game import _update_last_decision_wp_after

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)
        game.game_over = True
        game.score_home = 5
        game.score_away = 3

        decision_log = [
            {"wp_before": 0.80, "wp_after": None, "wpa": None}
        ]

        _update_last_decision_wp_after(decision_log, game, "home")

        assert decision_log[0]["wp_after"] == 1.0
        assert decision_log[0]["wpa"] == pytest.approx(0.20)

    def test_updates_wp_after_game_over_loss(self):
        from simulation import SimulationEngine, load_rosters
        from game import _update_last_decision_wp_after

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)
        game.game_over = True
        game.score_home = 2
        game.score_away = 5

        decision_log = [
            {"wp_before": 0.30, "wp_after": None, "wpa": None}
        ]

        _update_last_decision_wp_after(decision_log, game, "home")

        assert decision_log[0]["wp_after"] == 0.0
        assert decision_log[0]["wpa"] == pytest.approx(-0.30)

    def test_does_not_overwrite_existing_wp_after(self):
        from simulation import SimulationEngine, load_rosters
        from game import _update_last_decision_wp_after

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)

        decision_log = [
            {"wp_before": 0.55, "wp_after": 0.60, "wpa": 0.05}
        ]

        _update_last_decision_wp_after(decision_log, game, "home")

        # Should not have been changed
        assert decision_log[0]["wp_after"] == 0.60
        assert decision_log[0]["wpa"] == 0.05

    def test_handles_empty_log(self):
        from simulation import SimulationEngine, load_rosters
        from game import _update_last_decision_wp_after

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)

        decision_log = []
        # Should not raise
        _update_last_decision_wp_after(decision_log, game, "home")

    def test_handles_missing_wp_before(self):
        from simulation import SimulationEngine, load_rosters
        from game import _update_last_decision_wp_after

        rosters = load_rosters()
        engine = SimulationEngine(seed=42)
        game = engine.initialize_game(rosters)

        decision_log = [
            {"wp_before": None, "wp_after": None, "wpa": None}
        ]

        _update_last_decision_wp_after(decision_log, game, "home")

        assert decision_log[0]["wp_after"] is not None
        assert decision_log[0]["wpa"] is None  # can't compute WPA without wp_before


# ---------------------------------------------------------------------------
# 14. NO_ACTION_TYPES consistency
# ---------------------------------------------------------------------------

class TestNoActionTypes:
    """The NO_ACTION_TYPES set should be consistent between modules."""

    def test_no_action_types_matches_game_module(self):
        from game import NO_ACTION_TYPES as game_no_actions
        from decision_quality_wpa import NO_ACTION_TYPES as wpa_no_actions
        assert game_no_actions == wpa_no_actions

    def test_no_action_types_is_frozenset(self):
        assert isinstance(NO_ACTION_TYPES, frozenset)

    def test_known_no_action_types(self):
        expected = {"NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
                    "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER"}
        assert NO_ACTION_TYPES == expected


# ---------------------------------------------------------------------------
# 15. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_wp_with_all_base_states(self):
        """All 8 base states should return valid WP values."""
        base_states = ["000", "100", "010", "001", "110", "101", "011", "111"]
        for state in base_states:
            wp = lookup_wp(inning=5, half="TOP", outs=1, runner_key=state,
                           score_diff=0, is_home=True)
            assert 0.01 <= wp <= 0.99, f"Invalid WP for state {state}: {wp}"

    def test_wp_with_all_out_counts(self):
        """All 3 out counts should return valid WP values."""
        for outs in (0, 1, 2):
            wp = lookup_wp(inning=5, half="TOP", outs=outs, runner_key="000",
                           score_diff=0, is_home=True)
            assert 0.01 <= wp <= 0.99, f"Invalid WP for outs={outs}: {wp}"

    def test_wp_with_all_half_innings(self):
        """Both TOP and BOTTOM should return valid WP values."""
        for half in ("TOP", "BOTTOM"):
            wp = lookup_wp(inning=5, half=half, outs=1, runner_key="000",
                           score_diff=0, is_home=True)
            assert 0.01 <= wp <= 0.99, f"Invalid WP for half={half}: {wp}"

    def test_score_decision_with_extreme_wp_values(self):
        """Decisions at extreme WP boundaries."""
        dwpa = score_decision(
            wp_before=0.01, wp_after=0.99,
            decision_dict={"decision": "PINCH_HIT", "action_details": "miracle"},
            leverage_index=10.0,
        )
        assert dwpa.wpa == pytest.approx(0.98)

    def test_game_log_single_decision(self):
        """Game log with a single decision should still produce a report."""
        log = {
            "game_info": {
                "managed_team": "home",
                "final_score": {"home": 1, "away": 0},
                "winner": "home",
            },
            "decisions": [
                {
                    "turn": 1,
                    "game_state": {
                        "inning": 9,
                        "half": "BOTTOM",
                        "outs": 2,
                        "score": {"home": 0, "away": 0},
                        "runners": {},
                    },
                    "decision": {"decision": "PINCH_HIT", "action_details": "Walk-off setup"},
                    "is_active_decision": True,
                },
            ],
        }
        report = score_game_log(log, "home")
        assert report.total_decisions == 1
        # Single decision: wp_after = final WP = 1.0 (home won)
        assert report.decision_scores[0].wp_after == 1.0

    def test_wpa_symmetry_for_zero_impact(self):
        """A decision that doesn't change WP should have WPA of 0."""
        dwpa = score_decision(
            wp_before=0.50, wp_after=0.50,
            decision_dict={"decision": "DEFENSIVE_POSITIONING", "action_details": "shift"},
            leverage_index=1.0,
        )
        assert dwpa.wpa == 0.0
        assert dwpa.is_active is True
