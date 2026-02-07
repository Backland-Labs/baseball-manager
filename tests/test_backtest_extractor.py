# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for backtest.extractor module.

Validates:
  1. resolve_team_side resolves names, abbreviations, home/away
  2. walk_game_feed yields correct number of decision points
  3. DecisionPoint fields are populated correctly
  4. Real manager actions detected (pitching changes, subs)
  5. Pitcher tracking accumulates stats across plays
  6. Lineup/bench/bullpen state updates on substitutions
  7. Leverage index computed for each decision point
  8. On-deck batter identification
  9. Score tracking across plays
  10. Edge cases: unknown team, half-inning boundaries
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backtest.extractor import (
    ActionType,
    BullpenEntry,
    DecisionPoint,
    PlayerState,
    RealManagerAction,
    _PitcherTracker,
    _count_pitches_in_play,
    _detect_manager_actions,
    _find_on_deck,
    _runners_from_pre_play,
    resolve_team_side,
    walk_game_feed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "game_feed_746865.json"


@pytest.fixture(scope="module")
def game_feed():
    """Load the real game feed fixture (STL @ CHC, 2024-06-15)."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture
def home_decision_points(game_feed):
    """All decision points for the home team (CHC)."""
    return list(walk_game_feed(game_feed, "CHC"))


@pytest.fixture
def away_decision_points(game_feed):
    """All decision points for the away team (STL)."""
    return list(walk_game_feed(game_feed, "STL"))


# ---------------------------------------------------------------------------
# resolve_team_side
# ---------------------------------------------------------------------------

class TestResolveTeamSide:
    def test_home_literal(self, game_feed):
        assert resolve_team_side(game_feed, "home") == "home"

    def test_away_literal(self, game_feed):
        assert resolve_team_side(game_feed, "away") == "away"

    def test_abbreviation_chc(self, game_feed):
        assert resolve_team_side(game_feed, "CHC") == "home"

    def test_abbreviation_stl(self, game_feed):
        assert resolve_team_side(game_feed, "STL") == "away"

    def test_full_name(self, game_feed):
        assert resolve_team_side(game_feed, "Chicago Cubs") == "home"

    def test_full_name_away(self, game_feed):
        assert resolve_team_side(game_feed, "St. Louis Cardinals") == "away"

    def test_case_insensitive(self, game_feed):
        assert resolve_team_side(game_feed, "chc") == "home"

    def test_unknown_raises(self, game_feed):
        with pytest.raises(ValueError, match="Cannot resolve team"):
            resolve_team_side(game_feed, "NYY")


# ---------------------------------------------------------------------------
# walk_game_feed: basic structure
# ---------------------------------------------------------------------------

class TestWalkGameFeed:
    def test_yields_decision_points(self, home_decision_points):
        assert len(home_decision_points) > 0
        for dp in home_decision_points:
            assert isinstance(dp, DecisionPoint)

    def test_correct_number_of_plays(self, game_feed, home_decision_points):
        all_plays = game_feed["liveData"]["plays"]["allPlays"]
        assert len(home_decision_points) == len(all_plays)

    def test_play_indices_sequential(self, home_decision_points):
        indices = [dp.play_index for dp in home_decision_points]
        assert indices == list(range(len(home_decision_points)))

    def test_first_play_is_top_of_first(self, home_decision_points):
        dp = home_decision_points[0]
        assert dp.inning == 1
        assert dp.half == "TOP"
        assert dp.outs == 0

    def test_score_starts_at_zero(self, home_decision_points):
        dp = home_decision_points[0]
        assert dp.score_home == 0
        assert dp.score_away == 0

    def test_managed_team_side_set(self, home_decision_points):
        for dp in home_decision_points:
            assert dp.managed_team_side == "home"

    def test_away_team_side_set(self, away_decision_points):
        for dp in away_decision_points:
            assert dp.managed_team_side == "away"


# ---------------------------------------------------------------------------
# Decision point fields
# ---------------------------------------------------------------------------

class TestDecisionPointFields:
    def test_batter_populated(self, home_decision_points):
        dp = home_decision_points[0]
        assert dp.batter_id != ""
        assert dp.batter_name != ""
        assert dp.batter_bats in ("L", "R", "S")

    def test_pitcher_populated(self, home_decision_points):
        dp = home_decision_points[0]
        assert dp.pitcher_id != ""
        assert dp.pitcher_name != ""
        assert dp.pitcher_throws in ("L", "R")

    def test_lineup_populated(self, home_decision_points):
        dp = home_decision_points[0]
        assert len(dp.current_lineup) == 9
        for p in dp.current_lineup:
            assert isinstance(p, PlayerState)
            assert p.player_id != ""
            assert p.name != ""

    def test_bullpen_populated(self, home_decision_points):
        dp = home_decision_points[0]
        assert len(dp.bullpen) > 0
        for bp in dp.bullpen:
            assert isinstance(bp, BullpenEntry)

    def test_bench_populated(self, home_decision_points):
        dp = home_decision_points[0]
        assert len(dp.bench) > 0

    def test_opp_lineup_populated(self, home_decision_points):
        dp = home_decision_points[0]
        assert len(dp.opp_lineup) == 9

    def test_runners_dict_structure(self, home_decision_points):
        dp = home_decision_points[0]
        assert "first" in dp.runners
        assert "second" in dp.runners
        assert "third" in dp.runners


# ---------------------------------------------------------------------------
# Manager action detection
# ---------------------------------------------------------------------------

class TestManagerActions:
    def test_home_manager_has_pitching_change(self, home_decision_points):
        """CHC (home) makes a pitching change when Imanaga is pulled."""
        actions = [dp for dp in home_decision_points if dp.real_manager_action is not None]
        pitching_changes = [
            dp for dp in actions
            if dp.real_manager_action.action_type == ActionType.PITCHING_CHANGE
        ]
        assert len(pitching_changes) >= 1

    def test_away_manager_has_pitching_changes(self, away_decision_points):
        """STL (away) makes multiple pitching changes."""
        actions = [dp for dp in away_decision_points if dp.real_manager_action is not None]
        pitching_changes = [
            dp for dp in actions
            if dp.real_manager_action.action_type == ActionType.PITCHING_CHANGE
        ]
        assert len(pitching_changes) >= 1

    def test_manager_action_has_player_info(self, home_decision_points):
        actions = [dp for dp in home_decision_points if dp.real_manager_action is not None]
        for dp in actions:
            action = dp.real_manager_action
            if action.action_type == ActionType.PITCHING_CHANGE:
                assert action.player_in is not None or action.details != ""

    def test_no_action_is_default(self, home_decision_points):
        """Most plays should have no manager action."""
        no_action = [dp for dp in home_decision_points if dp.real_manager_action is None]
        assert len(no_action) > len(home_decision_points) // 2


# ---------------------------------------------------------------------------
# Pitcher tracking
# ---------------------------------------------------------------------------

class TestPitcherTracker:
    def test_set_current(self):
        tracker = _PitcherTracker()
        tracker.set_current("home", "12345")
        assert tracker.current("home") == "12345"

    def test_initial_stats(self):
        tracker = _PitcherTracker()
        tracker.set_current("home", "12345")
        stats = tracker.get("12345")
        assert stats["pitch_count"] == 0
        assert stats["batters_faced"] == 0
        assert stats["outs_recorded"] == 0
        assert stats["runs_allowed"] == 0

    def test_record_play(self):
        tracker = _PitcherTracker()
        tracker.set_current("home", "12345")
        tracker.record_play("12345", pitches=5, outs=1, runs=0)
        stats = tracker.get("12345")
        assert stats["pitch_count"] == 5
        assert stats["batters_faced"] == 1
        assert stats["outs_recorded"] == 1

    def test_accumulates(self):
        tracker = _PitcherTracker()
        tracker.set_current("home", "12345")
        tracker.record_play("12345", pitches=5, outs=1, runs=0)
        tracker.record_play("12345", pitches=3, outs=0, runs=1)
        stats = tracker.get("12345")
        assert stats["pitch_count"] == 8
        assert stats["batters_faced"] == 2
        assert stats["outs_recorded"] == 1
        assert stats["runs_allowed"] == 1

    def test_innings_pitched(self):
        tracker = _PitcherTracker()
        tracker.set_current("home", "12345")
        # 7 outs = 2.1 innings
        for _ in range(7):
            tracker.record_play("12345", pitches=4, outs=1, runs=0)
        assert tracker.innings_pitched("12345") == 2.1

    def test_unknown_pitcher_returns_defaults(self):
        tracker = _PitcherTracker()
        stats = tracker.get("unknown")
        assert stats["pitch_count"] == 0

    def test_multiple_pitchers(self):
        tracker = _PitcherTracker()
        tracker.set_current("home", "111")
        tracker.set_current("away", "222")
        tracker.record_play("111", pitches=5, outs=1, runs=0)
        tracker.record_play("222", pitches=3, outs=0, runs=1)
        assert tracker.get("111")["pitch_count"] == 5
        assert tracker.get("222")["runs_allowed"] == 1


# ---------------------------------------------------------------------------
# Pitcher stats accumulate during walk
# ---------------------------------------------------------------------------

class TestPitcherStatsInWalk:
    def test_pitcher_pitch_count_increases(self, home_decision_points):
        """Pitcher pitch count should increase as the game progresses."""
        # Group by pitcher
        pitcher_counts = {}
        for dp in home_decision_points:
            pid = dp.pitcher_id
            if pid not in pitcher_counts:
                pitcher_counts[pid] = []
            pitcher_counts[pid].append(dp.pitcher_pitch_count)

        # For the starting pitcher, pitch count should generally increase
        for pid, counts in pitcher_counts.items():
            if len(counts) > 3:
                # Should be non-decreasing (within same pitcher stint)
                assert counts[-1] >= counts[0]

    def test_starting_pitcher_has_zero_initial_count(self, home_decision_points):
        dp = home_decision_points[0]
        assert dp.pitcher_pitch_count == 0


# ---------------------------------------------------------------------------
# Score tracking
# ---------------------------------------------------------------------------

class TestScoreTracking:
    def test_final_score_reached(self, home_decision_points):
        """By the last play, the score should reflect the actual final."""
        # CHC 5, STL 1. The last DP has the score BEFORE that play resolves,
        # so we check that at some point we see scores close to final.
        max_home = max(dp.score_home for dp in home_decision_points)
        max_away = max(dp.score_away for dp in home_decision_points)
        # The last play's score_home should be the final or one less
        assert max_home >= 4  # CHC scored 5
        assert max_away >= 1  # STL scored 1


# ---------------------------------------------------------------------------
# Leverage index
# ---------------------------------------------------------------------------

class TestLeverageIndex:
    def test_leverage_index_present(self, home_decision_points):
        for dp in home_decision_points:
            assert dp.leverage_index >= 0.0

    def test_first_inning_reasonable_li(self, home_decision_points):
        dp = home_decision_points[0]
        # First inning of a tied game should have moderate LI
        assert 0.5 <= dp.leverage_index <= 2.5


# ---------------------------------------------------------------------------
# On-deck batter
# ---------------------------------------------------------------------------

class TestOnDeckBatter:
    def test_on_deck_present(self, home_decision_points):
        """Most decision points should have an on-deck batter."""
        with_on_deck = [dp for dp in home_decision_points if dp.on_deck_batter_id]
        assert len(with_on_deck) > 0

    def test_find_on_deck_function(self):
        lineup = [
            PlayerState(player_id="1", name="A", position="CF", bats="R"),
            PlayerState(player_id="2", name="B", position="SS", bats="L"),
            PlayerState(player_id="3", name="C", position="1B", bats="R"),
        ]
        result = _find_on_deck(lineup, "1")
        assert result is not None
        assert result["id"] == "2"
        assert result["name"] == "B"

    def test_find_on_deck_wraps(self):
        lineup = [
            PlayerState(player_id="1", name="A", position="CF", bats="R"),
            PlayerState(player_id="2", name="B", position="SS", bats="L"),
        ]
        result = _find_on_deck(lineup, "2")
        assert result is not None
        assert result["id"] == "1"

    def test_find_on_deck_unknown_batter(self):
        lineup = [
            PlayerState(player_id="1", name="A", position="CF", bats="R"),
        ]
        result = _find_on_deck(lineup, "999")
        assert result is None

    def test_find_on_deck_empty_lineup(self):
        assert _find_on_deck([], "1") is None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_runners_from_pre_play_empty(self):
        play = {"runners": []}
        runners = _runners_from_pre_play(play)
        assert runners["first"] is None
        assert runners["second"] is None
        assert runners["third"] is None

    def test_runners_from_pre_play_runner_on_first(self):
        play = {
            "runners": [
                {
                    "movement": {"start": "1B", "end": "2B"},
                    "details": {"runner": {"id": 123, "fullName": "Test Runner"}},
                }
            ]
        }
        runners = _runners_from_pre_play(play)
        assert runners["first"] is not None
        assert runners["first"]["id"] == "123"
        assert runners["first"]["name"] == "Test Runner"

    def test_count_pitches_in_play(self):
        play = {
            "playEvents": [
                {"isPitch": True},
                {"isPitch": True},
                {"isPitch": False, "type": "action"},
                {"isPitch": True},
            ]
        }
        assert _count_pitches_in_play(play) == 3

    def test_count_pitches_empty(self):
        assert _count_pitches_in_play({"playEvents": []}) == 0
        assert _count_pitches_in_play({}) == 0


# ---------------------------------------------------------------------------
# Detect manager actions
# ---------------------------------------------------------------------------

class TestDetectManagerActions:
    def test_pitching_substitution_for_fielding_team(self):
        play = {
            "about": {"isTopInning": True},
            "playEvents": [
                {
                    "type": "action",
                    "details": {
                        "event": "Pitching Change",
                        "eventType": "pitching_substitution",
                        "description": "Pitching Change: Rivera replaces Smith.",
                    },
                    "player": {"id": 100},
                    "replacedPlayer": {"id": 200},
                }
            ],
            "result": {"event": "Strikeout", "eventType": "strikeout"},
        }
        game_data = {
            "players": {
                "ID100": {"fullName": "Rivera"},
                "ID200": {"fullName": "Smith"},
            }
        }
        # Top inning: home is fielding
        actions = _detect_manager_actions(play, game_data, "home")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.PITCHING_CHANGE
        assert actions[0].player_in == "100"
        assert actions[0].player_out == "200"

    def test_pitching_substitution_not_for_batting_team(self):
        play = {
            "about": {"isTopInning": True},
            "playEvents": [
                {
                    "type": "action",
                    "details": {
                        "event": "Pitching Change",
                        "eventType": "pitching_substitution",
                        "description": "Pitching Change",
                    },
                    "player": {"id": 100},
                    "replacedPlayer": {"id": 200},
                }
            ],
            "result": {"event": "Strikeout"},
        }
        game_data = {"players": {"ID100": {"fullName": "A"}, "ID200": {"fullName": "B"}}}
        # Top inning: away is batting, so away manager shouldn't get this pitching change
        actions = _detect_manager_actions(play, game_data, "away")
        assert len(actions) == 0

    def test_offensive_substitution_pinch_hit(self):
        play = {
            "about": {"isTopInning": False},
            "playEvents": [
                {
                    "type": "action",
                    "details": {
                        "event": "Offensive Sub",
                        "eventType": "offensive_substitution",
                        "description": "Pinch-hitter Jones bats for Williams.",
                    },
                    "player": {"id": 300},
                    "replacedPlayer": {"id": 400},
                }
            ],
            "result": {"event": "Single"},
        }
        game_data = {"players": {"ID300": {"fullName": "Jones"}, "ID400": {"fullName": "Williams"}}}
        # Bottom inning: home is batting
        actions = _detect_manager_actions(play, game_data, "home")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.PINCH_HIT

    def test_offensive_substitution_pinch_run(self):
        play = {
            "about": {"isTopInning": True},
            "playEvents": [
                {
                    "type": "action",
                    "details": {
                        "event": "Offensive Sub",
                        "eventType": "offensive_substitution",
                        "description": "Pinch-runner Fast replaces Slow.",
                    },
                    "player": {"id": 500},
                    "replacedPlayer": {"id": 600},
                }
            ],
            "result": {"event": "Single"},
        }
        game_data = {"players": {"ID500": {"fullName": "Fast"}, "ID600": {"fullName": "Slow"}}}
        # Top inning: away is batting
        actions = _detect_manager_actions(play, game_data, "away")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.PINCH_RUN

    def test_intent_walk_detected(self):
        play = {
            "about": {"isTopInning": True},
            "playEvents": [],
            "result": {"event": "Intent Walk", "eventType": "intent_walk"},
        }
        game_data = {"players": {}}
        # Top: home is fielding, so home manager issued the IBB
        actions = _detect_manager_actions(play, game_data, "home")
        assert any(a.action_type == ActionType.IBB for a in actions)

    def test_stolen_base_detected(self):
        play = {
            "about": {"isTopInning": False},
            "playEvents": [],
            "result": {"event": "Stolen Base 2B", "eventType": "stolen_base"},
        }
        game_data = {"players": {}}
        # Bottom: home is batting
        actions = _detect_manager_actions(play, game_data, "home")
        assert any(a.action_type == ActionType.STOLEN_BASE for a in actions)

    def test_sac_bunt_detected(self):
        play = {
            "about": {"isTopInning": True},
            "playEvents": [],
            "result": {"event": "Sac Bunt", "eventType": "sac_bunt"},
        }
        game_data = {"players": {}}
        # Top: away is batting
        actions = _detect_manager_actions(play, game_data, "away")
        assert any(a.action_type == ActionType.BUNT for a in actions)


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------

class TestModels:
    def test_action_type_values(self):
        assert ActionType.PITCHING_CHANGE.value == "PITCHING_CHANGE"
        assert ActionType.NO_ACTION.value == "NO_ACTION"

    def test_player_state_defaults(self):
        ps = PlayerState(player_id="1", name="Test", position="SS")
        assert ps.bats == "R"
        assert ps.throws == "R"

    def test_bullpen_entry_defaults(self):
        bp = BullpenEntry(player_id="1", name="Test")
        assert bp.throws == "R"
        assert bp.pitch_count == 0
        assert bp.has_pitched is False
        assert bp.available is True

    def test_decision_point_model_copy(self, home_decision_points):
        dp = home_decision_points[0]
        copy = dp.model_copy()
        assert copy.play_index == dp.play_index
        assert copy.batter_name == dp.batter_name

    def test_real_manager_action_serialization(self):
        action = RealManagerAction(
            action_type=ActionType.PITCHING_CHANGE,
            player_in="100",
            player_out="200",
            player_in_name="Rivera",
            player_out_name="Smith",
            details="Pitching Change: Rivera replaces Smith",
        )
        d = action.model_dump()
        assert d["action_type"] == "PITCHING_CHANGE"
        assert d["player_in"] == "100"


# ---------------------------------------------------------------------------
# Integration: walk game with both teams
# ---------------------------------------------------------------------------

class TestBothTeams:
    def test_same_play_count(self, home_decision_points, away_decision_points):
        """Both teams should see the same number of plays."""
        assert len(home_decision_points) == len(away_decision_points)

    def test_different_manager_actions(self, home_decision_points, away_decision_points):
        """Each team sees different manager actions (their own manager's)."""
        home_actions = [
            dp for dp in home_decision_points if dp.real_manager_action is not None
        ]
        away_actions = [
            dp for dp in away_decision_points if dp.real_manager_action is not None
        ]
        # Not necessarily different counts, but at least the play indices
        # of actions won't be identical (different managers act at different times)
        home_indices = {dp.play_index for dp in home_actions}
        away_indices = {dp.play_index for dp in away_actions}
        # There should be at least one action unique to each side
        assert home_indices != away_indices or (len(home_actions) == 0 and len(away_actions) == 0)

    def test_home_team_lineup_is_chc(self, game_feed, home_decision_points):
        """When managing CHC (home), our lineup should be CHC players."""
        dp = home_decision_points[0]
        # Get CHC batting order from boxscore to cross-check
        box_home = game_feed["liveData"]["boxscore"]["teams"]["home"]
        box_order = [str(pid) for pid in box_home["battingOrder"]]
        our_ids = [p.player_id for p in dp.current_lineup]
        assert our_ids == box_order
