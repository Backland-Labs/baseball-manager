# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the decision_logging feature.

Validates that every agent invocation is logged with full context:
  1. Game state at time of decision (inning, score, outs, runners, batter/pitcher)
  2. Tool calls the agent made (name + parameters)
  3. The agent's full decision response
  4. Whether the decision was active or no-action
  5. Token usage and latency
  6. Structured JSON files organized by game
  7. Timestamps for correlating with live game timeline
"""

import json
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from simulation import SimulationEngine, GameState, load_rosters, game_state_to_scenario
from game import (
    build_decision_log_entry,
    write_game_log,
    NO_ACTION_TYPES,
    _peek_validate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rosters():
    return load_rosters()


@pytest.fixture
def engine():
    return SimulationEngine(seed=42)


@pytest.fixture
def game_state(rosters, engine):
    return engine.initialize_game(rosters)


@pytest.fixture
def advanced_game_state(rosters, engine):
    """A game state partway through the game with runners on base."""
    game = engine.initialize_game(rosters)
    # Simulate a few PAs to get a more interesting state
    for _ in range(15):
        if game.game_over:
            break
        pa_result = engine.simulate_plate_appearance(game)
        engine.apply_pa_result(game, pa_result)
    return game


@pytest.fixture
def sample_decision_dict():
    return {
        "decision": "PITCHING_CHANGE",
        "action_details": "Bring in Greg Foster to face the lefty",
        "confidence": 0.85,
        "reasoning": "Starter has faced the order 3 times, velocity down 2mph",
        "win_probability_before": 0.52,
        "win_probability_after_expected": 0.57,
        "key_factors": ["TTO penalty", "velocity decline", "platoon advantage"],
        "alternatives_considered": [],
        "risks": ["Closer unavailable for 9th"],
    }


@pytest.fixture
def sample_no_action_dict():
    return {
        "decision": "NO_ACTION",
        "action_details": "Let the batter hit, low leverage situation",
        "confidence": 0.95,
        "reasoning": "Early in the game, no compelling reason to intervene",
        "key_factors": ["low leverage"],
        "risks": [],
    }


@pytest.fixture
def sample_metadata():
    return {
        "tool_calls": [
            {"tool_name": "get_pitcher_fatigue_assessment", "tool_input": {"pitcher_id": "a_sp1"}},
            {"tool_name": "get_bullpen_status", "tool_input": {}},
            {"tool_name": "get_win_probability", "tool_input": {"inning": 7, "half": "BOTTOM", "outs": 1}},
        ],
        "token_usage": {
            "input_tokens": 3500,
            "output_tokens": 850,
            "total_tokens": 4350,
        },
        "latency_ms": 2340,
        "agent_turns": 3,
        "retries": 0,
    }


@pytest.fixture
def empty_metadata():
    return {
        "tool_calls": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "latency_ms": 0,
        "agent_turns": 0,
        "retries": 0,
    }


# ---------------------------------------------------------------------------
# Step 1: Game state at time of decision
# ---------------------------------------------------------------------------

class TestLogEntryGameState:
    """Each log entry includes the game state at the time of the decision."""

    def test_entry_has_game_state(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "game_state" in entry

    def test_game_state_has_inning(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "inning" in entry["game_state"]
        assert entry["game_state"]["inning"] == game_state.inning

    def test_game_state_has_score(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "score" in entry["game_state"]
        assert entry["game_state"]["score"]["home"] == game_state.score_home
        assert entry["game_state"]["score"]["away"] == game_state.score_away

    def test_game_state_has_outs(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "outs" in entry["game_state"]
        assert entry["game_state"]["outs"] == game_state.outs

    def test_game_state_has_half(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "half" in entry["game_state"]
        assert entry["game_state"]["half"] == game_state.half

    def test_game_state_has_batter(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        batter = entry["game_state"]["batter"]
        assert "player_id" in batter
        assert "name" in batter
        assert batter["player_id"] == game_state.batting_team().current_batter().player_id

    def test_game_state_has_pitcher(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        pitcher = entry["game_state"]["pitcher"]
        assert "player_id" in pitcher
        assert "name" in pitcher
        assert pitcher["player_id"] == game_state.current_pitcher().player_id

    def test_game_state_has_runners(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "runners" in entry["game_state"]
        assert isinstance(entry["game_state"]["runners"], dict)

    def test_game_state_has_situation_display(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "situation" in entry["game_state"]
        assert isinstance(entry["game_state"]["situation"], str)
        assert len(entry["game_state"]["situation"]) > 0

    def test_runners_captured_when_present(self, advanced_game_state, sample_decision_dict, sample_metadata):
        """When runners are on base, their info should be in the log entry."""
        # Advance the state until there are runners (or use whatever state we got)
        entry = build_decision_log_entry(1, advanced_game_state, "home", sample_decision_dict, sample_metadata, time.time())
        runners = entry["game_state"]["runners"]
        # runners dict has string keys for bases that have runners
        for base_key, runner_info in runners.items():
            assert "player_id" in runner_info
            assert "name" in runner_info


# ---------------------------------------------------------------------------
# Step 2: Tool calls with parameters
# ---------------------------------------------------------------------------

class TestLogEntryToolCalls:
    """The entry includes which tools the agent called and with what parameters."""

    def test_entry_has_tool_calls(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "tool_calls" in entry

    def test_tool_calls_is_list(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert isinstance(entry["tool_calls"], list)

    def test_tool_calls_have_name_and_input(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        for tc in entry["tool_calls"]:
            assert "tool_name" in tc
            assert "tool_input" in tc

    def test_tool_calls_count_matches(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert len(entry["tool_calls"]) == 3

    def test_tool_names_are_strings(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        for tc in entry["tool_calls"]:
            assert isinstance(tc["tool_name"], str)

    def test_tool_inputs_are_dicts(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        for tc in entry["tool_calls"]:
            assert isinstance(tc["tool_input"], dict)

    def test_no_tool_calls_when_metadata_empty(self, game_state, sample_decision_dict, empty_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, empty_metadata, time.time())
        assert entry["tool_calls"] == []

    def test_specific_tool_names_logged(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        tool_names = [tc["tool_name"] for tc in entry["tool_calls"]]
        assert "get_pitcher_fatigue_assessment" in tool_names
        assert "get_bullpen_status" in tool_names
        assert "get_win_probability" in tool_names

    def test_tool_input_parameters_preserved(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        fatigue_call = [tc for tc in entry["tool_calls"] if tc["tool_name"] == "get_pitcher_fatigue_assessment"][0]
        assert fatigue_call["tool_input"]["pitcher_id"] == "a_sp1"


# ---------------------------------------------------------------------------
# Step 3: Full decision response
# ---------------------------------------------------------------------------

class TestLogEntryDecisionResponse:
    """The entry includes the agent's full response (decision text and reasoning)."""

    def test_entry_has_decision(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "decision" in entry

    def test_decision_has_type(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["decision"]["decision"] == "PITCHING_CHANGE"

    def test_decision_has_action_details(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "action_details" in entry["decision"]
        assert len(entry["decision"]["action_details"]) > 0

    def test_decision_has_reasoning(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "reasoning" in entry["decision"]
        assert len(entry["decision"]["reasoning"]) > 0

    def test_decision_has_confidence(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "confidence" in entry["decision"]
        assert 0 <= entry["decision"]["confidence"] <= 1

    def test_decision_has_key_factors(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "key_factors" in entry["decision"]
        assert isinstance(entry["decision"]["key_factors"], list)

    def test_full_decision_dict_preserved(self, game_state, sample_decision_dict, sample_metadata):
        """The full decision dict is stored, not a subset."""
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        for key in sample_decision_dict:
            assert key in entry["decision"]
            assert entry["decision"][key] == sample_decision_dict[key]


# ---------------------------------------------------------------------------
# Step 4: Active vs no-action classification
# ---------------------------------------------------------------------------

class TestLogEntryActiveVsNoAction:
    """The entry includes whether the decision was active or no-action."""

    def test_entry_has_is_active_decision(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "is_active_decision" in entry

    def test_pitching_change_is_active(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["is_active_decision"] is True

    def test_no_action_is_not_active(self, game_state, sample_no_action_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_no_action_dict, sample_metadata, time.time())
        assert entry["is_active_decision"] is False

    def test_swing_away_is_not_active(self, game_state, sample_metadata):
        decision = {"decision": "SWING_AWAY", "action_details": "let him hit", "confidence": 0.9, "reasoning": "ok", "key_factors": [], "risks": []}
        entry = build_decision_log_entry(1, game_state, "home", decision, sample_metadata, time.time())
        assert entry["is_active_decision"] is False

    def test_stolen_base_is_active(self, game_state, sample_metadata):
        decision = {"decision": "STOLEN_BASE", "action_details": "send runner", "confidence": 0.7, "reasoning": "fast", "key_factors": [], "risks": []}
        entry = build_decision_log_entry(1, game_state, "home", decision, sample_metadata, time.time())
        assert entry["is_active_decision"] is True

    def test_pinch_hit_is_active(self, game_state, sample_metadata):
        decision = {"decision": "PINCH_HIT", "action_details": "send Tanaka", "confidence": 0.8, "reasoning": "platoon", "key_factors": [], "risks": []}
        entry = build_decision_log_entry(1, game_state, "home", decision, sample_metadata, time.time())
        assert entry["is_active_decision"] is True

    def test_empty_decision_is_not_active(self, game_state, sample_metadata):
        decision = {"decision": "", "action_details": "", "confidence": 0.0, "reasoning": "", "key_factors": [], "risks": []}
        entry = build_decision_log_entry(1, game_state, "home", decision, sample_metadata, time.time())
        assert entry["is_active_decision"] is False

    def test_all_no_action_types_classified_correctly(self, game_state, sample_metadata):
        for no_action_type in NO_ACTION_TYPES:
            decision = {"decision": no_action_type, "action_details": "", "confidence": 0.9, "reasoning": "ok", "key_factors": [], "risks": []}
            entry = build_decision_log_entry(1, game_state, "home", decision, sample_metadata, time.time())
            assert entry["is_active_decision"] is False, f"{no_action_type} should be classified as no-action"


# ---------------------------------------------------------------------------
# Step 5: Token usage and latency
# ---------------------------------------------------------------------------

class TestLogEntryTokenUsageAndLatency:
    """The entry includes token usage and latency."""

    def test_entry_has_token_usage(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "token_usage" in entry

    def test_token_usage_has_input_tokens(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["token_usage"]["input_tokens"] == 3500

    def test_token_usage_has_output_tokens(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["token_usage"]["output_tokens"] == 850

    def test_token_usage_has_total_tokens(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["token_usage"]["total_tokens"] == 4350

    def test_entry_has_latency_ms(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "latency_ms" in entry
        assert entry["latency_ms"] == 2340

    def test_entry_has_agent_turns(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "agent_turns" in entry
        assert entry["agent_turns"] == 3

    def test_entry_has_retries(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert "retries" in entry
        assert entry["retries"] == 0

    def test_zero_token_usage_for_empty_metadata(self, game_state, sample_decision_dict, empty_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, empty_metadata, time.time())
        assert entry["token_usage"]["input_tokens"] == 0
        assert entry["token_usage"]["output_tokens"] == 0
        assert entry["token_usage"]["total_tokens"] == 0
        assert entry["latency_ms"] == 0


# ---------------------------------------------------------------------------
# Step 6: Structured JSON files organized by game
# ---------------------------------------------------------------------------

class TestWriteGameLog:
    """Logs are written to structured JSON files organized by game."""

    def test_write_creates_file(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            assert path.exists()

    def test_write_file_is_valid_json(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_write_file_named_by_seed(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            assert "game_42" in path.name

    def test_write_has_game_info(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert "game_info" in data
            assert data["game_info"]["seed"] == 42
            assert data["game_info"]["managed_team"] == "home"

    def test_write_has_home_and_away_teams(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert "home_team" in data["game_info"]
            assert "away_team" in data["game_info"]

    def test_write_has_final_score(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert "final_score" in data["game_info"]
            assert "home" in data["game_info"]["final_score"]
            assert "away" in data["game_info"]["final_score"]

    def test_write_has_summary(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert "summary" in data
            assert "total_decisions" in data["summary"]
            assert "active_decisions" in data["summary"]
            assert "no_action_decisions" in data["summary"]

    def test_write_summary_counts_correct(self, game_state, sample_decision_dict, sample_no_action_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            active_entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            passive_entry = build_decision_log_entry(2, game_state, "home", sample_no_action_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [active_entry, passive_entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["total_decisions"] == 2
            assert data["summary"]["active_decisions"] == 1
            assert data["summary"]["no_action_decisions"] == 1

    def test_write_summary_has_token_totals(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert data["summary"]["total_input_tokens"] == 3500
            assert data["summary"]["total_output_tokens"] == 850
            assert data["summary"]["total_tokens"] == 4350
            assert data["summary"]["total_latency_ms"] == 2340

    def test_write_has_decisions_list(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert "decisions" in data
            assert len(data["decisions"]) == 1

    def test_write_has_errors_list(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            errors = [{"error": "test error", "timestamp": time.time()}]
            path = write_game_log(game_state, [entry], errors, seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert "errors" in data
            assert len(data["errors"]) == 1
            assert data["summary"]["total_errors"] == 1

    def test_write_multiple_decisions(self, game_state, sample_decision_dict, sample_no_action_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entries = [
                build_decision_log_entry(i + 1, game_state, "home",
                                         sample_decision_dict if i % 2 == 0 else sample_no_action_dict,
                                         sample_metadata, time.time())
                for i in range(10)
            ]
            path = write_game_log(game_state, entries, [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            assert len(data["decisions"]) == 10

    def test_write_creates_log_directory(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "game_logs"
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=log_dir)
            assert path.exists()
            assert log_dir.exists()

    def test_different_seeds_produce_different_files(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path1 = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            path2 = write_game_log(game_state, [entry], [], seed=99, managed_team="home", log_dir=Path(tmpdir))
            assert path1 != path2
            assert path1.exists()
            assert path2.exists()


# ---------------------------------------------------------------------------
# Step 7: Timestamps
# ---------------------------------------------------------------------------

class TestLogEntryTimestamps:
    """Timestamps are included for correlating with the live game timeline."""

    def test_entry_has_timestamp(self, game_state, sample_decision_dict, sample_metadata):
        ts = time.time()
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, ts)
        assert "timestamp" in entry
        assert entry["timestamp"] == ts

    def test_timestamp_is_numeric(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert isinstance(entry["timestamp"], (int, float))

    def test_sequential_entries_have_increasing_timestamps(self, game_state, sample_decision_dict, sample_metadata):
        entries = []
        for i in range(5):
            ts = time.time()
            entry = build_decision_log_entry(i + 1, game_state, "home", sample_decision_dict, sample_metadata, ts)
            entries.append(entry)
        for i in range(1, len(entries)):
            assert entries[i]["timestamp"] >= entries[i - 1]["timestamp"]


# ---------------------------------------------------------------------------
# Additional: Turn numbering and managed team
# ---------------------------------------------------------------------------

class TestLogEntryTurnAndTeam:
    """Each entry has a turn number and managed team identifier."""

    def test_entry_has_turn(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(5, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["turn"] == 5

    def test_entry_has_managed_team(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        assert entry["managed_team"] == "home"

    def test_away_team_managed(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "away", sample_decision_dict, sample_metadata, time.time())
        assert entry["managed_team"] == "away"


# ---------------------------------------------------------------------------
# JSON serialization round-trip
# ---------------------------------------------------------------------------

class TestLogEntrySerialization:
    """Log entries must be fully JSON-serializable."""

    def test_entry_serializes_to_json(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        json_str = json.dumps(entry, default=str)
        assert len(json_str) > 0

    def test_entry_round_trips_through_json(self, game_state, sample_decision_dict, sample_metadata):
        entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
        json_str = json.dumps(entry, default=str)
        parsed = json.loads(json_str)
        assert parsed["turn"] == entry["turn"]
        assert parsed["decision"]["decision"] == "PITCHING_CHANGE"
        assert parsed["game_state"]["inning"] == entry["game_state"]["inning"]

    def test_game_log_file_round_trips(self, game_state, sample_decision_dict, sample_metadata):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = build_decision_log_entry(1, game_state, "home", sample_decision_dict, sample_metadata, time.time())
            path = write_game_log(game_state, [entry], [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)
            # Re-write and compare
            path2 = Path(tmpdir) / "rewritten.json"
            with open(path2, "w") as f:
                json.dump(data, f, indent=2, default=str)
            with open(path2) as f:
                data2 = json.load(f)
            assert data == data2


# ---------------------------------------------------------------------------
# Integration: multiple decision types
# ---------------------------------------------------------------------------

class TestLogEntryDecisionTypes:
    """Verify various decision types produce correct log entries."""

    @pytest.mark.parametrize("decision_type,expected_active", [
        ("PITCHING_CHANGE", True),
        ("PULL_STARTER", True),
        ("PINCH_HIT", True),
        ("STOLEN_BASE", True),
        ("SACRIFICE_BUNT", True),
        ("INTENTIONAL_WALK", True),
        ("DEFENSIVE_POSITIONING", True),
        ("MOUND_VISIT", True),
        ("PINCH_RUN", True),
        ("REPLAY_CHALLENGE", True),
        ("NO_ACTION", False),
        ("SWING_AWAY", False),
        ("LET_HIM_HIT", False),
        ("CONTINUE", False),
        ("HOLD", False),
    ])
    def test_decision_type_active_classification(self, game_state, sample_metadata, decision_type, expected_active):
        decision = {
            "decision": decision_type,
            "action_details": "test",
            "confidence": 0.5,
            "reasoning": "test",
            "key_factors": [],
            "risks": [],
        }
        entry = build_decision_log_entry(1, game_state, "home", decision, sample_metadata, time.time())
        assert entry["is_active_decision"] is expected_active


# ---------------------------------------------------------------------------
# Integration: full game log structure
# ---------------------------------------------------------------------------

class TestFullGameLogStructure:
    """End-to-end test of game log structure with multiple entries."""

    def test_full_game_log_structure(self, game_state, sample_metadata):
        decisions_data = [
            ("PITCHING_CHANGE", True),
            ("NO_ACTION", False),
            ("STOLEN_BASE", True),
            ("SWING_AWAY", False),
            ("PINCH_HIT", True),
        ]
        entries = []
        for i, (dec_type, _) in enumerate(decisions_data):
            decision = {
                "decision": dec_type,
                "action_details": f"test action {i}",
                "confidence": 0.5 + i * 0.1,
                "reasoning": f"reason {i}",
                "key_factors": [f"factor_{i}"],
                "risks": [],
            }
            entry = build_decision_log_entry(i + 1, game_state, "home", decision, sample_metadata, time.time())
            entries.append(entry)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_game_log(game_state, entries, [], seed=42, managed_team="home", log_dir=Path(tmpdir))
            with open(path) as f:
                data = json.load(f)

            # Verify structure
            assert data["game_info"]["seed"] == 42
            assert data["summary"]["total_decisions"] == 5
            assert data["summary"]["active_decisions"] == 3
            assert data["summary"]["no_action_decisions"] == 2

            # Each decision entry preserves full context
            for i, entry_data in enumerate(data["decisions"]):
                assert entry_data["turn"] == i + 1
                assert "game_state" in entry_data
                assert "tool_calls" in entry_data
                assert "decision" in entry_data
                assert "is_active_decision" in entry_data
                assert "token_usage" in entry_data
                assert "latency_ms" in entry_data
                assert "timestamp" in entry_data
