# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the repo_setup feature.

Validates the project bootstrap:
  1. Agent entry point script has PEP 723 inline metadata declaring dependencies
  2. tools/ directory with __init__.py and one .py file per tool (all 12 tools)
  3. MatchupState, RosterState, and OpponentRosterState Pydantic models in models.py
  4. DecisionOutput model (ManagerDecision) in models.py
  5. Entry point wires up Claude Agent SDK client, registers tools, accepts game state
  6. Project runs with 'uv run game.py' (dry-run validation)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Step 1: Agent entry point has PEP 723 inline metadata
# ---------------------------------------------------------------------------

class TestStep1PEP723Metadata:
    """The entry point script must have a PEP 723 inline metadata block."""

    def test_game_py_exists(self):
        assert (PROJECT_ROOT / "game.py").exists()

    def test_game_py_has_pep723_block(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "# /// script" in text
        assert "# ///" in text

    def test_game_py_declares_anthropic_dependency(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "anthropic" in text

    def test_game_py_declares_pydantic_dependency(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "pydantic" in text

    def test_models_py_has_pep723_block(self):
        text = (PROJECT_ROOT / "models.py").read_text()
        assert "# /// script" in text

    def test_models_py_declares_pydantic_dependency(self):
        text = (PROJECT_ROOT / "models.py").read_text()
        assert "pydantic" in text


# ---------------------------------------------------------------------------
# Step 2: tools/ directory with __init__.py and 12 tool files
# ---------------------------------------------------------------------------

class TestStep2ToolsDirectory:
    """The tools/ directory must contain __init__.py and one .py per tool."""

    EXPECTED_TOOLS = [
        "get_batter_stats",
        "get_pitcher_stats",
        "get_matchup_data",
        "get_run_expectancy",
        "get_win_probability",
        "evaluate_stolen_base",
        "evaluate_sacrifice_bunt",
        "get_bullpen_status",
        "get_pitcher_fatigue_assessment",
        "get_defensive_positioning",
        "get_defensive_replacement_value",
        "get_platoon_comparison",
    ]

    def test_tools_directory_exists(self):
        assert (PROJECT_ROOT / "tools").is_dir()

    def test_tools_init_py_exists(self):
        assert (PROJECT_ROOT / "tools" / "__init__.py").exists()

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_tool_file_exists(self, tool_name):
        assert (PROJECT_ROOT / "tools" / f"{tool_name}.py").exists()

    def test_exactly_12_tool_files(self):
        tool_files = [
            f for f in (PROJECT_ROOT / "tools").glob("*.py")
            if f.name != "__init__.py"
        ]
        assert len(tool_files) == 12

    def test_all_tools_importable(self):
        from tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 12

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_tool_registered_in_all_tools(self, tool_name):
        from tools import ALL_TOOLS
        names = [t.name for t in ALL_TOOLS]
        assert tool_name in names


# ---------------------------------------------------------------------------
# Step 3: Pydantic models for MatchupState, RosterState, OpponentRosterState
# ---------------------------------------------------------------------------

class TestStep3PydanticModels:
    """models.py must define the three game state input models."""

    def test_matchup_state_importable(self):
        from models import MatchupState
        assert MatchupState is not None

    def test_roster_state_importable(self):
        from models import RosterState
        assert RosterState is not None

    def test_opponent_roster_state_importable(self):
        from models import OpponentRosterState
        assert OpponentRosterState is not None

    def test_matchup_state_has_required_fields(self):
        from models import MatchupState
        fields = set(MatchupState.model_fields.keys())
        required = {"inning", "half", "outs", "count", "runners", "score",
                     "batting_team", "batter", "pitcher", "on_deck_batter"}
        assert required.issubset(fields)

    def test_roster_state_has_required_fields(self):
        from models import RosterState
        fields = set(RosterState.model_fields.keys())
        required = {"our_lineup", "our_lineup_position", "bench", "bullpen",
                     "mound_visits_remaining", "challenge_available"}
        assert required.issubset(fields)

    def test_opponent_roster_state_has_required_fields(self):
        from models import OpponentRosterState
        fields = set(OpponentRosterState.model_fields.keys())
        required = {"their_lineup", "their_lineup_position", "their_bench",
                     "their_bullpen"}
        assert required.issubset(fields)

    def test_matchup_state_is_pydantic_model(self):
        from models import MatchupState
        from pydantic import BaseModel
        assert issubclass(MatchupState, BaseModel)

    def test_roster_state_is_pydantic_model(self):
        from models import RosterState
        from pydantic import BaseModel
        assert issubclass(RosterState, BaseModel)

    def test_opponent_roster_state_is_pydantic_model(self):
        from models import OpponentRosterState
        from pydantic import BaseModel
        assert issubclass(OpponentRosterState, BaseModel)

    def test_matchup_state_serialization(self):
        """MatchupState can be constructed and serialized to JSON."""
        from models import (
            MatchupState, Half, BattingTeam, Count, Runners,
            Score, BatterInfo, PitcherInfo, OnDeckBatter, Hand, ThrowHand,
        )
        ms = MatchupState(
            inning=1, half=Half.TOP, outs=0,
            count=Count(balls=0, strikes=0),
            runners=Runners(),
            score=Score(home=0, away=0),
            batting_team=BattingTeam.AWAY,
            batter=BatterInfo(player_id="p1", name="Test", bats=Hand.R, lineup_position=1),
            pitcher=PitcherInfo(player_id="p2", name="Pitcher", throws=ThrowHand.R),
            on_deck_batter=OnDeckBatter(player_id="p3", name="OnDeck", bats=Hand.L),
        )
        data = json.loads(ms.model_dump_json())
        assert data["inning"] == 1
        assert data["half"] == "TOP"


# ---------------------------------------------------------------------------
# Step 4: DecisionOutput model (ManagerDecision) in models.py
# ---------------------------------------------------------------------------

class TestStep4DecisionOutput:
    """models.py must define a decision output model for the agent."""

    def test_manager_decision_importable(self):
        from models import ManagerDecision
        assert ManagerDecision is not None

    def test_manager_decision_is_pydantic_model(self):
        from models import ManagerDecision
        from pydantic import BaseModel
        assert issubclass(ManagerDecision, BaseModel)

    def test_manager_decision_has_decision_field(self):
        from models import ManagerDecision
        assert "decision" in ManagerDecision.model_fields

    def test_manager_decision_has_action_details(self):
        from models import ManagerDecision
        assert "action_details" in ManagerDecision.model_fields

    def test_manager_decision_has_reasoning(self):
        from models import ManagerDecision
        assert "reasoning" in ManagerDecision.model_fields

    def test_manager_decision_construction(self):
        from models import ManagerDecision
        d = ManagerDecision(
            decision="NO_ACTION",
            action_details="No strategic intervention needed",
            confidence=0.9,
            reasoning="Routine at-bat",
        )
        assert d.decision == "NO_ACTION"
        assert d.confidence == 0.9

    def test_manager_decision_serialization(self):
        from models import ManagerDecision
        d = ManagerDecision(
            decision="PITCHING_CHANGE",
            action_details="Bring in Rivera",
            confidence=0.85,
            reasoning="Starter fatigued",
            key_factors=["velocity down", "3rd TTO"],
            risks=["Rivera pitched yesterday"],
        )
        data = json.loads(d.model_dump_json())
        assert data["decision"] == "PITCHING_CHANGE"
        assert len(data["key_factors"]) == 2


# ---------------------------------------------------------------------------
# Step 5: Entry point wires up Claude SDK, tools, and accepts input
# ---------------------------------------------------------------------------

class TestStep5EntryPointWiring:
    """game.py must instantiate the SDK client, register tools, and accept input."""

    def test_game_py_imports_anthropic(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "from anthropic import Anthropic" in text

    def test_game_py_imports_all_tools(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "from tools import ALL_TOOLS" in text

    def test_game_py_imports_models(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "MatchupState" in text
        assert "RosterState" in text
        assert "OpponentRosterState" in text
        assert "ManagerDecision" in text

    def test_game_py_has_system_prompt(self):
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "SYSTEM_PROMPT" in text

    def test_game_py_registers_tools_with_sdk(self):
        """The entry point must pass ALL_TOOLS to the Claude Agent SDK."""
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "tools=ALL_TOOLS" in text

    def test_game_py_uses_sdk_tool_runner(self):
        """The entry point must use the SDK's tool_runner for the agent loop."""
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "tool_runner" in text

    def test_game_py_uses_manager_decision_output(self):
        """The entry point must use ManagerDecision as the output format."""
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "output_format=ManagerDecision" in text

    def test_build_sample_scenario_returns_valid_structure(self):
        from game import build_sample_scenario
        scenario = build_sample_scenario()
        assert "matchup_state" in scenario
        assert "roster_state" in scenario
        assert "opponent_roster_state" in scenario
        assert "decision_prompt" in scenario

    def test_sample_scenario_serializes_to_json(self):
        from game import build_sample_scenario
        scenario = build_sample_scenario()
        j = json.dumps(scenario, default=str)
        assert len(j) > 100

    def test_12_tools_have_names(self):
        from tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert hasattr(tool, "name")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0


# ---------------------------------------------------------------------------
# Step 6: Project runs with uv run (dry-run)
# ---------------------------------------------------------------------------

class TestStep6RunsWithUV:
    """The project must be runnable via 'uv run game.py --dry-run'."""

    def test_dry_run_succeeds(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Dry run failed:\n{result.stderr}"
        assert "ALL VALIDATIONS PASSED" in result.stdout

    def test_dry_run_validates_models(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert "MatchupState" in result.stdout
        assert "RosterState" in result.stdout
        assert "OpponentRosterState" in result.stdout
        assert "ManagerDecision" in result.stdout

    def test_dry_run_validates_tools(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert "get_batter_stats: OK" in result.stdout
        assert "get_pitcher_stats: OK" in result.stdout
        assert "get_matchup_data: OK" in result.stdout
        assert "get_run_expectancy: OK" in result.stdout

    def test_dry_run_validates_agent_config(self):
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert "Tools registered: 12" in result.stdout
        assert "Output schema: ManagerDecision" in result.stdout

    def test_sim_mode_runs(self):
        """Simulation mode runs without API key."""
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--sim", "--seed", "42"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Sim failed:\n{result.stderr}"
        assert "FINAL" in result.stdout or "Total plate appearances" in result.stdout
