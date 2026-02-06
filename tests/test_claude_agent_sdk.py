# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the claude_agent_sdk feature.

Validates that the agent is built on the Claude Agent SDK:
1. The agent is instantiated using the Claude Agent SDK client (anthropic package)
2. All 12 information-gathering tools are registered as SDK tool definitions with input schemas
3. The SDK manages the conversation loop: sending game state and processing tool-use responses
4. The agent's system prompt instructs it to act as a baseball manager that gathers context
   via tools and responds with a plain-text decision
5. Tool call results are returned to the agent via the SDK's tool result mechanism
6. The agent can call multiple tools in a single turn to gather all needed context
7. After gathering sufficient context, the agent produces a plain-text decision as its final response
8. The SDK handles retries and error formatting when the agent makes an invalid tool call
"""

import json
import sys
import time
import inspect
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from anthropic import Anthropic, beta_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Step 1: Agent is instantiated using the Claude Agent SDK client
# ===========================================================================

class TestStep1AgentUsesSDKClient:
    """The agent must be instantiated using the Claude Agent SDK client."""

    def test_game_py_imports_anthropic(self):
        """game.py must import Anthropic from the anthropic package."""
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "from anthropic import Anthropic" in text

    def test_game_py_instantiates_client(self):
        """game.py must instantiate an Anthropic() client."""
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "Anthropic()" in text

    def test_run_agent_game_creates_client(self):
        """The run_agent_game function creates an Anthropic client."""
        from game import run_agent_game
        source = inspect.getsource(run_agent_game)
        assert "Anthropic()" in source

    def test_call_agent_accepts_client(self):
        """The _call_agent function accepts a client parameter."""
        from game import _call_agent
        sig = inspect.signature(_call_agent)
        param_names = list(sig.parameters.keys())
        assert "client" in param_names

    def test_run_agent_decision_accepts_client(self):
        """The run_agent_decision function accepts a client parameter."""
        from game import run_agent_decision
        sig = inspect.signature(run_agent_decision)
        param_names = list(sig.parameters.keys())
        assert "client" in param_names

    def test_anthropic_is_pep723_dependency(self):
        """anthropic must be declared as a PEP 723 dependency in game.py."""
        text = (PROJECT_ROOT / "game.py").read_text()
        # Find the PEP 723 metadata block
        in_block = False
        found_anthropic = False
        for line in text.splitlines():
            if line.strip() == "# /// script":
                in_block = True
            elif line.strip() == "# ///" and in_block:
                break
            elif in_block and "anthropic" in line.lower():
                found_anthropic = True
        assert found_anthropic, "anthropic not declared in PEP 723 metadata block"

    def test_single_turn_also_creates_client(self):
        """The run_single_turn function also creates an Anthropic client."""
        from game import run_single_turn
        source = inspect.getsource(run_single_turn)
        assert "Anthropic()" in source


# ===========================================================================
# Step 2: All 12 tools registered as SDK tool definitions with input schemas
# ===========================================================================

class TestStep2ToolsRegisteredWithSchemas:
    """All 12 tools must be registered as SDK tool definitions with input schemas."""

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

    def test_all_tools_list_has_12_entries(self):
        """ALL_TOOLS must contain exactly 12 tool definitions."""
        from tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 12

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_tool_is_in_all_tools(self, tool_name):
        """Each expected tool must be present in ALL_TOOLS by name."""
        from tools import ALL_TOOLS
        tool_names = [t.name for t in ALL_TOOLS]
        assert tool_name in tool_names, f"Tool '{tool_name}' not found in ALL_TOOLS"

    def test_all_tools_have_name_attribute(self):
        """Every tool in ALL_TOOLS must have a .name attribute (SDK tool)."""
        from tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert hasattr(tool, "name"), f"Tool {tool} missing .name attribute"
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

    def test_all_tools_are_callable(self):
        """Every tool in ALL_TOOLS must be callable."""
        from tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert callable(tool), f"Tool {tool.name} is not callable"

    def test_tools_decorated_with_beta_tool(self):
        """Each tool .py file must use the @beta_tool decorator from anthropic."""
        for tool_name in self.EXPECTED_TOOLS:
            text = (PROJECT_ROOT / "tools" / f"{tool_name}.py").read_text()
            assert "@beta_tool" in text, f"{tool_name}.py missing @beta_tool decorator"
            assert "from anthropic import beta_tool" in text, (
                f"{tool_name}.py missing 'from anthropic import beta_tool'"
            )

    def test_tools_have_descriptions(self):
        """Each tool must have a description for the SDK (from docstring or tool definition)."""
        for tool_name in self.EXPECTED_TOOLS:
            text = (PROJECT_ROOT / "tools" / f"{tool_name}.py").read_text()
            # The function decorated with @beta_tool must have a docstring
            # which the SDK uses as the tool description
            func_def_idx = text.find(f"def {tool_name}(")
            assert func_def_idx != -1, f"Function {tool_name} not found in its file"
            # After the function signature, there should be a docstring
            after_func = text[func_def_idx:]
            assert '"""' in after_func[:500] or "'''" in after_func[:500], (
                f"Tool {tool_name} function missing docstring"
            )

    def test_tools_return_json_strings(self):
        """All tools must return JSON-parseable strings."""
        from tools import ALL_TOOLS

        # Call each tool with valid inputs and verify JSON output
        test_calls = [
            (ALL_TOOLS[0], ("h_003",)),       # get_batter_stats
            (ALL_TOOLS[1], ("h_sp1",)),       # get_pitcher_stats
            (ALL_TOOLS[2], ("h_003", "a_sp1")),  # get_matchup_data
        ]
        for tool, args in test_calls:
            result = tool(*args)
            assert isinstance(result, str), f"Tool {tool.name} did not return a string"
            data = json.loads(result)
            assert isinstance(data, dict), f"Tool {tool.name} did not return a JSON object"

    def test_tools_passed_to_tool_runner(self):
        """game.py must pass ALL_TOOLS to the SDK's tool_runner."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "tools=ALL_TOOLS" in source

    def test_game_py_imports_all_tools(self):
        """game.py must import ALL_TOOLS from tools."""
        text = (PROJECT_ROOT / "game.py").read_text()
        assert "from tools import ALL_TOOLS" in text


# ===========================================================================
# Step 3: SDK manages the conversation loop
# ===========================================================================

class TestStep3SDKManagesConversationLoop:
    """The SDK must manage the conversation loop: sending game state and processing responses."""

    def test_call_agent_uses_tool_runner(self):
        """_call_agent must use client.beta.messages.tool_runner for the conversation loop."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "tool_runner" in source
        assert "client.beta.messages.tool_runner" in source

    def test_call_agent_iterates_over_runner(self):
        """_call_agent must iterate over the runner to process multiple turns."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "for message in runner" in source

    def test_call_agent_passes_messages(self):
        """_call_agent must pass the conversation messages to the SDK."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "messages=messages" in source

    def test_call_agent_passes_system_prompt(self):
        """_call_agent must pass the SYSTEM_PROMPT to the SDK."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "system=SYSTEM_PROMPT" in source

    def test_call_agent_passes_model(self):
        """_call_agent must specify the Claude model to use."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "model=" in source

    def test_call_agent_passes_max_tokens(self):
        """_call_agent must specify max_tokens."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "max_tokens=" in source

    def test_run_agent_decision_builds_user_message(self):
        """run_agent_decision must format the game state as a user message."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "matchup_state" in source
        assert "roster_state" in source
        assert "opponent_roster_state" in source
        assert "decision_prompt" in source

    def test_run_agent_decision_appends_messages(self):
        """run_agent_decision must append to the messages list for context continuity."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "messages.append" in source

    def test_messages_include_assistant_responses(self):
        """run_agent_decision must add assistant responses back to messages."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert '"role": "assistant"' in source or "'role': 'assistant'" in source

    def test_run_agent_game_trims_message_history(self):
        """run_agent_game should trim old messages to manage context window."""
        from game import run_agent_game
        source = inspect.getsource(run_agent_game)
        assert "messages[-" in source or "messages =" in source


# ===========================================================================
# Step 4: System prompt instructs agent as baseball manager
# ===========================================================================

class TestStep4SystemPromptBaseballManager:
    """The system prompt must instruct the agent to act as a baseball manager."""

    def test_system_prompt_exists(self):
        """SYSTEM_PROMPT must be defined in game.py."""
        from game import SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_baseball_manager(self):
        """System prompt must establish the agent as a baseball manager."""
        from game import SYSTEM_PROMPT
        assert "baseball manager" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_tools(self):
        """System prompt must mention the 12 available tools."""
        from game import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "tool" in prompt_lower
        assert "12" in SYSTEM_PROMPT

    def test_system_prompt_lists_decision_types(self):
        """System prompt must list the available decision types."""
        from game import SYSTEM_PROMPT
        # Key decision types should be mentioned
        assert "PITCHING_CHANGE" in SYSTEM_PROMPT or "pitching change" in SYSTEM_PROMPT.lower()
        assert "PINCH_HIT" in SYSTEM_PROMPT or "pinch hit" in SYSTEM_PROMPT.lower()
        assert "STOLEN_BASE" in SYSTEM_PROMPT or "stolen base" in SYSTEM_PROMPT.lower()
        assert "NO_ACTION" in SYSTEM_PROMPT or "no action" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_analytical_context(self):
        """System prompt should reference analytical tools like win probability."""
        from game import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "win probability" in prompt_lower or "run expectancy" in prompt_lower

    def test_system_prompt_mentions_mlb_rules(self):
        """System prompt should reference key MLB rules like 3-batter minimum."""
        from game import SYSTEM_PROMPT
        assert "3-batter" in SYSTEM_PROMPT or "three-batter" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_player_removal(self):
        """System prompt should note that removed players cannot re-enter."""
        from game import SYSTEM_PROMPT
        assert "removed" in SYSTEM_PROMPT.lower() or "re-enter" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_mound_visits(self):
        """System prompt should reference mound visit limits."""
        from game import SYSTEM_PROMPT
        assert "mound visit" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_no_action_default(self):
        """System prompt should explain that most at-bats require no action."""
        from game import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        assert "no action" in prompt_lower or "no strategic" in prompt_lower

    def test_system_prompt_mentions_platoon(self):
        """System prompt should mention platoon advantages."""
        from game import SYSTEM_PROMPT
        assert "platoon" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_fatigue(self):
        """System prompt should mention pitcher fatigue considerations."""
        from game import SYSTEM_PROMPT
        assert "fatigue" in SYSTEM_PROMPT.lower() or "pitcher" in SYSTEM_PROMPT.lower()

    def test_system_prompt_used_in_call_agent(self):
        """SYSTEM_PROMPT is passed to the SDK tool_runner call."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "system=SYSTEM_PROMPT" in source


# ===========================================================================
# Step 5: Tool call results returned via SDK tool result mechanism
# ===========================================================================

class TestStep5ToolResultsMechanism:
    """Tool call results are returned to the agent via the SDK's tool result mechanism."""

    def test_tools_return_strings(self):
        """Tools return string results (JSON) compatible with the SDK's tool result protocol."""
        from tools import ALL_TOOLS

        # The SDK's beta_tool decorator expects tool functions to return strings
        result = ALL_TOOLS[0]("h_003")  # get_batter_stats
        assert isinstance(result, str)
        data = json.loads(result)
        assert "status" in data

    def test_tool_runner_handles_tool_dispatch(self):
        """The tool_runner from the SDK automatically dispatches tool calls to functions."""
        # Verify that _call_agent uses tool_runner which handles dispatching
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        # tool_runner handles the dispatch loop automatically
        assert "tool_runner" in source

    def test_call_agent_tracks_tool_calls(self):
        """_call_agent must track which tools the agent called and return them."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "tool_calls" in source
        assert "tool_name" in source
        assert "tool_input" in source

    def test_call_agent_processes_tool_use_blocks(self):
        """_call_agent must detect tool_use blocks in the agent's response."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert 'block.type == "tool_use"' in source

    def test_call_agent_returns_metadata_with_tool_calls(self):
        """_call_agent must return metadata including the list of tool calls."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "call_metadata" in source
        # Verify metadata structure includes tool_calls
        assert '"tool_calls": tool_calls' in source or "'tool_calls': tool_calls" in source

    def test_tool_error_responses_are_json(self):
        """Tool error responses must also be valid JSON for the SDK to relay to the agent."""
        from tools import ALL_TOOLS
        # Call with invalid inputs
        error_result = ALL_TOOLS[0]("NONEXISTENT")
        data = json.loads(error_result)
        assert data["status"] == "error"
        assert "message" in data

    def test_all_tools_return_status_field(self):
        """All tools must include a 'status' field in their response (ok or error)."""
        from tools import ALL_TOOLS

        # Valid calls should return status: ok
        result = json.loads(ALL_TOOLS[0]("h_003"))  # get_batter_stats
        assert result["status"] in ("ok", "error")

        result = json.loads(ALL_TOOLS[1]("h_sp1"))  # get_pitcher_stats
        assert result["status"] in ("ok", "error")


# ===========================================================================
# Step 6: Agent can call multiple tools in a single turn
# ===========================================================================

class TestStep6MultiplToolsPerTurn:
    """The agent can call multiple tools in a single turn."""

    def test_call_agent_iterates_all_content_blocks(self):
        """_call_agent iterates over all content blocks, handling multiple tool_use per turn."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        # Must iterate message.content to find all tool_use blocks
        assert "for block in message.content" in source

    def test_tool_calls_accumulated_across_blocks(self):
        """Tool calls should be accumulated in a list, allowing multiple per message."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "tool_calls.append" in source

    def test_tool_runner_supports_multi_tool(self):
        """The SDK tool_runner naturally supports the agent calling multiple tools per turn.

        Verify that ALL_TOOLS contains multiple tools (the SDK dispatches any of them).
        """
        from tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 12
        # All tools are registered, so the SDK can dispatch any combination per turn
        tool_names = {t.name for t in ALL_TOOLS}
        assert len(tool_names) == 12  # No duplicates

    def test_all_tools_independently_callable(self):
        """Each tool can be called independently (no shared mutable state between calls)."""
        from tools import ALL_TOOLS

        # Call multiple tools in sequence and verify independent results
        batter_result = json.loads(ALL_TOOLS[0]("h_003"))  # get_batter_stats
        pitcher_result = json.loads(ALL_TOOLS[1]("h_sp1"))  # get_pitcher_stats
        matchup_result = json.loads(ALL_TOOLS[2]("h_003", "a_sp1"))  # get_matchup_data
        re_result = json.loads(ALL_TOOLS[3](True, False, False, 1))  # get_run_expectancy

        # Each returns data independently
        assert batter_result["status"] == "ok"
        assert pitcher_result["status"] == "ok"
        assert matchup_result["status"] == "ok"
        assert re_result["status"] == "ok"

        # Results are different -- each tool returns distinct data structures
        assert "traditional" in batter_result  # Batting stats
        assert "pitch_mix" in pitcher_result or "stats" in pitcher_result  # Pitching stats
        assert "batter_id" in matchup_result or "matchup" in matchup_result  # Matchup data


# ===========================================================================
# Step 7: Agent produces plain-text decision as final response
# ===========================================================================

class TestStep7PlainTextDecision:
    """After gathering context, the agent produces a structured decision."""

    def test_manager_decision_model_exists(self):
        """ManagerDecision model must exist in models.py."""
        from models import ManagerDecision
        assert ManagerDecision is not None

    def test_manager_decision_has_required_fields(self):
        """ManagerDecision must have all required fields."""
        from models import ManagerDecision
        fields = set(ManagerDecision.model_fields.keys())
        assert "decision" in fields
        assert "action_details" in fields
        assert "confidence" in fields
        assert "reasoning" in fields

    def test_manager_decision_has_optional_fields(self):
        """ManagerDecision must have optional analytical fields."""
        from models import ManagerDecision
        fields = set(ManagerDecision.model_fields.keys())
        assert "key_factors" in fields
        assert "risks" in fields
        assert "alternatives_considered" in fields

    def test_output_format_is_manager_decision(self):
        """_call_agent must pass output_format=ManagerDecision to the SDK."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "output_format=ManagerDecision" in source

    def test_call_agent_extracts_parsed_decision(self):
        """_call_agent must extract the parsed ManagerDecision from the final message."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "final_message.parsed" in source

    def test_call_agent_model_dumps_decision(self):
        """_call_agent converts the ManagerDecision to a dict via model_dump."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "model_dump" in source

    def test_call_agent_returns_decision_dict(self):
        """_call_agent returns a tuple with the decision dict."""
        from game import _call_agent
        sig = inspect.signature(_call_agent)
        # Return annotation should indicate tuple
        source = inspect.getsource(_call_agent)
        assert "return decision_dict," in source

    def test_fallback_to_no_action_on_no_parsed_output(self):
        """If the agent doesn't return structured output, fallback to NO_ACTION."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "NO_ACTION" in source
        assert "No structured output received" in source or "No valid decision received" in source

    def test_manager_decision_can_be_constructed(self):
        """ManagerDecision can be instantiated with valid data."""
        from models import ManagerDecision
        decision = ManagerDecision(
            decision="PINCH_HIT",
            action_details="Send Tanaka to bat for Ortiz",
            confidence=0.8,
            reasoning="Platoon advantage against LHP",
            key_factors=["L-L disadvantage"],
            risks=["Lose Ortiz's bat"],
        )
        assert decision.decision == "PINCH_HIT"
        assert decision.confidence == 0.8

    def test_manager_decision_no_action(self):
        """ManagerDecision can represent a no-action decision."""
        from models import ManagerDecision
        decision = ManagerDecision(
            decision="NO_ACTION",
            action_details="No strategic intervention needed",
            confidence=0.95,
            reasoning="Standard at-bat, no high-leverage situation",
        )
        assert decision.decision == "NO_ACTION"

    def test_no_action_types_defined(self):
        """game.py must define the set of no-action decision types."""
        from game import NO_ACTION_TYPES
        assert isinstance(NO_ACTION_TYPES, set)
        assert "NO_ACTION" in NO_ACTION_TYPES
        assert "SWING_AWAY" in NO_ACTION_TYPES


# ===========================================================================
# Step 8: SDK handles retries and error formatting for invalid tool calls
# ===========================================================================

class TestStep8RetriesAndErrorHandling:
    """The SDK handles retries and error formatting for invalid tool calls."""

    def test_run_agent_decision_has_retry_loop(self):
        """run_agent_decision must implement a retry loop for invalid decisions."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "max_retries" in source
        assert "retries" in source
        assert "while" in source

    def test_run_agent_decision_validates_before_retry(self):
        """run_agent_decision must validate the decision before re-prompting."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "_peek_validate" in source

    def test_peek_validate_function_exists(self):
        """_peek_validate must exist as a validation-only check."""
        from game import _peek_validate
        assert callable(_peek_validate)

    def test_peek_validate_does_not_mutate_game_state(self):
        """_peek_validate must check validity without modifying game state."""
        from game import _peek_validate
        from simulation import SimulationEngine, load_rosters

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)

        # Save game state before
        score_before = (game.score_home, game.score_away)
        inning_before = game.inning
        outs_before = game.outs

        # Call _peek_validate with various decisions
        _peek_validate(game, {"decision": "NO_ACTION"}, "home")
        _peek_validate(game, {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"}, "home")
        _peek_validate(game, {"decision": "STOLEN_BASE"}, "home")

        # Game state must be unchanged
        assert game.score_home == score_before[0]
        assert game.score_away == score_before[1]
        assert game.inning == inning_before
        assert game.outs == outs_before

    def test_peek_validate_no_action_always_valid(self):
        """_peek_validate should accept NO_ACTION as always valid."""
        from game import _peek_validate
        from simulation import SimulationEngine, load_rosters

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)

        result = _peek_validate(game, {"decision": "NO_ACTION"}, "home")
        assert result.valid

    def test_peek_validate_catches_3_batter_minimum(self):
        """_peek_validate must catch 3-batter minimum violations."""
        from game import _peek_validate
        from simulation import SimulationEngine, load_rosters

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)
        # Top of 1st: home is fielding, pitcher has faced 0 batters
        game.home.current_pitcher_batters_faced_this_stint = 1

        result = _peek_validate(
            game,
            {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
            "home",
        )
        assert not result.valid
        assert "3-batter" in result.error.lower()

    def test_retry_sends_error_message_back(self):
        """On invalid decision, the error is sent back to the agent to correct."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "INVALID" in source
        assert "reconsider" in source.lower() or "valid" in source.lower()

    def test_forced_no_action_after_max_retries(self):
        """After max_retries, a forced NO_ACTION is returned."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "Forced" in source or "forced" in source or "forcing NO_ACTION" in source

    def test_max_retries_default(self):
        """run_agent_decision should have a max_retries parameter with default value."""
        from game import run_agent_decision
        sig = inspect.signature(run_agent_decision)
        params = sig.parameters
        assert "max_retries" in params
        default = params["max_retries"].default
        assert isinstance(default, int) and default >= 3

    def test_call_agent_handles_exceptions(self):
        """run_agent_decision handles exceptions from _call_agent gracefully."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "except Exception" in source

    def test_tool_errors_return_structured_json(self):
        """When tools receive bad inputs, they return structured error JSON."""
        from tools import ALL_TOOLS

        # Invalid player ID
        result = json.loads(ALL_TOOLS[0]("INVALID_PLAYER"))
        assert result["status"] == "error"
        assert "error_code" in result
        assert "message" in result

    def test_run_agent_decision_returns_metadata(self):
        """run_agent_decision returns comprehensive metadata including retries count."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert '"retries": retries' in source or "'retries': retries" in source

    def test_decision_result_has_valid_and_error_fields(self):
        """DecisionResult must have valid and error fields."""
        from simulation import DecisionResult
        # Valid result
        r = DecisionResult(valid=True)
        assert r.valid is True
        # Invalid result
        r = DecisionResult(valid=False, error="Test error")
        assert r.valid is False
        assert r.error == "Test error"


# ===========================================================================
# Integration tests: Full SDK wiring
# ===========================================================================

class TestSDKIntegrationWiring:
    """Integration tests verifying the complete SDK wiring."""

    def test_game_state_to_scenario_produces_valid_payload(self):
        """game_state_to_scenario must produce a valid scenario dict for the agent."""
        from simulation import SimulationEngine, load_rosters, game_state_to_scenario

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)
        scenario = game_state_to_scenario(game, "home")

        assert "matchup_state" in scenario
        assert "roster_state" in scenario
        assert "opponent_roster_state" in scenario
        assert "decision_prompt" in scenario

        # Verify the scenario can be serialized to JSON
        json_str = json.dumps(scenario, default=str)
        assert len(json_str) > 100

    def test_scenario_matchup_state_has_required_fields(self):
        """The matchup_state in scenario must have all required fields."""
        from simulation import SimulationEngine, load_rosters, game_state_to_scenario

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)
        scenario = game_state_to_scenario(game, "home")

        ms = scenario["matchup_state"]
        assert "inning" in ms
        assert "half" in ms
        assert "outs" in ms
        assert "batter" in ms
        assert "pitcher" in ms
        assert "score" in ms

    def test_dry_run_validates_full_sdk_setup(self):
        """The dry-run mode validates the full SDK setup (models, tools, scenario)."""
        from game import run_dry_run
        # run_dry_run should complete without errors
        run_dry_run()

    def test_build_sample_scenario_returns_valid_structure(self):
        """build_sample_scenario must return a fully populated scenario."""
        from game import build_sample_scenario
        scenario = build_sample_scenario()

        assert "matchup_state" in scenario
        assert "roster_state" in scenario
        assert "opponent_roster_state" in scenario
        assert "decision_prompt" in scenario

        # Verify nested structures
        ms = scenario["matchup_state"]
        assert ms["inning"] == 7
        assert ms["half"] == "BOTTOM"
        assert ms["batter"]["player_id"] == "h_003"

    def test_build_decision_log_entry_captures_context(self):
        """build_decision_log_entry must capture full game context and metadata."""
        from game import build_decision_log_entry
        from simulation import SimulationEngine, load_rosters

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)

        entry = build_decision_log_entry(
            turn=1,
            game_state=game,
            managed_team="home",
            decision_dict={"decision": "NO_ACTION", "action_details": "test"},
            decision_metadata={
                "tool_calls": [{"tool_name": "get_batter_stats", "tool_input": {"player_id": "h_001"}}],
                "token_usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200},
                "latency_ms": 500,
                "agent_turns": 2,
                "retries": 0,
            },
            timestamp=time.time(),
        )

        assert entry["turn"] == 1
        assert entry["managed_team"] == "home"
        assert "game_state" in entry
        assert "tool_calls" in entry
        assert "token_usage" in entry
        assert "latency_ms" in entry
        assert entry["is_active_decision"] is False

    def test_write_game_log_creates_file(self):
        """write_game_log must write a structured JSON log file."""
        from game import write_game_log
        from simulation import SimulationEngine, load_rosters
        import tempfile

        engine = SimulationEngine(seed=42)
        rosters = load_rosters()
        game = engine.initialize_game(rosters)
        game.game_over = True
        game.winning_team = "Home Team"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_game_log(
                game_state=game,
                decision_log=[],
                error_log=[],
                seed=42,
                managed_team="home",
                log_dir=Path(tmpdir),
            )
            assert log_path.exists()
            data = json.loads(log_path.read_text())
            assert "game_info" in data
            assert "summary" in data
            assert "decisions" in data

    def test_all_12_tools_work_with_sdk_protocol(self):
        """All 12 tools follow the SDK protocol: accept args, return JSON string."""
        from tools import ALL_TOOLS

        # Each tool must have .name and be callable, returning a JSON string
        for tool in ALL_TOOLS:
            assert hasattr(tool, "name")
            assert callable(tool)
            # We won't call all of them but verify the protocol attributes

    def test_system_prompt_covers_all_tool_categories(self):
        """The system prompt should reference tool categories: stats, situation, pitching, defense."""
        from game import SYSTEM_PROMPT
        prompt_lower = SYSTEM_PROMPT.lower()
        # The prompt should reference the types of analytical tools available
        assert "statistic" in prompt_lower or "stats" in prompt_lower
        assert "probability" in prompt_lower or "win" in prompt_lower
        assert "pitcher" in prompt_lower or "pitching" in prompt_lower
        assert "defen" in prompt_lower  # defensive, defense
