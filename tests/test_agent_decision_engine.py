# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the agent_decision_engine feature.

Verifies all feature requirements from features.json:
1. Accept MatchupState, RosterState, and OpponentRosterState as input
2. Format the game state as a natural-language scenario description for the agent
3. Send the scenario to the Claude agent with all 12 tools registered
4. The agent may call zero or more tools to gather analytical context
5. The agent produces a plain-text response: either 'no action' or an active decision with reasoning
6. Return the agent's response along with metadata (tools called, token usage)
7. Handle agent errors gracefully: if the agent fails to produce a valid response after retries,
   return a safe 'no action' default
"""

import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ManagerDecision
from simulation import (
    SimulationEngine,
    GameState,
    load_rosters,
    game_state_to_scenario,
    validate_and_apply_decision,
    DecisionResult,
    BaseRunner,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def make_mock_message(decision_dict=None, tool_uses=None, text_blocks=None,
                      input_tokens=100, output_tokens=50):
    """Create a mock message object that looks like an anthropic API response.

    Args:
        decision_dict: Dict to set as the .parsed ManagerDecision.
        tool_uses: List of (tool_name, tool_input) tuples for tool_use blocks.
        text_blocks: List of text strings for text blocks.
        input_tokens: Simulated input token count.
        output_tokens: Simulated output token count.
    """
    blocks = []

    if tool_uses:
        for name, inp in tool_uses:
            block = SimpleNamespace(type="tool_use", name=name, input=inp)
            blocks.append(block)

    if text_blocks:
        for text in text_blocks:
            block = SimpleNamespace(type="text", text=text)
            blocks.append(block)

    if not blocks:
        blocks.append(SimpleNamespace(type="text", text="No action needed."))

    msg = SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )

    if decision_dict:
        msg.parsed = ManagerDecision(**decision_dict)
    else:
        msg.parsed = None

    return msg


def make_no_action_message(**kwargs):
    """Create a mock message with a NO_ACTION decision."""
    return make_mock_message(
        decision_dict={
            "decision": "NO_ACTION",
            "action_details": "No strategic intervention needed",
            "confidence": 0.95,
            "reasoning": "Standard at-bat, low leverage",
            "key_factors": ["low leverage"],
            "risks": [],
        },
        text_blocks=["No action needed."],
        **kwargs,
    )


def make_active_decision_message(decision_type="PITCHING_CHANGE",
                                 action_details="Bring in Greg Foster",
                                 tool_uses=None, **kwargs):
    """Create a mock message with an active decision."""
    return make_mock_message(
        decision_dict={
            "decision": decision_type,
            "action_details": action_details,
            "confidence": 0.80,
            "reasoning": "Pitcher showing fatigue signs",
            "key_factors": ["velocity decline", "high pitch count"],
            "risks": ["reliever may struggle"],
            "win_probability_before": 0.45,
            "win_probability_after_expected": 0.52,
        },
        tool_uses=tool_uses,
        text_blocks=["Making a pitching change."],
        **kwargs,
    )


def make_mock_runner(messages_to_yield):
    """Create a mock tool_runner that yields the given messages."""
    return iter(messages_to_yield)


def make_mock_client(messages_sequence):
    """Create a mock Anthropic client that returns messages in sequence.

    Args:
        messages_sequence: List of lists-of-messages. Each outer list element
            is one call to tool_runner; each inner list is yielded by the runner.
    """
    client = MagicMock()
    call_count = [0]

    def fake_tool_runner(**kwargs):
        idx = min(call_count[0], len(messages_sequence) - 1)
        call_count[0] += 1
        return iter(messages_sequence[idx])

    client.beta.messages.tool_runner = fake_tool_runner
    return client


def make_test_game(seed=42):
    """Create a test game state."""
    engine = SimulationEngine(seed=seed)
    rosters = load_rosters()
    return engine, engine.initialize_game(rosters)


# ===========================================================================
# Step 1: Accept MatchupState, RosterState, OpponentRosterState as input
# ===========================================================================

class TestStep1AcceptGameStateModels:
    """The decision engine accepts parsed game state models as input."""

    def test_run_agent_decision_accepts_game_state(self):
        """run_agent_decision accepts a GameState parameter."""
        from game import run_agent_decision
        sig = inspect.signature(run_agent_decision)
        params = list(sig.parameters.keys())
        assert "game_state" in params

    def test_run_agent_decision_accepts_managed_team(self):
        """run_agent_decision accepts a managed_team parameter."""
        from game import run_agent_decision
        sig = inspect.signature(run_agent_decision)
        params = list(sig.parameters.keys())
        assert "managed_team" in params

    def test_run_agent_decision_accepts_messages(self):
        """run_agent_decision accepts a messages list for conversation context."""
        from game import run_agent_decision
        sig = inspect.signature(run_agent_decision)
        params = list(sig.parameters.keys())
        assert "messages" in params

    def test_game_state_to_scenario_extracts_models(self):
        """game_state_to_scenario extracts matchup, roster, and opponent models."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")

        assert "matchup_state" in scenario
        assert "roster_state" in scenario
        assert "opponent_roster_state" in scenario

        # Matchup state has current batter and pitcher
        ms = scenario["matchup_state"]
        assert "batter" in ms
        assert "pitcher" in ms
        assert "inning" in ms
        assert "outs" in ms

    def test_scenario_includes_matchup_state_fields(self):
        """MatchupState should have all required fields for the agent."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        ms = scenario["matchup_state"]

        required_fields = ["inning", "half", "outs", "count", "runners",
                           "score", "batting_team", "batter", "pitcher",
                           "on_deck_batter"]
        for field in required_fields:
            assert field in ms, f"Missing MatchupState field: {field}"

    def test_scenario_includes_roster_state_fields(self):
        """RosterState should have lineup, bench, bullpen, and management fields."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        rs = scenario["roster_state"]

        required_fields = ["our_lineup", "our_lineup_position", "bench",
                           "bullpen", "mound_visits_remaining", "challenge_available"]
        for field in required_fields:
            assert field in rs, f"Missing RosterState field: {field}"

    def test_scenario_includes_opponent_roster_fields(self):
        """OpponentRosterState should have lineup, bench, and bullpen."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        ors = scenario["opponent_roster_state"]

        required_fields = ["their_lineup", "their_lineup_position",
                           "their_bench", "their_bullpen"]
        for field in required_fields:
            assert field in ors, f"Missing OpponentRosterState field: {field}"

    def test_run_agent_decision_converts_game_state_to_scenario(self):
        """run_agent_decision internally calls game_state_to_scenario."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "game_state_to_scenario" in source

    def test_scenario_for_home_team(self):
        """Scenario for home team should have home roster as ours."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        rs = scenario["roster_state"]
        ors = scenario["opponent_roster_state"]

        # Home team lineup should be in our_lineup
        assert rs["our_lineup"][0]["player_id"] == game.home.lineup[0].player_id
        # Away team should be in their_lineup
        assert ors["their_lineup"][0]["player_id"] == game.away.lineup[0].player_id

    def test_scenario_for_away_team(self):
        """Scenario for away team should have away roster as ours."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "away")
        rs = scenario["roster_state"]
        ors = scenario["opponent_roster_state"]

        # Away team lineup should be in our_lineup
        assert rs["our_lineup"][0]["player_id"] == game.away.lineup[0].player_id
        # Home team should be in their_lineup
        assert ors["their_lineup"][0]["player_id"] == game.home.lineup[0].player_id


# ===========================================================================
# Step 2: Format game state as natural-language scenario description
# ===========================================================================

class TestStep2FormatScenarioDescription:
    """The game state is formatted as a natural-language scenario for the agent."""

    def test_scenario_has_decision_prompt(self):
        """Scenario should include a natural-language decision prompt."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        assert "decision_prompt" in scenario
        prompt = scenario["decision_prompt"]
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_decision_prompt_describes_situation(self):
        """Decision prompt should describe the current game situation."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        prompt = scenario["decision_prompt"]
        # Should mention inning or game context
        assert "1st" in prompt or "Top" in prompt or "inning" in prompt.lower()

    def test_decision_prompt_indicates_batting_fielding(self):
        """Decision prompt should indicate whether team is batting or fielding."""
        engine, game = make_test_game()
        # Top of 1st: home is fielding
        home_scenario = game_state_to_scenario(game, "home")
        assert "FIELDING" in home_scenario["decision_prompt"]

        # Away is batting
        away_scenario = game_state_to_scenario(game, "away")
        assert "BATTING" in away_scenario["decision_prompt"]

    def test_user_message_includes_all_scenario_parts(self):
        """run_agent_decision formats user message with all scenario parts."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "matchup_state" in source
        assert "roster_state" in source
        assert "opponent_roster_state" in source
        assert "decision_prompt" in source

    def test_user_message_formatted_as_json(self):
        """The game state should be formatted as JSON in the user message."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "json.dumps" in source

    def test_scenario_is_json_serializable(self):
        """The entire scenario should be JSON-serializable."""
        engine, game = make_test_game()
        scenario = game_state_to_scenario(game, "home")
        json_str = json.dumps(scenario, default=str)
        assert len(json_str) > 200
        parsed = json.loads(json_str)
        assert parsed["matchup_state"]["inning"] == 1

    def test_user_message_appended_to_messages(self):
        """run_agent_decision appends the user message to the messages list."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "messages.append" in source
        assert '"role": "user"' in source or "'role': 'user'" in source


# ===========================================================================
# Step 3: Send scenario to Claude agent with all 12 tools registered
# ===========================================================================

class TestStep3SendToClaudeWithTools:
    """The scenario is sent to the Claude agent with all 12 tools registered."""

    def test_call_agent_uses_tool_runner(self):
        """_call_agent must use the SDK's tool_runner for tool-use loops."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "client.beta.messages.tool_runner" in source

    def test_call_agent_passes_all_tools(self):
        """_call_agent must pass ALL_TOOLS to the SDK."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "tools=ALL_TOOLS" in source

    def test_all_tools_count(self):
        """ALL_TOOLS should contain exactly 12 tools."""
        from tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 12

    def test_call_agent_passes_system_prompt(self):
        """_call_agent must pass the SYSTEM_PROMPT."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "system=SYSTEM_PROMPT" in source

    def test_call_agent_specifies_model(self):
        """_call_agent must specify a Claude model."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "model=" in source

    def test_call_agent_specifies_max_tokens(self):
        """_call_agent must specify max_tokens."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "max_tokens=" in source

    def test_call_agent_uses_structured_output(self):
        """_call_agent must request structured output as ManagerDecision."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "output_format=ManagerDecision" in source

    def test_system_prompt_loaded_from_file(self):
        """SYSTEM_PROMPT should be loaded from AGENT_PROMPT.md at module level."""
        from game import SYSTEM_PROMPT, load_system_prompt
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100
        # Verify it matches the file
        prompt_path = PROJECT_ROOT / "AGENT_PROMPT.md"
        assert prompt_path.exists()
        file_content = prompt_path.read_text()
        assert SYSTEM_PROMPT == file_content

    def test_run_agent_decision_calls_agent_with_mock(self):
        """run_agent_decision should call _call_agent and return decision."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        no_action_msg = make_no_action_message()
        client = make_mock_client([[no_action_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, updated_messages, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "NO_ACTION"
        assert len(updated_messages) > 0

    def test_call_agent_iterates_runner(self):
        """_call_agent must iterate over the tool_runner to process all turns."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert "for message in runner" in source


# ===========================================================================
# Step 4: Agent may call zero or more tools
# ===========================================================================

class TestStep4AgentCallsTools:
    """The agent may call zero or more tools to gather analytical context."""

    def test_agent_zero_tool_calls(self):
        """Agent can make a decision without calling any tools."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # Message with no tool_use blocks
        no_tool_msg = make_no_action_message()
        client = make_mock_client([[no_tool_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "NO_ACTION"
        assert metadata["tool_calls"] == []

    def test_agent_single_tool_call(self):
        """Agent can call a single tool before deciding."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # First message: tool call
        tool_msg = make_mock_message(
            tool_uses=[("get_batter_stats", {"player_id": "h_003"})],
            input_tokens=200, output_tokens=100,
        )
        # Second message: decision
        decision_msg = make_no_action_message(input_tokens=300, output_tokens=150)
        client = make_mock_client([[tool_msg, decision_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "NO_ACTION"
        assert len(metadata["tool_calls"]) == 1
        assert metadata["tool_calls"][0]["tool_name"] == "get_batter_stats"

    def test_agent_multiple_tool_calls(self):
        """Agent can call multiple tools across multiple turns."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # Turn 1: two tool calls in one message
        turn1 = make_mock_message(
            tool_uses=[
                ("get_batter_stats", {"player_id": "h_003"}),
                ("get_pitcher_stats", {"player_id": "a_sp1"}),
            ],
            input_tokens=200, output_tokens=100,
        )
        # Turn 2: one more tool call
        turn2 = make_mock_message(
            tool_uses=[("get_matchup_data", {"batter_id": "h_003", "pitcher_id": "a_sp1"})],
            input_tokens=300, output_tokens=150,
        )
        # Turn 3: decision
        turn3 = make_no_action_message(input_tokens=400, output_tokens=200)

        client = make_mock_client([[turn1, turn2, turn3]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert len(metadata["tool_calls"]) == 3
        tool_names = [tc["tool_name"] for tc in metadata["tool_calls"]]
        assert "get_batter_stats" in tool_names
        assert "get_pitcher_stats" in tool_names
        assert "get_matchup_data" in tool_names

    def test_tool_calls_tracked_with_parameters(self):
        """Tool calls should include the input parameters."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        tool_msg = make_mock_message(
            tool_uses=[("get_win_probability", {
                "inning": 7, "half": "BOTTOM", "outs": 1,
                "runner_on_first": False, "runner_on_second": False,
                "runner_on_third": False, "score_differential": -1,
            })],
        )
        decision_msg = make_no_action_message()
        client = make_mock_client([[tool_msg, decision_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert metadata["tool_calls"][0]["tool_input"]["inning"] == 7
        assert metadata["tool_calls"][0]["tool_input"]["half"] == "BOTTOM"

    def test_call_agent_detects_tool_use_blocks(self):
        """_call_agent processes tool_use blocks in message content."""
        from game import _call_agent
        source = inspect.getsource(_call_agent)
        assert 'block.type == "tool_use"' in source
        assert "block.name" in source
        assert "block.input" in source


# ===========================================================================
# Step 5: Agent produces plain-text response (no action or active decision)
# ===========================================================================

class TestStep5AgentProducesDecision:
    """The agent produces either a 'no action' or an active decision with reasoning."""

    def test_no_action_decision(self):
        """Agent can produce a NO_ACTION decision."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "NO_ACTION"
        assert "reasoning" in decision_dict
        assert "confidence" in decision_dict

    def test_active_decision_pitching_change(self):
        """Agent can produce a PITCHING_CHANGE decision."""
        from game import run_agent_decision

        engine, game = make_test_game()
        # Ensure we've faced enough batters for a valid pitching change
        game.home.current_pitcher_batters_faced_this_stint = 4
        messages = []

        msg = make_active_decision_message(
            decision_type="PITCHING_CHANGE",
            action_details="Bring in Greg Foster",
            tool_uses=[("get_pitcher_fatigue_assessment", {"pitcher_id": "h_sp1"})],
        )
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "PITCHING_CHANGE"
        assert decision_dict["action_details"] == "Bring in Greg Foster"
        assert decision_dict["confidence"] > 0

    def test_active_decision_pinch_hit(self):
        """Agent can produce a PINCH_HIT decision."""
        from game import run_agent_decision

        engine, game = make_test_game()
        game.half = "BOTTOM"  # Home team is batting
        messages = []

        bench_player = game.home.bench[0]
        current_batter = game.home.current_batter()

        msg = make_active_decision_message(
            decision_type="PINCH_HIT",
            action_details=f"Send in {bench_player.name} to bat for {current_batter.name}",
        )
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "PINCH_HIT"
        assert bench_player.name in decision_dict["action_details"]

    def test_active_decision_stolen_base(self):
        """Agent can produce a STOLEN_BASE decision."""
        from game import run_agent_decision

        engine, game = make_test_game()
        game.half = "BOTTOM"  # Home team is batting
        # Put a runner on first
        runner = BaseRunner(player=game.home.lineup[0], start_base=1)
        game.runners = [runner]
        messages = []

        msg = make_active_decision_message(
            decision_type="STOLEN_BASE",
            action_details="Send the runner to steal second",
        )
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "STOLEN_BASE"

    def test_decision_includes_reasoning(self):
        """Decision should include reasoning for the agent's choice."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "reasoning" in decision_dict
        assert len(decision_dict["reasoning"]) > 0

    def test_decision_includes_confidence(self):
        """Decision should include a confidence score."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "confidence" in decision_dict
        assert 0.0 <= decision_dict["confidence"] <= 1.0

    def test_decision_includes_key_factors(self):
        """Decision should include key factors that drove it."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "key_factors" in decision_dict
        assert isinstance(decision_dict["key_factors"], list)

    def test_fallback_to_no_action_when_no_parsed(self):
        """If agent returns no structured output, default to NO_ACTION."""
        from game import _call_agent

        # Message with no parsed decision
        msg = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I think we should wait.")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            parsed=None,
        )

        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = _call_agent(client, [], verbose=False)

        assert decision_dict["decision"] == "NO_ACTION"

    def test_no_action_types_recognized(self):
        """All no-action synonyms should be recognized."""
        from game import NO_ACTION_TYPES
        expected = {"NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
                    "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER"}
        for t in expected:
            assert t in NO_ACTION_TYPES


# ===========================================================================
# Step 6: Return agent response with metadata (tools called, token usage)
# ===========================================================================

class TestStep6ReturnMetadata:
    """Return the agent's response along with metadata."""

    def test_metadata_includes_tool_calls(self):
        """Metadata should include the list of tool calls made."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        tool_msg = make_mock_message(
            tool_uses=[("get_batter_stats", {"player_id": "h_003"})],
        )
        decision_msg = make_no_action_message()
        client = make_mock_client([[tool_msg, decision_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "tool_calls" in metadata
        assert len(metadata["tool_calls"]) == 1
        assert metadata["tool_calls"][0]["tool_name"] == "get_batter_stats"
        assert metadata["tool_calls"][0]["tool_input"]["player_id"] == "h_003"

    def test_metadata_includes_token_usage(self):
        """Metadata should include token usage information."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message(input_tokens=500, output_tokens=200)
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "token_usage" in metadata
        usage = metadata["token_usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage
        assert usage["input_tokens"] == 500
        assert usage["output_tokens"] == 200
        assert usage["total_tokens"] == 700

    def test_metadata_accumulates_tokens_across_turns(self):
        """Token usage should be accumulated across multiple agent turns."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        turn1 = make_mock_message(
            tool_uses=[("get_batter_stats", {"player_id": "h_003"})],
            input_tokens=200, output_tokens=100,
        )
        turn2 = make_no_action_message(input_tokens=300, output_tokens=150)
        client = make_mock_client([[turn1, turn2]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert metadata["token_usage"]["input_tokens"] == 500
        assert metadata["token_usage"]["output_tokens"] == 250
        assert metadata["token_usage"]["total_tokens"] == 750

    def test_metadata_includes_latency(self):
        """Metadata should include latency in milliseconds."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "latency_ms" in metadata
        assert isinstance(metadata["latency_ms"], (int, float))
        assert metadata["latency_ms"] >= 0

    def test_metadata_includes_agent_turns(self):
        """Metadata should include the number of agent turns."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        turn1 = make_mock_message(
            tool_uses=[("get_batter_stats", {"player_id": "h_003"})],
        )
        turn2 = make_no_action_message()
        client = make_mock_client([[turn1, turn2]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "agent_turns" in metadata
        assert metadata["agent_turns"] == 2

    def test_metadata_includes_retries(self):
        """Metadata should include the retry count."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert "retries" in metadata
        assert metadata["retries"] == 0

    def test_return_type_is_triple(self):
        """run_agent_decision returns a 3-tuple."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            result = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert len(result) == 3
        decision_dict, updated_messages, metadata = result
        assert isinstance(decision_dict, dict)
        assert isinstance(updated_messages, list)
        assert isinstance(metadata, dict)

    def test_messages_updated_with_conversation(self):
        """Messages list should be updated with the agent conversation."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, updated_messages, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        # Should have at least the user message + assistant response
        assert len(updated_messages) >= 2
        assert updated_messages[0]["role"] == "user"
        assert updated_messages[1]["role"] == "assistant"

    def test_call_agent_return_structure(self):
        """_call_agent returns (decision_dict, final_message, call_metadata)."""
        from game import _call_agent

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            result = _call_agent(client, [], verbose=False)

        assert len(result) == 3
        decision_dict, final_message, call_metadata = result
        assert isinstance(decision_dict, dict)
        assert "tool_calls" in call_metadata
        assert "token_usage" in call_metadata
        assert "agent_turns" in call_metadata


# ===========================================================================
# Step 7: Handle agent errors gracefully with retries and NO_ACTION fallback
# ===========================================================================

class TestStep7ErrorHandlingAndRetries:
    """Handle agent errors: retries on invalid decisions, forced NO_ACTION."""

    def test_invalid_decision_triggers_retry(self):
        """An invalid decision should trigger a retry with error message."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # First call returns invalid: pitching change while batting (away is batting top 1st)
        invalid_msg = make_active_decision_message(
            decision_type="PITCHING_CHANGE",
            action_details="Pull the starter",
        )
        # Second call returns valid NO_ACTION
        valid_msg = make_no_action_message()

        client = make_mock_client([[invalid_msg], [valid_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, updated_messages, metadata = run_agent_decision(
                client, game, "away", messages, verbose=False,
            )

        # Should have retried and got NO_ACTION
        assert decision_dict["decision"] == "NO_ACTION"
        assert metadata["retries"] == 1

    def test_forced_no_action_after_max_retries(self):
        """After max retries, a forced NO_ACTION is returned."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # All calls return invalid decisions (pitching change while batting)
        invalid_msg = make_active_decision_message(
            decision_type="PITCHING_CHANGE",
            action_details="Pull the starter",
        )

        # Create enough invalid responses for max_retries + 1
        all_invalid = [[invalid_msg]] * 10
        client = make_mock_client(all_invalid)

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, metadata = run_agent_decision(
                client, game, "away", messages, verbose=False, max_retries=3,
            )

        # Should be forced NO_ACTION
        assert decision_dict["decision"] == "NO_ACTION"
        assert "forced" in decision_dict["action_details"].lower() or \
               "forced" in decision_dict["reasoning"].lower() or \
               metadata["retries"] >= 3

    def test_agent_call_exception_handled(self):
        """Exceptions from the Claude API should be caught gracefully."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        client = MagicMock()
        client.beta.messages.tool_runner.side_effect = Exception("API connection failed")

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        # Should return NO_ACTION on failure
        assert decision_dict["decision"] == "NO_ACTION"

    def test_retry_exception_handled(self):
        """Exception during retry should not crash."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # First call returns invalid (pitching change while batting)
        invalid_msg = make_active_decision_message(
            decision_type="PITCHING_CHANGE",
            action_details="Pull the starter",
        )

        call_count = [0]

        def fake_tool_runner(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([invalid_msg])
            else:
                raise Exception("API rate limit exceeded")

        client = MagicMock()
        client.beta.messages.tool_runner = fake_tool_runner

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "away", messages, verbose=False, max_retries=2,
            )

        # Should eventually return NO_ACTION
        assert decision_dict["decision"] == "NO_ACTION"

    def test_peek_validate_does_not_mutate_state(self):
        """_peek_validate should check validity without changing game state."""
        from game import _peek_validate

        engine, game = make_test_game()

        # Snapshot
        score_home = game.score_home
        score_away = game.score_away
        inning = game.inning
        outs = game.outs
        pitcher_id = game.home.current_pitcher.player_id

        # Validate various decisions
        _peek_validate(game, {"decision": "NO_ACTION"}, "home")
        _peek_validate(game, {"decision": "PITCHING_CHANGE", "action_details": "change"}, "home")
        _peek_validate(game, {"decision": "STOLEN_BASE"}, "away")
        _peek_validate(game, {"decision": "PINCH_HIT", "action_details": "send in someone"}, "away")

        # Verify no mutation
        assert game.score_home == score_home
        assert game.score_away == score_away
        assert game.inning == inning
        assert game.outs == outs
        assert game.home.current_pitcher.player_id == pitcher_id

    def test_peek_validate_no_action_always_valid(self):
        """_peek_validate should accept NO_ACTION as always valid."""
        from game import _peek_validate

        engine, game = make_test_game()

        for dec_type in ["NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
                         "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER"]:
            result = _peek_validate(game, {"decision": dec_type}, "home")
            assert result.valid, f"{dec_type} should be valid"

    def test_peek_validate_catches_batting_fielding_errors(self):
        """_peek_validate catches decisions inappropriate for batting/fielding state."""
        from game import _peek_validate

        engine, game = make_test_game()

        # Top of 1st: away bats, home fields
        # PITCHING_CHANGE while batting (away) should be invalid
        result = _peek_validate(
            game, {"decision": "PITCHING_CHANGE", "action_details": "change"}, "away",
        )
        assert not result.valid
        assert "batting" in result.error.lower()

        # PINCH_HIT while fielding (home) should be invalid
        result = _peek_validate(
            game, {"decision": "PINCH_HIT", "action_details": "hit"}, "home",
        )
        assert not result.valid
        assert "fielding" in result.error.lower()

    def test_peek_validate_catches_3_batter_minimum(self):
        """_peek_validate catches 3-batter minimum violations."""
        from game import _peek_validate

        engine, game = make_test_game()
        game.home.current_pitcher_batters_faced_this_stint = 1

        result = _peek_validate(
            game,
            {"decision": "PITCHING_CHANGE", "action_details": "bring in reliever"},
            "home",
        )
        assert not result.valid
        assert "3-batter" in result.error.lower()

    def test_peek_validate_accepts_valid_pitching_change(self):
        """_peek_validate accepts a valid pitching change after 3 batters."""
        from game import _peek_validate

        engine, game = make_test_game()
        game.home.current_pitcher_batters_faced_this_stint = 3

        result = _peek_validate(
            game,
            {"decision": "PITCHING_CHANGE", "action_details": "bring in Greg Foster"},
            "home",
        )
        assert result.valid

    def test_peek_validate_catches_no_eligible_runner(self):
        """_peek_validate catches stolen base with no eligible runner."""
        from game import _peek_validate

        engine, game = make_test_game()
        game.half = "BOTTOM"
        game.runners = []

        result = _peek_validate(
            game, {"decision": "STOLEN_BASE", "action_details": "steal"}, "home",
        )
        assert not result.valid
        assert "no eligible runner" in result.error.lower()

    def test_peek_validate_catches_no_mound_visits(self):
        """_peek_validate catches mound visit with no visits remaining."""
        from game import _peek_validate

        engine, game = make_test_game()
        game.home.mound_visits_remaining = 0

        result = _peek_validate(
            game, {"decision": "MOUND_VISIT"}, "home",
        )
        assert not result.valid
        assert "no mound visits" in result.error.lower()

    def test_peek_validate_catches_no_challenge(self):
        """_peek_validate catches replay challenge with no challenge available."""
        from game import _peek_validate

        engine, game = make_test_game()
        game.home.challenge_available = False

        result = _peek_validate(
            game, {"decision": "REPLAY_CHALLENGE"}, "home",
        )
        assert not result.valid
        assert "no challenge" in result.error.lower()

    def test_retry_sends_error_back_to_agent(self):
        """The retry loop sends the validation error back to the agent."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        # Invalid then valid
        invalid_msg = make_active_decision_message(
            decision_type="PITCHING_CHANGE",
            action_details="Pull the starter",
        )
        valid_msg = make_no_action_message()
        client = make_mock_client([[invalid_msg], [valid_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            _, updated_messages, _ = run_agent_decision(
                client, game, "away", messages, verbose=False,
            )

        # Should have error feedback in messages
        error_messages = [m for m in updated_messages
                          if m.get("role") == "user" and "INVALID" in m.get("content", "")]
        assert len(error_messages) >= 1

    def test_max_retries_parameter(self):
        """run_agent_decision has a max_retries parameter with a reasonable default."""
        from game import run_agent_decision
        sig = inspect.signature(run_agent_decision)
        assert "max_retries" in sig.parameters
        default = sig.parameters["max_retries"].default
        assert isinstance(default, int)
        assert default >= 3

    def test_run_agent_game_wraps_agent_errors(self):
        """run_agent_game catches agent call failures and logs errors."""
        from game import run_agent_game
        source = inspect.getsource(run_agent_game)
        assert "except Exception" in source
        assert "error_log" in source


# ===========================================================================
# Integration tests: Decision engine as complete pipeline
# ===========================================================================

class TestIntegrationDecisionPipeline:
    """Integration tests for the complete decision engine pipeline."""

    def test_full_pipeline_no_action(self):
        """Full pipeline: game state -> scenario -> agent -> NO_ACTION."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, updated_messages, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        # Decision
        assert decision_dict["decision"] == "NO_ACTION"
        assert decision_dict["confidence"] > 0
        assert len(decision_dict["reasoning"]) > 0

        # Messages updated
        assert len(updated_messages) >= 2

        # Metadata complete
        assert "tool_calls" in metadata
        assert "token_usage" in metadata
        assert "latency_ms" in metadata
        assert "agent_turns" in metadata
        assert "retries" in metadata

    def test_full_pipeline_active_decision_with_tools(self):
        """Full pipeline: game state -> scenario -> tools -> active decision."""
        from game import run_agent_decision

        engine, game = make_test_game()
        game.home.current_pitcher_batters_faced_this_stint = 5
        messages = []

        # Agent calls tools then makes a decision
        tool_msg = make_mock_message(
            tool_uses=[
                ("get_pitcher_fatigue_assessment", {"pitcher_id": "h_sp1"}),
                ("get_bullpen_status", {}),
            ],
            input_tokens=300, output_tokens=200,
        )
        decision_msg = make_active_decision_message(
            decision_type="PITCHING_CHANGE",
            action_details="Bring in Greg Foster",
            input_tokens=400, output_tokens=300,
        )
        client = make_mock_client([[tool_msg, decision_msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        assert decision_dict["decision"] == "PITCHING_CHANGE"
        assert len(metadata["tool_calls"]) == 2
        assert metadata["token_usage"]["total_tokens"] == 1200
        assert metadata["agent_turns"] == 2

    def test_decision_validated_after_agent_returns(self):
        """Decision from agent goes through _peek_validate before returning."""
        from game import run_agent_decision
        source = inspect.getsource(run_agent_decision)
        assert "_peek_validate" in source

    def test_decision_log_entry_from_pipeline(self):
        """build_decision_log_entry captures complete pipeline output."""
        from game import build_decision_log_entry, run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message(input_tokens=500, output_tokens=200)
        client = make_mock_client([[msg]])

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, metadata = run_agent_decision(
                client, game, "home", messages, verbose=False,
            )

        entry = build_decision_log_entry(
            turn=1,
            game_state=game,
            managed_team="home",
            decision_dict=decision_dict,
            decision_metadata=metadata,
            timestamp=time.time(),
        )

        assert entry["turn"] == 1
        assert entry["managed_team"] == "home"
        assert "game_state" in entry
        assert entry["game_state"]["inning"] == 1
        assert "tool_calls" in entry
        assert "decision" in entry
        assert entry["decision"]["decision"] == "NO_ACTION"
        assert "is_active_decision" in entry
        assert entry["is_active_decision"] is False
        assert "token_usage" in entry
        assert entry["token_usage"]["input_tokens"] == 500
        assert "latency_ms" in entry

    def test_active_decision_logged_as_active(self):
        """Active decisions are marked as is_active_decision=True in logs."""
        from game import build_decision_log_entry

        engine, game = make_test_game()

        entry = build_decision_log_entry(
            turn=1,
            game_state=game,
            managed_team="home",
            decision_dict={
                "decision": "PITCHING_CHANGE",
                "action_details": "Bring in reliever",
                "confidence": 0.8,
                "reasoning": "Pitcher is tired",
            },
            decision_metadata={
                "tool_calls": [{"tool_name": "get_pitcher_fatigue_assessment",
                                "tool_input": {"pitcher_id": "h_sp1"}}],
                "token_usage": {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700},
                "latency_ms": 1500,
                "agent_turns": 2,
                "retries": 0,
            },
            timestamp=time.time(),
        )

        assert entry["is_active_decision"] is True

    def test_game_log_aggregates_decisions(self):
        """write_game_log aggregates multiple decisions into a game log file."""
        from game import write_game_log, build_decision_log_entry
        import tempfile

        engine, game = make_test_game()

        # Build a few decision entries
        entries = []
        for i in range(3):
            entry = build_decision_log_entry(
                turn=i + 1,
                game_state=game,
                managed_team="home",
                decision_dict={
                    "decision": "NO_ACTION" if i < 2 else "PITCHING_CHANGE",
                    "action_details": "test",
                    "confidence": 0.9,
                    "reasoning": "test",
                },
                decision_metadata={
                    "tool_calls": [],
                    "token_usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                    "latency_ms": 500,
                    "agent_turns": 1,
                    "retries": 0,
                },
                timestamp=time.time(),
            )
            entries.append(entry)

        game.game_over = True
        game.winning_team = "Home Team"

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = write_game_log(
                game_state=game,
                decision_log=entries,
                error_log=[],
                seed=42,
                managed_team="home",
                log_dir=Path(tmpdir),
            )
            assert log_path.exists()
            data = json.loads(log_path.read_text())

            assert data["summary"]["total_decisions"] == 3
            assert data["summary"]["active_decisions"] == 1
            assert data["summary"]["no_action_decisions"] == 2
            assert data["summary"]["total_input_tokens"] == 300
            assert data["summary"]["total_output_tokens"] == 150
            assert data["summary"]["total_tokens"] == 450
            assert data["summary"]["total_latency_ms"] == 1500

    def test_multiple_decision_points_in_game(self):
        """The decision engine can be called at multiple decision points."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]] * 5)

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            for i in range(5):
                decision_dict, messages, metadata = run_agent_decision(
                    client, game, "home", messages, verbose=False,
                )
                assert decision_dict["decision"] == "NO_ACTION"

    def test_context_continuity_across_decisions(self):
        """Messages accumulate across decision points for context continuity."""
        from game import run_agent_decision

        engine, game = make_test_game()
        messages = []

        msg = make_no_action_message()
        client = make_mock_client([[msg]] * 3)

        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            for i in range(3):
                _, messages, _ = run_agent_decision(
                    client, game, "home", messages, verbose=False,
                )

        # Messages should have accumulated (user + assistant pairs)
        assert len(messages) >= 6  # 3 decisions * 2 messages each

    def test_decision_engine_with_different_managed_teams(self):
        """Decision engine works for both home and away managed teams."""
        from game import run_agent_decision

        engine, game = make_test_game()

        msg = make_no_action_message()

        # Home team
        client = make_mock_client([[msg]])
        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "home", [], verbose=False,
            )
        assert decision_dict["decision"] == "NO_ACTION"

        # Away team
        client = make_mock_client([[msg]])
        with patch("game.SYSTEM_PROMPT", "Test prompt"):
            decision_dict, _, _ = run_agent_decision(
                client, game, "away", [], verbose=False,
            )
        assert decision_dict["decision"] == "NO_ACTION"

    def test_build_sample_scenario_is_valid(self):
        """build_sample_scenario creates a valid scenario dict."""
        from game import build_sample_scenario
        scenario = build_sample_scenario()

        assert "matchup_state" in scenario
        assert "roster_state" in scenario
        assert "opponent_roster_state" in scenario
        assert "decision_prompt" in scenario

        ms = scenario["matchup_state"]
        assert ms["inning"] == 7
        assert ms["half"] == "BOTTOM"
        assert ms["outs"] == 1

    def test_dry_run_validates_complete_setup(self):
        """Dry-run validates the complete agent setup without API calls."""
        from game import run_dry_run
        # Should complete without errors
        run_dry_run()


# ===========================================================================
# Main test runner
# ===========================================================================

def run_all_tests():
    print("=" * 72)
    print("AGENT DECISION ENGINE TESTS")
    print("=" * 72)

    test_classes = [
        ("Step 1: Accept Game State Models", TestStep1AcceptGameStateModels),
        ("Step 2: Format Scenario Description", TestStep2FormatScenarioDescription),
        ("Step 3: Send to Claude with Tools", TestStep3SendToClaudeWithTools),
        ("Step 4: Agent Calls Tools", TestStep4AgentCallsTools),
        ("Step 5: Agent Produces Decision", TestStep5AgentProducesDecision),
        ("Step 6: Return Metadata", TestStep6ReturnMetadata),
        ("Step 7: Error Handling and Retries", TestStep7ErrorHandlingAndRetries),
        ("Integration: Decision Pipeline", TestIntegrationDecisionPipeline),
    ]

    passed = 0
    failed = 0
    failures = []

    for category, cls in test_classes:
        print(f"\n[{category}]")
        instance = cls()
        for method_name in sorted(dir(instance)):
            if not method_name.startswith("test_"):
                continue
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  {method_name}: PASSED")
            except Exception as e:
                failed += 1
                failures.append((method_name, str(e)))
                print(f"  {method_name}: FAILED - {e}")

    print(f"\n{'=' * 72}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    if failures:
        print(f"\nFAILURES:")
        for name, err in failures:
            print(f"  {name}: {err}")
    print(f"{'=' * 72}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
