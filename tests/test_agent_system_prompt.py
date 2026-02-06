# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the agent_system_prompt feature.

Validates that the system prompt:
  1. Establishes the agent's identity as the manager of a specific MLB team
  2. Lists all 12 available tools with brief descriptions
  3. Instructs the agent to assess the situation before calling tools
  4. Embeds key analytical constants (TTO penalty, platoon splits, breakeven rates, LI thresholds)
  5. Includes all MLB rules that constrain decisions
  6. Instructs that output is plain-text describing the decision, suitable for tweeting
  7. Explains that most at-bats require no action
  8. Notes that players removed from the game cannot return
  9. Stores the system prompt in AGENT_PROMPT.md loaded at runtime
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_PROMPT_PATH = PROJECT_ROOT / "AGENT_PROMPT.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prompt_text() -> str:
    """Load the AGENT_PROMPT.md file once for all tests."""
    return AGENT_PROMPT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def system_prompt() -> str:
    """Load SYSTEM_PROMPT from game.py to verify runtime loading."""
    from game import SYSTEM_PROMPT
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Step 1: Agent identity as manager of a specific MLB team
# ---------------------------------------------------------------------------

class TestStep1AgentIdentity:
    """The system prompt must establish the agent's identity as a baseball manager."""

    def test_prompt_file_exists(self):
        assert AGENT_PROMPT_PATH.exists(), "AGENT_PROMPT.md must exist in project root"

    def test_prompt_is_not_empty(self, prompt_text):
        assert len(prompt_text.strip()) > 500, "System prompt should be substantial"

    def test_prompt_establishes_manager_identity(self, prompt_text):
        text = prompt_text.lower()
        assert "manager" in text, "Prompt must establish agent as a manager"

    def test_prompt_mentions_mlb(self, prompt_text):
        assert "MLB" in prompt_text, "Prompt must mention MLB"

    def test_prompt_mentions_team(self, prompt_text):
        text = prompt_text.lower()
        assert "team" in text, "Prompt must reference a team"

    def test_prompt_describes_game_state_input(self, prompt_text):
        assert "MatchupState" in prompt_text, "Prompt must describe MatchupState input"
        assert "RosterState" in prompt_text, "Prompt must describe RosterState input"
        assert "OpponentRosterState" in prompt_text, "Prompt must describe OpponentRosterState input"


# ---------------------------------------------------------------------------
# Step 2: Lists all 12 available tools with descriptions
# ---------------------------------------------------------------------------

class TestStep2ToolDescriptions:
    """The prompt must list all 12 tools with brief descriptions."""

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

    @pytest.mark.parametrize("tool_name", EXPECTED_TOOLS)
    def test_tool_mentioned_in_prompt(self, prompt_text, tool_name):
        assert tool_name in prompt_text, f"Tool '{tool_name}' must be listed in the system prompt"

    def test_all_12_tools_present(self, prompt_text):
        count = sum(1 for t in self.EXPECTED_TOOLS if t in prompt_text)
        assert count == 12, f"All 12 tools must be in the prompt, found {count}"

    def test_tool_descriptions_present(self, prompt_text):
        """Each tool should have some descriptive text near it."""
        # Check that each tool name is followed by descriptive content
        for tool in self.EXPECTED_TOOLS:
            idx = prompt_text.find(tool)
            assert idx != -1, f"Tool {tool} not found in prompt"
            # The surrounding context should have descriptive words
            surrounding = prompt_text[idx:idx + 300]
            assert len(surrounding) > len(tool) + 20, \
                f"Tool {tool} should have a description"


# ---------------------------------------------------------------------------
# Step 3: Instructs agent to assess situation before calling tools
# ---------------------------------------------------------------------------

class TestStep3SituationAssessment:
    """The prompt must instruct the agent to assess the situation
    and determine whether action is needed before calling tools."""

    def test_instructs_to_assess_situation(self, prompt_text):
        text = prompt_text.lower()
        # The prompt should tell the agent to evaluate/assess before deciding
        has_assess = any(word in text for word in [
            "assess", "evaluate", "analyze", "gather data before",
            "build analytical context",
        ])
        assert has_assess, "Prompt must instruct agent to assess the situation"

    def test_mentions_decision_framework(self, prompt_text):
        text = prompt_text.lower()
        # Should mention that the agent should use tools for data
        has_tools_instruction = any(phrase in text for phrase in [
            "use tools", "use them to", "call tool", "look up",
            "gather", "information-gathering",
        ])
        assert has_tools_instruction, "Prompt must instruct agent to use tools for information"


# ---------------------------------------------------------------------------
# Step 4: Embeds key analytical constants
# ---------------------------------------------------------------------------

class TestStep4AnalyticalConstants:
    """The prompt must embed key analytical constants: TTO penalty,
    platoon splits, breakeven rates, and leverage index thresholds."""

    def test_tto_penalty_present(self, prompt_text):
        text = prompt_text.lower()
        # TTO = Times Through Order
        has_tto = "times through" in text or "tto" in text
        assert has_tto, "Prompt must include Times Through Order (TTO) penalty data"

    def test_tto_woba_values(self, prompt_text):
        # Should contain specific wOBA increase values for TTO
        assert "wOBA" in prompt_text or "woba" in prompt_text.lower(), \
            "Prompt must reference wOBA metric"
        # Check for the specific TTO values from DESIGN.md
        assert "+0.010" in prompt_text or "+0.015" in prompt_text, \
            "Prompt must include 2nd TTO wOBA penalty values"
        assert "+0.020" in prompt_text or "+0.035" in prompt_text, \
            "Prompt must include 3rd TTO wOBA penalty values"

    def test_platoon_splits_present(self, prompt_text):
        text = prompt_text.lower()
        assert "platoon" in text, "Prompt must discuss platoon advantages"

    def test_platoon_woba_values(self, prompt_text):
        # The platoon gap should be quantified
        has_platoon_data = "0.030" in prompt_text or "0.040" in prompt_text
        assert has_platoon_data, "Prompt must include platoon wOBA gap values (~0.030-0.040)"

    def test_platoon_same_hand_disadvantage(self, prompt_text):
        text = prompt_text.lower()
        assert "same-hand" in text or "same hand" in text, \
            "Prompt must describe same-hand matchup disadvantage"

    def test_stolen_base_breakeven_rates(self, prompt_text):
        text = prompt_text.lower()
        assert "breakeven" in text or "break-even" in text or "break even" in text, \
            "Prompt must include stolen base breakeven rates"

    def test_breakeven_specific_values(self, prompt_text):
        # Check for specific breakeven values from DESIGN.md
        has_values = "71.5" in prompt_text or "74.4" in prompt_text or "77.5" in prompt_text
        assert has_values, "Prompt must include specific breakeven rate values (71.5%, 74.4%, 77.5%)"

    def test_leverage_index_thresholds(self, prompt_text):
        text = prompt_text.lower()
        assert "leverage" in text, "Prompt must discuss leverage index"

    def test_leverage_index_specific_thresholds(self, prompt_text):
        # Check for the specific LI threshold values from DESIGN.md
        has_li_thresholds = all(val in prompt_text for val in ["0.5", "1.5", "2.5"])
        assert has_li_thresholds, "Prompt must include LI threshold values (0.5, 1.5, 2.5)"

    def test_leverage_index_reliever_mapping(self, prompt_text):
        text = prompt_text.lower()
        # Should map LI to reliever quality
        has_reliever_mapping = any(phrase in text for phrase in [
            "setup", "middle reliever", "closer", "best available",
            "weakest available", "mop-up",
        ])
        assert has_reliever_mapping, "Prompt must map LI thresholds to reliever deployment"

    def test_matchup_projection_formula(self, prompt_text):
        # Should include the odds ratio formula
        has_odds_ratio = "P_batter" in prompt_text or "P_league" in prompt_text or "odds ratio" in prompt_text.lower()
        assert has_odds_ratio, "Prompt should include the matchup projection (odds ratio) formula"


# ---------------------------------------------------------------------------
# Step 5: Includes all MLB rules that constrain decisions
# ---------------------------------------------------------------------------

class TestStep5MLBRules:
    """The prompt must include all MLB rules that constrain decisions."""

    def test_three_batter_minimum(self, prompt_text):
        text = prompt_text.lower()
        assert "3-batter" in text or "three-batter" in text or "3 batter" in text, \
            "Prompt must include 3-batter minimum rule"

    def test_runner_on_second_extras(self, prompt_text):
        text = prompt_text.lower()
        has_rule = ("10th inning" in text or "extra inning" in text or "extras" in text) and \
                   ("runner" in text and "2nd" in text or "second" in text)
        assert has_rule, "Prompt must include runner on 2nd in extras rule"

    def test_mound_visit_limit(self, prompt_text):
        text = prompt_text.lower()
        assert "mound visit" in text, "Prompt must include mound visit rules"
        assert "5" in prompt_text, "Prompt must mention 5 mound visits per game"

    def test_pickoff_limit(self, prompt_text):
        text = prompt_text.lower()
        has_pickoff = "pickoff" in text or "disengagement" in text
        assert has_pickoff, "Prompt must include pickoff/disengagement limit"

    def test_infield_positioning_rule(self, prompt_text):
        text = prompt_text.lower()
        has_positioning = "2 infielders" in text or "infield" in text
        assert has_positioning, "Prompt must include infield positioning rules"

    def test_universal_dh(self, prompt_text):
        text = prompt_text.lower()
        assert "dh" in text or "designated hitter" in text, \
            "Prompt must mention Universal DH rule"

    def test_replay_challenge_rule(self, prompt_text):
        text = prompt_text.lower()
        assert "challenge" in text or "replay" in text, \
            "Prompt must include replay challenge rules"

    def test_pitch_timer(self, prompt_text):
        text = prompt_text.lower()
        has_timer = "pitch timer" in text or "pitch clock" in text or \
                    ("15 second" in text and "20 second" in text)
        assert has_timer, "Prompt must include pitch timer rules"


# ---------------------------------------------------------------------------
# Step 6: Output is plain-text for tweeting
# ---------------------------------------------------------------------------

class TestStep6OutputFormat:
    """The prompt must instruct that output is plain-text suitable for tweeting."""

    def test_output_format_described(self, prompt_text):
        text = prompt_text.lower()
        # The prompt should describe the output format
        has_output = any(phrase in text for phrase in [
            "output", "response", "decision", "managerdecision",
        ])
        assert has_output, "Prompt must describe the expected output format"

    def test_decision_types_listed(self, prompt_text):
        # Should list the main decision types
        decision_types = ["NO_ACTION", "PITCHING_CHANGE", "PINCH_HIT", "STOLEN_BASE"]
        found = sum(1 for dt in decision_types if dt in prompt_text)
        assert found >= 3, "Prompt must list the main decision types"

    def test_output_includes_reasoning(self, prompt_text):
        text = prompt_text.lower()
        assert "reasoning" in text, "Prompt must mention that reasoning is included in output"

    def test_output_includes_confidence(self, prompt_text):
        text = prompt_text.lower()
        assert "confidence" in text, "Prompt must mention confidence in the output"


# ---------------------------------------------------------------------------
# Step 7: Most at-bats require no action
# ---------------------------------------------------------------------------

class TestStep7NoActionDefault:
    """The prompt must explain that most at-bats require no action
    and the agent should only make active decisions when warranted."""

    def test_most_at_bats_no_action(self, prompt_text):
        text = prompt_text.lower()
        has_no_action_guidance = any(phrase in text for phrase in [
            "most at-bats require no action",
            "most at-bats require no strategic",
            "majority of at-bats",
            "no action",
        ])
        assert has_no_action_guidance, "Prompt must state that most at-bats require no action"

    def test_only_decide_when_warranted(self, prompt_text):
        text = prompt_text.lower()
        has_guidance = any(phrase in text for phrase in [
            "when the situation warrants",
            "when the situation clearly",
            "only make active decisions when",
            "do not overthink",
            "routine situations",
        ])
        assert has_guidance, "Prompt must instruct agent to only act when situation warrants it"


# ---------------------------------------------------------------------------
# Step 8: Players removed cannot return
# ---------------------------------------------------------------------------

class TestStep8PlayersRemovedCannotReturn:
    """The prompt must note that players removed from the game cannot return."""

    def test_removed_players_rule(self, prompt_text):
        text = prompt_text.lower()
        has_rule = any(phrase in text for phrase in [
            "removed from the game cannot return",
            "removed cannot return",
            "cannot re-enter",
            "cannot reenter",
            "substituted out of the game",
        ])
        assert has_rule, "Prompt must state that removed players cannot return to the game"


# ---------------------------------------------------------------------------
# Step 9: System prompt stored in AGENT_PROMPT.md and loaded at runtime
# ---------------------------------------------------------------------------

class TestStep9RuntimeLoading:
    """The system prompt must be stored in AGENT_PROMPT.md and loaded at runtime."""

    def test_agent_prompt_md_exists(self):
        assert AGENT_PROMPT_PATH.exists(), "AGENT_PROMPT.md must exist"

    def test_agent_prompt_md_is_markdown(self):
        assert AGENT_PROMPT_PATH.suffix == ".md", "Prompt file must be a markdown file"

    def test_game_py_loads_from_file(self):
        """game.py must load the system prompt from AGENT_PROMPT.md, not hardcode it."""
        game_text = (PROJECT_ROOT / "game.py").read_text()
        assert "AGENT_PROMPT.md" in game_text, \
            "game.py must reference AGENT_PROMPT.md"

    def test_game_py_has_load_function(self):
        """game.py must have a load_system_prompt function."""
        from game import load_system_prompt
        assert callable(load_system_prompt)

    def test_load_system_prompt_returns_string(self):
        from game import load_system_prompt
        result = load_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 500

    def test_system_prompt_matches_file(self, prompt_text, system_prompt):
        """SYSTEM_PROMPT in game.py must match the contents of AGENT_PROMPT.md."""
        assert system_prompt == prompt_text, \
            "SYSTEM_PROMPT loaded at runtime must match AGENT_PROMPT.md contents"

    def test_system_prompt_used_in_agent_call(self):
        """The SYSTEM_PROMPT must be used when calling the Claude agent."""
        game_text = (PROJECT_ROOT / "game.py").read_text()
        assert "system=SYSTEM_PROMPT" in game_text, \
            "game.py must pass SYSTEM_PROMPT as the system parameter to the agent"

    def test_load_system_prompt_accepts_custom_path(self, tmp_path):
        """load_system_prompt should accept a custom path for testing."""
        from game import load_system_prompt
        custom_prompt = "You are a test prompt."
        custom_path = tmp_path / "test_prompt.md"
        custom_path.write_text(custom_prompt, encoding="utf-8")
        result = load_system_prompt(path=custom_path)
        assert result == custom_prompt

    def test_load_system_prompt_raises_on_missing_file(self, tmp_path):
        """load_system_prompt should raise FileNotFoundError for missing file."""
        from game import load_system_prompt
        missing_path = tmp_path / "nonexistent.md"
        with pytest.raises(FileNotFoundError):
            load_system_prompt(path=missing_path)


# ---------------------------------------------------------------------------
# Integration: Verify dry-run still passes with the new system prompt
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests verifying the system prompt works with game.py."""

    def test_dry_run_succeeds_with_loaded_prompt(self):
        import subprocess
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"Dry run failed:\n{result.stderr}"
        assert "ALL VALIDATIONS PASSED" in result.stdout

    def test_system_prompt_length_in_dry_run(self):
        """Dry run should report the system prompt length."""
        import subprocess
        result = subprocess.run(
            ["uv", "run", str(PROJECT_ROOT / "game.py"), "--dry-run"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert "System prompt:" in result.stdout
        # The loaded prompt should be substantially longer than the old hardcoded one
        # Extract the character count from the output
        for line in result.stdout.splitlines():
            if "System prompt:" in line:
                # Extract the number (e.g., "System prompt: 5432 chars")
                parts = line.strip().split()
                for part in parts:
                    if part.isdigit():
                        char_count = int(part)
                        assert char_count > 3000, \
                            f"System prompt should be >3000 chars (from AGENT_PROMPT.md), got {char_count}"
                        break
                break
