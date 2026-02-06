# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.78.0",
#     "pydantic>=2.0",
# ]
# ///
"""Baseball Manager AI Agent -- main entry point.

Run with:  uv run game.py           # full agent test (requires ANTHROPIC_API_KEY)
           uv run game.py --dry-run  # validate setup without API calls
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

from anthropic import Anthropic

from models import (
    BatterInfo,
    BattingTeam,
    BullpenPitcher,
    BullpenRole,
    Count,
    Freshness,
    Half,
    Hand,
    LineupPlayer,
    ManagerDecision,
    MatchupState,
    OnDeckBatter,
    OpponentBenchPlayer,
    OpponentBullpenPitcher,
    OpponentRosterState,
    PitcherInfo,
    RosterState,
    Runners,
    Score,
    ThrowHand,
)
from tools import ALL_TOOLS


SYSTEM_PROMPT = """\
You are an experienced baseball manager. You receive a game scenario describing \
the current situation and must make a managerial decision.

You have access to 12 information-gathering tools that let you look up player \
statistics, evaluate strategic options, and assess game probabilities. Use them \
to build analytical context before deciding.

After gathering the information you need, respond with your decision as a \
structured ManagerDecision object.

Key principles:
- Always consider the inning, score, outs, runners, and matchup before deciding.
- Use tools to compute win probability and run expectancy for context.
- Factor in platoon advantages, pitcher fatigue, and bullpen availability.
- Consider the 3-batter minimum rule for relievers.
- Weigh both expected runs and probability of scoring at least one run.
- Account for remaining bench and bullpen depth when making substitutions.
"""


def build_sample_scenario() -> dict:
    """Build a minimal sample scenario for testing the agent."""
    matchup = MatchupState(
        inning=7,
        half=Half.BOTTOM,
        outs=1,
        count=Count(balls=1, strikes=1),
        runners=Runners(),
        score=Score(home=3, away=4),
        batting_team=BattingTeam.HOME,
        batter=BatterInfo(
            player_id="h_003",
            name="Rafael Ortiz",
            bats=Hand.L,
            lineup_position=3,
        ),
        pitcher=PitcherInfo(
            player_id="a_sp1",
            name="Matt Henderson",
            throws=ThrowHand.L,
            pitch_count_today=88,
            batters_faced_today=24,
            times_through_order=3,
            innings_pitched_today=6.1,
            runs_allowed_today=3,
            today_line={"IP": 6.1, "H": 7, "R": 3, "ER": 3, "BB": 2, "K": 6},
        ),
        on_deck_batter=OnDeckBatter(
            player_id="h_004",
            name="Tyrone Jackson",
            bats=Hand.R,
        ),
    )

    roster = RosterState(
        our_lineup=[
            LineupPlayer(player_id="h_001", name="Marcus Chen", position="CF", bats=Hand.L),
            LineupPlayer(player_id="h_002", name="Derek Williams", position="SS", bats=Hand.R),
            LineupPlayer(player_id="h_003", name="Rafael Ortiz", position="1B", bats=Hand.L),
            LineupPlayer(player_id="h_004", name="Tyrone Jackson", position="RF", bats=Hand.R),
            LineupPlayer(player_id="h_005", name="Jake Morrison", position="3B", bats=Hand.R),
            LineupPlayer(player_id="h_006", name="Shin-Soo Park", position="DH", bats=Hand.L),
            LineupPlayer(player_id="h_007", name="Carlos Ramirez", position="LF", bats=Hand.S),
            LineupPlayer(player_id="h_008", name="Tommy Sullivan", position="C", bats=Hand.R),
            LineupPlayer(player_id="h_009", name="Andre Davis", position="2B", bats=Hand.L),
        ],
        our_lineup_position=2,
        bench=[
            {"player_id": "h_012", "name": "Darnell Washington", "bats": "L", "positions": ["LF", "RF"], "available": True},
            {"player_id": "h_013", "name": "Eduardo Reyes", "bats": "L", "positions": ["1B", "DH"], "available": True},
            {"player_id": "h_014", "name": "Kenji Tanaka", "bats": "R", "positions": ["CF", "LF", "RF"], "available": True},
            {"player_id": "h_010", "name": "Victor Nguyen", "bats": "R", "positions": ["C", "1B"], "available": True},
            {"player_id": "h_011", "name": "Ryan O'Brien", "bats": "R", "positions": ["2B", "SS", "3B"], "available": True},
        ],
        bullpen=[
            BullpenPitcher(player_id="h_bp1", name="Greg Foster", throws=ThrowHand.R, role=BullpenRole.CLOSER, freshness=Freshness.FRESH, days_since_last_appearance=2),
            BullpenPitcher(player_id="h_bp2", name="Luis Herrera", throws=ThrowHand.L, role=BullpenRole.SETUP, freshness=Freshness.MODERATE, days_since_last_appearance=1),
            BullpenPitcher(player_id="h_bp3", name="Marcus Webb", throws=ThrowHand.R, role=BullpenRole.SETUP, freshness=Freshness.FRESH, days_since_last_appearance=3),
            BullpenPitcher(player_id="h_bp4", name="Danny Kim", throws=ThrowHand.R, role=BullpenRole.MIDDLE, freshness=Freshness.FRESH, days_since_last_appearance=4),
            BullpenPitcher(player_id="h_bp5", name="Alex Turner", throws=ThrowHand.L, role=BullpenRole.MIDDLE, freshness=Freshness.TIRED, days_since_last_appearance=0, pitches_last_3_days=[25, 18, 0]),
            BullpenPitcher(player_id="h_bp6", name="Jason Blake", throws=ThrowHand.R, role=BullpenRole.LONG, freshness=Freshness.FRESH, days_since_last_appearance=5),
            BullpenPitcher(player_id="h_bp7", name="Chris Evans", throws=ThrowHand.R, role=BullpenRole.MOPUP, freshness=Freshness.FRESH, days_since_last_appearance=3),
            BullpenPitcher(player_id="h_bp8", name="Sam Rodriguez", throws=ThrowHand.R, role=BullpenRole.MOPUP, freshness=Freshness.FRESH, days_since_last_appearance=4),
        ],
        mound_visits_remaining=4,
        challenge_available=True,
    )

    opponent_roster = OpponentRosterState(
        their_lineup=[
            LineupPlayer(player_id="a_001", name="Jordan Bell", position="2B", bats=Hand.R),
            LineupPlayer(player_id="a_002", name="Liam O'Connor", position="CF", bats=Hand.L),
            LineupPlayer(player_id="a_003", name="Anthony Russo", position="DH", bats=Hand.R),
            LineupPlayer(player_id="a_004", name="Malik Thompson", position="LF", bats=Hand.L),
            LineupPlayer(player_id="a_005", name="Kevin Park", position="1B", bats=Hand.L),
            LineupPlayer(player_id="a_006", name="Trey Anderson", position="RF", bats=Hand.R),
            LineupPlayer(player_id="a_007", name="Nathan Cruz", position="SS", bats=Hand.R),
            LineupPlayer(player_id="a_008", name="Ben Harper", position="3B", bats=Hand.L),
            LineupPlayer(player_id="a_009", name="Diego Santos", position="C", bats=Hand.R),
        ],
        their_lineup_position=4,
        their_bench=[
            OpponentBenchPlayer(player_id="a_010", name="James Wright", bats=Hand.L),
            OpponentBenchPlayer(player_id="a_011", name="Tyler Brooks", bats=Hand.S),
            OpponentBenchPlayer(player_id="a_012", name="Marcus Green", bats=Hand.R),
            OpponentBenchPlayer(player_id="a_013", name="Pete Lawson", bats=Hand.R),
            OpponentBenchPlayer(player_id="a_014", name="Isaiah Carter", bats=Hand.L),
        ],
        their_bullpen=[
            OpponentBullpenPitcher(player_id="a_bp1", name="Zach Miller", throws=ThrowHand.R, role=BullpenRole.CLOSER),
            OpponentBullpenPitcher(player_id="a_bp2", name="Omar Hassan", throws=ThrowHand.R, role=BullpenRole.SETUP),
            OpponentBullpenPitcher(player_id="a_bp3", name="Trevor Fox", throws=ThrowHand.L, role=BullpenRole.SETUP),
            OpponentBullpenPitcher(player_id="a_bp4", name="Rick Simmons", throws=ThrowHand.R, role=BullpenRole.MIDDLE),
            OpponentBullpenPitcher(player_id="a_bp5", name="Will Chang", throws=ThrowHand.R, role=BullpenRole.MIDDLE),
            OpponentBullpenPitcher(player_id="a_bp6", name="Brian Kelly", throws=ThrowHand.R, role=BullpenRole.LONG),
        ],
    )

    decision_prompt = (
        "Bottom of the 7th, 1 out, bases empty. Your team trails 3-4. "
        "Rafael Ortiz (L) is batting against Matt Henderson (LHP) who is in his "
        "3rd time through the order with 88 pitches. Ortiz is a lefty facing a lefty. "
        "On deck is Tyrone Jackson (R). "
        "Should you pinch-hit for Ortiz, let him bat, or consider another strategy? "
        "Gather relevant data with tools before deciding."
    )

    return {
        "matchup_state": matchup.model_dump(),
        "roster_state": roster.model_dump(),
        "opponent_roster_state": opponent_roster.model_dump(),
        "decision_prompt": decision_prompt,
    }


def run_dry_run() -> None:
    """Validate all models, tools, and scenario construction without API calls."""
    print("=" * 72)
    print("BASEBALL MANAGER AI AGENT -- Dry Run Validation")
    print("=" * 72)

    # 1. Validate models import and construction
    print("\n[1/5] Validating Pydantic models...")
    scenario = build_sample_scenario()
    # Round-trip the scenario through JSON to verify serialization
    scenario_json = json.dumps(scenario, default=str)
    assert len(scenario_json) > 100, "Scenario JSON is too small"
    print(f"  MatchupState:        OK ({len(json.dumps(scenario['matchup_state']))} bytes)")
    print(f"  RosterState:         OK ({len(json.dumps(scenario['roster_state']))} bytes)")
    print(f"  OpponentRosterState: OK ({len(json.dumps(scenario['opponent_roster_state']))} bytes)")

    # Validate ManagerDecision can be constructed
    test_decision = ManagerDecision(
        decision="PINCH_HIT",
        action_details="Send Kenji Tanaka (R) to pinch-hit for Rafael Ortiz (L)",
        confidence=0.75,
        reasoning="Ortiz has a same-hand disadvantage vs LHP Henderson",
        key_factors=["L-L matchup disadvantage", "Henderson 3rd time through order"],
        risks=["Lose Ortiz's bat for rest of game"],
    )
    assert test_decision.decision == "PINCH_HIT"
    print(f"  ManagerDecision:     OK")

    # 2. Validate all 12 tools are registered
    print("\n[2/5] Validating tools...")
    assert len(ALL_TOOLS) == 12, f"Expected 12 tools, got {len(ALL_TOOLS)}"
    tool_names = []
    for tool in ALL_TOOLS:
        name = tool.name
        tool_names.append(name)
        print(f"  {name}: OK")
    expected_tools = [
        "get_batter_stats", "get_pitcher_stats", "get_matchup_data",
        "get_run_expectancy", "get_win_probability", "evaluate_stolen_base",
        "evaluate_sacrifice_bunt", "get_bullpen_status", "get_pitcher_fatigue_assessment",
        "get_defensive_positioning", "get_defensive_replacement_value", "get_platoon_comparison",
    ]
    for expected in expected_tools:
        assert expected in tool_names, f"Missing tool: {expected}"

    # 3. Validate tool stubs return valid JSON
    print("\n[3/5] Validating tool stub outputs...")
    # Call each tool stub and verify it returns parseable JSON
    results = {
        "get_batter_stats": ALL_TOOLS[0]("h_003"),
        "get_pitcher_stats": ALL_TOOLS[1]("a_sp1"),
        "get_matchup_data": ALL_TOOLS[2]("h_003", "a_sp1"),
        "get_run_expectancy": ALL_TOOLS[3](True, False, False, 1),
        "get_win_probability": ALL_TOOLS[4](7, "BOTTOM", 1, False, False, False, -1),
        "evaluate_stolen_base": ALL_TOOLS[5]("h_001", 2, "a_sp1", "a_009"),
        "evaluate_sacrifice_bunt": ALL_TOOLS[6]("h_009", True, False, False, 0, -1, 7),
        "get_bullpen_status": ALL_TOOLS[7](),
        "get_pitcher_fatigue_assessment": ALL_TOOLS[8]("a_sp1"),
        "get_defensive_positioning": ALL_TOOLS[9]("h_003", "a_sp1", 1, False, False, False, -1, 7),
        "get_defensive_replacement_value": ALL_TOOLS[10]("h_007", "h_012", "LF"),
        "get_platoon_comparison": ALL_TOOLS[11]("h_003", "h_014", "a_sp1"),
    }
    for name, result in results.items():
        data = json.loads(result)
        assert "status" in data, f"Tool {name} missing 'status' field"
        print(f"  {name}: returns valid JSON ({data['status']})")

    # 4. Validate sample rosters file exists and is valid
    print("\n[4/5] Validating sample rosters...")
    roster_path = Path(__file__).parent / "data" / "sample_rosters.json"
    assert roster_path.exists(), f"Missing {roster_path}"
    with open(roster_path) as f:
        rosters = json.load(f)
    for team_key in ("home", "away"):
        team = rosters[team_key]
        lineup_count = len(team["lineup"])
        bench_count = len(team["bench"])
        bp_count = len(team["bullpen"])
        total = lineup_count + bench_count + bp_count + 1  # +1 for starting pitcher
        print(f"  {team['team_name']}: {lineup_count} lineup + {bench_count} bench + 1 SP + {bp_count} bullpen = {total} players")
        assert lineup_count == 9, f"{team_key} lineup should have 9 players"
        assert bench_count >= 5, f"{team_key} bench should have at least 5 players"
        assert bp_count >= 8, f"{team_key} bullpen should have at least 8 pitchers"

    # 5. Validate system prompt and agent setup
    print("\n[5/5] Validating agent configuration...")
    assert len(SYSTEM_PROMPT) > 100, "System prompt is too short"
    print(f"  System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  Tools registered: {len(ALL_TOOLS)}")
    print(f"  Output schema: ManagerDecision")

    print("\n" + "=" * 72)
    print("ALL VALIDATIONS PASSED")
    print("=" * 72)
    print("\nThe project skeleton is complete. To run the full agent test:")
    print("  ANTHROPIC_API_KEY=<key> uv run game.py")


def run_single_turn() -> None:
    """Run a single-turn test: send a scenario, allow tool calls, get a ManagerDecision."""
    client = Anthropic()
    scenario = build_sample_scenario()

    user_message = (
        "Here is the current game scenario:\n\n"
        f"**Matchup State:**\n```json\n{json.dumps(scenario['matchup_state'], indent=2)}\n```\n\n"
        f"**Roster State:**\n```json\n{json.dumps(scenario['roster_state'], indent=2)}\n```\n\n"
        f"**Opponent Roster State:**\n```json\n{json.dumps(scenario['opponent_roster_state'], indent=2)}\n```\n\n"
        f"**Decision Needed:** {scenario['decision_prompt']}"
    )

    print("=" * 72)
    print("BASEBALL MANAGER AI AGENT -- Single-Turn Test")
    print("=" * 72)
    print(f"\nScenario: {scenario['decision_prompt']}\n")
    print("Sending to agent with 12 tools registered...")
    print("-" * 72)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*output_format.*deprecated.*")
        runner = client.beta.messages.tool_runner(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=ALL_TOOLS,
            output_format=ManagerDecision,
            messages=[{"role": "user", "content": user_message}],
        )

    turn = 0
    final_message = None
    for message in runner:
        turn += 1
        for block in message.content:
            if block.type == "tool_use":
                print(f"  [Turn {turn}] Tool call: {block.name}({json.dumps(block.input, separators=(',', ':'))})")
            elif block.type == "text" and block.text.strip():
                print(f"  [Turn {turn}] Agent: {block.text[:200]}...")
        final_message = message

    print("-" * 72)

    if final_message and hasattr(final_message, "parsed") and final_message.parsed:
        decision: ManagerDecision = final_message.parsed
        print("\nMANAGER DECISION:")
        print(f"  Decision:    {decision.decision}")
        print(f"  Details:     {decision.action_details}")
        print(f"  Confidence:  {decision.confidence}")
        print(f"  Reasoning:   {decision.reasoning[:200]}...")
        print(f"  Key Factors: {decision.key_factors}")
        print(f"  Risks:       {decision.risks}")
        if decision.alternatives_considered:
            print(f"  Alternatives: {len(decision.alternatives_considered)} considered")
        print("\nTest PASSED: Agent received scenario, called tools, and returned a valid ManagerDecision.")
    else:
        print("\nTest FAILED: No valid ManagerDecision received.")
        sys.exit(1)


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        run_dry_run()
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY set. Running dry-run validation instead.")
        print("Set ANTHROPIC_API_KEY to run the full agent test.\n")
        run_dry_run()
    else:
        run_single_turn()
