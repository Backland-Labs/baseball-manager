# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.78.0",
#     "pydantic>=2.0",
#     "httpx",
# ]
# ///
"""Targeted test: compare agent decisions against real manager decisions.

Loads a real game, picks decision points where the manager acted (plus a few
high-leverage NO_ACTION points), runs the agent on each, and prints a detailed
comparison including every tool call the agent made.

Usage:
    ANTHROPIC_KEY=sk-... uv run test_agent_vs_manager.py
    ANTHROPIC_KEY=sk-... uv run test_agent_vs_manager.py --game-pk 746865 --team CHC
    ANTHROPIC_KEY=sk-... uv run test_agent_vs_manager.py --sample 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config import require_api_key, create_anthropic_client
from game import _call_agent, load_system_prompt
from backtest.extractor import walk_game_feed, resolve_team_side
from backtest.runner import (
    build_scenario,
    compare_decisions,
    _normalize_decision_type,
)
from backtest.tools import (
    BACKTEST_TOOLS,
    set_backtest_context,
    clear_backtest_context,
)
from data.mlb_api import get_live_game_feed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_situation(dp) -> str:
    half_str = "T" if dp.half == "TOP" else "B"
    runner_parts = []
    for base, label in [("first", "1B"), ("second", "2B"), ("third", "3B")]:
        if dp.runners.get(base):
            runner_parts.append(label)
    runners_str = ",".join(runner_parts) if runner_parts else "---"
    return (
        f"{half_str}{dp.inning} {dp.outs} out | "
        f"{dp.score_away}-{dp.score_home} | "
        f"Runners: {runners_str} | "
        f"LI={dp.leverage_index:.2f}"
    )


def _format_tool_calls(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "  (no tool calls)"
    lines = []
    for i, tc in enumerate(tool_calls, 1):
        name = tc["tool_name"]
        args = tc["tool_input"]
        args_str = json.dumps(args, separators=(",", ":"))
        if len(args_str) > 100:
            args_str = args_str[:100] + "..."
        lines.append(f"  {i}. {name}({args_str})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Targeted agent vs real manager decision test."
    )
    parser.add_argument(
        "--game-pk", type=int, default=None,
        help="MLB gamePk. If omitted, uses the fixture file (game 746865).",
    )
    parser.add_argument(
        "--team", default="CHC",
        help='Team to manage. Default: CHC.',
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Limit to N decision points total (always includes real actions).",
    )
    parser.add_argument(
        "--hi-li", type=int, default=3,
        help="Number of high-leverage NO_ACTION points to include (default: 3).",
    )
    args = parser.parse_args()

    require_api_key("ANTHROPIC_KEY required to run agent. Set it and retry.")

    # Load game feed
    if args.game_pk is None:
        fixture_path = Path(__file__).resolve().parent / "tests" / "fixtures" / "game_feed_746865.json"
        if not fixture_path.exists():
            print(f"Fixture not found: {fixture_path}", file=sys.stderr)
            return 1
        print(f"Loading fixture: game 746865")
        with open(fixture_path) as f:
            feed = json.load(f)
        game_pk = 746865
    else:
        game_pk = args.game_pk
        print(f"Fetching game feed for gamePk={game_pk}...")
        feed = get_live_game_feed(game_pk)

    # Validate
    status = feed.get("gameData", {}).get("status", {}).get("detailedState", "")
    if status != "Final":
        print(f"Game {game_pk} is not final (status: {status})", file=sys.stderr)
        return 1

    managed_side = resolve_team_side(feed, args.team)
    game_data = feed.get("gameData", {})
    teams = game_data.get("teams", {})
    home_name = teams.get("home", {}).get("name", "?")
    away_name = teams.get("away", {}).get("name", "?")
    game_date = game_data.get("datetime", {}).get("officialDate", "?")
    linescore = feed.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_score = linescore.get("home", {}).get("runs", 0)
    away_score = linescore.get("away", {}).get("runs", 0)

    print(f"Game: {away_name} @ {home_name}, {game_date}")
    print(f"Final: {away_name} {away_score}, {home_name} {home_score}")
    print(f"Managing: {managed_side.title()} ({teams.get(managed_side, {}).get('name', '?')})")
    print()

    # Extract decision points
    all_dps = list(walk_game_feed(feed, args.team))
    active_dps = [dp for dp in all_dps if dp.real_manager_action is not None]
    no_action_dps = [dp for dp in all_dps if dp.real_manager_action is None]

    # Sort NO_ACTION by leverage index descending, pick top N
    no_action_dps.sort(key=lambda dp: dp.leverage_index, reverse=True)
    hi_li_picks = no_action_dps[:args.hi_li]

    # Combine and sort by play_index
    selected = active_dps + hi_li_picks
    selected.sort(key=lambda dp: dp.play_index)

    # Apply sample limit if set
    if args.sample is not None and args.sample < len(selected):
        # Keep all active, trim hi-li
        if args.sample >= len(active_dps):
            remaining = args.sample - len(active_dps)
            selected = active_dps + hi_li_picks[:remaining]
            selected.sort(key=lambda dp: dp.play_index)
        else:
            selected = selected[:args.sample]

    print(f"Decision points: {len(all_dps)} total, {len(active_dps)} with real manager actions")
    print(f"Selected for test: {len(selected)} ({len(active_dps)} active + {len(selected) - len(active_dps)} high-LI)")
    print("=" * 80)

    # Run agent on each
    client = create_anthropic_client()
    system_prompt = load_system_prompt()

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tool_calls = 0

    for i, dp in enumerate(selected):
        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{len(selected)}] {_format_situation(dp)}")
        print(f"  Batter: {dp.batter_name} ({dp.batter_bats}) vs Pitcher: {dp.pitcher_name} ({dp.pitcher_throws}, {dp.pitcher_pitch_count} pitches)")

        real_type = "NO_ACTION"
        real_details = ""
        if dp.real_manager_action:
            real_type = dp.real_manager_action.action_type.value
            real_details = dp.real_manager_action.details
        print(f"  Real manager: {real_type}")
        if real_details:
            print(f"    -> {real_details[:120]}")

        # Set context and build scenario
        set_backtest_context(game_date, feed, dp)
        scenario = build_scenario(dp)

        user_message = (
            "Here is the current game scenario:\n\n"
            f"**Matchup State:**\n```json\n{json.dumps(scenario['matchup_state'], indent=2)}\n```\n\n"
            f"**Roster State:**\n```json\n{json.dumps(scenario['roster_state'], indent=2)}\n```\n\n"
            f"**Opponent Roster State:**\n```json\n{json.dumps(scenario['opponent_roster_state'], indent=2)}\n```\n\n"
            f"**Decision Needed:** {scenario['decision_prompt']}"
        )
        messages = [{"role": "user", "content": user_message}]

        # Call agent
        start = time.time()
        try:
            decision_dict, final_message, call_meta = _call_agent(
                client, messages,
                tools=BACKTEST_TOOLS,
                system=system_prompt,
                verbose=False,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            decision_dict = {"decision": "ERROR", "action_details": str(e)}
            call_meta = {"tool_calls": [], "token_usage": {}}
        elapsed = time.time() - start

        clear_backtest_context()

        # Compare
        comparison = compare_decisions(decision_dict, dp.real_manager_action)
        tool_calls = call_meta.get("tool_calls", [])
        usage = call_meta.get("token_usage", {})

        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)
        total_tool_calls += len(tool_calls)

        agent_type = comparison["agent_type"]
        match_str = "MATCH" if comparison["category_match"] else "DISAGREE"

        print(f"\n  Agent decision: {agent_type} [{match_str}]")
        print(f"    Details: {decision_dict.get('action_details', '')[:150]}")
        print(f"    Confidence: {decision_dict.get('confidence', 0):.2f}")

        # Reasoning
        reasoning = decision_dict.get("reasoning", "")
        if reasoning:
            # Wrap to ~100 chars
            words = reasoning.split()
            line = "    Reasoning: "
            for w in words:
                if len(line) + len(w) + 1 > 100:
                    print(line)
                    line = "      "
                line += w + " "
            if line.strip():
                print(line.rstrip())

        # Key factors
        factors = decision_dict.get("key_factors", [])
        if factors:
            print(f"    Key factors:")
            for f in factors[:5]:
                print(f"      - {f[:120]}")

        # Tool calls
        print(f"\n  Tool calls ({len(tool_calls)}):")
        print(_format_tool_calls(tool_calls))

        print(f"\n  Tokens: {usage.get('input_tokens', 0):,} in / {usage.get('output_tokens', 0):,} out | {elapsed:.1f}s")

        results.append({
            "play_index": dp.play_index,
            "situation": _format_situation(dp),
            "real_type": real_type,
            "real_details": real_details,
            "agent_type": agent_type,
            "agent_details": decision_dict.get("action_details", ""),
            "agent_confidence": decision_dict.get("confidence", 0),
            "category_match": comparison["category_match"],
            "action_match": comparison["action_match"],
            "tool_calls": tool_calls,
            "reasoning": reasoning,
            "key_factors": factors,
            "token_usage": usage,
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    total = len(results)
    matches = sum(1 for r in results if r["category_match"])
    active_results = [r for r in results if r["real_type"] != "NO_ACTION"]
    active_matches = sum(1 for r in active_results if r["category_match"])

    print(f"Category agreement: {matches}/{total} ({100*matches/total:.0f}%)")
    if active_results:
        print(f"Active decision matches: {active_matches}/{len(active_results)} ({100*active_matches/len(active_results):.0f}%)")

    # Disagreements
    disagrees = [r for r in results if not r["category_match"]]
    if disagrees:
        print(f"\nDisagreements ({len(disagrees)}):")
        for r in disagrees:
            print(f"  {r['situation']}")
            print(f"    Real: {r['real_type']} | Agent: {r['agent_type']}")

    # Tool call summary
    tool_counts: dict[str, int] = {}
    for r in results:
        for tc in r["tool_calls"]:
            name = tc["tool_name"]
            tool_counts[name] = tool_counts.get(name, 0) + 1

    print(f"\nTool usage across {total} decisions ({total_tool_calls} total calls):")
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

    print(f"\nTotal tokens: {total_input_tokens + total_output_tokens:,} "
          f"(input: {total_input_tokens:,}, output: {total_output_tokens:,})")

    # Save results
    output_dir = Path("data/game_logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    team_slug = args.team.lower().replace(" ", "_")
    output_path = output_dir / f"targeted_test_{game_pk}_{team_slug}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
