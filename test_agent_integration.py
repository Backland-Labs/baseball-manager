# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0", "pydantic>=2.0"]
# ///
#!/usr/bin/env python
"""Test agent integration with a limited number of decision points."""

import json
import sys

from config import create_anthropic_client

from backtest.extractor import walk_game_feed
from backtest.runner import build_scenario, compare_decisions
from backtest.tools import set_backtest_context, clear_backtest_context, BACKTEST_TOOLS
from data.mlb_api import get_live_game_feed
from game import _call_agent, load_system_prompt

def main():
    print('=== Limited Agent Backtest (3 decision points) ===')
    print()

    # Fetch game
    print('Fetching game feed...')
    feed = get_live_game_feed(746865)
    game_date = feed['gameData']['datetime']['officialDate']
    all_dps = list(walk_game_feed(feed, 'CHC'))

    print(f'Game: STL @ CHC, {game_date}')
    print(f'Total decision points: {len(all_dps)}')
    print()

    # Test with first, middle, and last decision points
    test_indices = [0, len(all_dps) // 2, len(all_dps) - 1]
    test_dps = [all_dps[i] for i in test_indices]

    print(f'Testing 3 decision points: indices {test_indices}')
    print()

    # Initialize Anthropic client
    client = create_anthropic_client()
    system_prompt = load_system_prompt()

    results = []
    for i, dp in enumerate(test_dps):
        idx = test_indices[i]
        print(f'{i+1}. Decision Point {idx}: Inn {dp.inning} {dp.half}')
        print(f'   Batter: {dp.batter_name} vs Pitcher: {dp.pitcher_name}')
        print(f'   Score: {dp.score_away}-{dp.score_home}, Outs: {dp.outs}')

        # Set context
        set_backtest_context(game_date, feed, dp)

        # Build scenario
        scenario = build_scenario(dp)

        # Format user message
        ms = scenario['matchup_state']
        rs = scenario['roster_state']
        user_message = {
            'role': 'user',
            'content': (
                'Here is the current game scenario:\n\n'
                f'**Matchup State:**\n```json\n{json.dumps(ms, indent=2)}\n```\n\n'
                f'**Roster State:**\n```json\n{json.dumps(rs, indent=2)}\n```\n\n'
                f'**Decision Needed:** {scenario["decision_prompt"]}'
            )
        }

        # Call agent
        print(f'   Calling agent...')
        decision_dict, final_message, metadata = _call_agent(
            client,
            [user_message],
            tools=BACKTEST_TOOLS,
            system=system_prompt,
            verbose=False
        )

        clear_backtest_context()

        # Compare
        comparison = compare_decisions(decision_dict, dp.real_manager_action)

        print(f'   Agent decision: {decision_dict["decision"]}')
        print(f'   Real manager: {dp.real_manager_action.action_type.value if dp.real_manager_action else "NO_ACTION"}')
        print(f'   Match: {comparison["category_match"]}')
        print(f'   Tokens: {metadata["token_usage"]["total_tokens"]}')
        print(f'   Tool calls: {len(metadata["tool_calls"])}')

        results.append({
            'dp_index': idx,
            'inning': dp.inning,
            'half': dp.half,
            'agent_decision': decision_dict['decision'],
            'real_action': dp.real_manager_action.action_type.value if dp.real_manager_action else 'NO_ACTION',
            'match': comparison['category_match'],
            'tokens': metadata['token_usage']['total_tokens'],
            'tool_calls': len(metadata['tool_calls']),
        })
        print()

    print('=== SUMMARY ===')
    print(f'Tested: {len(results)} decision points')
    print(f'Matches: {sum(1 for r in results if r["match"])} / {len(results)}')
    print(f'Total tokens: {sum(r["tokens"] for r in results):,}')
    print(f'Total tool calls: {sum(r["tool_calls"] for r in results)}')
    print()
    print('SUCCESS: Full agent pipeline works end-to-end')
    return 0

if __name__ == '__main__':
    sys.exit(main())
