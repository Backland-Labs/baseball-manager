# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.78.0",
#     "pydantic>=2.0",
#     "httpx",
# ]
# ///
"""Backtest the baseball manager agent against a real historical MLB game.

Usage:
    uv run backtest.py --game-pk 746865 --team CHC --dry-run
    uv run backtest.py --game-pk 746865 --team "Chicago Cubs"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from backtest.runner import (
    run_backtest,
    format_report,
)
from backtest.extractor import resolve_team_side
from data.mlb_api import get_live_game_feed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backtest the baseball manager agent against a real MLB game."
    )
    parser.add_argument(
        "--game-pk", type=int, required=True,
        help="MLB gamePk identifier for the game to backtest.",
    )
    parser.add_argument(
        "--team", required=True,
        help='Team name, abbreviation, or "home"/"away".',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract decision points only, no agent API calls.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress (default: True).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Sample N decision points instead of running all. "
             "Always includes real manager actions plus random NO_ACTION picks.",
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Pre-flight: validate API key unless dry-run
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        print("Set it or use --dry-run to skip agent calls.", file=sys.stderr)
        return 1

    # Fetch and validate game
    try:
        if verbose:
            print(f"Fetching game feed for gamePk={args.game_pk}...")
        feed = get_live_game_feed(args.game_pk)
    except Exception as e:
        print(f"Error fetching game {args.game_pk}: {e}", file=sys.stderr)
        return 1

    # Validate game is Final
    status = feed.get("gameData", {}).get("status", {}).get("detailedState", "")
    if status != "Final":
        print(f"Error: Game {args.game_pk} is not final (status: {status})", file=sys.stderr)
        return 1

    # Resolve team
    try:
        managed_side = resolve_team_side(feed, args.team)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Game info for report
    game_data = feed.get("gameData", {})
    teams = game_data.get("teams", {})
    linescore = feed.get("liveData", {}).get("linescore", {})
    linescore_teams = linescore.get("teams", {})

    game_info = {
        "home": teams.get("home", {}).get("name", "?"),
        "away": teams.get("away", {}).get("name", "?"),
        "date": game_data.get("datetime", {}).get("officialDate", "?"),
        "home_score": linescore_teams.get("home", {}).get("runs", 0),
        "away_score": linescore_teams.get("away", {}).get("runs", 0),
        "managed_side": managed_side,
        "game_pk": args.game_pk,
    }

    # Run backtest
    try:
        entries = run_backtest(
            args.game_pk, args.team,
            dry_run=args.dry_run,
            verbose=verbose,
            sample=args.sample,
        )
    except Exception as e:
        print(f"Error during backtest: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if args.dry_run:
        return 0

    # Print report
    report = format_report(entries, game_info)
    print("\n" + report)

    # Save results
    output_dir = Path("data/game_logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    team_slug = args.team.lower().replace(" ", "_")
    output_path = output_dir / f"backtest_{args.game_pk}_{team_slug}.json"

    output_data = {
        "game_info": game_info,
        "entries": [e.model_dump() for e in entries],
        "summary": {
            "total_decisions": len(entries),
            "category_matches": sum(1 for e in entries if e.category_match),
            "active_real": sum(1 for e in entries if e.real_manager_action_type != "NO_ACTION"),
            "active_agent": sum(1 for e in entries if e.agent_decision_type != "NO_ACTION"),
            "total_input_tokens": sum(e.token_usage.get("input_tokens", 0) for e in entries),
            "total_output_tokens": sum(e.token_usage.get("output_tokens", 0) for e in entries),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
