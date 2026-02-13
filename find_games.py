# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Find today's MLB games and their statuses.

Lists all MLB games for today (or a given date), showing game PKs,
teams, status, and score.  Use this to find the game-pk needed for
live_game_feed.py.

Usage::

    # List today's games
    uv run find_games.py

    # List games for a specific date
    uv run find_games.py --date 2025-07-04

    # Filter by team
    uv run find_games.py --team "Red Sox"

    # Show only live games
    uv run find_games.py --live
"""

from __future__ import annotations

import argparse
import datetime
import sys
from typing import Any


def format_game_status(game: dict[str, Any]) -> str:
    """Format a game's status for display."""
    status = game.get("status", {})
    abstract = status.get("abstractGameState", "Unknown")
    detailed = status.get("detailedState", "")

    if abstract == "Final":
        return "FINAL"
    elif abstract == "Live":
        linescore = game.get("linescore", {})
        inning = linescore.get("currentInning", "?")
        is_top = linescore.get("isTopInning", True)
        half = "Top" if is_top else "Bot"
        return f"LIVE ({half} {inning})"
    elif abstract == "Preview":
        game_date = game.get("gameDate", "")
        if game_date:
            try:
                dt = datetime.datetime.fromisoformat(game_date.replace("Z", "+00:00"))
                local_time = dt.astimezone()
                return f"Scheduled {local_time.strftime('%I:%M %p')}"
            except (ValueError, TypeError):
                pass
        return detailed or "Scheduled"
    else:
        return detailed or abstract


def format_score(game: dict[str, Any]) -> str:
    """Format the score for a game."""
    linescore = game.get("linescore", {})
    teams_score = linescore.get("teams", {})
    home_runs = teams_score.get("home", {}).get("runs")
    away_runs = teams_score.get("away", {}).get("runs")
    if home_runs is not None and away_runs is not None:
        return f"{away_runs}-{home_runs}"
    return "-"


def find_games(
    date: str | None = None,
    team: str | None = None,
    live_only: bool = False,
) -> list[dict[str, Any]]:
    """Find MLB games for a given date.

    Args:
        date: Date string in YYYY-MM-DD format. Defaults to today.
        team: Optional team name/abbreviation to filter by.
        live_only: If True, only return games that are currently live.

    Returns:
        List of game dicts from the MLB Stats API.
    """
    from data.mlb_api import get_schedule_by_date, lookup_team_id

    if date is None:
        date = datetime.date.today().strftime("%Y-%m-%d")

    team_id = None
    if team:
        try:
            team_id = lookup_team_id(team)
        except ValueError:
            print(f"Warning: Could not resolve team '{team}', showing all games",
                  file=sys.stderr)

    games = get_schedule_by_date(
        date=date,
        team_id=team_id,
        hydrate="linescore,probablePitcher",
    )

    if live_only:
        games = [
            g for g in games
            if g.get("status", {}).get("abstractGameState") == "Live"
        ]

    return games


def print_games(games: list[dict[str, Any]], date: str) -> None:
    """Print games in a formatted table."""
    if not games:
        print(f"No games found for {date}.")
        return

    print(f"\nMLB Games for {date}")
    print("=" * 72)
    print(f"{'Game PK':<10} {'Away':<22} {'Home':<22} {'Score':<7} {'Status'}")
    print("-" * 72)

    for game in games:
        game_pk = game.get("gamePk", "?")
        teams = game.get("teams", {})
        away = teams.get("away", {}).get("team", {}).get("name", "?")
        home = teams.get("home", {}).get("team", {}).get("name", "?")
        score = format_score(game)
        status = format_game_status(game)

        print(f"{game_pk:<10} {away:<22} {home:<22} {score:<7} {status}")

    print("-" * 72)
    print(f"Total: {len(games)} game(s)")

    # Print usage hint for live games
    live_games = [
        g for g in games
        if g.get("status", {}).get("abstractGameState") == "Live"
    ]
    if live_games:
        g = live_games[0]
        pk = g.get("gamePk")
        away_name = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "?")
        home_name = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "?")
        print(f"\nTo monitor a live game:")
        print(f"  ANTHROPIC_KEY=sk-... uv run live_game_feed.py --game-pk {pk} --team \"{home_name}\"")
        print(f"\nWith tweets:")
        print(f"  ANTHROPIC_KEY=sk-... uv run live_game_feed.py --game-pk {pk} --team \"{home_name}\" --tweet-dry-run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find today's MLB games and their statuses."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--team", type=str, default=None,
        help="Filter by team name or abbreviation",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Show only live (in-progress) games",
    )

    args = parser.parse_args()

    date = args.date or datetime.date.today().strftime("%Y-%m-%d")
    games = find_games(date=date, team=args.team, live_only=args.live)
    print_games(games, date)
