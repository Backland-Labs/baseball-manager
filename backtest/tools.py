# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0", "pydantic>=2.0"]
# ///
"""Backtesting tool implementations.

Provides 12 agent tools for backtesting: 5 real-stat tools backed by the
MLB Stats API, 5 pass-through wrappers around existing simulation tools,
and 2 unchanged static tools (run expectancy, win probability).

Module-level context must be set before agent calls via set_backtest_context().
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from anthropic import beta_tool

from data.mlb_api import (
    get_batter_vs_pitcher,
    extract_bvp_stats,
    get_player_info,
    extract_pitcher_game_stats,
)
from data.cache import Cache
from tools.response import success_response, error_response, player_ref, unavailable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backtest context -- set before each agent call
# ---------------------------------------------------------------------------

_backtest_context: dict | None = None

# Cache with effectively permanent TTL for historical stat lookups
_backtest_cache = Cache()
_PERMANENT_TTL = 10 * 365 * 86400  # ~10 years


def set_backtest_context(
    game_date: str,
    game_feed: dict,
    decision_point: Any = None,
) -> None:
    """Set the module-level context for backtest tool calls.

    Args:
        game_date: Game date as YYYY-MM-DD string.
        game_feed: Full MLB Stats API game feed dict.
        decision_point: Optional DecisionPoint for roster context.
    """
    global _backtest_context
    _backtest_context = {
        "game_date": game_date,
        "game_feed": game_feed,
        "decision_point": decision_point,
    }


def clear_backtest_context() -> None:
    """Clear the backtest context."""
    global _backtest_context
    _backtest_context = None


def _get_context() -> dict:
    """Return current context or raise if not set."""
    if _backtest_context is None:
        raise RuntimeError("Backtest context not set. Call set_backtest_context() first.")
    return _backtest_context


def _get_game_data() -> dict:
    return _get_context()["game_feed"].get("gameData", {})


def _get_player_meta(player_id: int) -> dict:
    """Get player metadata from gameData.players."""
    return _get_game_data().get("players", {}).get(f"ID{player_id}", {})


# ---------------------------------------------------------------------------
# 5 real-stat tools
# ---------------------------------------------------------------------------

@beta_tool
def get_batter_stats(
    player_id: str,
    vs_hand: Optional[str] = None,
    home_away: Optional[str] = None,
    recency_window: Optional[str] = None,
) -> str:
    """Retrieves batting statistics for a player, including traditional stats,
    advanced metrics, plate discipline, batted ball profile, and sprint speed.
    Supports splits by handedness, home/away, and recency windows.

    Args:
        player_id: The unique identifier of the batter.
        vs_hand: Optional split by pitcher handedness ('L' or 'R').
        home_away: Optional split by venue ('home' or 'away').
        recency_window: Optional recency filter ('last_7', 'last_14', 'last_30', 'season').
    Returns:
        JSON string with batting statistics.
    """
    TOOL_NAME = "get_batter_stats"

    ctx = _get_context()
    game_data = _get_game_data()
    dp = ctx.get("decision_point")

    pid = int(player_id)
    meta = _get_player_meta(pid)
    if not meta:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
                              f"Player '{player_id}' not found in game data.")

    name = meta.get("fullName", str(player_id))
    bats = meta.get("batSide", {}).get("code", "R")

    # Get season stats from the game feed boxscore (end-of-game totals)
    # For backtesting, the boxscore shows the player's game performance.
    # We use gameData.players for the player's profile info.
    feed = ctx["game_feed"]
    boxscore = feed.get("liveData", {}).get("boxscore", {})

    # Find player in boxscore for today's line
    today_line = {"AB": 0, "H": 0, "BB": 0, "K": 0, "RBI": 0}
    for side in ("home", "away"):
        players = boxscore.get("teams", {}).get(side, {}).get("players", {})
        key = f"ID{pid}"
        if key in players:
            batting = players[key].get("stats", {}).get("batting", {})
            if batting:
                today_line = {
                    "AB": batting.get("atBats", 0),
                    "H": batting.get("hits", 0),
                    "BB": batting.get("baseOnBalls", 0),
                    "K": batting.get("strikeOuts", 0),
                    "RBI": batting.get("rbi", 0),
                }
            break

    # For season stats, we return what's available from the player metadata.
    # In a real implementation, this would call the MLB Stats API season stats
    # endpoint with a date cutoff. For the MVP, we provide the player's profile
    # data and the in-game context.
    return success_response(TOOL_NAME, {
        **player_ref(player_id, name),
        "bats": bats,
        "splits": {
            "vs_hand": vs_hand,
            "home_away": home_away,
            "recency_window": recency_window,
        },
        "traditional": unavailable("Season stats require MLB Stats API season endpoint (not yet implemented for backtest MVP)"),
        "advanced": unavailable("Season stats require MLB Stats API season endpoint (not yet implemented for backtest MVP)"),
        "today": today_line,
    })


@beta_tool
def get_pitcher_stats(
    player_id: str,
    vs_hand: Optional[str] = None,
    home_away: Optional[str] = None,
    recency_window: Optional[str] = None,
) -> str:
    """Retrieves pitching statistics for a pitcher, including ERA/FIP/xFIP,
    strikeout and walk rates, ground ball rate, pitch mix with per-pitch metrics,
    and times-through-order splits.

    Args:
        player_id: The unique identifier of the pitcher.
        vs_hand: Optional split by batter handedness ('L' or 'R').
        home_away: Optional split by venue ('home' or 'away').
        recency_window: Optional recency filter ('last_7', 'last_14', 'last_30', 'season').
    Returns:
        JSON string with pitching statistics.
    """
    TOOL_NAME = "get_pitcher_stats"

    ctx = _get_context()
    game_data = _get_game_data()

    pid = int(player_id)
    meta = _get_player_meta(pid)
    if not meta:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
                              f"Player '{player_id}' not found in game data.")

    name = meta.get("fullName", str(player_id))
    throws = meta.get("pitchHand", {}).get("code", "R")

    # Get in-game pitching stats from boxscore
    feed = ctx["game_feed"]
    pitcher_game_stats = extract_pitcher_game_stats(feed, pid)

    today_line = {
        "IP": float(pitcher_game_stats.get("innings_pitched", "0.0")),
        "H": pitcher_game_stats.get("hits", 0),
        "R": pitcher_game_stats.get("runs", 0),
        "ER": pitcher_game_stats.get("earned_runs", 0),
        "BB": pitcher_game_stats.get("walks", 0),
        "K": pitcher_game_stats.get("strikeouts", 0),
    }

    return success_response(TOOL_NAME, {
        **player_ref(player_id, name),
        "throws": throws,
        "splits": {
            "vs_hand": vs_hand,
            "home_away": home_away,
            "recency_window": recency_window,
        },
        "traditional": unavailable("Season stats require MLB Stats API season endpoint (not yet implemented for backtest MVP)"),
        "rates": unavailable("Season stats require MLB Stats API season endpoint (not yet implemented for backtest MVP)"),
        "pitch_mix": unavailable("Pitch mix requires Statcast data (not yet implemented for backtest MVP)"),
        "times_through_order": unavailable("TTO splits require game log data (not yet implemented for backtest MVP)"),
        "today": today_line,
    })


@beta_tool
def get_matchup_data(batter_id: str, pitcher_id: str) -> str:
    """Retrieves head-to-head batter vs pitcher history and similarity-based
    projections. Returns direct matchup results, sample-size reliability,
    similarity-model projected wOBA, and pitch-type vulnerability breakdown.

    Args:
        batter_id: The unique identifier of the batter.
        pitcher_id: The unique identifier of the pitcher.
    Returns:
        JSON string with matchup data.
    """
    TOOL_NAME = "get_matchup_data"

    ctx = _get_context()

    bid = int(batter_id)
    pid_int = int(pitcher_id)

    batter_meta = _get_player_meta(bid)
    pitcher_meta = _get_player_meta(pid_int)

    if not batter_meta:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
                              f"Batter '{batter_id}' not found in game data.")
    if not pitcher_meta:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
                              f"Pitcher '{pitcher_id}' not found in game data.")

    # Fetch real BvP data from MLB Stats API (career totals, no date filter needed)
    try:
        bvp = get_batter_vs_pitcher(bid, pid_int)
        stats = extract_bvp_stats(bvp)
    except Exception as e:
        logger.warning("BvP lookup failed for %s vs %s: %s", batter_id, pitcher_id, e)
        stats = {
            "plate_appearances": 0, "at_bats": 0, "hits": 0,
            "doubles": 0, "triples": 0, "home_runs": 0,
            "walks": 0, "strikeouts": 0,
            "avg": None, "obp": None, "slg": None, "ops": None,
            "small_sample": True,
            "batter_name": batter_meta.get("fullName", ""),
            "pitcher_name": pitcher_meta.get("fullName", ""),
        }

    data: dict[str, Any] = {
        "batter_id": batter_id,
        "batter_name": batter_meta.get("fullName", ""),
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_meta.get("fullName", ""),
        "career_pa": stats["plate_appearances"],
        "sample_size_reliability": (
            "none" if stats["plate_appearances"] == 0
            else "small" if stats["plate_appearances"] < 10
            else "medium" if stats["plate_appearances"] < 30
            else "large"
        ),
    }

    if stats["plate_appearances"] > 0:
        data["matchup_stats"] = {
            "AVG": stats["avg"],
            "OBP": stats["obp"],
            "SLG": stats["slg"],
            "OPS": stats["ops"],
            "AB": stats["at_bats"],
            "H": stats["hits"],
            "HR": stats["home_runs"],
            "BB": stats["walks"],
            "K": stats["strikeouts"],
        }
    else:
        data["matchup_stats"] = unavailable(
            "No prior matchup history between these players.")

    return success_response(TOOL_NAME, data)


@beta_tool
def get_bullpen_status(
    team: str = "home",
    used_pitcher_ids: str | None = None,
    warming_pitcher_ids: str | None = None,
    ready_pitcher_ids: str | None = None,
) -> str:
    """Returns detailed status of all bullpen pitchers for the managed team
    including availability, stats, freshness, rest days, recent pitch counts,
    platoon splits, and warm-up status.

    Args:
        team: Which team's bullpen to return. Either "home" or "away".
        used_pitcher_ids: Comma-separated list of pitcher IDs already used and removed in this game. These pitchers are excluded from the results.
        warming_pitcher_ids: Comma-separated list of pitcher IDs currently warming up in the bullpen.
        ready_pitcher_ids: Comma-separated list of pitcher IDs that are warmed up and ready to enter.
    Returns:
        JSON string with bullpen status for all available relievers.
    """
    TOOL_NAME = "get_bullpen_status"

    ctx = _get_context()
    dp = ctx.get("decision_point")
    game_data = _get_game_data()

    if team not in ("home", "away"):
        return error_response(TOOL_NAME, "INVALID_PARAMETER",
                              f"Invalid team value: '{team}'. Must be 'home' or 'away'.")

    # Parse comma-separated ID lists
    def _parse_ids(csv: str | None) -> set[str]:
        return {p.strip() for p in csv.split(",") if p.strip()} if csv else set()

    used_ids = _parse_ids(used_pitcher_ids)

    # Get bullpen from decision point if available, otherwise from boxscore
    if dp is not None:
        managed_side = dp.managed_team_side
        if team == managed_side:
            bullpen_entries = dp.bullpen
        else:
            bullpen_entries = dp.opp_bullpen

        pitchers = []
        for bp in bullpen_entries:
            if bp.player_id in used_ids:
                continue
            pitchers.append({
                "player_id": bp.player_id,
                "name": bp.name,
                "throws": bp.throws,
                "role": "MIDDLE",  # Role not available from game feed
                "available": bp.available,
                "has_pitched": bp.has_pitched,
                "pitch_count_today": bp.pitch_count,
            })
    else:
        # Fall back to boxscore bullpen
        feed = ctx["game_feed"]
        boxscore = feed.get("liveData", {}).get("boxscore", {})
        team_box = boxscore.get("teams", {}).get(team, {})
        bullpen_ids = team_box.get("bullpen", [])

        pitchers = []
        for pid in bullpen_ids:
            if str(pid) in used_ids:
                continue
            meta = _get_player_meta(pid)
            pitchers.append({
                "player_id": str(pid),
                "name": meta.get("fullName", str(pid)),
                "throws": meta.get("pitchHand", {}).get("code", "R"),
                "role": "MIDDLE",
                "available": True,
            })

    teams = game_data.get("teams", {})
    team_name = teams.get(team, {}).get("name", team)

    return success_response(TOOL_NAME, {
        "team": team,
        "team_name": team_name,
        "bullpen_count": len(pitchers),
        "available_count": sum(1 for p in pitchers if p.get("available", True)),
        "bullpen": pitchers,
    })


@beta_tool
def get_pitcher_fatigue_assessment(
    pitcher_id: str,
    pitch_count: int = 0,
    innings_pitched: float = 0.0,
    times_through_order: int = 1,
    runs_allowed: int = 0,
    in_current_game: bool = True,
) -> str:
    """Assesses the current pitcher's fatigue based on in-game trends: velocity
    changes, spin rate decline, batted ball quality trend, pitch count, times
    through order, and an overall fatigue rating.

    Args:
        pitcher_id: The unique identifier of the pitcher to assess.
        pitch_count: Total pitches thrown so far in this game. Defaults to 0 (start of game).
        innings_pitched: Innings pitched so far (e.g., 5.0 for 5 complete innings, 5.2 for 5 and 2 outs).
        times_through_order: Times through the batting order (1 = first time, 2 = second, etc.).
        runs_allowed: Runs allowed in the current game.
        in_current_game: Whether the pitcher is currently in the game. Set to false to get assessment for a pitcher not yet in the game.
    Returns:
        JSON string with fatigue assessment.
    """
    TOOL_NAME = "get_pitcher_fatigue_assessment"

    ctx = _get_context()
    dp = ctx.get("decision_point")

    pid = int(pitcher_id)
    meta = _get_player_meta(pid)
    if not meta:
        return error_response(TOOL_NAME, "INVALID_PLAYER_ID",
                              f"Player '{pitcher_id}' not found in game data.")

    name = meta.get("fullName", str(pitcher_id))
    throws = meta.get("pitchHand", {}).get("code", "R")

    # Use decision point data if available and this is the current pitcher
    if dp is not None and pitcher_id == dp.pitcher_id:
        actual_pc = dp.pitcher_pitch_count
        actual_ip = dp.pitcher_innings_pitched
        actual_bf = dp.pitcher_batters_faced
        actual_ra = dp.pitcher_runs_allowed
    else:
        actual_pc = pitch_count
        actual_ip = innings_pitched
        actual_bf = 0
        actual_ra = runs_allowed

    # Derive fatigue level
    if actual_pc <= 30:
        fatigue_level = "fresh"
    elif actual_pc <= 75:
        fatigue_level = "normal"
    elif actual_pc <= 100:
        fatigue_level = "fatigued"
    else:
        fatigue_level = "gassed"

    # TTO penalty
    effective_tto = max(1, times_through_order)
    if actual_bf > 0:
        effective_tto = max(1, min(4, (actual_bf // 9) + 1))

    if effective_tto >= 3 and actual_pc > 75:
        fatigue_level = "fatigued"
    if effective_tto >= 3 and actual_pc > 90:
        fatigue_level = "gassed"

    return success_response(TOOL_NAME, {
        "pitcher_id": pitcher_id,
        "pitcher_name": name,
        "throws": throws,
        "pitch_count": actual_pc,
        "innings_pitched": actual_ip,
        "batters_faced": actual_bf,
        "times_through_order": effective_tto,
        "runs_allowed": actual_ra,
        "fatigue_level": fatigue_level,
    })


# ---------------------------------------------------------------------------
# 5 pass-through tools (wrap existing simulation tools)
# ---------------------------------------------------------------------------

from tools.evaluate_stolen_base import evaluate_stolen_base as _sim_evaluate_stolen_base
from tools.evaluate_sacrifice_bunt import evaluate_sacrifice_bunt as _sim_evaluate_sacrifice_bunt
from tools.get_defensive_positioning import get_defensive_positioning as _sim_get_defensive_positioning
from tools.get_defensive_replacement_value import get_defensive_replacement_value as _sim_get_defensive_replacement_value
from tools.get_platoon_comparison import get_platoon_comparison as _sim_get_platoon_comparison

# Re-export simulation tools directly -- they use static data and don't need
# backtest-specific implementations. The agent can use them as-is.
evaluate_stolen_base = _sim_evaluate_stolen_base
evaluate_sacrifice_bunt = _sim_evaluate_sacrifice_bunt
get_defensive_positioning = _sim_get_defensive_positioning
get_defensive_replacement_value = _sim_get_defensive_replacement_value
get_platoon_comparison = _sim_get_platoon_comparison


# ---------------------------------------------------------------------------
# 2 unchanged static tools
# ---------------------------------------------------------------------------

from tools.get_run_expectancy import get_run_expectancy
from tools.get_win_probability import get_win_probability


# ---------------------------------------------------------------------------
# Collected tool list for backtest agent calls
# ---------------------------------------------------------------------------

BACKTEST_TOOLS = [
    # 5 real-stat tools
    get_batter_stats,
    get_pitcher_stats,
    get_matchup_data,
    get_bullpen_status,
    get_pitcher_fatigue_assessment,
    # 2 static tools (unchanged)
    get_run_expectancy,
    get_win_probability,
    # 5 pass-through tools
    evaluate_stolen_base,
    evaluate_sacrifice_bunt,
    get_defensive_positioning,
    get_defensive_replacement_value,
    get_platoon_comparison,
]
