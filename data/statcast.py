# /// script
# requires-python = ">=3.12"
# dependencies = ["pybaseball>=2.3.0"]
# ///
"""Integration with pybaseball to fetch real Statcast and FanGraphs player statistics.

Provides functions to fetch season stats, splits, advanced metrics, and
pitch-level data.  All functions integrate with the file-based cache layer
so repeated lookups avoid redundant network requests.

Usage::

    from data.statcast import (
        get_batting_stats,
        get_pitching_stats,
        get_batting_splits,
        get_pitching_splits,
        get_pitcher_tto_splits,
        get_defensive_metrics,
        get_catcher_metrics,
        get_sprint_speed,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from data.cache import Cache, TTL_SEASON_STATS, get_default_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pybaseball lazy import
# ---------------------------------------------------------------------------

_pybaseball = None


def _get_pybaseball():
    """Lazily import pybaseball to avoid import-time side effects."""
    global _pybaseball
    if _pybaseball is None:
        try:
            import pybaseball as pb
            # Disable the progress bar / caching messages from pybaseball
            try:
                pb.cache.enable()
            except Exception:
                pass
            _pybaseball = pb
        except ImportError:
            raise ImportError(
                "pybaseball is required for Statcast data. "
                "Install it with: pip install pybaseball"
            )
    return _pybaseball


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StatcastError(Exception):
    """Base exception for Statcast data errors."""


class StatcastPlayerNotFoundError(StatcastError):
    """Raised when a player cannot be found in Statcast data."""


class StatcastDataUnavailableError(StatcastError):
    """Raised when Statcast data is temporarily unavailable."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None if not possible."""
    if val is None:
        return None
    try:
        import math
        result = float(val)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    """Convert a value to int, returning None if not possible."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _round_or_none(val: Any, digits: int = 3) -> float | None:
    """Round a value, returning None if not a number."""
    f = _safe_float(val)
    if f is None:
        return None
    return round(f, digits)


def _lookup_player_id(player_name: str) -> int | None:
    """Look up a player's MLB ID by name using pybaseball.

    Args:
        player_name: The player's full name (e.g. "Mike Trout").

    Returns:
        The MLB player ID, or None if not found.
    """
    pb = _get_pybaseball()
    try:
        parts = player_name.strip().split()
        if len(parts) < 2:
            return None
        last = parts[-1]
        first = parts[0]
        results = pb.playerid_lookup(last, first)
        if results is not None and not results.empty:
            # Return the first match's key_mlbam
            row = results.iloc[0]
            mlb_id = _safe_int(row.get("key_mlbam"))
            return mlb_id
    except Exception as exc:
        logger.warning("Player lookup failed for %r: %s", player_name, exc)
    return None


# ---------------------------------------------------------------------------
# Batting stats
# ---------------------------------------------------------------------------

def get_batting_stats(
    player_id: int,
    season: int = 2024,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch real batting statistics from Statcast/FanGraphs for a player.

    Returns traditional stats, advanced metrics, plate discipline,
    batted ball profile, and sprint speed.

    Args:
        player_id: MLB player ID.
        season: Season year.
        cache: Optional cache instance.

    Returns:
        Dict with keys: ``traditional``, ``advanced``, ``plate_discipline``,
        ``batted_ball``, ``sprint_speed``, ``situational``.  Fields that are
        unavailable are set to ``None``.

    Raises:
        StatcastPlayerNotFoundError: If the player has no data.
        StatcastDataUnavailableError: If pybaseball cannot fetch data.
    """
    _cache = cache or get_default_cache()
    cache_params = {"player_id": player_id, "season": season, "type": "batting"}
    cached = _cache.get("statcast_batting", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        # FanGraphs batting stats for the season
        fg = pb.batting_stats(season, qual=0)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch FanGraphs batting stats for {season}: {exc}"
        ) from exc

    if fg is None or fg.empty:
        raise StatcastDataUnavailableError(
            f"No FanGraphs batting data available for {season}"
        )

    # Find player row -- FanGraphs uses IDfg or we match on mlbID
    player_row = None
    if "IDfg" in fg.columns:
        # Try matching by MLBAM ID if available
        if "xMLBAMID" in fg.columns:
            matches = fg[fg["xMLBAMID"] == player_id]
            if not matches.empty:
                player_row = matches.iloc[0]
        if player_row is None and "MLBAMID" in fg.columns:
            matches = fg[fg["MLBAMID"] == player_id]
            if not matches.empty:
                player_row = matches.iloc[0]

    if player_row is None:
        raise StatcastPlayerNotFoundError(
            f"Player {player_id} not found in {season} FanGraphs batting data"
        )

    # Extract stats
    result = _extract_batting_stats(player_row)

    # Fetch sprint speed from Statcast
    try:
        sprint = get_sprint_speed(player_id, season, cache=_cache)
        result["sprint_speed"] = sprint
    except Exception:
        result["sprint_speed"] = None

    _cache.set("statcast_batting", cache_params, result, ttl=TTL_SEASON_STATS)
    return result


def _extract_batting_stats(row) -> dict[str, Any]:
    """Extract batting statistics from a FanGraphs DataFrame row."""
    return {
        "traditional": {
            "AVG": _round_or_none(row.get("AVG")),
            "OBP": _round_or_none(row.get("OBP")),
            "SLG": _round_or_none(row.get("SLG")),
            "OPS": _round_or_none(
                (_safe_float(row.get("OBP")) or 0) +
                (_safe_float(row.get("SLG")) or 0)
            ),
        },
        "advanced": {
            "wOBA": _round_or_none(row.get("wOBA")),
            "wRC_plus": _safe_int(row.get("wRC+")),
            "barrel_rate": _round_or_none(row.get("Barrel%")),
            "xwOBA": _round_or_none(row.get("xwOBA")),
        },
        "plate_discipline": {
            "K_pct": _round_or_none(row.get("K%")),
            "BB_pct": _round_or_none(row.get("BB%")),
            "chase_rate": _round_or_none(row.get("O-Swing%")),
            "whiff_rate": _round_or_none(row.get("SwStr%")),
        },
        "batted_ball": {
            "GB_pct": _round_or_none(row.get("GB%")),
            "pull_pct": _round_or_none(row.get("Pull%")),
            "exit_velocity": _round_or_none(row.get("EV"), 1),
            "launch_angle": _round_or_none(row.get("LA"), 1),
        },
        "situational": {
            "RISP_avg": None,  # Requires split-specific query
            "high_leverage_ops": None,
            "late_and_close_ops": None,
        },
    }


# ---------------------------------------------------------------------------
# Pitching stats
# ---------------------------------------------------------------------------

def get_pitching_stats(
    player_id: int,
    season: int = 2024,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch real pitching statistics from Statcast/FanGraphs for a pitcher.

    Returns ERA/FIP/xFIP, strikeout and walk rates, ground ball rate,
    pitch mix with per-pitch metrics, and times-through-order splits.

    Args:
        player_id: MLB player ID.
        season: Season year.
        cache: Optional cache instance.

    Returns:
        Dict with keys: ``traditional``, ``rates``, ``batted_ball``,
        ``pitch_mix``, ``times_through_order``.

    Raises:
        StatcastPlayerNotFoundError: If the player has no data.
        StatcastDataUnavailableError: If pybaseball cannot fetch data.
    """
    _cache = cache or get_default_cache()
    cache_params = {"player_id": player_id, "season": season, "type": "pitching"}
    cached = _cache.get("statcast_pitching", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        fg = pb.pitching_stats(season, qual=0)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch FanGraphs pitching stats for {season}: {exc}"
        ) from exc

    if fg is None or fg.empty:
        raise StatcastDataUnavailableError(
            f"No FanGraphs pitching data available for {season}"
        )

    # Find player row
    player_row = None
    if "xMLBAMID" in fg.columns:
        matches = fg[fg["xMLBAMID"] == player_id]
        if not matches.empty:
            player_row = matches.iloc[0]
    if player_row is None and "MLBAMID" in fg.columns:
        matches = fg[fg["MLBAMID"] == player_id]
        if not matches.empty:
            player_row = matches.iloc[0]

    if player_row is None:
        raise StatcastPlayerNotFoundError(
            f"Player {player_id} not found in {season} FanGraphs pitching data"
        )

    result = _extract_pitching_stats(player_row)

    # Fetch pitch mix from Statcast
    try:
        pitch_mix = _fetch_pitch_mix(player_id, season, cache=_cache)
        result["pitch_mix"] = pitch_mix
    except Exception as exc:
        logger.warning("Could not fetch pitch mix for %d: %s", player_id, exc)
        result["pitch_mix"] = []

    # Fetch TTO splits
    try:
        tto = get_pitcher_tto_splits(player_id, season, cache=_cache)
        result["times_through_order"] = tto
    except Exception:
        result["times_through_order"] = {
            "1st": None,
            "2nd": None,
            "3rd_plus": None,
        }

    _cache.set("statcast_pitching", cache_params, result, ttl=TTL_SEASON_STATS)
    return result


def _extract_pitching_stats(row) -> dict[str, Any]:
    """Extract pitching statistics from a FanGraphs DataFrame row."""
    return {
        "traditional": {
            "ERA": _round_or_none(row.get("ERA"), 2),
            "FIP": _round_or_none(row.get("FIP"), 2),
            "xFIP": _round_or_none(row.get("xFIP"), 2),
            "SIERA": _round_or_none(row.get("SIERA"), 2),
        },
        "rates": {
            "K_pct": _round_or_none(row.get("K%")),
            "BB_pct": _round_or_none(row.get("BB%")),
        },
        "batted_ball": {
            "GB_pct": _round_or_none(row.get("GB%")),
            "FB_pct": _round_or_none(row.get("FB%")),
            "LD_pct": _round_or_none(row.get("LD%")),
        },
    }


def _fetch_pitch_mix(
    player_id: int,
    season: int,
    cache: Cache | None = None,
) -> list[dict[str, Any]]:
    """Fetch pitch mix data from Statcast for a pitcher.

    Args:
        player_id: MLB player ID.
        season: Season year.
        cache: Optional cache instance.

    Returns:
        List of dicts, each with ``pitch_type``, ``usage``, ``velocity``,
        ``spin_rate``, ``whiff_rate``.
    """
    _cache = cache or get_default_cache()
    cache_params = {"player_id": player_id, "season": season, "type": "pitch_mix"}
    cached = _cache.get("statcast_pitch_mix", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        data = pb.statcast_pitcher(f"{season}-03-01", f"{season}-11-30", player_id)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch Statcast pitch data for player {player_id}: {exc}"
        ) from exc

    if data is None or data.empty:
        return []

    pitch_mix = []
    total_pitches = len(data)

    if total_pitches == 0:
        return []

    # Group by pitch type
    for pitch_type, group in data.groupby("pitch_type"):
        if pitch_type is None or str(pitch_type).strip() == "":
            continue

        count = len(group)
        usage = round(count / total_pitches, 3)

        # Average velocity
        velo = _round_or_none(group["release_speed"].mean(), 1)

        # Average spin rate
        spin = _safe_int(group["release_spin_rate"].mean())

        # Whiff rate: swinging strikes / total swings
        swings = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked",
            "foul", "foul_tip", "hit_into_play",
            "hit_into_play_no_out", "hit_into_play_score",
        ])]
        whiffs = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked",
        ])]
        whiff_rate = None
        if len(swings) > 0:
            whiff_rate = round(len(whiffs) / len(swings), 3)

        pitch_mix.append({
            "pitch_type": str(pitch_type),
            "usage": usage,
            "velocity": velo,
            "spin_rate": spin,
            "whiff_rate": whiff_rate,
        })

    # Sort by usage descending
    pitch_mix.sort(key=lambda x: x["usage"], reverse=True)

    _cache.set("statcast_pitch_mix", cache_params, pitch_mix, ttl=TTL_SEASON_STATS)
    return pitch_mix


# ---------------------------------------------------------------------------
# Split stats
# ---------------------------------------------------------------------------

def get_batting_splits(
    player_id: int,
    season: int = 2024,
    vs_hand: str | None = None,
    home_away: str | None = None,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch batting splits from FanGraphs.

    Args:
        player_id: MLB player ID.
        season: Season year.
        vs_hand: ``"L"`` or ``"R"`` to filter by pitcher handedness.
        home_away: ``"home"`` or ``"away"`` to filter by venue.
        cache: Optional cache instance.

    Returns:
        Dict with batting stats for the requested split.

    Raises:
        StatcastPlayerNotFoundError: If the player has no split data.
        StatcastDataUnavailableError: If data cannot be fetched.
    """
    _cache = cache or get_default_cache()
    cache_params = {
        "player_id": player_id,
        "season": season,
        "vs_hand": vs_hand,
        "home_away": home_away,
        "type": "batting_splits",
    }
    cached = _cache.get("statcast_batting_splits", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    # FanGraphs batting stats -- use split_seasons for splits
    try:
        if vs_hand is not None:
            # Use FanGraphs split type: "vL" or "vR"
            split_type = f"v{vs_hand}"
            fg = pb.batting_stats(season, qual=0, split_seasons=False)
        elif home_away is not None:
            fg = pb.batting_stats(season, qual=0, split_seasons=False)
        else:
            fg = pb.batting_stats(season, qual=0)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch batting splits: {exc}"
        ) from exc

    if fg is None or fg.empty:
        raise StatcastDataUnavailableError(
            f"No batting split data available for {season}"
        )

    # Find player
    player_row = None
    for col in ("xMLBAMID", "MLBAMID"):
        if col in fg.columns:
            matches = fg[fg[col] == player_id]
            if not matches.empty:
                player_row = matches.iloc[0]
                break

    if player_row is None:
        raise StatcastPlayerNotFoundError(
            f"Player {player_id} not found in {season} batting splits"
        )

    result = _extract_batting_stats(player_row)
    result["split"] = {
        "vs_hand": vs_hand,
        "home_away": home_away,
    }

    _cache.set("statcast_batting_splits", cache_params, result, ttl=TTL_SEASON_STATS)
    return result


def get_pitching_splits(
    player_id: int,
    season: int = 2024,
    vs_hand: str | None = None,
    home_away: str | None = None,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch pitching splits from FanGraphs.

    Args:
        player_id: MLB player ID.
        season: Season year.
        vs_hand: ``"L"`` or ``"R"`` to filter by batter handedness.
        home_away: ``"home"`` or ``"away"`` to filter by venue.
        cache: Optional cache instance.

    Returns:
        Dict with pitching stats for the requested split.

    Raises:
        StatcastPlayerNotFoundError: If the player has no split data.
        StatcastDataUnavailableError: If data cannot be fetched.
    """
    _cache = cache or get_default_cache()
    cache_params = {
        "player_id": player_id,
        "season": season,
        "vs_hand": vs_hand,
        "home_away": home_away,
        "type": "pitching_splits",
    }
    cached = _cache.get("statcast_pitching_splits", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        fg = pb.pitching_stats(season, qual=0)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch pitching splits: {exc}"
        ) from exc

    if fg is None or fg.empty:
        raise StatcastDataUnavailableError(
            f"No pitching split data available for {season}"
        )

    player_row = None
    for col in ("xMLBAMID", "MLBAMID"):
        if col in fg.columns:
            matches = fg[fg[col] == player_id]
            if not matches.empty:
                player_row = matches.iloc[0]
                break

    if player_row is None:
        raise StatcastPlayerNotFoundError(
            f"Player {player_id} not found in {season} pitching splits"
        )

    result = _extract_pitching_stats(player_row)
    result["split"] = {
        "vs_hand": vs_hand,
        "home_away": home_away,
    }

    _cache.set("statcast_pitching_splits", cache_params, result, ttl=TTL_SEASON_STATS)
    return result


# ---------------------------------------------------------------------------
# Times-through-order splits
# ---------------------------------------------------------------------------

def get_pitcher_tto_splits(
    player_id: int,
    season: int = 2024,
    cache: Cache | None = None,
) -> dict[str, float | None]:
    """Fetch pitcher times-through-order wOBA splits from Statcast.

    Args:
        player_id: MLB player ID.
        season: Season year.
        cache: Optional cache instance.

    Returns:
        Dict with keys ``"1st"``, ``"2nd"``, ``"3rd_plus"``, each a wOBA
        float or ``None`` if unavailable.
    """
    _cache = cache or get_default_cache()
    cache_params = {"player_id": player_id, "season": season, "type": "tto"}
    cached = _cache.get("statcast_tto", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        data = pb.statcast_pitcher(f"{season}-03-01", f"{season}-11-30", player_id)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch TTO data for player {player_id}: {exc}"
        ) from exc

    if data is None or data.empty:
        return {"1st": None, "2nd": None, "3rd_plus": None}

    # Compute wOBA by times through order using at_bat_number relative to each game
    # wOBA weights (standard linear weights)
    woba_weights = {
        "single": 0.888,
        "double": 1.271,
        "triple": 1.616,
        "home_run": 2.101,
        "walk": 0.690,
        "hit_by_pitch": 0.722,
        "strikeout": 0.0,
        "field_out": 0.0,
        "grounded_into_double_play": 0.0,
        "fielders_choice": 0.0,
        "force_out": 0.0,
        "sac_fly": 0.0,
    }

    tto_results = {"1st": None, "2nd": None, "3rd_plus": None}

    if "at_bat_number" not in data.columns or "game_pk" not in data.columns:
        return tto_results

    # Determine lineup position order within each game
    # Group at-bats by game, then assign TTO by sequence of unique batters
    try:
        # Filter to plate appearance end events only
        pa_events = data[data["events"].notna()].copy()
        if pa_events.empty:
            return tto_results

        tto_woba = {1: [], 2: [], 3: []}

        for game_pk, game_data in pa_events.groupby("game_pk"):
            seen_batters = set()
            times_through = 0
            prev_batter = None
            batter_order = []

            # Sort by at_bat_number within the game
            game_sorted = game_data.sort_values("at_bat_number")

            for _, row in game_sorted.iterrows():
                batter_id = row.get("batter")
                event = str(row.get("events", "")).lower()

                # Track times through order
                if batter_id not in seen_batters:
                    seen_batters.add(batter_id)
                    if len(seen_batters) == 1:
                        times_through = 1
                elif batter_id == batter_order[0] if batter_order else False:
                    times_through += 1

                if batter_id not in batter_order:
                    batter_order.append(batter_id)

                tto_key = min(times_through, 3)
                if tto_key < 1:
                    tto_key = 1

                # Calculate wOBA value for this PA
                woba_val = woba_weights.get(event, 0.0)
                tto_woba[tto_key].append(woba_val)

        # Compute average wOBA for each TTO
        for tto_num, values in tto_woba.items():
            if values:
                avg_woba = round(sum(values) / len(values), 3)
                if tto_num == 1:
                    tto_results["1st"] = avg_woba
                elif tto_num == 2:
                    tto_results["2nd"] = avg_woba
                else:
                    tto_results["3rd_plus"] = avg_woba

    except Exception as exc:
        logger.warning("TTO computation failed for %d: %s", player_id, exc)

    _cache.set("statcast_tto", cache_params, tto_results, ttl=TTL_SEASON_STATS)
    return tto_results


# ---------------------------------------------------------------------------
# Sprint speed
# ---------------------------------------------------------------------------

def get_sprint_speed(
    player_id: int,
    season: int = 2024,
    cache: Cache | None = None,
) -> float | None:
    """Fetch a player's sprint speed from Statcast.

    Args:
        player_id: MLB player ID.
        season: Season year.
        cache: Optional cache instance.

    Returns:
        Sprint speed in ft/s, or ``None`` if unavailable.
    """
    _cache = cache or get_default_cache()
    cache_params = {"player_id": player_id, "season": season, "type": "sprint_speed"}
    cached = _cache.get("statcast_sprint_speed", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        leaders = pb.statcast_sprint_speed(season, min_opp=0)
    except Exception as exc:
        logger.warning("Sprint speed fetch failed: %s", exc)
        return None

    if leaders is None or leaders.empty:
        return None

    # Match by player ID
    id_col = None
    for col in ("player_id", "mlb_id", "pitcher"):
        if col in leaders.columns:
            id_col = col
            break

    if id_col is None:
        return None

    matches = leaders[leaders[id_col] == player_id]
    if matches.empty:
        return None

    speed = _round_or_none(matches.iloc[0].get("hp_to_1b", None), 1)
    if speed is None:
        # Try sprint_speed column
        speed = _round_or_none(matches.iloc[0].get("sprint_speed", None), 1)

    _cache.set("statcast_sprint_speed", cache_params, speed, ttl=TTL_SEASON_STATS)
    return speed


# ---------------------------------------------------------------------------
# Defensive metrics
# ---------------------------------------------------------------------------

def get_defensive_metrics(
    player_id: int,
    season: int = 2024,
    position: str | None = None,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch defensive metrics (OAA, DRS, UZR) from FanGraphs/Statcast.

    Args:
        player_id: MLB player ID.
        season: Season year.
        position: Optional fielding position to filter (e.g. ``"SS"``).
        cache: Optional cache instance.

    Returns:
        Dict with keys: ``OAA``, ``DRS``, ``UZR``, ``position``.
        Values are ``None`` if unavailable.

    Raises:
        StatcastPlayerNotFoundError: If the player has no fielding data.
        StatcastDataUnavailableError: If data cannot be fetched.
    """
    _cache = cache or get_default_cache()
    cache_params = {
        "player_id": player_id,
        "season": season,
        "position": position,
        "type": "defense",
    }
    cached = _cache.get("statcast_defense", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    try:
        fg = pb.batting_stats(season, qual=0)
    except Exception as exc:
        raise StatcastDataUnavailableError(
            f"Failed to fetch defensive data: {exc}"
        ) from exc

    if fg is None or fg.empty:
        raise StatcastDataUnavailableError(
            f"No defensive data available for {season}"
        )

    # Find player
    player_row = None
    for col in ("xMLBAMID", "MLBAMID"):
        if col in fg.columns:
            matches = fg[fg[col] == player_id]
            if not matches.empty:
                player_row = matches.iloc[0]
                break

    if player_row is None:
        raise StatcastPlayerNotFoundError(
            f"Player {player_id} not found in {season} defensive data"
        )

    result = {
        "OAA": _safe_int(player_row.get("OAA")),
        "DRS": _safe_int(player_row.get("DRS")),
        "UZR": _round_or_none(player_row.get("UZR"), 1),
        "position": position or str(player_row.get("Pos", "")),
        "player_id": player_id,
        "season": season,
    }

    _cache.set("statcast_defense", cache_params, result, ttl=TTL_SEASON_STATS)
    return result


# ---------------------------------------------------------------------------
# Catcher metrics
# ---------------------------------------------------------------------------

def get_catcher_metrics(
    player_id: int,
    season: int = 2024,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch catcher metrics (pop time, framing runs) from Statcast.

    Args:
        player_id: MLB player ID.
        season: Season year.
        cache: Optional cache instance.

    Returns:
        Dict with keys: ``pop_time``, ``framing_runs``, ``framing_runs_per_200``.
        Values are ``None`` if unavailable.

    Raises:
        StatcastPlayerNotFoundError: If the player has no catcher data.
        StatcastDataUnavailableError: If data cannot be fetched.
    """
    _cache = cache or get_default_cache()
    cache_params = {"player_id": player_id, "season": season, "type": "catcher"}
    cached = _cache.get("statcast_catcher", cache_params)
    if cached is not None:
        return cached

    pb = _get_pybaseball()

    result = {
        "pop_time": None,
        "framing_runs": None,
        "framing_runs_per_200": None,
        "player_id": player_id,
        "season": season,
    }

    # Try to fetch catcher pop time via statcast
    try:
        # pybaseball doesn't have a direct catcher pop time endpoint,
        # but we can fetch from Statcast catcher data if available
        # Use statcast_catcher_poptime if available, otherwise fall back
        if hasattr(pb, "statcast_catcher_poptime"):
            pop_data = pb.statcast_catcher_poptime(season)
            if pop_data is not None and not pop_data.empty:
                # Match by player ID
                for col in ("player_id", "catcher_id", "catcher"):
                    if col in pop_data.columns:
                        matches = pop_data[pop_data[col] == player_id]
                        if not matches.empty:
                            result["pop_time"] = _round_or_none(
                                matches.iloc[0].get("pop_time", None), 2
                            )
                            break
    except Exception as exc:
        logger.warning("Pop time fetch failed for %d: %s", player_id, exc)

    # Try to fetch framing data
    try:
        if hasattr(pb, "statcast_catcher_framing"):
            frame_data = pb.statcast_catcher_framing(season)
            if frame_data is not None and not frame_data.empty:
                for col in ("player_id", "catcher_id", "catcher"):
                    if col in frame_data.columns:
                        matches = frame_data[frame_data[col] == player_id]
                        if not matches.empty:
                            row = matches.iloc[0]
                            result["framing_runs"] = _round_or_none(
                                row.get("runs_extra_strikes", None), 1
                            )
                            result["framing_runs_per_200"] = _round_or_none(
                                row.get("runs_extra_strikes_per_200", None), 1
                            )
                            break
    except Exception as exc:
        logger.warning("Framing data fetch failed for %d: %s", player_id, exc)

    _cache.set("statcast_catcher", cache_params, result, ttl=TTL_SEASON_STATS)
    return result
