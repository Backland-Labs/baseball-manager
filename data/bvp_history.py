# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Batter-vs-pitcher matchup history data layer.

Fetches real batter-vs-pitcher career head-to-head plate appearance results
from the MLB Stats API.  Provides a consistent JSON schema regardless of
data availability, flags small samples, and handles rookies gracefully.

All lookups are cached in ``data/cache/matchups/`` with a 24-hour TTL to
avoid redundant API calls during a game.

Usage::

    from data.bvp_history import get_matchup_history, MatchupResult

    result = get_matchup_history(batter_id=545361, pitcher_id=434378)
    print(result.plate_appearances)
    print(result.small_sample)
    print(result.to_dict())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from data.cache import Cache, TTL_MATCHUP, get_default_cache
from data.mlb_api import (
    MLBApiError,
    get_batter_vs_pitcher,
    extract_bvp_stats,
)

logger = logging.getLogger(__name__)

# Small-sample threshold
SMALL_SAMPLE_THRESHOLD = 10

# Cache endpoint name (produces data/cache/matchups/<hash>.json)
CACHE_ENDPOINT = "matchups"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BvPHistoryError(Exception):
    """Base exception for batter-vs-pitcher history errors."""


class BvPPlayerNotFoundError(BvPHistoryError):
    """Raised when a player cannot be found in the MLB Stats API."""


class BvPDataUnavailableError(BvPHistoryError):
    """Raised when BvP data is temporarily unavailable."""


# ---------------------------------------------------------------------------
# MatchupResult data class
# ---------------------------------------------------------------------------

@dataclass
class MatchupResult:
    """Consistent schema for batter-vs-pitcher matchup history.

    All fields are always present.  When no matchup history exists,
    counting fields are 0 and rate stats are ``None``.
    """

    batter_id: int
    pitcher_id: int
    batter_name: str
    pitcher_name: str
    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    walks: int = 0
    strikeouts: int = 0
    batting_average: float | None = None
    slugging: float | None = None
    ops: float | None = None
    obp: float | None = None
    small_sample: bool = True
    has_history: bool = False
    seasons: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict with consistent schema."""
        return {
            "batter_id": self.batter_id,
            "pitcher_id": self.pitcher_id,
            "batter_name": self.batter_name,
            "pitcher_name": self.pitcher_name,
            "plate_appearances": self.plate_appearances,
            "at_bats": self.at_bats,
            "hits": self.hits,
            "doubles": self.doubles,
            "triples": self.triples,
            "home_runs": self.home_runs,
            "walks": self.walks,
            "strikeouts": self.strikeouts,
            "batting_average": self.batting_average,
            "slugging": self.slugging,
            "ops": self.ops,
            "obp": self.obp,
            "small_sample": self.small_sample,
            "has_history": self.has_history,
            "seasons": self.seasons,
        }

    @classmethod
    def empty(cls, batter_id: int, pitcher_id: int,
              batter_name: str = "", pitcher_name: str = "") -> MatchupResult:
        """Create an empty result for players with no matchup history."""
        return cls(
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            batter_name=batter_name,
            pitcher_name=pitcher_name,
        )

    @classmethod
    def from_api_response(cls, batter_id: int, pitcher_id: int,
                          api_data: dict[str, Any]) -> MatchupResult:
        """Create a MatchupResult from the MLB Stats API response.

        Args:
            batter_id: MLB player ID for the batter.
            pitcher_id: MLB player ID for the pitcher.
            api_data: Result dict from ``mlb_api.get_batter_vs_pitcher``.
        """
        stats = extract_bvp_stats(api_data)

        pa = stats.get("plate_appearances", 0)
        has_history = pa > 0
        small_sample = pa < SMALL_SAMPLE_THRESHOLD

        seasons = []
        for s in api_data.get("seasons", []):
            season_stats = s.get("stats", {})
            season_entry = {
                "season": s.get("season"),
                "plate_appearances": _safe_int(season_stats.get("plateAppearances")),
                "at_bats": _safe_int(season_stats.get("atBats")),
                "hits": _safe_int(season_stats.get("hits")),
                "batting_average": _safe_float(season_stats.get("avg")),
                "slugging": _safe_float(season_stats.get("slg")),
                "ops": _safe_float(season_stats.get("ops")),
            }
            seasons.append(season_entry)

        return cls(
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            batter_name=stats.get("batter_name", ""),
            pitcher_name=stats.get("pitcher_name", ""),
            plate_appearances=pa,
            at_bats=stats.get("at_bats", 0),
            hits=stats.get("hits", 0),
            doubles=stats.get("doubles", 0),
            triples=stats.get("triples", 0),
            home_runs=stats.get("home_runs", 0),
            walks=stats.get("walks", 0),
            strikeouts=stats.get("strikeouts", 0),
            batting_average=stats.get("avg"),
            slugging=stats.get("slg"),
            ops=stats.get("ops"),
            obp=stats.get("obp"),
            small_sample=small_sample,
            has_history=has_history,
            seasons=seasons,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_int(val: Any) -> int:
    """Convert a value to int, returning 0 on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None or val == "-.--":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _get_matchups_cache(cache: Cache | None = None) -> Cache:
    """Return the cache instance to use for matchup data.

    If no explicit cache is provided, uses the default cache.
    The cache stores entries under the ``matchups`` endpoint, which
    produces files at ``data/cache/matchups/<hash>.json``.
    """
    return cache or get_default_cache()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_matchup_history(
    batter_id: int,
    pitcher_id: int,
    season: int | None = None,
    cache: Cache | None = None,
) -> MatchupResult:
    """Fetch batter-vs-pitcher matchup history from the MLB Stats API.

    Returns a :class:`MatchupResult` with a consistent schema regardless
    of whether matchup data exists.  Results are cached under the
    ``matchups`` endpoint (``data/cache/matchups/``) with a 24-hour TTL.

    Args:
        batter_id: MLB player ID for the batter.
        pitcher_id: MLB player ID for the pitcher.
        season: Optional season filter.  If ``None``, returns career totals.
        cache: Optional :class:`Cache` instance.  Defaults to the
            module-level default cache.

    Returns:
        A :class:`MatchupResult` with all fields populated.

    Raises:
        BvPPlayerNotFoundError: When a player cannot be found.
        BvPDataUnavailableError: When the API is temporarily unavailable.
    """
    _cache = _get_matchups_cache(cache)

    # Check cache first
    cache_params = {
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "season": season,
    }
    cached = _cache.get(CACHE_ENDPOINT, cache_params)
    if cached is not None:
        return MatchupResult(**cached)

    # Fetch from MLB Stats API
    try:
        api_data = get_batter_vs_pitcher(
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            season=season,
            cache=cache,
        )
    except MLBApiError as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 404:
            logger.warning(
                "Player not found in BvP lookup: batter=%s pitcher=%s",
                batter_id, pitcher_id,
            )
            raise BvPPlayerNotFoundError(
                f"Player not found: batter_id={batter_id}, "
                f"pitcher_id={pitcher_id}"
            ) from exc
        logger.error(
            "MLB API error fetching BvP data: %s (batter=%s pitcher=%s)",
            exc, batter_id, pitcher_id,
        )
        raise BvPDataUnavailableError(
            f"BvP data unavailable: {exc}"
        ) from exc

    result = MatchupResult.from_api_response(batter_id, pitcher_id, api_data)

    # Cache the result under the matchups endpoint
    _cache.set(CACHE_ENDPOINT, cache_params, result.to_dict(), ttl=TTL_MATCHUP)

    return result


def get_matchup_history_safe(
    batter_id: int,
    pitcher_id: int,
    season: int | None = None,
    cache: Cache | None = None,
) -> MatchupResult:
    """Like :func:`get_matchup_history` but never raises.

    On any error, returns an empty :class:`MatchupResult`.  Useful when
    the caller prefers degraded data over a failure.
    """
    try:
        return get_matchup_history(batter_id, pitcher_id, season, cache)
    except (BvPHistoryError, Exception) as exc:
        logger.warning(
            "Failed to fetch BvP history (returning empty): %s", exc,
        )
        return MatchupResult.empty(batter_id, pitcher_id)
