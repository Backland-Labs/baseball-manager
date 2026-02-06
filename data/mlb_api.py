# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Core client for the MLB Stats API (statsapi.mlb.com).

Provides functions to fetch rosters, player metadata, live game state,
schedules, and batter-vs-pitcher career stats.  All functions return
plain Python dicts/lists parsed from the API's JSON responses.

The module integrates with the file-based cache layer (``data.cache``)
so repeated lookups during a game avoid redundant network requests.

Usage::

    from data.mlb_api import (
        get_team_roster,
        get_player_info,
        get_live_game_feed,
        get_schedule_by_date,
        get_batter_vs_pitcher,
        lookup_team_id,
    )
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import json
from typing import Any

from data.cache import (
    Cache,
    TTL_LIVE_GAME,
    TTL_MATCHUP,
    TTL_SEASON_STATS,
    get_default_cache,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://statsapi.mlb.com/api"
DEFAULT_TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds; actual delay = base * 2^attempt

# MLB team ID lookup table (all 30 teams)
TEAM_IDS: dict[str, int] = {
    "angels": 108, "los angeles angels": 108, "laa": 108,
    "diamondbacks": 109, "arizona diamondbacks": 109, "az": 109, "ari": 109,
    "orioles": 110, "baltimore orioles": 110, "bal": 110,
    "red sox": 111, "boston red sox": 111, "bos": 111,
    "cubs": 112, "chicago cubs": 112, "chc": 112,
    "reds": 113, "cincinnati reds": 113, "cin": 113,
    "guardians": 114, "cleveland guardians": 114, "cle": 114,
    "rockies": 115, "colorado rockies": 115, "col": 115,
    "tigers": 116, "detroit tigers": 116, "det": 116,
    "astros": 117, "houston astros": 117, "hou": 117,
    "royals": 118, "kansas city royals": 118, "kc": 118,
    "dodgers": 119, "los angeles dodgers": 119, "lad": 119,
    "nationals": 120, "washington nationals": 120, "wsh": 120, "was": 120,
    "mets": 121, "new york mets": 121, "nym": 121,
    "athletics": 133, "oakland athletics": 133, "oak": 133, "ath": 133,
    "pirates": 134, "pittsburgh pirates": 134, "pit": 134,
    "padres": 135, "san diego padres": 135, "sd": 135,
    "mariners": 136, "seattle mariners": 136, "sea": 136,
    "giants": 137, "san francisco giants": 137, "sf": 137,
    "cardinals": 138, "st. louis cardinals": 138, "stl": 138,
    "rays": 139, "tampa bay rays": 139, "tb": 139,
    "rangers": 140, "texas rangers": 140, "tex": 140,
    "blue jays": 141, "toronto blue jays": 141, "tor": 141,
    "twins": 142, "minnesota twins": 142, "min": 142,
    "phillies": 143, "philadelphia phillies": 143, "phi": 143,
    "braves": 144, "atlanta braves": 144, "atl": 144,
    "white sox": 145, "chicago white sox": 145, "cws": 145, "chw": 145,
    "marlins": 146, "miami marlins": 146, "mia": 146,
    "yankees": 147, "new york yankees": 147, "nyy": 147,
    "brewers": 158, "milwaukee brewers": 158, "mil": 158,
}

# Reverse lookup: team_id -> canonical name
TEAM_NAMES: dict[int, str] = {
    108: "Los Angeles Angels",
    109: "Arizona Diamondbacks",
    110: "Baltimore Orioles",
    111: "Boston Red Sox",
    112: "Chicago Cubs",
    113: "Cincinnati Reds",
    114: "Cleveland Guardians",
    115: "Colorado Rockies",
    116: "Detroit Tigers",
    117: "Houston Astros",
    118: "Kansas City Royals",
    119: "Los Angeles Dodgers",
    120: "Washington Nationals",
    121: "New York Mets",
    133: "Oakland Athletics",
    134: "Pittsburgh Pirates",
    135: "San Diego Padres",
    136: "Seattle Mariners",
    137: "San Francisco Giants",
    138: "St. Louis Cardinals",
    139: "Tampa Bay Rays",
    140: "Texas Rangers",
    141: "Toronto Blue Jays",
    142: "Minnesota Twins",
    143: "Philadelphia Phillies",
    144: "Atlanta Braves",
    145: "Chicago White Sox",
    146: "Miami Marlins",
    147: "New York Yankees",
    158: "Milwaukee Brewers",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MLBApiError(Exception):
    """Base exception for MLB Stats API errors."""

    def __init__(self, message: str, status_code: int | None = None,
                 url: str | None = None):
        self.status_code = status_code
        self.url = url
        super().__init__(message)


class MLBApiNotFoundError(MLBApiError):
    """Raised when a resource is not found (404)."""


class MLBApiConnectionError(MLBApiError):
    """Raised when a connection to the API cannot be established."""


class MLBApiTimeoutError(MLBApiError):
    """Raised when a request to the API times out."""


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT,
                max_retries: int = MAX_RETRIES) -> dict[str, Any]:
    """Fetch JSON from *url* with retry logic for transient failures.

    Retries on connection errors and 5xx responses using exponential
    backoff.  Raises :class:`MLBApiError` subclasses for non-retryable
    failures.

    Args:
        url: Full URL to fetch.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        MLBApiNotFoundError: If the server returns 404.
        MLBApiTimeoutError: If all attempts time out.
        MLBApiConnectionError: If the server is unreachable after retries.
        MLBApiError: For other HTTP errors.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return json.loads(data)

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise MLBApiNotFoundError(
                    f"Resource not found: {url}",
                    status_code=404,
                    url=url,
                ) from exc
            if exc.code == 400:
                # Bad request -- not retryable
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise MLBApiError(
                    f"Bad request ({exc.code}): {body}",
                    status_code=exc.code,
                    url=url,
                ) from exc
            if exc.code >= 500:
                # Server error -- retryable
                last_error = exc
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt)
                continue
            # Other client errors -- not retryable
            raise MLBApiError(
                f"HTTP {exc.code} from {url}",
                status_code=exc.code,
                url=url,
            ) from exc

        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(
                exc.reason, TimeoutError
            ):
                last_error = MLBApiTimeoutError(
                    f"Request timed out: {url}", url=url
                )
            elif isinstance(exc, TimeoutError):
                last_error = MLBApiTimeoutError(
                    f"Request timed out: {url}", url=url
                )
            else:
                last_error = MLBApiConnectionError(
                    f"Connection failed: {exc}", url=url
                )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)
            continue

        except OSError as exc:
            last_error = MLBApiConnectionError(
                f"Connection error: {exc}", url=url
            )
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)
            continue

    # All retries exhausted
    if isinstance(last_error, (MLBApiTimeoutError, MLBApiConnectionError)):
        raise last_error
    raise MLBApiConnectionError(
        f"Failed after {max_retries} retries: {last_error}", url=url
    )


def _backoff_sleep(attempt: int) -> None:
    """Sleep with exponential backoff."""
    delay = RETRY_BACKOFF_BASE * (2 ** attempt)
    time.sleep(delay)


def _build_url(version: str, path: str,
               params: dict[str, Any] | None = None) -> str:
    """Build a full MLB Stats API URL.

    Args:
        version: API version (e.g. ``"v1"`` or ``"v1.1"``).
        path: Resource path (e.g. ``"teams/111/roster"``).
        params: Optional query parameters.

    Returns:
        The full URL string.
    """
    url = f"{BASE_URL}/{version}/{path}"
    if params:
        # Filter out None values
        filtered = {k: v for k, v in params.items() if v is not None}
        if filtered:
            query = "&".join(f"{k}={v}" for k, v in filtered.items())
            url = f"{url}?{query}"
    return url


# ---------------------------------------------------------------------------
# Team lookup
# ---------------------------------------------------------------------------

def lookup_team_id(team: str | int) -> int:
    """Resolve a team name, abbreviation, or ID to an MLB team ID.

    Args:
        team: Team name (e.g. ``"Red Sox"``), abbreviation (``"BOS"``),
            or numeric team ID (``111``).

    Returns:
        The integer MLB team ID.

    Raises:
        ValueError: If the team cannot be resolved.
    """
    if isinstance(team, int):
        if team in TEAM_NAMES:
            return team
        raise ValueError(f"Unknown team ID: {team}")

    key = team.strip().lower()
    if key in TEAM_IDS:
        return TEAM_IDS[key]

    # Try numeric string
    try:
        team_id = int(key)
        if team_id in TEAM_NAMES:
            return team_id
    except ValueError:
        pass

    raise ValueError(
        f"Unknown team: {team!r}. Use a team name (e.g. 'Red Sox'), "
        f"abbreviation (e.g. 'BOS'), or numeric team ID (e.g. 111)."
    )


def get_team_name(team_id: int) -> str:
    """Return the canonical team name for a team ID.

    Args:
        team_id: MLB team ID.

    Returns:
        Canonical team name string.

    Raises:
        ValueError: If the team ID is unknown.
    """
    if team_id in TEAM_NAMES:
        return TEAM_NAMES[team_id]
    raise ValueError(f"Unknown team ID: {team_id}")


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

def get_team_roster(
    team: str | int,
    roster_type: str = "active",
    season: int | None = None,
    hydrate: str = "person",
    cache: Cache | None = None,
) -> list[dict[str, Any]]:
    """Fetch the roster for a team.

    Args:
        team: Team name, abbreviation, or numeric ID.
        roster_type: Roster type. Common values: ``"active"`` (26-man),
            ``"40Man"``, ``"fullRoster"``, ``"depthChart"``.
        season: Season year. Defaults to the current season.
        hydrate: Hydration string. ``"person"`` inlines full player
            metadata into each roster entry.
        cache: Optional :class:`Cache` instance. Defaults to the
            module-level default cache.

    Returns:
        A list of roster entry dicts, each containing ``"person"``,
        ``"jerseyNumber"``, ``"position"``, and ``"status"`` keys.

    Raises:
        MLBApiError: On API errors.
        ValueError: If the team cannot be resolved.
    """
    team_id = lookup_team_id(team)
    _cache = cache or get_default_cache()

    cache_params = {"team_id": team_id, "roster_type": roster_type,
                    "season": season}
    cached = _cache.get("roster", cache_params)
    if cached is not None:
        return cached

    params: dict[str, Any] = {"rosterType": roster_type}
    if season is not None:
        params["season"] = season
    if hydrate:
        params["hydrate"] = hydrate

    url = _build_url("v1", f"teams/{team_id}/roster", params)
    data = _fetch_json(url)

    roster = data.get("roster", [])
    _cache.set("roster", cache_params, roster, ttl=TTL_SEASON_STATS)
    return roster


# ---------------------------------------------------------------------------
# Player metadata
# ---------------------------------------------------------------------------

def get_player_info(
    player_id: int,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch metadata for a single player.

    Returns a dict with keys: ``id``, ``fullName``, ``firstName``,
    ``lastName``, ``primaryNumber``, ``primaryPosition`` (dict),
    ``batSide`` (dict), ``pitchHand`` (dict), ``active``, and other
    biographical data.

    Args:
        player_id: MLB player ID.
        cache: Optional cache instance.

    Returns:
        Player metadata dict.

    Raises:
        MLBApiNotFoundError: If the player ID does not exist.
        MLBApiError: On other API errors.
    """
    _cache = cache or get_default_cache()

    cache_params = {"player_id": player_id}
    cached = _cache.get("player_info", cache_params)
    if cached is not None:
        return cached

    url = _build_url("v1", f"people/{player_id}")
    data = _fetch_json(url)

    people = data.get("people", [])
    if not people:
        raise MLBApiNotFoundError(
            f"Player {player_id} not found",
            status_code=404,
        )

    player = people[0]
    _cache.set("player_info", cache_params, player, ttl=TTL_SEASON_STATS)
    return player


def get_players_info(
    player_ids: list[int],
    cache: Cache | None = None,
) -> list[dict[str, Any]]:
    """Fetch metadata for multiple players in a single request.

    Args:
        player_ids: List of MLB player IDs.
        cache: Optional cache instance.

    Returns:
        List of player metadata dicts.

    Raises:
        MLBApiError: On API errors.
    """
    if not player_ids:
        return []

    _cache = cache or get_default_cache()

    # Check cache for each player; collect uncached IDs
    results: dict[int, dict] = {}
    uncached_ids: list[int] = []

    for pid in player_ids:
        cached = _cache.get("player_info", {"player_id": pid})
        if cached is not None:
            results[pid] = cached
        else:
            uncached_ids.append(pid)

    if uncached_ids:
        ids_str = ",".join(str(pid) for pid in uncached_ids)
        url = _build_url("v1", "people", {"personIds": ids_str})
        data = _fetch_json(url)

        for person in data.get("people", []):
            pid = person["id"]
            results[pid] = person
            _cache.set("player_info", {"player_id": pid}, person,
                        ttl=TTL_SEASON_STATS)

    # Return in the order requested
    return [results[pid] for pid in player_ids if pid in results]


# ---------------------------------------------------------------------------
# Player metadata extraction helpers
# ---------------------------------------------------------------------------

def extract_player_metadata(player: dict[str, Any]) -> dict[str, Any]:
    """Extract key fields from a player info dict into a flat structure.

    Args:
        player: A player dict as returned by :func:`get_player_info`.

    Returns:
        Dict with keys: ``player_id``, ``name``, ``position``,
        ``position_abbreviation``, ``bats``, ``throws``,
        ``jersey_number``, ``active``.
    """
    position = player.get("primaryPosition", {})
    bat_side = player.get("batSide", {})
    pitch_hand = player.get("pitchHand", {})

    return {
        "player_id": player.get("id"),
        "name": player.get("fullName", ""),
        "first_name": player.get("firstName", ""),
        "last_name": player.get("lastName", ""),
        "position": position.get("name", ""),
        "position_abbreviation": position.get("abbreviation", ""),
        "position_code": position.get("code", ""),
        "bats": bat_side.get("code", ""),  # L, R, or S
        "throws": pitch_hand.get("code", ""),  # L or R
        "jersey_number": player.get("primaryNumber", ""),
        "active": player.get("active", False),
    }


def extract_roster_metadata(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract player metadata from a hydrated roster response.

    Args:
        roster: Roster list as returned by :func:`get_team_roster`
            (with ``hydrate="person"``).

    Returns:
        List of flat metadata dicts (one per player).
    """
    results = []
    for entry in roster:
        person = entry.get("person", {})
        position = entry.get("position", {})
        jersey = entry.get("jerseyNumber", "")

        # Merge person-level and roster-entry-level data
        meta = extract_player_metadata(person)
        # Override position from roster entry (more accurate for current role)
        if position.get("abbreviation"):
            meta["position"] = position.get("name", meta["position"])
            meta["position_abbreviation"] = position.get(
                "abbreviation", meta["position_abbreviation"]
            )
            meta["position_code"] = position.get(
                "code", meta["position_code"]
            )
        if jersey:
            meta["jersey_number"] = jersey
        results.append(meta)
    return results


# ---------------------------------------------------------------------------
# Live game feed
# ---------------------------------------------------------------------------

def get_live_game_feed(
    game_pk: int,
    fields: str | None = None,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch the live game feed for a game.

    This is the most data-rich endpoint in the API, returning the full
    play-by-play, linescore, boxscore, and game metadata.

    Args:
        game_pk: The unique game identifier (gamePk).
        fields: Optional comma-separated field filter to reduce response
            size (e.g. ``"gameData,liveData.linescore"``).
        cache: Optional cache instance.

    Returns:
        The full game feed response dict with keys ``gameData`` and
        ``liveData``.

    Raises:
        MLBApiNotFoundError: If the game does not exist.
        MLBApiError: On other API errors.
    """
    _cache = cache or get_default_cache()

    cache_params = {"game_pk": game_pk}
    cached = _cache.get("live_game", cache_params)
    if cached is not None:
        return cached

    params: dict[str, Any] = {}
    if fields:
        params["fields"] = fields

    url = _build_url("v1.1", f"game/{game_pk}/feed/live", params or None)
    data = _fetch_json(url)

    _cache.set("live_game", cache_params, data, ttl=TTL_LIVE_GAME)
    return data


def extract_game_situation(feed: dict[str, Any]) -> dict[str, Any]:
    """Extract the current game situation from a live game feed.

    Pulls inning, outs, count, runners, score, and current batter/pitcher
    from the linescore and current play data.

    Args:
        feed: Live game feed dict from :func:`get_live_game_feed`.

    Returns:
        A dict with keys: ``inning``, ``half``, ``outs``, ``count``,
        ``runners``, ``score``, ``batter``, ``pitcher``,
        ``on_deck_batter``, ``game_status``.
    """
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    linescore = live_data.get("linescore", {})

    # Game status
    status = game_data.get("status", {})
    game_status = status.get("abstractGameState", "Unknown")

    # Inning / half
    inning = linescore.get("currentInning", 0)
    is_top = linescore.get("isTopInning", True)
    half = "TOP" if is_top else "BOTTOM"

    # Outs
    outs = linescore.get("outs", 0)

    # Score
    teams_data = linescore.get("teams", {})
    score = {
        "home": teams_data.get("home", {}).get("runs", 0),
        "away": teams_data.get("away", {}).get("runs", 0),
    }

    # Current play for count, batter, pitcher
    plays = live_data.get("plays", {})
    current_play = plays.get("currentPlay", {})
    matchup = current_play.get("matchup", {})
    count_data = current_play.get("count", {})

    count = {
        "balls": count_data.get("balls", 0),
        "strikes": count_data.get("strikes", 0),
    }

    batter = matchup.get("batter", {})
    pitcher = matchup.get("pitcher", {})

    # Runners
    offense = linescore.get("offense", {})
    runners = {
        "first": offense.get("first"),
        "second": offense.get("second"),
        "third": offense.get("third"),
    }

    # On-deck batter
    on_deck = offense.get("ondeck", None)

    return {
        "inning": inning,
        "half": half,
        "outs": outs,
        "count": count,
        "runners": runners,
        "score": score,
        "batter": batter,
        "pitcher": pitcher,
        "on_deck_batter": on_deck,
        "game_status": game_status,
    }


def extract_pitcher_game_stats(feed: dict[str, Any],
                                pitcher_id: int) -> dict[str, Any]:
    """Extract a pitcher's in-game stats from the live feed boxscore.

    Args:
        feed: Live game feed dict.
        pitcher_id: MLB player ID for the pitcher.

    Returns:
        Dict with keys: ``innings_pitched``, ``hits``, ``runs``,
        ``earned_runs``, ``walks``, ``strikeouts``, ``pitch_count``,
        ``batters_faced``, ``home_runs_allowed``.
        Returns an empty dict if the pitcher is not found in the boxscore.
    """
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})

    for side in ("away", "home"):
        team_box = boxscore.get("teams", {}).get(side, {})
        players = team_box.get("players", {})
        key = f"ID{pitcher_id}"
        if key in players:
            player_data = players[key]
            pitching = player_data.get("stats", {}).get("pitching", {})
            if pitching:
                return {
                    "innings_pitched": pitching.get("inningsPitched", "0.0"),
                    "hits": pitching.get("hits", 0),
                    "runs": pitching.get("runs", 0),
                    "earned_runs": pitching.get("earnedRuns", 0),
                    "walks": pitching.get("baseOnBalls", 0),
                    "strikeouts": pitching.get("strikeOuts", 0),
                    "pitch_count": pitching.get("numberOfPitches", 0),
                    "batters_faced": pitching.get("battersFaced", 0),
                    "home_runs_allowed": pitching.get("homeRuns", 0),
                }
    return {}


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def get_schedule_by_date(
    date: str,
    sport_id: int = 1,
    team_id: int | None = None,
    hydrate: str | None = None,
    cache: Cache | None = None,
) -> list[dict[str, Any]]:
    """Fetch the game schedule for a given date.

    Args:
        date: Date string in ``YYYY-MM-DD`` format.
        sport_id: Sport ID (``1`` for MLB).
        team_id: Optional team ID to filter games.
        hydrate: Optional hydration string (e.g. ``"probablePitcher,linescore"``).
        cache: Optional cache instance.

    Returns:
        A flat list of game dicts from all matching dates. Each game
        dict contains ``gamePk``, ``gameDate``, ``status``, ``teams``
        (with ``away`` and ``home``), ``venue``, etc.

    Raises:
        MLBApiError: On API errors.
    """
    _cache = cache or get_default_cache()

    cache_params = {"date": date, "sport_id": sport_id, "team_id": team_id}
    cached = _cache.get("schedule", cache_params)
    if cached is not None:
        return cached

    params: dict[str, Any] = {"sportId": sport_id, "date": date}
    if team_id is not None:
        params["teamId"] = team_id
    if hydrate:
        params["hydrate"] = hydrate

    url = _build_url("v1", "schedule", params)
    data = _fetch_json(url)

    # Flatten: collect all games from all dates
    games: list[dict[str, Any]] = []
    for date_entry in data.get("dates", []):
        games.extend(date_entry.get("games", []))

    _cache.set("schedule", cache_params, games, ttl=TTL_LIVE_GAME)
    return games


def find_active_game_pks(
    date: str,
    team: str | int | None = None,
) -> list[int]:
    """Find gamePk values for games on a given date.

    Args:
        date: Date string in ``YYYY-MM-DD`` format.
        team: Optional team name/abbreviation/ID to filter.

    Returns:
        List of gamePk integers.
    """
    team_id = lookup_team_id(team) if team is not None else None
    games = get_schedule_by_date(date, team_id=team_id)
    return [g["gamePk"] for g in games if "gamePk" in g]


# ---------------------------------------------------------------------------
# Batter vs. Pitcher
# ---------------------------------------------------------------------------

def get_batter_vs_pitcher(
    batter_id: int,
    pitcher_id: int,
    season: int | None = None,
    cache: Cache | None = None,
) -> dict[str, Any]:
    """Fetch batter-vs-pitcher career stats from the MLB Stats API.

    The response contains career totals and per-season breakdowns.

    Args:
        batter_id: MLB player ID for the batter.
        pitcher_id: MLB player ID for the pitcher.
        season: Optional season to filter. If ``None``, returns all
            seasons.
        cache: Optional cache instance.

    Returns:
        A dict with keys:

        - ``career``: Career aggregate stats (dict) or ``None`` if no
          matchup history exists.
        - ``seasons``: List of per-season stat dicts.
        - ``plate_appearances``: Total career plate appearances.
        - ``small_sample``: ``True`` if fewer than 10 PAs.
        - ``batter``: Batter info dict (``id``, ``fullName``).
        - ``pitcher``: Pitcher info dict (``id``, ``fullName``).

    Raises:
        MLBApiError: On API errors.
    """
    _cache = cache or get_default_cache()

    cache_params = {"batter_id": batter_id, "pitcher_id": pitcher_id,
                    "season": season}
    cached = _cache.get("bvp", cache_params)
    if cached is not None:
        return cached

    params: dict[str, Any] = {
        "stats": "vsPlayer",
        "group": "hitting",
        "opposingPlayerId": pitcher_id,
    }
    if season is not None:
        params["season"] = season

    url = _build_url("v1", f"people/{batter_id}/stats", params)
    data = _fetch_json(url)

    career_stats = None
    season_splits: list[dict] = []
    batter_info: dict = {}
    pitcher_info: dict = {}

    for stat_group in data.get("stats", []):
        display_name = stat_group.get("type", {}).get("displayName", "")
        splits = stat_group.get("splits", [])

        if display_name == "vsPlayerTotal" and splits:
            split = splits[0]
            career_stats = split.get("stat", {})
            batter_info = split.get("batter", {})
            pitcher_info = split.get("pitcher", {})
        elif display_name == "vsPlayer":
            for split in splits:
                entry = {
                    "season": split.get("season"),
                    "stats": split.get("stat", {}),
                    "team": split.get("team", {}),
                    "opponent": split.get("opponent", {}),
                }
                season_splits.append(entry)
                if not batter_info:
                    batter_info = split.get("batter", {})
                if not pitcher_info:
                    pitcher_info = split.get("pitcher", {})

    pa = 0
    if career_stats:
        pa = int(career_stats.get("plateAppearances", 0))

    result = {
        "career": career_stats,
        "seasons": season_splits,
        "plate_appearances": pa,
        "small_sample": pa < 10,
        "batter": batter_info,
        "pitcher": pitcher_info,
    }

    _cache.set("bvp", cache_params, result, ttl=TTL_MATCHUP)
    return result


# ---------------------------------------------------------------------------
# Model mapping helpers
# ---------------------------------------------------------------------------

def map_roster_to_model_fields(
    roster: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map hydrated roster entries to the fields expected by the agent models.

    Produces a list of dicts suitable for constructing ``LineupPlayer``,
    ``BenchPlayer``, or ``BullpenPitcher`` model instances.

    Args:
        roster: Hydrated roster list from :func:`get_team_roster`.

    Returns:
        List of dicts with keys: ``player_id``, ``name``, ``position``,
        ``bats``, ``throws``, ``jersey_number``, ``is_pitcher``.
    """
    results = []
    for entry in roster:
        person = entry.get("person", {})
        position = entry.get("position", {})
        bat_side = person.get("batSide", {})
        pitch_hand = person.get("pitchHand", {})
        pos_type = position.get("type", "")

        results.append({
            "player_id": str(person.get("id", "")),
            "name": person.get("fullName", ""),
            "position": position.get("abbreviation", ""),
            "bats": bat_side.get("code", "R"),  # L, R, or S
            "throws": pitch_hand.get("code", "R"),  # L or R
            "jersey_number": entry.get("jerseyNumber", ""),
            "is_pitcher": pos_type == "Pitcher",
        })
    return results


def map_game_situation_to_matchup_state(
    feed: dict[str, Any],
    managed_team_side: str = "home",
) -> dict[str, Any]:
    """Map a live game feed to the fields expected by MatchupState.

    This is a convenience function that combines
    :func:`extract_game_situation` and :func:`extract_pitcher_game_stats`
    to produce a dict that can be passed to the ``MatchupState`` Pydantic
    model constructor.

    Args:
        feed: Live game feed dict from :func:`get_live_game_feed`.
        managed_team_side: ``"home"`` or ``"away"`` (which team the agent
            manages).

    Returns:
        A dict with all MatchupState fields populated from the feed.
    """
    situation = extract_game_situation(feed)
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})

    # Determine batting team
    is_top = situation["half"] == "TOP"
    batting_team = "AWAY" if is_top else "HOME"

    # Batter info
    batter = situation.get("batter", {})
    batter_person = _get_person_from_game_data(game_data, batter.get("id"))
    batter_bat_side = batter_person.get("batSide", {}).get("code", "R")

    # Pitcher info
    pitcher = situation.get("pitcher", {})
    pitcher_id = pitcher.get("id")
    pitcher_person = _get_person_from_game_data(game_data, pitcher_id)
    pitcher_throws = pitcher_person.get("pitchHand", {}).get("code", "R")

    # Pitcher in-game stats
    pitcher_stats = extract_pitcher_game_stats(feed, pitcher_id) if pitcher_id else {}

    # On-deck batter
    on_deck = situation.get("on_deck_batter")
    on_deck_info = None
    if on_deck:
        on_deck_person = _get_person_from_game_data(game_data, on_deck.get("id"))
        on_deck_info = {
            "player_id": str(on_deck.get("id", "")),
            "name": on_deck.get("fullName", ""),
            "bats": on_deck_person.get("batSide", {}).get("code", "R"),
        }

    # Runner info
    runners = {}
    for base_name in ("first", "second", "third"):
        runner = situation["runners"].get(base_name)
        if runner:
            runners[base_name] = {
                "player_id": str(runner.get("id", "")),
                "name": runner.get("fullName", ""),
                "sprint_speed": None,
                "sb_success_rate": None,
            }

    return {
        "inning": situation["inning"],
        "half": situation["half"],
        "outs": situation["outs"],
        "count": situation["count"],
        "runners": runners,
        "score": situation["score"],
        "batting_team": batting_team,
        "managed_team": managed_team_side.upper(),
        "batter": {
            "player_id": str(batter.get("id", "")),
            "name": batter.get("fullName", ""),
            "bats": batter_bat_side,
            "lineup_position": 0,  # Would need lineup data to determine
        },
        "pitcher": {
            "player_id": str(pitcher.get("id", "")),
            "name": pitcher.get("fullName", ""),
            "throws": pitcher_throws,
            "pitch_count_today": pitcher_stats.get("pitch_count", 0),
            "batters_faced_today": pitcher_stats.get("batters_faced", 0),
            "times_through_order": 1,  # Not directly available; would need calculation
            "innings_pitched_today": float(
                pitcher_stats.get("innings_pitched", "0.0")
            ),
            "runs_allowed_today": pitcher_stats.get("runs", 0),
            "today_line": {
                "IP": float(pitcher_stats.get("innings_pitched", "0.0")),
                "H": pitcher_stats.get("hits", 0),
                "R": pitcher_stats.get("runs", 0),
                "ER": pitcher_stats.get("earned_runs", 0),
                "BB": pitcher_stats.get("walks", 0),
                "K": pitcher_stats.get("strikeouts", 0),
            },
        },
        "on_deck_batter": on_deck_info,
        "game_status": situation["game_status"],
    }


def _get_person_from_game_data(game_data: dict, player_id: int | None) -> dict:
    """Look up a player's metadata from the gameData.players section.

    Args:
        game_data: The ``gameData`` portion of a live feed.
        player_id: MLB player ID.

    Returns:
        Player metadata dict, or empty dict if not found.
    """
    if player_id is None:
        return {}
    players = game_data.get("players", {})
    key = f"ID{player_id}"
    return players.get(key, {})


# ---------------------------------------------------------------------------
# BvP stats extraction
# ---------------------------------------------------------------------------

def extract_bvp_stats(bvp: dict[str, Any]) -> dict[str, Any]:
    """Extract key batter-vs-pitcher stats into a flat structure.

    Useful for feeding into tool responses or the agent's context.

    Args:
        bvp: Result dict from :func:`get_batter_vs_pitcher`.

    Returns:
        Dict with keys: ``plate_appearances``, ``at_bats``, ``hits``,
        ``doubles``, ``triples``, ``home_runs``, ``walks``,
        ``strikeouts``, ``avg``, ``obp``, ``slg``, ``ops``,
        ``small_sample``, ``batter_name``, ``pitcher_name``.
        Values are ``None`` if no matchup history exists.
    """
    career = bvp.get("career")
    if not career:
        return {
            "plate_appearances": 0,
            "at_bats": 0,
            "hits": 0,
            "doubles": 0,
            "triples": 0,
            "home_runs": 0,
            "walks": 0,
            "strikeouts": 0,
            "avg": None,
            "obp": None,
            "slg": None,
            "ops": None,
            "small_sample": True,
            "batter_name": bvp.get("batter", {}).get("fullName", ""),
            "pitcher_name": bvp.get("pitcher", {}).get("fullName", ""),
        }

    def _safe_int(val: Any) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    def _safe_float(val: Any) -> float | None:
        if val is None or val == "-.--":
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    return {
        "plate_appearances": _safe_int(career.get("plateAppearances")),
        "at_bats": _safe_int(career.get("atBats")),
        "hits": _safe_int(career.get("hits")),
        "doubles": _safe_int(career.get("doubles")),
        "triples": _safe_int(career.get("triples")),
        "home_runs": _safe_int(career.get("homeRuns")),
        "walks": _safe_int(career.get("baseOnBalls")),
        "strikeouts": _safe_int(career.get("strikeOuts")),
        "avg": _safe_float(career.get("avg")),
        "obp": _safe_float(career.get("obp")),
        "slg": _safe_float(career.get("slg")),
        "ops": _safe_float(career.get("ops")),
        "small_sample": bvp.get("small_sample", True),
        "batter_name": bvp.get("batter", {}).get("fullName", ""),
        "pitcher_name": bvp.get("pitcher", {}).get("fullName", ""),
    }
