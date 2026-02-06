# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Game state ingestion module.

Handles parsing raw game state JSON payloads into the agent's input models
(MatchupState, RosterState, OpponentRosterState).  Supports two input
formats:

1. **MLB Stats API live game feed** -- the raw JSON from
   ``statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live``, optionally
   accompanied by roster data.
2. **Pre-parsed intermediate format** -- a dict with keys
   ``matchup_state``, ``roster_state``, and ``opponent_roster_state``
   that map directly to the Pydantic models (e.g. from the simulation
   engine's ``game_state_to_scenario()``).

All ingestion paths validate the result with the Pydantic models and return
clear errors for missing or invalid data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from models import (
    BatterInfo,
    BattingTeam,
    BenchPlayer,
    BullpenPitcher,
    BullpenRole,
    Count,
    Freshness,
    Half,
    Hand,
    LineupPlayer,
    MatchupState,
    OnDeckBatter,
    OpponentBenchPlayer,
    OpponentBullpenPitcher,
    OpponentRosterState,
    PitcherInfo,
    RosterState,
    Runner,
    Runners,
    Score,
    ThrowHand,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Raised when a game state payload cannot be ingested."""

    def __init__(self, message: str, field: str | None = None,
                 details: list[str] | None = None):
        self.field = field
        self.details = details or []
        super().__init__(message)


class IngestionValidationError(IngestionError):
    """Raised when a parsed game state fails Pydantic validation."""

    def __init__(self, message: str, validation_errors: list[dict]):
        self.validation_errors = validation_errors
        super().__init__(message, details=[
            f"{e.get('loc', '?')}: {e.get('msg', '?')}" for e in validation_errors
        ])


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(payload: dict[str, Any]) -> str:
    """Detect the format of a game state payload.

    Returns:
        ``"mlb_api"`` if the payload looks like an MLB Stats API live feed,
        ``"intermediate"`` if it has the pre-parsed keys, or
        ``"unknown"`` otherwise.
    """
    if "gameData" in payload and "liveData" in payload:
        return "mlb_api"
    if "matchup_state" in payload and "roster_state" in payload:
        return "intermediate"
    return "unknown"


# ---------------------------------------------------------------------------
# MLB Stats API feed -> intermediate dict helpers
# ---------------------------------------------------------------------------

def _extract_half(is_top: bool) -> str:
    return "TOP" if is_top else "BOTTOM"


def _extract_batting_team(half: str) -> str:
    return "AWAY" if half == "TOP" else "HOME"


def _get_person(game_data: dict, player_id: int | None) -> dict:
    """Lookup a player in the gameData.players section."""
    if player_id is None:
        return {}
    players = game_data.get("players", {})
    return players.get(f"ID{player_id}", {})


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compute_times_through_order(batters_faced: int) -> int:
    """Estimate times through the order from batters faced."""
    if batters_faced <= 0:
        return 1
    return max(1, (batters_faced - 1) // 9 + 1)


def _extract_pitcher_stats_from_boxscore(
    feed: dict[str, Any], pitcher_id: int
) -> dict[str, Any]:
    """Extract a pitcher's in-game stats from the boxscore."""
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})

    for side in ("away", "home"):
        team_box = boxscore.get("teams", {}).get(side, {})
        players = team_box.get("players", {})
        key = f"ID{pitcher_id}"
        if key in players:
            pitching = players[key].get("stats", {}).get("pitching", {})
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


def _extract_lineup_from_boxscore(
    feed: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    """Extract a team's active lineup from the boxscore.

    Returns a list of player dicts in batting order, each with
    player_id, name, position, bats, and in_game status.
    """
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    game_data = feed.get("gameData", {})

    team_box = boxscore.get("teams", {}).get(side, {})
    batting_order = team_box.get("battingOrder", [])
    players_data = team_box.get("players", {})

    lineup = []
    for pid in batting_order:
        key = f"ID{pid}"
        player_box = players_data.get(key, {})
        person = player_box.get("person", {})
        position = player_box.get("position", {})

        # Get handedness from gameData.players
        person_detail = _get_person(game_data, pid)
        bats = person_detail.get("batSide", {}).get("code", "R")

        lineup.append({
            "player_id": str(pid),
            "name": person.get("fullName", ""),
            "position": position.get("abbreviation", ""),
            "bats": bats,
            "in_game": True,
        })

    return lineup


def _extract_bullpen_from_boxscore(
    feed: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    """Extract bullpen pitchers from the boxscore.

    Returns a list of reliever dicts with availability info.
    """
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    game_data = feed.get("gameData", {})

    team_box = boxscore.get("teams", {}).get(side, {})
    bullpen_ids = team_box.get("bullpen", [])
    pitchers_used = team_box.get("pitchers", [])
    players_data = team_box.get("players", {})

    relievers = []
    for pid in bullpen_ids:
        key = f"ID{pid}"
        player_box = players_data.get(key, {})
        person = player_box.get("person", {})

        person_detail = _get_person(game_data, pid)
        throws = person_detail.get("pitchHand", {}).get("code", "R")

        # Pitcher is available if not already used in this game
        used = pid in pitchers_used

        relievers.append({
            "player_id": str(pid),
            "name": person.get("fullName", ""),
            "throws": throws,
            "role": "MIDDLE",  # Role not available from API; default
            "available": not used,
            "freshness": "FRESH",
            "pitches_last_3_days": [0, 0, 0],
            "days_since_last_appearance": 5,
            "is_warming_up": False,
        })

    return relievers


def _extract_bench_from_boxscore(
    feed: dict[str, Any], side: str
) -> list[dict[str, Any]]:
    """Extract bench players from the boxscore.

    Returns a list of bench player dicts with availability info.
    """
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    game_data = feed.get("gameData", {})

    team_box = boxscore.get("teams", {}).get(side, {})
    bench_ids = team_box.get("bench", [])
    players_data = team_box.get("players", {})

    bench = []
    for pid in bench_ids:
        key = f"ID{pid}"
        player_box = players_data.get(key, {})
        person = player_box.get("person", {})

        person_detail = _get_person(game_data, pid)
        bats = person_detail.get("batSide", {}).get("code", "R")
        position = person_detail.get("primaryPosition", {})
        pos_abbrev = position.get("abbreviation", "")

        bench.append({
            "player_id": str(pid),
            "name": person.get("fullName", ""),
            "bats": bats,
            "positions": [pos_abbrev] if pos_abbrev else [],
            "available": True,
        })

    return bench


def _extract_runner(
    runner_data: dict | None, game_data: dict
) -> dict[str, Any] | None:
    """Extract runner info from a linescore offense entry.

    Populates sprint_speed and sb_success_rate with defaults when
    real Statcast data is not available.
    """
    if runner_data is None:
        return None

    pid = runner_data.get("id")
    person = _get_person(game_data, pid)
    name = runner_data.get("fullName", person.get("fullName", ""))

    return {
        "player_id": str(pid) if pid else "",
        "name": name,
        "sprint_speed": 27.0,  # League-average default (ft/s)
        "sb_success_rate": 0.70,  # League-average default
    }


# ---------------------------------------------------------------------------
# MLB Stats API feed ingestion
# ---------------------------------------------------------------------------

def ingest_mlb_api_feed(
    feed: dict[str, Any],
    managed_team_side: str = "home",
    roster_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an MLB Stats API live game feed to agent input models.

    Args:
        feed: Live game feed dict (from ``/api/v1.1/game/{pk}/feed/live``).
        managed_team_side: ``"home"`` or ``"away"``.
        roster_overrides: Optional dict with keys ``bench``, ``bullpen``,
            ``mound_visits_remaining``, ``challenge_available`` to
            supplement data not available from the feed alone.

    Returns:
        A dict with ``matchup_state``, ``roster_state``, and
        ``opponent_roster_state`` keys, each containing a validated
        Pydantic model instance.

    Raises:
        IngestionError: If required data is missing from the feed.
        IngestionValidationError: If the parsed data fails Pydantic validation.
    """
    if "gameData" not in feed or "liveData" not in feed:
        raise IngestionError(
            "Feed missing required top-level keys: gameData, liveData",
            field="feed",
        )

    managed_team_side = managed_team_side.lower()
    if managed_team_side not in ("home", "away"):
        raise IngestionError(
            f"managed_team_side must be 'home' or 'away', got '{managed_team_side}'",
            field="managed_team_side",
        )

    overrides = roster_overrides or {}
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    linescore = live_data.get("linescore", {})

    # Game status check
    status = game_data.get("status", {})
    game_status = status.get("abstractGameState", "Unknown")

    # Inning / half
    inning = linescore.get("currentInning")
    if inning is None or inning < 1:
        raise IngestionError(
            "Cannot determine current inning from feed",
            field="linescore.currentInning",
        )

    is_top = linescore.get("isTopInning", True)
    half = _extract_half(is_top)
    batting_team = _extract_batting_team(half)

    # Outs
    outs = _safe_int(linescore.get("outs"), 0)

    # Score
    teams_data = linescore.get("teams", {})
    score_home = _safe_int(teams_data.get("home", {}).get("runs"), 0)
    score_away = _safe_int(teams_data.get("away", {}).get("runs"), 0)

    # Current play for count, batter, pitcher
    plays = live_data.get("plays", {})
    current_play = plays.get("currentPlay", {})
    matchup_data = current_play.get("matchup", {})
    count_data = current_play.get("count", {})

    # Batter
    batter_data = matchup_data.get("batter", {})
    batter_id = batter_data.get("id")
    if batter_id is None:
        raise IngestionError(
            "Cannot determine current batter from feed",
            field="currentPlay.matchup.batter",
        )
    batter_person = _get_person(game_data, batter_id)
    batter_bats = batter_person.get("batSide", {}).get("code", "R")

    # Pitcher
    pitcher_data = matchup_data.get("pitcher", {})
    pitcher_id = pitcher_data.get("id")
    if pitcher_id is None:
        raise IngestionError(
            "Cannot determine current pitcher from feed",
            field="currentPlay.matchup.pitcher",
        )
    pitcher_person = _get_person(game_data, pitcher_id)
    pitcher_throws = pitcher_person.get("pitchHand", {}).get("code", "R")

    # Pitcher in-game stats
    pitcher_stats = _extract_pitcher_stats_from_boxscore(feed, pitcher_id)
    pitch_count = _safe_int(pitcher_stats.get("pitch_count"), 0)
    batters_faced = _safe_int(pitcher_stats.get("batters_faced"), 0)
    ip_str = pitcher_stats.get("innings_pitched", "0.0")
    innings_pitched = _safe_float(ip_str, 0.0)
    runs_allowed = _safe_int(pitcher_stats.get("runs"), 0)
    tto = _compute_times_through_order(batters_faced)

    # On-deck batter
    offense = linescore.get("offense", {})
    on_deck_data = offense.get("ondeck")
    if on_deck_data:
        on_deck_person = _get_person(game_data, on_deck_data.get("id"))
        on_deck_bats = on_deck_person.get("batSide", {}).get("code", "R")
        on_deck_dict = {
            "player_id": str(on_deck_data.get("id", "")),
            "name": on_deck_data.get("fullName", ""),
            "bats": on_deck_bats,
        }
    else:
        on_deck_dict = {
            "player_id": "unknown",
            "name": "Unknown",
            "bats": "R",
        }

    # Runners
    runners_dict: dict[str, Any] = {}
    for base_name in ("first", "second", "third"):
        runner_data = offense.get(base_name)
        extracted = _extract_runner(runner_data, game_data)
        if extracted:
            runners_dict[base_name] = extracted

    # Determine lineup position from batting order
    our_side = managed_team_side
    opp_side = "away" if our_side == "home" else "home"

    # Build matchup_state
    matchup_state_dict = {
        "inning": inning,
        "half": half,
        "outs": outs,
        "count": {
            "balls": _safe_int(count_data.get("balls"), 0),
            "strikes": _safe_int(count_data.get("strikes"), 0),
        },
        "runners": runners_dict,
        "score": {"home": score_home, "away": score_away},
        "batting_team": batting_team,
        "batter": {
            "player_id": str(batter_id),
            "name": batter_data.get("fullName", ""),
            "bats": batter_bats,
            "lineup_position": 1,  # Updated below if available
        },
        "pitcher": {
            "player_id": str(pitcher_id),
            "name": pitcher_data.get("fullName", ""),
            "throws": pitcher_throws,
            "pitch_count_today": pitch_count,
            "batters_faced_today": batters_faced,
            "times_through_order": tto,
            "innings_pitched_today": innings_pitched,
            "runs_allowed_today": runs_allowed,
            "today_line": {
                "IP": innings_pitched,
                "H": _safe_int(pitcher_stats.get("hits"), 0),
                "R": runs_allowed,
                "ER": _safe_int(pitcher_stats.get("earned_runs"), 0),
                "BB": _safe_int(pitcher_stats.get("walks"), 0),
                "K": _safe_int(pitcher_stats.get("strikeouts"), 0),
            },
        },
        "on_deck_batter": on_deck_dict,
    }

    # Determine batting lineup position from boxscore batting order
    batting_side = "away" if is_top else "home"
    batting_box = live_data.get("boxscore", {}).get("teams", {}).get(batting_side, {})
    batting_order = batting_box.get("battingOrder", [])
    for idx, pid in enumerate(batting_order):
        if pid == batter_id:
            matchup_state_dict["batter"]["lineup_position"] = idx + 1
            break

    # Build roster_state
    our_lineup = _extract_lineup_from_boxscore(feed, our_side)
    our_bench = overrides.get("bench", _extract_bench_from_boxscore(feed, our_side))
    our_bullpen = overrides.get("bullpen", _extract_bullpen_from_boxscore(feed, our_side))

    # Determine lineup position for our team
    our_box = live_data.get("boxscore", {}).get("teams", {}).get(our_side, {})
    our_batting_order = our_box.get("battingOrder", [])
    our_lineup_position = 0  # default

    roster_state_dict = {
        "our_lineup": our_lineup,
        "our_lineup_position": our_lineup_position,
        "bench": our_bench,
        "bullpen": our_bullpen,
        "mound_visits_remaining": overrides.get("mound_visits_remaining", 5),
        "challenge_available": overrides.get("challenge_available", True),
    }

    # Build opponent_roster_state
    opp_lineup = _extract_lineup_from_boxscore(feed, opp_side)
    opp_bench = overrides.get("their_bench", _extract_bench_from_boxscore(feed, opp_side))
    opp_bullpen = overrides.get("their_bullpen", _extract_bullpen_from_boxscore(feed, opp_side))

    # Convert opponent bench to the simpler format
    opp_bench_simple = []
    for bp in opp_bench:
        opp_bench_simple.append({
            "player_id": bp.get("player_id", ""),
            "name": bp.get("name", ""),
            "bats": bp.get("bats", "R"),
            "available": bp.get("available", True),
        })

    # Convert opponent bullpen to simpler format
    opp_bullpen_simple = []
    for bp in opp_bullpen:
        opp_bullpen_simple.append({
            "player_id": bp.get("player_id", ""),
            "name": bp.get("name", ""),
            "throws": bp.get("throws", "R"),
            "role": bp.get("role", "MIDDLE"),
            "available": bp.get("available", True),
            "freshness": bp.get("freshness", "FRESH"),
        })

    opponent_roster_dict = {
        "their_lineup": opp_lineup,
        "their_lineup_position": 0,
        "their_bench": opp_bench_simple,
        "their_bullpen": opp_bullpen_simple,
    }

    return _validate_and_build(
        matchup_state_dict,
        roster_state_dict,
        opponent_roster_dict,
    )


# ---------------------------------------------------------------------------
# Intermediate format ingestion
# ---------------------------------------------------------------------------

def ingest_intermediate(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a pre-parsed intermediate payload to validated models.

    The intermediate format is the dict produced by the simulation engine's
    ``game_state_to_scenario()`` function, with keys ``matchup_state``,
    ``roster_state``, and ``opponent_roster_state``.

    Args:
        payload: Dict with the three required keys.

    Returns:
        A dict with ``matchup_state``, ``roster_state``, and
        ``opponent_roster_state`` keys, each containing a validated
        Pydantic model instance.

    Raises:
        IngestionError: If required keys are missing.
        IngestionValidationError: If data fails Pydantic validation.
    """
    missing = []
    for key in ("matchup_state", "roster_state", "opponent_roster_state"):
        if key not in payload:
            missing.append(key)
    if missing:
        raise IngestionError(
            f"Intermediate payload missing required keys: {', '.join(missing)}",
            field="payload",
            details=missing,
        )

    return _validate_and_build(
        payload["matchup_state"],
        payload["roster_state"],
        payload["opponent_roster_state"],
    )


# ---------------------------------------------------------------------------
# Validation and model construction
# ---------------------------------------------------------------------------

def _validate_and_build(
    matchup_dict: dict[str, Any],
    roster_dict: dict[str, Any],
    opponent_dict: dict[str, Any],
) -> dict[str, Any]:
    """Validate dicts and build Pydantic model instances.

    Returns:
        Dict with ``matchup_state``, ``roster_state``, and
        ``opponent_roster_state`` values as Pydantic models.

    Raises:
        IngestionValidationError: On any validation failure.
    """
    errors: list[dict] = []

    # --- MatchupState ---
    matchup_state = None
    try:
        matchup_state = MatchupState(**matchup_dict)
    except ValidationError as exc:
        for e in exc.errors():
            errors.append({
                "model": "MatchupState",
                "loc": str(e.get("loc", "")),
                "msg": e.get("msg", ""),
                "type": e.get("type", ""),
            })

    # --- RosterState ---
    roster_state = None
    try:
        roster_state = RosterState(**roster_dict)
    except ValidationError as exc:
        for e in exc.errors():
            errors.append({
                "model": "RosterState",
                "loc": str(e.get("loc", "")),
                "msg": e.get("msg", ""),
                "type": e.get("type", ""),
            })

    # --- OpponentRosterState ---
    opponent_state = None
    try:
        opponent_state = OpponentRosterState(**opponent_dict)
    except ValidationError as exc:
        for e in exc.errors():
            errors.append({
                "model": "OpponentRosterState",
                "loc": str(e.get("loc", "")),
                "msg": e.get("msg", ""),
                "type": e.get("type", ""),
            })

    if errors:
        raise IngestionValidationError(
            f"Validation failed with {len(errors)} error(s)",
            validation_errors=errors,
        )

    return {
        "matchup_state": matchup_state,
        "roster_state": roster_state,
        "opponent_roster_state": opponent_state,
    }


# ---------------------------------------------------------------------------
# Unified ingestion entry point
# ---------------------------------------------------------------------------

def ingest_game_state(
    payload: dict[str, Any] | str,
    managed_team_side: str = "home",
    roster_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a game state payload and return validated Pydantic models.

    This is the primary entry point. It auto-detects the payload format
    and delegates to the appropriate parser.

    Args:
        payload: Either a JSON string or a dict. Accepted formats:
            - MLB Stats API live game feed (has ``gameData``, ``liveData``)
            - Pre-parsed intermediate (has ``matchup_state``, ``roster_state``,
              ``opponent_roster_state``)
        managed_team_side: ``"home"`` or ``"away"`` (for MLB API feeds).
        roster_overrides: Optional overrides for roster data not in the
            live feed (e.g. bench availability, bullpen rest days).

    Returns:
        Dict with keys:
            - ``matchup_state``: validated :class:`MatchupState`
            - ``roster_state``: validated :class:`RosterState`
            - ``opponent_roster_state``: validated :class:`OpponentRosterState`

    Raises:
        IngestionError: If the payload format is not recognized or required
            data is missing.
        IngestionValidationError: If parsed data fails Pydantic validation.
    """
    # Parse JSON string if needed
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"Invalid JSON payload: {exc}",
                field="payload",
            ) from exc

    if not isinstance(payload, dict):
        raise IngestionError(
            f"Payload must be a dict or JSON string, got {type(payload).__name__}",
            field="payload",
        )

    fmt = detect_format(payload)

    if fmt == "mlb_api":
        return ingest_mlb_api_feed(payload, managed_team_side, roster_overrides)
    elif fmt == "intermediate":
        return ingest_intermediate(payload)
    else:
        raise IngestionError(
            "Unrecognized payload format. Expected MLB Stats API live feed "
            "(with 'gameData' and 'liveData') or intermediate format "
            "(with 'matchup_state', 'roster_state', 'opponent_roster_state').",
            field="payload",
        )


# ---------------------------------------------------------------------------
# JSON file ingestion convenience
# ---------------------------------------------------------------------------

def ingest_from_file(
    path: str | Path,
    managed_team_side: str = "home",
    roster_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a game state JSON file and ingest it.

    Args:
        path: Path to a JSON file containing a game state payload.
        managed_team_side: ``"home"`` or ``"away"``.
        roster_overrides: Optional roster overrides.

    Returns:
        Same as :func:`ingest_game_state`.

    Raises:
        FileNotFoundError: If the file does not exist.
        IngestionError: On parse/validation errors.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Game state file not found: {path}")

    with open(p) as f:
        data = json.load(f)

    return ingest_game_state(data, managed_team_side, roster_overrides)
