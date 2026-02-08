# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Decision point extraction from MLB Stats API game feeds.

Walks the play-by-play of a completed game, reconstructing game state at each
at-bat boundary and identifying the real manager's decisions (substitutions,
intentional walks, etc.).  Yields DecisionPoint objects suitable for feeding
to the agent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from enum import Enum
from typing import Any

from pydantic import BaseModel

from decision_quality_wpa import lookup_li, _runners_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and models
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    PITCHING_CHANGE = "PITCHING_CHANGE"
    PINCH_HIT = "PINCH_HIT"
    PINCH_RUN = "PINCH_RUN"
    STOLEN_BASE = "STOLEN_BASE"
    BUNT = "BUNT"
    IBB = "IBB"
    NO_ACTION = "NO_ACTION"


class RealManagerAction(BaseModel):
    action_type: ActionType
    player_in: str | None = None
    player_out: str | None = None
    player_in_name: str | None = None
    player_out_name: str | None = None
    details: str = ""


class PlayerState(BaseModel):
    player_id: str
    name: str
    position: str
    bats: str = "R"
    throws: str = "R"


class BullpenEntry(BaseModel):
    player_id: str
    name: str
    throws: str = "R"
    pitch_count: int = 0
    has_pitched: bool = False
    available: bool = True


class DecisionPoint(BaseModel):
    play_index: int
    inning: int
    half: str  # "TOP" or "BOTTOM"
    outs: int
    runners: dict  # {"first": {...} | None, "second": ..., "third": ...}
    score_home: int
    score_away: int
    batter_id: str
    batter_name: str
    batter_bats: str
    pitcher_id: str
    pitcher_name: str
    pitcher_throws: str
    pitcher_pitch_count: int
    pitcher_batters_faced: int
    pitcher_innings_pitched: float
    pitcher_runs_allowed: int
    current_lineup: list[PlayerState]
    bench: list[PlayerState]
    bullpen: list[BullpenEntry]
    opp_lineup: list[PlayerState]
    opp_bench: list[PlayerState]
    opp_bullpen: list[BullpenEntry]
    on_deck_batter_id: str | None = None
    on_deck_batter_name: str | None = None
    on_deck_batter_bats: str = "R"
    real_manager_action: RealManagerAction | None = None
    real_outcome: str = ""
    leverage_index: float = 1.0
    managed_team_side: str = "home"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_player_meta(game_data: dict, player_id: int) -> dict[str, Any]:
    """Look up player from gameData.players."""
    return game_data.get("players", {}).get(f"ID{player_id}", {})


def _bat_side(game_data: dict, player_id: int) -> str:
    return _get_player_meta(game_data, player_id).get("batSide", {}).get("code", "R")


def _pitch_hand(game_data: dict, player_id: int) -> str:
    return _get_player_meta(game_data, player_id).get("pitchHand", {}).get("code", "R")


def _player_name(game_data: dict, player_id: int) -> str:
    return _get_player_meta(game_data, player_id).get("fullName", str(player_id))


def _parse_ip(ip_str: str) -> float:
    """Parse innings pitched string like '6.1' to float (6.333...)."""
    try:
        return float(ip_str)
    except (ValueError, TypeError):
        return 0.0


def resolve_team_side(feed: dict, managed_team: str) -> str:
    """Resolve managed_team to 'home' or 'away'."""
    if managed_team.lower() in ("home", "away"):
        return managed_team.lower()

    teams = feed.get("gameData", {}).get("teams", {})
    key = managed_team.strip().lower()

    for side in ("home", "away"):
        team = teams.get(side, {})
        team_name = team.get("name", "").lower()
        team_abbrev = team.get("abbreviation", "").lower()
        team_short = team.get("shortName", "").lower()
        team_club = team.get("clubName", "").lower()

        if key in (team_name, team_abbrev, team_short, team_club):
            return side

    raise ValueError(
        f"Cannot resolve team '{managed_team}' to home or away. "
        f"Home: {teams.get('home', {}).get('name', '?')}, "
        f"Away: {teams.get('away', {}).get('name', '?')}"
    )


def _build_initial_lineup(
    boxscore_team: dict, game_data: dict,
) -> list[PlayerState]:
    """Build starting lineup from boxscore batting order."""
    batting_order = boxscore_team.get("battingOrder", [])
    players_dict = boxscore_team.get("players", {})
    lineup = []
    for pid in batting_order:
        key = f"ID{pid}"
        p = players_dict.get(key, {})
        person = p.get("person", {})
        pos = p.get("position", {})
        meta = _get_player_meta(game_data, pid)
        lineup.append(PlayerState(
            player_id=str(pid),
            name=person.get("fullName", meta.get("fullName", str(pid))),
            position=pos.get("abbreviation", ""),
            bats=meta.get("batSide", {}).get("code", "R"),
            throws=meta.get("pitchHand", {}).get("code", "R"),
        ))
    return lineup


def _build_initial_bullpen(
    boxscore_team: dict, game_data: dict, starting_pitcher_id: int,
) -> list[BullpenEntry]:
    """Build bullpen from boxscore bullpen list, excluding starting pitcher."""
    bullpen_ids = boxscore_team.get("bullpen", [])
    entries = []
    for pid in bullpen_ids:
        if pid == starting_pitcher_id:
            continue
        meta = _get_player_meta(game_data, pid)
        entries.append(BullpenEntry(
            player_id=str(pid),
            name=meta.get("fullName", str(pid)),
            throws=meta.get("pitchHand", {}).get("code", "R"),
        ))
    return entries


def _build_initial_bench(
    boxscore_team: dict, game_data: dict,
) -> list[PlayerState]:
    """Build bench from boxscore bench list."""
    bench_ids = boxscore_team.get("bench", [])
    entries = []
    for pid in bench_ids:
        meta = _get_player_meta(game_data, pid)
        pos = meta.get("primaryPosition", {})
        entries.append(PlayerState(
            player_id=str(pid),
            name=meta.get("fullName", str(pid)),
            position=pos.get("abbreviation", ""),
            bats=meta.get("batSide", {}).get("code", "R"),
            throws=meta.get("pitchHand", {}).get("code", "R"),
        ))
    return entries


def _runners_from_pre_play(play: dict) -> dict:
    """Extract runner positions before the play from play.runners movements.

    Returns dict like {"first": {"id": "123", "name": "..."}, "second": None, ...}
    """
    runners: dict[str, dict | None] = {"first": None, "second": None, "third": None}
    base_map = {"1B": "first", "2B": "second", "3B": "third"}

    for r in play.get("runners", []):
        movement = r.get("movement", {})
        start = movement.get("start")
        if start and start in base_map:
            details = r.get("details", {})
            runner_info = details.get("runner", r.get("details", {}).get("runner", {}))
            runner_id = runner_info.get("id", "")
            runner_name = runner_info.get("fullName", "")
            runners[base_map[start]] = {
                "id": str(runner_id),
                "name": runner_name,
            }

    return runners


def _count_pitches_in_play(play: dict) -> int:
    """Count pitches (isPitch=true events) in a play."""
    return sum(
        1 for ev in play.get("playEvents", [])
        if ev.get("isPitch", False)
    )


def _detect_manager_actions(
    play: dict, game_data: dict, managed_side: str,
) -> list[RealManagerAction]:
    """Detect real manager actions from playEvents in a play.

    Only returns actions for the managed team's manager.
    """
    actions: list[RealManagerAction] = []
    about = play.get("about", {})
    is_top = about.get("isTopInning", True)
    # When batting: top = away batting, bottom = home batting
    # When fielding: top = home fielding, bottom = away fielding
    batting_side = "away" if is_top else "home"
    fielding_side = "home" if is_top else "away"

    for ev in play.get("playEvents", []):
        if ev.get("type") != "action":
            continue

        event_name = ev.get("details", {}).get("event", "")
        event_type = ev.get("details", {}).get("eventType", "")
        desc = ev.get("details", {}).get("description", "")
        player_in = ev.get("player", {})
        replaced = ev.get("replacedPlayer", {})

        player_in_id = str(player_in.get("id", "")) if player_in else None
        player_out_id = str(replaced.get("id", "")) if replaced else None
        player_in_name = _player_name(game_data, int(player_in_id)) if player_in_id and player_in_id != "" else None
        player_out_name = _player_name(game_data, int(player_out_id)) if player_out_id and player_out_id != "" else None

        if event_type == "pitching_substitution":
            # Pitching changes are made by the fielding team's manager
            if fielding_side == managed_side:
                actions.append(RealManagerAction(
                    action_type=ActionType.PITCHING_CHANGE,
                    player_in=player_in_id,
                    player_out=player_out_id,
                    player_in_name=player_in_name,
                    player_out_name=player_out_name,
                    details=desc,
                ))

        elif event_type == "offensive_substitution":
            # Offensive subs are by the batting team's manager
            if batting_side == managed_side:
                # Determine if pinch hit or pinch run from description
                if "pinch-runner" in desc.lower() or "pinch runner" in desc.lower():
                    action_type = ActionType.PINCH_RUN
                else:
                    action_type = ActionType.PINCH_HIT
                actions.append(RealManagerAction(
                    action_type=action_type,
                    player_in=player_in_id,
                    player_out=player_out_id,
                    player_in_name=player_in_name,
                    player_out_name=player_out_name,
                    details=desc,
                ))

        elif event_type == "defensive_substitution" or event_type == "defensive_switch":
            # Defensive subs are by the fielding team's manager
            # We track these for state but they're generally not major decisions
            # unless paired with a pitching change (already captured above)
            pass

    # Also check play result for special events
    result_event = play.get("result", {}).get("event", "")
    result_event_type = play.get("result", {}).get("eventType", "")

    if result_event == "Intent Walk" and fielding_side == managed_side:
        actions.append(RealManagerAction(
            action_type=ActionType.IBB,
            details=play.get("result", {}).get("description", ""),
        ))

    if "Stolen Base" in result_event and batting_side == managed_side:
        actions.append(RealManagerAction(
            action_type=ActionType.STOLEN_BASE,
            details=play.get("result", {}).get("description", ""),
        ))

    if "Sac Bunt" in result_event and batting_side == managed_side:
        actions.append(RealManagerAction(
            action_type=ActionType.BUNT,
            details=play.get("result", {}).get("description", ""),
        ))

    return actions


def _apply_substitution_to_lineup(
    lineup: list[PlayerState],
    bench: list[PlayerState],
    bullpen: list[BullpenEntry],
    ev: dict,
    game_data: dict,
) -> None:
    """Mutate lineup/bench/bullpen to reflect a substitution event."""
    event_type = ev.get("details", {}).get("eventType", "")
    player_in = ev.get("player", {})
    replaced = ev.get("replacedPlayer", {})
    position = ev.get("position", {})

    if not player_in:
        return

    player_in_id = str(player_in.get("id", ""))
    replaced_id = str(replaced.get("id", "")) if replaced else ""
    pos_abbrev = position.get("abbreviation", "") if position else ""
    meta = _get_player_meta(game_data, int(player_in_id)) if player_in_id else {}

    if event_type == "pitching_substitution":
        # New pitcher comes in -- update bullpen
        for i, bp in enumerate(bullpen):
            if bp.player_id == player_in_id:
                bullpen[i] = bp.model_copy(update={"has_pitched": True})
                break

    elif event_type in ("offensive_substitution", "defensive_substitution"):
        # Replace player in lineup
        if replaced_id:
            for i, p in enumerate(lineup):
                if p.player_id == replaced_id:
                    new_player = PlayerState(
                        player_id=player_in_id,
                        name=meta.get("fullName", player_in_id),
                        position=pos_abbrev or p.position,
                        bats=meta.get("batSide", {}).get("code", "R"),
                        throws=meta.get("pitchHand", {}).get("code", "R"),
                    )
                    lineup[i] = new_player
                    # Remove from bench if present
                    bench[:] = [b for b in bench if b.player_id != player_in_id]
                    break

    elif event_type == "defensive_switch":
        # Player changes position in lineup (no personnel change)
        if pos_abbrev:
            for i, p in enumerate(lineup):
                if p.player_id == player_in_id:
                    lineup[i] = p.model_copy(update={"position": pos_abbrev})
                    break


# ---------------------------------------------------------------------------
# Pitcher tracking state
# ---------------------------------------------------------------------------

class _PitcherTracker:
    """Track per-pitcher in-game stats as we walk the play-by-play."""

    def __init__(self) -> None:
        self._stats: dict[str, dict] = {}
        self._current: dict[str, str] = {}  # side -> pitcher_id

    def set_current(self, side: str, pitcher_id: str) -> None:
        self._current[side] = pitcher_id
        if pitcher_id not in self._stats:
            self._stats[pitcher_id] = {
                "pitch_count": 0,
                "batters_faced": 0,
                "outs_recorded": 0,
                "runs_allowed": 0,
            }

    def current(self, side: str) -> str:
        return self._current.get(side, "")

    def record_play(self, pitcher_id: str, pitches: int, outs: int,
                    runs: int, faced: int = 1) -> None:
        if pitcher_id not in self._stats:
            self._stats[pitcher_id] = {
                "pitch_count": 0, "batters_faced": 0,
                "outs_recorded": 0, "runs_allowed": 0,
            }
        s = self._stats[pitcher_id]
        s["pitch_count"] += pitches
        s["batters_faced"] += faced
        s["outs_recorded"] += outs
        s["runs_allowed"] += runs

    def get(self, pitcher_id: str) -> dict:
        return self._stats.get(pitcher_id, {
            "pitch_count": 0, "batters_faced": 0,
            "outs_recorded": 0, "runs_allowed": 0,
        })

    def innings_pitched(self, pitcher_id: str) -> float:
        outs = self.get(pitcher_id)["outs_recorded"]
        full = outs // 3
        partial = outs % 3
        return full + partial / 10.0


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def walk_game_feed(
    feed: dict,
    managed_team: str,
) -> Iterator[DecisionPoint]:
    """Walk a completed game feed and yield DecisionPoint at each at-bat.

    Args:
        feed: Full MLB Stats API live game feed dict.
        managed_team: Team name, abbreviation, or "home"/"away".

    Yields:
        DecisionPoint for each plate appearance in the game.
    """
    managed_side = resolve_team_side(feed, managed_team)
    opp_side = "away" if managed_side == "home" else "home"

    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    all_plays = live_data.get("plays", {}).get("allPlays", [])
    linescore = live_data.get("linescore", {})

    # Build initial state from boxscore
    managed_box = boxscore.get("teams", {}).get(managed_side, {})
    opp_box = boxscore.get("teams", {}).get(opp_side, {})

    managed_lineup = _build_initial_lineup(managed_box, game_data)
    managed_bench = _build_initial_bench(managed_box, game_data)
    managed_bullpen = _build_initial_bullpen(managed_box, game_data, 0)

    opp_lineup = _build_initial_lineup(opp_box, game_data)
    opp_bench = _build_initial_bench(opp_box, game_data)
    opp_bullpen = _build_initial_bullpen(opp_box, game_data, 0)

    # Identify starting pitchers from first play
    if all_plays:
        first_play = all_plays[0]
        first_is_top = first_play.get("about", {}).get("isTopInning", True)
        first_pitcher_id = first_play.get("matchup", {}).get("pitcher", {}).get("id")

        if first_is_top:
            # Top of 1st: home team is fielding, so home pitcher
            home_sp_id = first_pitcher_id
        else:
            home_sp_id = None

        # Find away SP from the first bottom-half play
        away_sp_id = None
        for p in all_plays:
            if not p.get("about", {}).get("isTopInning", True):
                away_sp_id = p.get("matchup", {}).get("pitcher", {}).get("id")
                break
            if p.get("about", {}).get("isTopInning", True) and away_sp_id is None:
                pass
        if first_is_top:
            away_sp_id_2 = None
            for p in all_plays:
                if not p.get("about", {}).get("isTopInning", True):
                    away_sp_id_2 = p.get("matchup", {}).get("pitcher", {}).get("id")
                    break
            away_sp_id = away_sp_id_2
            # The first play pitcher is the home SP
        else:
            home_sp_id = None
            for p in all_plays:
                if p.get("about", {}).get("isTopInning", True):
                    home_sp_id = p.get("matchup", {}).get("pitcher", {}).get("id")
                    break
            away_sp_id = first_pitcher_id

        # Rebuild bullpens excluding starting pitchers
        if managed_side == "home":
            managed_bullpen = _build_initial_bullpen(managed_box, game_data, home_sp_id or 0)
            opp_bullpen = _build_initial_bullpen(opp_box, game_data, away_sp_id or 0)
        else:
            managed_bullpen = _build_initial_bullpen(managed_box, game_data, away_sp_id or 0)
            opp_bullpen = _build_initial_bullpen(opp_box, game_data, home_sp_id or 0)

    # Pitcher tracking
    pitcher_tracker = _PitcherTracker()

    # Score tracking
    score_home = 0
    score_away = 0
    current_outs = 0
    prev_inning = 0
    prev_half = ""

    for play_idx, play in enumerate(all_plays):
        about = play.get("about", {})
        inning = about.get("inning", 1)
        is_top = about.get("isTopInning", True)
        half = "TOP" if is_top else "BOTTOM"

        # Half-inning boundary: reset outs
        if inning != prev_inning or half != prev_half:
            current_outs = 0
            prev_inning = inning
            prev_half = half

        # Determine batting/fielding sides for this half-inning
        batting_side = "away" if is_top else "home"
        fielding_side = "home" if is_top else "away"

        # Set current pitcher if first play of a pitching stint
        pitcher_id = play.get("matchup", {}).get("pitcher", {}).get("id")
        if pitcher_id:
            pitcher_tracker.set_current(fielding_side, str(pitcher_id))

        # Process substitution events BEFORE yielding the decision point
        for ev in play.get("playEvents", []):
            if ev.get("type") != "action" or not ev.get("isSubstitution"):
                continue

            event_type = ev.get("details", {}).get("eventType", "")

            if event_type == "pitching_substitution":
                new_pitcher = ev.get("player", {})
                if new_pitcher:
                    pitcher_tracker.set_current(fielding_side, str(new_pitcher["id"]))

            # Apply to appropriate lineup
            if fielding_side == managed_side or batting_side == managed_side:
                if event_type in ("pitching_substitution", "defensive_substitution", "defensive_switch"):
                    if fielding_side == managed_side:
                        _apply_substitution_to_lineup(
                            managed_lineup, managed_bench, managed_bullpen, ev, game_data)
                    else:
                        _apply_substitution_to_lineup(
                            opp_lineup, opp_bench, opp_bullpen, ev, game_data)
                elif event_type == "offensive_substitution":
                    if batting_side == managed_side:
                        _apply_substitution_to_lineup(
                            managed_lineup, managed_bench, managed_bullpen, ev, game_data)
                    else:
                        _apply_substitution_to_lineup(
                            opp_lineup, opp_bench, opp_bullpen, ev, game_data)

        # Detect real manager actions
        actions = _detect_manager_actions(play, game_data, managed_side)
        real_action = actions[0] if actions else None

        # Get current pitcher for this at-bat
        current_pitcher_id = pitcher_tracker.current(fielding_side)
        pitcher_stats = pitcher_tracker.get(current_pitcher_id)

        # Batter info
        matchup = play.get("matchup", {})
        batter = matchup.get("batter", {})
        batter_id = str(batter.get("id", ""))
        batter_name = batter.get("fullName", "")
        batter_bats = matchup.get("batSide", {}).get("code", _bat_side(game_data, int(batter_id) if batter_id else 0))
        pitcher_throws = matchup.get("pitchHand", {}).get("code", _pitch_hand(game_data, int(current_pitcher_id) if current_pitcher_id else 0))

        # Runners pre-play
        runners = _runners_from_pre_play(play)

        # On-deck batter: next batter in the lineup
        # Determine from batting order
        if batting_side == managed_side:
            on_deck = _find_on_deck(managed_lineup, batter_id)
        else:
            on_deck = _find_on_deck(opp_lineup, batter_id)

        # Compute leverage index
        is_home = managed_side == "home"
        score_diff = (score_home - score_away) if is_home else (score_away - score_home)
        runner_key = _runners_key(
            runners.get("first") is not None,
            runners.get("second") is not None,
            runners.get("third") is not None,
        )
        li = lookup_li(inning, half, min(current_outs, 2), runner_key, score_diff, is_home)

        # Determine which lineup/bench/bullpen to show
        if managed_side == "home":
            cur_lineup = managed_lineup if batting_side == "home" or fielding_side == "home" else managed_lineup
            cur_bench = managed_bench
            cur_bullpen = managed_bullpen
            cur_opp_lineup = opp_lineup
            cur_opp_bench = opp_bench
            cur_opp_bullpen = opp_bullpen
        else:
            cur_lineup = managed_lineup
            cur_bench = managed_bench
            cur_bullpen = managed_bullpen
            cur_opp_lineup = opp_lineup
            cur_opp_bench = opp_bench
            cur_opp_bullpen = opp_bullpen

        # Build and yield the decision point
        dp = DecisionPoint(
            play_index=play_idx,
            inning=inning,
            half=half,
            outs=min(current_outs, 2),
            runners=runners,
            score_home=score_home,
            score_away=score_away,
            batter_id=batter_id,
            batter_name=batter_name,
            batter_bats=batter_bats,
            pitcher_id=current_pitcher_id,
            pitcher_name=_player_name(game_data, int(current_pitcher_id)) if current_pitcher_id else "",
            pitcher_throws=pitcher_throws,
            pitcher_pitch_count=pitcher_stats["pitch_count"],
            pitcher_batters_faced=pitcher_stats["batters_faced"],
            pitcher_innings_pitched=pitcher_tracker.innings_pitched(current_pitcher_id),
            pitcher_runs_allowed=pitcher_stats["runs_allowed"],
            current_lineup=[p.model_copy() for p in cur_lineup],
            bench=[p.model_copy() for p in cur_bench],
            bullpen=[p.model_copy() for p in cur_bullpen],
            opp_lineup=[p.model_copy() for p in cur_opp_lineup],
            opp_bench=[p.model_copy() for p in cur_opp_bench],
            opp_bullpen=[p.model_copy() for p in cur_opp_bullpen],
            on_deck_batter_id=on_deck.get("id") if on_deck else None,
            on_deck_batter_name=on_deck.get("name") if on_deck else None,
            on_deck_batter_bats=on_deck.get("bats", "R") if on_deck else "R",
            real_manager_action=real_action,
            real_outcome=play.get("result", {}).get("event", ""),
            leverage_index=li,
            managed_team_side=managed_side,
        )
        yield dp

        # After yielding: update state from play result
        pitches_in_play = _count_pitches_in_play(play)

        # Count outs produced in this play
        result = play.get("result", {})
        play_outs = 0
        for r in play.get("runners", []):
            if r.get("movement", {}).get("isOut", False):
                play_outs += 1

        # Runs scored
        runs_scored = 0
        for r in play.get("runners", []):
            if r.get("movement", {}).get("end") == "score":
                runs_scored += 1
        # Also check result rbi / awayScore / homeScore
        new_away = result.get("awayScore", score_away)
        new_home = result.get("homeScore", score_home)
        runs_this_play_away = new_away - score_away
        runs_this_play_home = new_home - score_home

        # Update pitcher stats
        if current_pitcher_id:
            pitcher_tracker.record_play(
                current_pitcher_id,
                pitches=pitches_in_play,
                outs=play_outs,
                runs=runs_this_play_away if fielding_side == "home" else runs_this_play_home,
            )

        # Update score and outs
        score_away = new_away
        score_home = new_home
        current_outs += play_outs


def _find_on_deck(lineup: list[PlayerState], current_batter_id: str) -> dict | None:
    """Find the on-deck batter given the current batter."""
    if not lineup:
        return None
    for i, p in enumerate(lineup):
        if p.player_id == current_batter_id:
            next_idx = (i + 1) % len(lineup)
            nxt = lineup[next_idx]
            return {"id": nxt.player_id, "name": nxt.name, "bats": nxt.bats}
    return None
