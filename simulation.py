# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Baseball game simulation engine.

Resolves play outcomes at the pitch level based on player attributes,
situational context, and probability distributions. Maintains the
authoritative game state, enforces baseball rules, and applies
managerial decisions.

All randomness is seeded for deterministic replay.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Load roster data
# ---------------------------------------------------------------------------

_ROSTER_PATH = Path(__file__).resolve().parent / "data" / "sample_rosters.json"


def load_rosters(path: Path | None = None) -> dict:
    """Load both team rosters from JSON."""
    p = path or _ROSTER_PATH
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Player runtime representation
# ---------------------------------------------------------------------------

@dataclass
class SimPlayer:
    """Runtime player data for the simulation."""
    player_id: str
    name: str
    primary_position: str
    bats: str  # "L", "R", "S"
    throws: str  # "L", "R"
    # Batter attributes (None for pitcher-only)
    contact: float = 65.0
    power: float = 50.0
    speed: float = 50.0
    eye: float = 60.0
    avg_vs_l: float = 0.250
    avg_vs_r: float = 0.250
    # Pitcher attributes (None for position players)
    stuff: float = 0.0
    control: float = 0.0
    stamina: float = 0.0
    velocity: float = 90.0
    era_vs_l: float = 4.50
    era_vs_r: float = 4.50
    is_pitcher: bool = False
    # Fielder attributes
    fielder_range: float = 50.0
    arm_strength: float = 60.0
    error_rate: float = 0.04
    positions: list[str] = field(default_factory=list)
    # Catcher specific
    pop_time: float = 2.0
    framing: float = 50.0
    # Bullpen role info
    role: str = ""

    @classmethod
    def from_roster_dict(cls, d: dict) -> SimPlayer:
        """Build a SimPlayer from a roster JSON entry."""
        batter = d.get("batter", {})
        pitcher = d.get("pitcher", {})
        fielder = d.get("fielder", {})
        catcher = d.get("catcher", {})
        return cls(
            player_id=d["player_id"],
            name=d["name"],
            primary_position=d.get("primary_position", "DH"),
            bats=d.get("bats", "R"),
            throws=d.get("throws", "R"),
            contact=batter.get("contact", 65.0),
            power=batter.get("power", 50.0),
            speed=batter.get("speed", 50.0),
            eye=batter.get("eye", 60.0),
            avg_vs_l=batter.get("avg_vs_l", 0.250),
            avg_vs_r=batter.get("avg_vs_r", 0.250),
            stuff=pitcher.get("stuff", 0.0),
            control=pitcher.get("control", 0.0),
            stamina=pitcher.get("stamina", 0.0),
            velocity=pitcher.get("velocity", 90.0),
            era_vs_l=pitcher.get("era_vs_l", 4.50),
            era_vs_r=pitcher.get("era_vs_r", 4.50),
            is_pitcher=bool(pitcher),
            fielder_range=fielder.get("range", 50.0),
            arm_strength=fielder.get("arm_strength", 60.0),
            error_rate=fielder.get("error_rate", 0.04),
            positions=fielder.get("positions", []),
            pop_time=catcher.get("pop_time", 2.0),
            framing=catcher.get("framing", 50.0),
            role=d.get("role", ""),
        )


# ---------------------------------------------------------------------------
# In-game stat tracking
# ---------------------------------------------------------------------------

@dataclass
class BatterGameStats:
    ab: int = 0
    hits: int = 0
    runs: int = 0
    rbi: int = 0
    bb: int = 0
    k: int = 0
    hbp: int = 0
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    sb: int = 0
    cs: int = 0

    @property
    def pa(self) -> int:
        return self.ab + self.bb + self.hbp

    def to_dict(self) -> dict:
        return {
            "AB": self.ab, "H": self.hits, "R": self.runs,
            "RBI": self.rbi, "BB": self.bb, "K": self.k,
            "2B": self.doubles, "3B": self.triples, "HR": self.hr,
            "SB": self.sb, "CS": self.cs, "HBP": self.hbp,
        }


@dataclass
class PitcherGameStats:
    ip_outs: int = 0  # outs recorded (3 = 1.0 IP)
    hits: int = 0
    runs: int = 0
    earned_runs: int = 0
    bb: int = 0
    k: int = 0
    pitches: int = 0
    batters_faced: int = 0
    hr_allowed: int = 0

    @property
    def ip(self) -> float:
        full = self.ip_outs // 3
        partial = self.ip_outs % 3
        return full + partial / 10.0

    def to_dict(self) -> dict:
        return {
            "IP": self.ip, "H": self.hits, "R": self.runs,
            "ER": self.earned_runs, "BB": self.bb, "K": self.k,
            "pitches": self.pitches, "HR": self.hr_allowed,
            "batters_faced": self.batters_faced,
        }


# ---------------------------------------------------------------------------
# Base runner representation
# ---------------------------------------------------------------------------

@dataclass
class BaseRunner:
    player: SimPlayer
    start_base: int  # 1, 2, or 3

    def __repr__(self) -> str:
        return f"Runner({self.player.name}@{self.start_base}B)"


# ---------------------------------------------------------------------------
# Play-by-play event
# ---------------------------------------------------------------------------

@dataclass
class PlayEvent:
    inning: int
    half: str  # "TOP" or "BOTTOM"
    outs_before: int
    description: str
    event_type: str  # "pitch", "in_play", "walk", "strikeout", "hbp", "steal", "decision", "inning_change", "game_end"
    score_home: int = 0
    score_away: int = 0
    runs_scored: int = 0
    batter_id: str = ""
    pitcher_id: str = ""

    def to_dict(self) -> dict:
        return {
            "inning": self.inning,
            "half": self.half,
            "outs_before": self.outs_before,
            "description": self.description,
            "event_type": self.event_type,
            "score": {"home": self.score_home, "away": self.score_away},
            "runs_scored": self.runs_scored,
        }


# ---------------------------------------------------------------------------
# Team game state
# ---------------------------------------------------------------------------

@dataclass
class TeamState:
    """Mutable state for one team during a game."""
    name: str
    lineup: list[SimPlayer]  # 9-player batting order
    lineup_positions: list[str]  # defensive position for each lineup slot
    bench: list[SimPlayer]
    bullpen: list[SimPlayer]
    starting_pitcher: SimPlayer
    current_pitcher: SimPlayer | None = None
    lineup_index: int = 0  # who bats next (0-8)

    # Track players already used
    used_pitchers: list[str] = field(default_factory=list)
    removed_players: list[str] = field(default_factory=list)

    # Pitcher tracking
    current_pitcher_batters_faced_this_stint: int = 0  # for 3-batter minimum

    # Stats tracking
    batter_stats: dict[str, BatterGameStats] = field(default_factory=dict)
    pitcher_stats: dict[str, PitcherGameStats] = field(default_factory=dict)

    # Mound visits and challenges
    mound_visits_remaining: int = 5
    challenge_available: bool = True

    # Innings scored tracking for box score
    inning_runs: list[int] = field(default_factory=list)

    def current_batter(self) -> SimPlayer:
        return self.lineup[self.lineup_index]

    def advance_batter(self) -> SimPlayer:
        self.lineup_index = (self.lineup_index + 1) % 9
        return self.lineup[self.lineup_index]

    def get_batter_stats(self, player_id: str) -> BatterGameStats:
        if player_id not in self.batter_stats:
            self.batter_stats[player_id] = BatterGameStats()
        return self.batter_stats[player_id]

    def get_pitcher_stats(self, player_id: str) -> PitcherGameStats:
        if player_id not in self.pitcher_stats:
            self.pitcher_stats[player_id] = PitcherGameStats()
        return self.pitcher_stats[player_id]

    def on_deck_batter(self) -> SimPlayer:
        return self.lineup[(self.lineup_index + 1) % 9]

    def get_player_by_id(self, player_id: str) -> SimPlayer | None:
        for p in self.lineup:
            if p.player_id == player_id:
                return p
        for p in self.bench:
            if p.player_id == player_id:
                return p
        for p in self.bullpen:
            if p.player_id == player_id:
                return p
        if self.starting_pitcher.player_id == player_id:
            return self.starting_pitcher
        if self.current_pitcher and self.current_pitcher.player_id == player_id:
            return self.current_pitcher
        return None


# ---------------------------------------------------------------------------
# Main game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Authoritative game state."""
    home: TeamState
    away: TeamState
    inning: int = 1
    half: str = "TOP"  # "TOP" = away bats, "BOTTOM" = home bats
    outs: int = 0
    score_home: int = 0
    score_away: int = 0
    runners: list[BaseRunner] = field(default_factory=list)  # runners currently on base
    play_log: list[PlayEvent] = field(default_factory=list)
    game_over: bool = False
    winning_team: str = ""
    seed: int = 0

    # Current inning runs (for tracking)
    _current_inning_runs: int = field(default=0, repr=False)

    def batting_team(self) -> TeamState:
        return self.away if self.half == "TOP" else self.home

    def fielding_team(self) -> TeamState:
        return self.home if self.half == "TOP" else self.away

    def current_pitcher(self) -> SimPlayer:
        ft = self.fielding_team()
        return ft.current_pitcher or ft.starting_pitcher

    def runner_on(self, base: int) -> BaseRunner | None:
        for r in self.runners:
            if r.start_base == base:
                return r
        return None

    def bases_string(self) -> str:
        """Return base state string like '110' for runners on 1st and 2nd."""
        return (
            ("1" if self.runner_on(1) else "0") +
            ("1" if self.runner_on(2) else "0") +
            ("1" if self.runner_on(3) else "0")
        )

    def score_display(self) -> str:
        return f"Away {self.score_away} - Home {self.score_home}"

    def situation_display(self) -> str:
        half_str = "Top" if self.half == "TOP" else "Bot"
        runners_str = "bases empty"
        on_bases = []
        if self.runner_on(1):
            on_bases.append("1st")
        if self.runner_on(2):
            on_bases.append("2nd")
        if self.runner_on(3):
            on_bases.append("3rd")
        if on_bases:
            runners_str = "runners on " + ", ".join(on_bases)
        return f"{half_str} {self.inning}, {self.outs} out, {runners_str}, {self.score_display()}"


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """Resolves play outcomes and manages game flow.

    Uses player attributes to compute probabilities for:
    - Strike/ball/foul/contact on each pitch
    - Type of contact (groundball, line drive, flyball, popup)
    - Hit direction and result
    - Baserunner advancement
    - Errors
    """

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.seed = seed
        self.rng = random.Random(seed)

    # -------------------------------------------------------------------
    # Probability helpers
    # -------------------------------------------------------------------

    def _clamp(self, v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    def _fatigue_factor(self, pitcher: SimPlayer, pstats: PitcherGameStats) -> float:
        """Returns a factor 0.0-1.0 representing pitcher fatigue.

        0.0 = fresh, 1.0 = completely gassed.
        Based on pitch count vs stamina, and batters faced.
        """
        # Stamina maps to expected pitch count capacity
        # Starter with stamina 70 can handle ~95 pitches
        # Reliever with stamina 30 can handle ~30 pitches
        capacity = 30 + (pitcher.stamina / 100) * 90  # 30-120 pitches
        pitch_fatigue = pstats.pitches / capacity if capacity > 0 else 1.0

        # Times-through-order penalty
        tto = pstats.batters_faced / 9.0  # approximate TTO
        tto_penalty = max(0.0, (tto - 1.0) * 0.05)  # 5% per additional TTO

        fatigue = self._clamp(pitch_fatigue + tto_penalty, 0.0, 1.0)
        return fatigue

    def _platoon_factor(self, batter: SimPlayer, pitcher: SimPlayer) -> float:
        """Returns a factor representing platoon advantage.

        Positive = batter has advantage, negative = pitcher has advantage.
        """
        batter_hand = batter.bats
        pitcher_hand = pitcher.throws

        # Switch hitters face the pitcher's opposite hand
        if batter_hand == "S":
            batter_hand = "R" if pitcher_hand == "L" else "L"

        # Same hand = pitcher advantage (~0.030 wOBA)
        if batter_hand == pitcher_hand:
            return -0.03
        else:
            return 0.03

    def _base_contact_rate(self, batter: SimPlayer, pitcher: SimPlayer,
                           fatigue: float) -> float:
        """Probability batter makes contact on a swing."""
        # Contact skill (0-100) drives contact rate
        # MLB avg contact rate on swings ~75%
        batter_contact = 0.55 + (batter.contact / 100) * 0.30  # 0.55-0.85

        # Pitcher stuff reduces contact rate
        pitcher_stuff_penalty = (pitcher.stuff / 100) * 0.15  # 0-0.15

        # Fatigue reduces pitcher effectiveness (more contact)
        fatigue_bonus = fatigue * 0.08

        rate = batter_contact - pitcher_stuff_penalty + fatigue_bonus
        return self._clamp(rate, 0.40, 0.95)

    def _swing_probability(self, batter: SimPlayer, pitcher: SimPlayer,
                           is_strike: bool, fatigue: float) -> float:
        """Probability batter swings at this pitch."""
        if is_strike:
            # Batters swing at ~70% of pitches in the zone
            base = 0.60 + (batter.contact / 100) * 0.15
            return self._clamp(base, 0.55, 0.80)
        else:
            # Chase rate: eye reduces chasing
            # MLB avg chase rate ~28%
            base = 0.42 - (batter.eye / 100) * 0.28
            # Pitcher with better control throws more borderline pitches
            control_boost = (pitcher.control / 100) * 0.05
            # Fatigue increases miss-location -> less chasing
            fatigue_penalty = fatigue * 0.05
            return self._clamp(base + control_boost - fatigue_penalty, 0.10, 0.45)

    def _strike_zone_probability(self, pitcher: SimPlayer, fatigue: float) -> float:
        """Probability the pitch is in the strike zone."""
        # Control determines zone rate (MLB avg ~45%)
        base = 0.30 + (pitcher.control / 100) * 0.30  # 0.30-0.60
        # Fatigue reduces control
        fatigue_penalty = fatigue * 0.10
        return self._clamp(base - fatigue_penalty, 0.25, 0.60)

    # -------------------------------------------------------------------
    # Pitch resolution
    # -------------------------------------------------------------------

    def resolve_pitch(self, batter: SimPlayer, pitcher: SimPlayer,
                      pstats: PitcherGameStats, balls: int, strikes: int
                      ) -> tuple[str, str | None]:
        """Resolve a single pitch.

        Returns:
            (outcome, detail) where outcome is one of:
            - "ball": ball
            - "called_strike": called strike
            - "swinging_strike": swinging strike
            - "foul": foul ball (strike if < 2 strikes)
            - "in_play": ball put in play (detail describes result)
            - "hbp": hit by pitch
        """
        fatigue = self._fatigue_factor(pitcher, pstats)

        # Small HBP chance (~1% of pitches)
        if self.rng.random() < 0.008 + fatigue * 0.004:
            return ("hbp", None)

        # Is pitch in the zone?
        is_strike = self.rng.random() < self._strike_zone_probability(pitcher, fatigue)

        # Does batter swing?
        swings = self.rng.random() < self._swing_probability(batter, pitcher, is_strike, fatigue)

        if not swings:
            if is_strike:
                return ("called_strike", None)
            else:
                return ("ball", None)

        # Batter swings -- does he make contact?
        contact_rate = self._base_contact_rate(batter, pitcher, fatigue)
        makes_contact = self.rng.random() < contact_rate

        if not makes_contact:
            return ("swinging_strike", None)

        # Contact made -- foul or in play?
        # Approximately 40% of contacted pitches are foul
        foul_rate = 0.38 + (pitcher.stuff / 100) * 0.05 - (batter.contact / 100) * 0.03
        foul_rate = self._clamp(foul_rate, 0.30, 0.50)

        if self.rng.random() < foul_rate:
            return ("foul", None)

        # Ball is in play
        return ("in_play", None)

    # -------------------------------------------------------------------
    # Ball-in-play resolution
    # -------------------------------------------------------------------

    def resolve_ball_in_play(self, batter: SimPlayer, pitcher: SimPlayer,
                             pstats: PitcherGameStats,
                             fielding_team: TeamState, game_state: GameState
                             ) -> dict:
        """Resolve a ball put in play.

        Returns dict with:
            result: "single", "double", "triple", "home_run", "groundout",
                    "flyout", "lineout", "popup", "fielders_choice",
                    "double_play", "error"
            description: human-readable description
            runs_scored: number of runs that scored
            runners_advanced: list of (runner, new_base) tuples
            outs_recorded: number of outs
        """
        fatigue = self._fatigue_factor(pitcher, pstats)
        platoon = self._platoon_factor(batter, pitcher)

        # Determine batted ball type
        bb_type = self._determine_batted_ball_type(batter, pitcher, fatigue)

        # Determine hit direction
        direction = self._determine_direction(batter)

        # Determine if it's a hit, based on batted ball type and player attributes
        is_hit, hit_type = self._determine_hit_or_out(
            batter, pitcher, bb_type, direction, fatigue, platoon, fielding_team
        )

        if is_hit:
            return self._resolve_hit(batter, hit_type, direction, game_state)
        else:
            return self._resolve_out(batter, bb_type, direction, game_state, fielding_team)

    def _determine_batted_ball_type(self, batter: SimPlayer, pitcher: SimPlayer,
                                     fatigue: float) -> str:
        """Determine batted ball type: groundball, line_drive, flyball, popup."""
        # GB% driven by power inversely (high power = more FB)
        gb_pct = 0.55 - (batter.power / 100) * 0.25  # 0.30-0.55
        # Pitcher stuff affects GB rate
        gb_pct += (pitcher.stuff / 100) * 0.05 - 0.025

        # Line drive rate ~20% for all hitters (relatively stable)
        ld_pct = 0.18 + (batter.contact / 100) * 0.04  # 0.18-0.22

        # Popup rate ~5-10%
        popup_pct = 0.08 - (batter.power / 100) * 0.03  # 0.05-0.08

        fb_pct = 1.0 - gb_pct - ld_pct - popup_pct

        roll = self.rng.random()
        if roll < gb_pct:
            return "groundball"
        elif roll < gb_pct + ld_pct:
            return "line_drive"
        elif roll < gb_pct + ld_pct + fb_pct:
            return "flyball"
        else:
            return "popup"

    def _determine_direction(self, batter: SimPlayer) -> str:
        """Determine hit direction: pull, center, opposite."""
        pull_pct = 0.30 + (batter.power / 100) * 0.18  # 0.30-0.48
        center_pct = 0.35
        # oppo_pct = 1.0 - pull - center

        roll = self.rng.random()
        if roll < pull_pct:
            return "pull"
        elif roll < pull_pct + center_pct:
            return "center"
        else:
            return "opposite"

    def _determine_hit_or_out(self, batter: SimPlayer, pitcher: SimPlayer,
                               bb_type: str, direction: str, fatigue: float,
                               platoon: float, fielding_team: TeamState
                               ) -> tuple[bool, str]:
        """Determine if a batted ball is a hit and what type.

        Returns (is_hit, hit_type_or_out_type).
        """
        # Determine effective batting average
        pitcher_hand = pitcher.throws
        if batter.bats == "S":
            effective_hand = "R" if pitcher_hand == "L" else "L"
        else:
            effective_hand = batter.bats

        if pitcher_hand == "L":
            base_avg = batter.avg_vs_l
        else:
            base_avg = batter.avg_vs_r

        # Adjust for fatigue (tired pitchers give up more hits)
        base_avg += fatigue * 0.040

        # Apply platoon factor
        base_avg += platoon

        # Hit rates by batted ball type (MLB averages)
        if bb_type == "groundball":
            # GB BABIP ~.240
            hit_prob = base_avg * 0.85
            # Speed helps on groundballs (infield hits)
            hit_prob += (batter.speed / 100) * 0.06
        elif bb_type == "line_drive":
            # LD BABIP ~.670
            hit_prob = 0.55 + base_avg * 0.40
        elif bb_type == "flyball":
            # FB BABIP ~.120 (excluding HR)
            hit_prob = base_avg * 0.45
        else:  # popup
            # Popup BABIP ~.010
            hit_prob = 0.01

        # Fielding range reduces hit probability
        avg_range = sum(
            p.fielder_range for p in fielding_team.lineup
        ) / len(fielding_team.lineup) if fielding_team.lineup else 60
        range_factor = (avg_range - 60) / 100 * 0.03  # +/- 3% for good/bad defense
        hit_prob -= range_factor

        hit_prob = self._clamp(hit_prob, 0.01, 0.95)

        is_hit = self.rng.random() < hit_prob

        if is_hit:
            hit_type = self._determine_hit_type(batter, pitcher, bb_type, fatigue)
            return (True, hit_type)
        else:
            return (False, bb_type)

    def _determine_hit_type(self, batter: SimPlayer, pitcher: SimPlayer,
                             bb_type: str, fatigue: float) -> str:
        """For a hit, determine single/double/triple/home_run."""
        if bb_type == "groundball":
            # Groundball hits are mostly singles, rarely doubles
            roll = self.rng.random()
            if roll < 0.90 + (batter.speed / 100) * 0.05:
                return "single"
            else:
                return "double"

        elif bb_type == "line_drive":
            # Line drives: singles, doubles, some HR
            roll = self.rng.random()
            hr_prob = 0.05 + (batter.power / 100) * 0.08 + fatigue * 0.02
            double_prob = 0.30 + (batter.speed / 100) * 0.05
            triple_prob = 0.03 + (batter.speed / 100) * 0.04
            if roll < hr_prob:
                return "home_run"
            elif roll < hr_prob + double_prob:
                return "double"
            elif roll < hr_prob + double_prob + triple_prob:
                return "triple"
            else:
                return "single"

        elif bb_type == "flyball":
            # Fly ball hits: HR most likely for power hitters
            roll = self.rng.random()
            hr_prob = 0.15 + (batter.power / 100) * 0.35 + fatigue * 0.05
            double_prob = 0.25
            triple_prob = 0.05 + (batter.speed / 100) * 0.05
            if roll < hr_prob:
                return "home_run"
            elif roll < hr_prob + double_prob:
                return "double"
            elif roll < hr_prob + double_prob + triple_prob:
                return "triple"
            else:
                return "single"

        else:  # popup (basically impossible to be a hit, but handle it)
            return "single"

    def _resolve_hit(self, batter: SimPlayer, hit_type: str, direction: str,
                     game_state: GameState) -> dict:
        """Resolve a hit: advance runners, score runs."""
        runs = 0
        new_runners: list[BaseRunner] = []
        descriptions = []
        outs = 0

        dir_text = {"pull": "left", "center": "center", "opposite": "right"}
        dir_str = dir_text.get(direction, "center")

        if hit_type == "home_run":
            # All runners score, batter scores
            runs = 1 + len(game_state.runners)
            for r in game_state.runners:
                descriptions.append(f"{r.player.name} scores")
            descriptions.append(f"{batter.name} scores")
            desc = f"{batter.name} homers to {dir_str} field"
            if runs > 1:
                desc += f" ({runs}-run homer)"
            return {
                "result": "home_run",
                "description": desc,
                "detail_descriptions": descriptions,
                "runs_scored": runs,
                "new_runners": [],
                "outs_recorded": 0,
                "hit_type": hit_type,
            }

        elif hit_type == "triple":
            # All runners score, batter to 3rd
            runs = len(game_state.runners)
            for r in game_state.runners:
                descriptions.append(f"{r.player.name} scores")
            new_runners = [BaseRunner(player=batter, start_base=3)]
            desc = f"{batter.name} triples to {dir_str} field"
            return {
                "result": "triple",
                "description": desc,
                "detail_descriptions": descriptions,
                "runs_scored": runs,
                "new_runners": new_runners,
                "outs_recorded": 0,
                "hit_type": hit_type,
            }

        elif hit_type == "double":
            runs = 0
            for r in game_state.runners:
                if r.start_base >= 2:
                    runs += 1
                    descriptions.append(f"{r.player.name} scores")
                else:
                    # Runner on 1st: fast runners score, slow go to 3rd
                    if r.player.speed >= 60 and self.rng.random() < 0.6:
                        runs += 1
                        descriptions.append(f"{r.player.name} scores from 1st")
                    else:
                        new_runners.append(BaseRunner(player=r.player, start_base=3))
                        descriptions.append(f"{r.player.name} to 3rd")
            new_runners.append(BaseRunner(player=batter, start_base=2))
            desc = f"{batter.name} doubles to {dir_str} field"
            return {
                "result": "double",
                "description": desc,
                "detail_descriptions": descriptions,
                "runs_scored": runs,
                "new_runners": new_runners,
                "outs_recorded": 0,
                "hit_type": hit_type,
            }

        else:  # single
            runs = 0
            for r in sorted(game_state.runners, key=lambda x: -x.start_base):
                if r.start_base == 3:
                    runs += 1
                    descriptions.append(f"{r.player.name} scores")
                elif r.start_base == 2:
                    # Fast runners score from 2nd on single
                    if r.player.speed >= 55 and self.rng.random() < 0.55:
                        runs += 1
                        descriptions.append(f"{r.player.name} scores from 2nd")
                    else:
                        new_runners.append(BaseRunner(player=r.player, start_base=3))
                        descriptions.append(f"{r.player.name} to 3rd")
                elif r.start_base == 1:
                    # Runner on 1st goes to 2nd or 3rd
                    if r.player.speed >= 70 and self.rng.random() < 0.3:
                        new_runners.append(BaseRunner(player=r.player, start_base=3))
                        descriptions.append(f"{r.player.name} to 3rd")
                    else:
                        new_runners.append(BaseRunner(player=r.player, start_base=2))
                        descriptions.append(f"{r.player.name} to 2nd")
            new_runners.append(BaseRunner(player=batter, start_base=1))

            hit_desc = {"pull": "to left field", "center": "up the middle",
                        "opposite": "to right field"}
            desc = f"{batter.name} singles {hit_desc.get(direction, 'up the middle')}"
            return {
                "result": "single",
                "description": desc,
                "detail_descriptions": descriptions,
                "runs_scored": runs,
                "new_runners": new_runners,
                "outs_recorded": 0,
                "hit_type": hit_type,
            }

    def _resolve_out(self, batter: SimPlayer, bb_type: str, direction: str,
                     game_state: GameState, fielding_team: TeamState) -> dict:
        """Resolve an out: handle DPs, fielder's choices, sac flies, errors."""
        runs = 0
        new_runners: list[BaseRunner] = list(game_state.runners)
        descriptions = []
        outs_recorded = 1

        # Error check
        avg_error = sum(p.error_rate for p in fielding_team.lineup) / max(len(fielding_team.lineup), 1)
        if self.rng.random() < avg_error * 1.5:
            # Error! Batter reaches, runners advance
            return self._resolve_error(batter, bb_type, direction, game_state, fielding_team)

        # Groundball out: check for double play
        if bb_type == "groundball":
            # Double play possibility with runner on 1st and < 2 outs
            r1 = game_state.runner_on(1)
            if r1 and game_state.outs < 2:
                # DP probability ~50% of GB with runner on 1st
                dp_prob = 0.40 - (batter.speed / 100) * 0.15 - (r1.player.speed / 100) * 0.10
                dp_prob = self._clamp(dp_prob, 0.10, 0.55)
                if self.rng.random() < dp_prob:
                    # Double play
                    outs_recorded = 2
                    # Remove runner on first
                    new_runners = [r for r in new_runners if r.start_base != 1]
                    # Runner on 3rd may score on DP if < 2 outs before
                    r3 = game_state.runner_on(3)
                    if r3 and game_state.outs == 0:
                        runs += 1
                        new_runners = [r for r in new_runners if r.start_base != 3]
                        descriptions.append(f"{r3.player.name} scores on double play")
                    # Runners on 2nd stay
                    desc = f"{batter.name} grounds into double play"
                    return {
                        "result": "double_play",
                        "description": desc,
                        "detail_descriptions": descriptions,
                        "runs_scored": runs,
                        "new_runners": new_runners,
                        "outs_recorded": outs_recorded,
                        "hit_type": "groundout",
                    }

            # Fielder's choice with runners on
            if game_state.runners and self.rng.random() < 0.25:
                # Get lead runner
                lead_runner = max(game_state.runners, key=lambda r: r.start_base)
                new_runners = [r for r in new_runners if r.start_base != lead_runner.start_base]
                new_runners.append(BaseRunner(player=batter, start_base=1))
                desc = f"{batter.name} reaches on fielder's choice, {lead_runner.player.name} out at {lead_runner.start_base + 1}"
                # Advance other runners
                advanced = []
                for r in new_runners:
                    if r.player.player_id != batter.player_id and r.start_base < 3:
                        advanced.append(r)
                return {
                    "result": "fielders_choice",
                    "description": desc,
                    "detail_descriptions": descriptions,
                    "runs_scored": runs,
                    "new_runners": new_runners,
                    "outs_recorded": 1,
                    "hit_type": "groundout",
                }

            # Regular groundout
            pos_map = {"pull": "short" if batter.bats in ("R", "S") else "second",
                       "center": "second", "opposite": "short" if batter.bats == "L" else "second"}
            pos = pos_map.get(direction, "short")
            desc = f"{batter.name} grounds out to {pos}"

            # Advance runners on groundout with < 2 outs
            if game_state.outs < 2:
                for r in sorted(new_runners, key=lambda x: -x.start_base):
                    if r.start_base == 3:
                        runs += 1
                        new_runners = [nr for nr in new_runners if nr.start_base != 3]
                        descriptions.append(f"{r.player.name} scores on groundout")
                    elif r.start_base == 2 and self.rng.random() < 0.3:
                        # Runner on 2nd sometimes advances to 3rd
                        idx = next(i for i, nr in enumerate(new_runners) if nr.start_base == 2)
                        new_runners[idx] = BaseRunner(player=r.player, start_base=3)

            return {
                "result": "groundout",
                "description": desc,
                "detail_descriptions": descriptions,
                "runs_scored": runs,
                "new_runners": [r for r in new_runners if r not in game_state.runners or r in new_runners],
                "outs_recorded": 1,
                "hit_type": "groundout",
            }

        elif bb_type == "flyball":
            dir_text = {"pull": "left", "center": "center", "opposite": "right"}
            pos = dir_text.get(direction, "center")
            desc = f"{batter.name} flies out to {pos} field"

            # Sac fly: runner on 3rd with < 2 outs
            r3 = game_state.runner_on(3)
            if r3 and game_state.outs < 2:
                # Tag up probability
                tag_prob = 0.70 + (r3.player.speed / 100) * 0.15
                if self.rng.random() < tag_prob:
                    runs += 1
                    new_runners = [r for r in new_runners if r.start_base != 3]
                    descriptions.append(f"{r3.player.name} tags up and scores")
                    desc = f"{batter.name} hits sacrifice fly to {pos} field"

            # Runner on 2nd might tag to 3rd on deep fly
            r2 = game_state.runner_on(2)
            if r2 and game_state.outs < 2 and self.rng.random() < 0.30:
                idx = next((i for i, r in enumerate(new_runners) if r.start_base == 2), None)
                if idx is not None:
                    new_runners[idx] = BaseRunner(player=r2.player, start_base=3)
                    descriptions.append(f"{r2.player.name} tags to 3rd")

            return {
                "result": "flyout",
                "description": desc,
                "detail_descriptions": descriptions,
                "runs_scored": runs,
                "new_runners": new_runners,
                "outs_recorded": 1,
                "hit_type": "flyout",
            }

        elif bb_type == "line_drive":
            desc = f"{batter.name} lines out"
            # Line drives caught rarely lead to advancement
            return {
                "result": "lineout",
                "description": desc,
                "detail_descriptions": [],
                "runs_scored": 0,
                "new_runners": list(game_state.runners),
                "outs_recorded": 1,
                "hit_type": "lineout",
            }

        else:  # popup
            desc = f"{batter.name} pops up"
            return {
                "result": "popup",
                "description": desc,
                "detail_descriptions": [],
                "runs_scored": 0,
                "new_runners": list(game_state.runners),
                "outs_recorded": 1,
                "hit_type": "popup",
            }

    def _resolve_error(self, batter: SimPlayer, bb_type: str, direction: str,
                       game_state: GameState, fielding_team: TeamState) -> dict:
        """Resolve an error: batter reaches, runners advance."""
        runs = 0
        new_runners: list[BaseRunner] = []
        descriptions = []

        for r in sorted(game_state.runners, key=lambda x: -x.start_base):
            if r.start_base == 3:
                runs += 1
                descriptions.append(f"{r.player.name} scores on error")
            elif r.start_base == 2:
                new_runners.append(BaseRunner(player=r.player, start_base=3))
                descriptions.append(f"{r.player.name} to 3rd on error")
            elif r.start_base == 1:
                new_runners.append(BaseRunner(player=r.player, start_base=2))
                descriptions.append(f"{r.player.name} to 2nd on error")

        new_runners.append(BaseRunner(player=batter, start_base=1))
        desc = f"{batter.name} reaches on error"

        return {
            "result": "error",
            "description": desc,
            "detail_descriptions": descriptions,
            "runs_scored": runs,
            "new_runners": new_runners,
            "outs_recorded": 0,
            "hit_type": "error",
        }

    # -------------------------------------------------------------------
    # Walk / HBP resolution
    # -------------------------------------------------------------------

    def resolve_walk(self, batter: SimPlayer, game_state: GameState) -> dict:
        """Resolve a walk: advance forced runners."""
        runs = 0
        new_runners: list[BaseRunner] = list(game_state.runners)
        descriptions = []

        # Force runners ahead if bases are occupied in sequence
        r1 = game_state.runner_on(1)
        r2 = game_state.runner_on(2)
        r3 = game_state.runner_on(3)

        if r1 and r2 and r3:
            # Bases loaded walk
            runs = 1
            new_runners = [r for r in new_runners if r.start_base != 3]
            descriptions.append(f"{r3.player.name} scores on walk")
            # Move r2 to 3rd
            for i, r in enumerate(new_runners):
                if r.start_base == 2:
                    new_runners[i] = BaseRunner(player=r.player, start_base=3)
            # Move r1 to 2nd
            for i, r in enumerate(new_runners):
                if r.start_base == 1:
                    new_runners[i] = BaseRunner(player=r.player, start_base=2)
        elif r1 and r2:
            # Runners on 1st and 2nd: both advance
            for i, r in enumerate(new_runners):
                if r.start_base == 2:
                    new_runners[i] = BaseRunner(player=r.player, start_base=3)
            for i, r in enumerate(new_runners):
                if r.start_base == 1:
                    new_runners[i] = BaseRunner(player=r.player, start_base=2)
        elif r1:
            # Runner on 1st: advances to 2nd
            for i, r in enumerate(new_runners):
                if r.start_base == 1:
                    new_runners[i] = BaseRunner(player=r.player, start_base=2)
        # Runner on 2nd/3rd only: no forced advancement

        new_runners.append(BaseRunner(player=batter, start_base=1))

        return {
            "runs_scored": runs,
            "new_runners": new_runners,
            "descriptions": descriptions,
        }

    # -------------------------------------------------------------------
    # Stolen base resolution
    # -------------------------------------------------------------------

    def resolve_stolen_base(self, runner: BaseRunner, target_base: int,
                            pitcher: SimPlayer, catcher: SimPlayer | None,
                            game_state: GameState) -> dict:
        """Resolve a stolen base attempt."""
        # Success probability based on speed, pitcher hold, catcher arm
        base_prob = 0.50 + (runner.player.speed / 100) * 0.30  # 0.50-0.80

        # Catcher pop time penalty
        if catcher:
            pop_penalty = (2.0 - catcher.pop_time) * 0.15  # Faster = harder to steal
            base_prob -= pop_penalty

        # Stealing 3rd is harder
        if target_base == 3:
            base_prob -= 0.10

        base_prob = self._clamp(base_prob, 0.30, 0.90)

        success = self.rng.random() < base_prob

        if success:
            # Advance runner
            new_runners = []
            for r in game_state.runners:
                if r.player.player_id == runner.player.player_id:
                    new_runners.append(BaseRunner(player=r.player, start_base=target_base))
                else:
                    new_runners.append(r)
            return {
                "success": True,
                "description": f"{runner.player.name} steals {target_base}{'nd' if target_base == 2 else 'rd'} base",
                "new_runners": new_runners,
                "outs_recorded": 0,
            }
        else:
            # Runner is out
            new_runners = [r for r in game_state.runners
                          if r.player.player_id != runner.player.player_id]
            return {
                "success": False,
                "description": f"{runner.player.name} caught stealing {target_base}{'nd' if target_base == 2 else 'rd'} base",
                "new_runners": new_runners,
                "outs_recorded": 1,
            }

    # -------------------------------------------------------------------
    # Full plate appearance
    # -------------------------------------------------------------------

    def simulate_plate_appearance(self, game_state: GameState) -> dict:
        """Simulate a complete plate appearance.

        Returns:
            dict with result, description, runs_scored, outs_recorded,
            new_runners, pitch_sequence, and event details.
        """
        bt = game_state.batting_team()
        ft = game_state.fielding_team()
        batter = bt.current_batter()
        pitcher = game_state.current_pitcher()
        pstats = ft.get_pitcher_stats(pitcher.player_id)

        pstats.batters_faced += 1

        balls = 0
        strikes = 0
        pitches = []

        while True:
            outcome, detail = self.resolve_pitch(batter, pitcher, pstats, balls, strikes)
            pstats.pitches += 1
            pitches.append(outcome)

            if outcome == "ball":
                balls += 1
                if balls >= 4:
                    # Walk
                    walk_result = self.resolve_walk(batter, game_state)
                    bt_stats = bt.get_batter_stats(batter.player_id)
                    bt_stats.bb += 1
                    pstats.bb += 1
                    return {
                        "result": "walk",
                        "description": f"{batter.name} walks",
                        "runs_scored": walk_result["runs_scored"],
                        "outs_recorded": 0,
                        "new_runners": walk_result["new_runners"],
                        "pitch_count": len(pitches),
                        "pitch_sequence": pitches,
                        "detail_descriptions": walk_result["descriptions"],
                        "hit_type": "walk",
                    }

            elif outcome in ("called_strike", "swinging_strike"):
                strikes += 1
                if strikes >= 3:
                    # Strikeout
                    bt_stats = bt.get_batter_stats(batter.player_id)
                    bt_stats.ab += 1
                    bt_stats.k += 1
                    pstats.k += 1
                    return {
                        "result": "strikeout",
                        "description": f"{batter.name} strikes out {'swinging' if outcome == 'swinging_strike' else 'looking'}",
                        "runs_scored": 0,
                        "outs_recorded": 1,
                        "new_runners": list(game_state.runners),
                        "pitch_count": len(pitches),
                        "pitch_sequence": pitches,
                        "detail_descriptions": [],
                        "hit_type": "strikeout",
                    }

            elif outcome == "foul":
                if strikes < 2:
                    strikes += 1

            elif outcome == "hbp":
                # Hit by pitch: same as walk but different stat
                walk_result = self.resolve_walk(batter, game_state)
                bt_stats = bt.get_batter_stats(batter.player_id)
                bt_stats.hbp += 1
                return {
                    "result": "hbp",
                    "description": f"{batter.name} hit by pitch",
                    "runs_scored": walk_result["runs_scored"],
                    "outs_recorded": 0,
                    "new_runners": walk_result["new_runners"],
                    "pitch_count": len(pitches),
                    "pitch_sequence": pitches,
                    "detail_descriptions": walk_result["descriptions"],
                    "hit_type": "hbp",
                }

            elif outcome == "in_play":
                bip_result = self.resolve_ball_in_play(
                    batter, pitcher, pstats, ft, game_state
                )

                bt_stats = bt.get_batter_stats(batter.player_id)
                bt_stats.ab += 1

                result = bip_result["result"]
                if result in ("single", "double", "triple", "home_run", "error"):
                    bt_stats.hits += 1
                    bt_stats.rbi += bip_result["runs_scored"]
                    if result == "double":
                        bt_stats.doubles += 1
                    elif result == "triple":
                        bt_stats.triples += 1
                    elif result == "home_run":
                        bt_stats.hr += 1
                        bt_stats.rbi += bip_result["runs_scored"]
                        bt_stats.rbi = bt_stats.rbi // 2 + bip_result["runs_scored"]  # fix double count
                        # Actually just fix: RBI was already set, HR counted batter too
                        bt_stats.rbi = bip_result["runs_scored"]
                    if result != "error":
                        pstats.hits += 1
                    if result == "home_run":
                        pstats.hr_allowed += 1
                elif result in ("groundout", "flyout", "lineout", "popup",
                                "double_play", "fielders_choice"):
                    if result == "fielders_choice":
                        bt_stats.hits -= 0  # no hit awarded
                    bt_stats.rbi += bip_result["runs_scored"]

                return {
                    "result": result,
                    "description": bip_result["description"],
                    "runs_scored": bip_result["runs_scored"],
                    "outs_recorded": bip_result["outs_recorded"],
                    "new_runners": bip_result["new_runners"],
                    "pitch_count": len(pitches),
                    "pitch_sequence": pitches,
                    "detail_descriptions": bip_result.get("detail_descriptions", []),
                    "hit_type": bip_result.get("hit_type", result),
                }

    # -------------------------------------------------------------------
    # Game flow
    # -------------------------------------------------------------------

    def apply_pa_result(self, game_state: GameState, pa_result: dict) -> list[PlayEvent]:
        """Apply a plate appearance result to the game state.

        Returns list of PlayEvent entries generated.
        """
        events = []
        bt = game_state.batting_team()
        ft = game_state.fielding_team()
        batter = bt.current_batter()
        pitcher = game_state.current_pitcher()

        # Record outs
        runs_scored = pa_result["runs_scored"]
        outs = pa_result["outs_recorded"]

        # Update pitcher stats for runs
        pstats = ft.get_pitcher_stats(pitcher.player_id)
        pstats.runs += runs_scored
        pstats.earned_runs += runs_scored  # simplified: all runs are earned
        pstats.ip_outs += outs

        # Update score
        if game_state.half == "TOP":
            game_state.score_away += runs_scored
        else:
            game_state.score_home += runs_scored

        game_state._current_inning_runs += runs_scored

        # Record batter runs scored
        if runs_scored > 0:
            # Mark runners who scored
            for desc in pa_result.get("detail_descriptions", []):
                if "scores" in desc:
                    for r in game_state.runners:
                        if r.player.name in desc:
                            r_stats = bt.get_batter_stats(r.player.player_id)
                            r_stats.runs += 1

            # If it's a home run, batter scores too
            if pa_result["result"] == "home_run":
                b_stats = bt.get_batter_stats(batter.player_id)
                b_stats.runs += 1

        # Create play event
        desc = pa_result["description"]
        for d in pa_result.get("detail_descriptions", []):
            desc += f". {d}"
        if runs_scored > 0:
            desc += f" [{game_state.score_display()}]"

        event = PlayEvent(
            inning=game_state.inning,
            half=game_state.half,
            outs_before=game_state.outs,
            description=desc,
            event_type="in_play" if pa_result["result"] not in ("walk", "strikeout", "hbp") else pa_result["result"],
            score_home=game_state.score_home,
            score_away=game_state.score_away,
            runs_scored=runs_scored,
            batter_id=batter.player_id,
            pitcher_id=pitcher.player_id,
        )
        events.append(event)
        game_state.play_log.append(event)

        # Update outs
        game_state.outs += outs

        # Update runners
        game_state.runners = pa_result["new_runners"]

        # Update pitcher batter count for 3-batter minimum
        ft.current_pitcher_batters_faced_this_stint += 1

        # Check for walk-off
        if (game_state.half == "BOTTOM" and game_state.inning >= 9 and
                game_state.score_home > game_state.score_away):
            game_state.game_over = True
            game_state.winning_team = game_state.home.name
            events.append(PlayEvent(
                inning=game_state.inning,
                half=game_state.half,
                outs_before=game_state.outs,
                description=f"Walk-off! {game_state.home.name} wins {game_state.score_home}-{game_state.score_away}!",
                event_type="game_end",
                score_home=game_state.score_home,
                score_away=game_state.score_away,
            ))
            game_state.play_log.append(events[-1])
            return events

        # Check for 3 outs -> half-inning over
        if game_state.outs >= 3:
            events.extend(self._end_half_inning(game_state))

        # Advance to next batter
        if not game_state.game_over:
            bt.advance_batter()

        return events

    def _end_half_inning(self, game_state: GameState) -> list[PlayEvent]:
        """Handle the transition between half-innings."""
        events = []

        bt = game_state.batting_team()

        # Record inning runs
        while len(bt.inning_runs) < game_state.inning:
            bt.inning_runs.append(0)
        bt.inning_runs[game_state.inning - 1] = game_state._current_inning_runs

        # Clear bases and outs
        game_state.runners = []
        game_state.outs = 0
        game_state._current_inning_runs = 0

        if game_state.half == "TOP":
            # Move to bottom of inning
            game_state.half = "BOTTOM"

            # Check if bottom of 9th+ is needed
            if game_state.inning >= 9 and game_state.score_home > game_state.score_away:
                # Home team already leads, game over
                game_state.game_over = True
                game_state.winning_team = game_state.home.name
                events.append(PlayEvent(
                    inning=game_state.inning,
                    half="BOTTOM",
                    outs_before=0,
                    description=f"Game over! {game_state.home.name} wins {game_state.score_home}-{game_state.score_away}!",
                    event_type="game_end",
                    score_home=game_state.score_home,
                    score_away=game_state.score_away,
                ))
                game_state.play_log.append(events[-1])
                return events

            events.append(PlayEvent(
                inning=game_state.inning,
                half="BOTTOM",
                outs_before=0,
                description=f"--- Bottom of the {self._ordinal(game_state.inning)} ---",
                event_type="inning_change",
                score_home=game_state.score_home,
                score_away=game_state.score_away,
            ))
            game_state.play_log.append(events[-1])

        else:
            # End of bottom of inning -> move to next inning top
            if game_state.inning >= 9:
                if game_state.score_home != game_state.score_away:
                    # Game is over
                    game_state.game_over = True
                    winner = game_state.home.name if game_state.score_home > game_state.score_away else game_state.away.name
                    game_state.winning_team = winner
                    events.append(PlayEvent(
                        inning=game_state.inning,
                        half="BOTTOM",
                        outs_before=3,
                        description=f"Game over! {winner} wins {max(game_state.score_home, game_state.score_away)}-{min(game_state.score_home, game_state.score_away)}!",
                        event_type="game_end",
                        score_home=game_state.score_home,
                        score_away=game_state.score_away,
                    ))
                    game_state.play_log.append(events[-1])
                    return events
                # Tie: go to extras
            game_state.inning += 1
            game_state.half = "TOP"

            events.append(PlayEvent(
                inning=game_state.inning,
                half="TOP",
                outs_before=0,
                description=f"--- Top of the {self._ordinal(game_state.inning)} ---",
                event_type="inning_change",
                score_home=game_state.score_home,
                score_away=game_state.score_away,
            ))
            game_state.play_log.append(events[-1])

        return events

    def _ordinal(self, n: int) -> str:
        """Return ordinal string for an integer (1st, 2nd, 3rd, etc.)."""
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    # -------------------------------------------------------------------
    # Game initialization
    # -------------------------------------------------------------------

    def initialize_game(self, rosters: dict, home_lineup_order: list[int] | None = None,
                        away_lineup_order: list[int] | None = None) -> GameState:
        """Create a GameState from roster data.

        Args:
            rosters: Dict with "home" and "away" team data.
            home_lineup_order: Optional custom batting order indices (0-8).
            away_lineup_order: Optional custom batting order indices (0-8).
        """
        def build_team(team_data: dict, lineup_order: list[int] | None) -> TeamState:
            lineup_players = [SimPlayer.from_roster_dict(p) for p in team_data["lineup"]]
            bench_players = [SimPlayer.from_roster_dict(p) for p in team_data["bench"]]
            bp_players = [SimPlayer.from_roster_dict(p) for p in team_data["bullpen"]]
            sp = SimPlayer.from_roster_dict(team_data["starting_pitcher"])

            if lineup_order:
                lineup_players = [lineup_players[i] for i in lineup_order]

            positions = [p.primary_position for p in lineup_players]

            team = TeamState(
                name=team_data["team_name"],
                lineup=lineup_players,
                lineup_positions=positions,
                bench=bench_players,
                bullpen=bp_players,
                starting_pitcher=sp,
                current_pitcher=sp,
            )
            team.used_pitchers.append(sp.player_id)
            return team

        home = build_team(rosters["home"], home_lineup_order)
        away = build_team(rosters["away"], away_lineup_order)

        state = GameState(
            home=home,
            away=away,
            seed=self.seed,
        )

        # Add initial event
        start_event = PlayEvent(
            inning=1,
            half="TOP",
            outs_before=0,
            description=f"--- Top of the 1st --- ({away.name} at {home.name})",
            event_type="inning_change",
            score_home=0,
            score_away=0,
        )
        state.play_log.append(start_event)

        return state

    # -------------------------------------------------------------------
    # Simulate full game (no agent, automated both sides)
    # -------------------------------------------------------------------

    def simulate_game(self, rosters: dict,
                      home_lineup_order: list[int] | None = None,
                      away_lineup_order: list[int] | None = None,
                      max_innings: int = 15,
                      verbose: bool = False) -> GameState:
        """Simulate a complete game with no agent intervention.

        Uses automated management for both teams.
        """
        game = self.initialize_game(rosters, home_lineup_order, away_lineup_order)

        while not game.game_over:
            if game.inning > max_innings:
                # Safety valve for extra innings
                game.game_over = True
                if game.score_home == game.score_away:
                    game.winning_team = "TIE (innings limit)"
                else:
                    game.winning_team = game.home.name if game.score_home > game.score_away else game.away.name
                break

            # Automated pitcher management
            self._auto_manage_pitcher(game)

            # Simulate plate appearance
            pa_result = self.simulate_plate_appearance(game)

            if verbose:
                bt = game.batting_team()
                print(f"  {pa_result['description']}")
                for d in pa_result.get("detail_descriptions", []):
                    if d:
                        print(f"    {d}")

            # Apply result
            events = self.apply_pa_result(game, pa_result)

            if verbose:
                for e in events:
                    if e.event_type in ("inning_change", "game_end"):
                        print(f"\n{e.description}")
                    if e.runs_scored > 0:
                        print(f"  Score: Away {game.score_away} - Home {game.score_home}")

        return game

    def _auto_manage_pitcher(self, game_state: GameState) -> None:
        """Simple automated pitcher management.

        Pull pitcher when:
        - Pitch count exceeds stamina-based threshold
        - Given up too many runs this inning
        - 3-batter minimum is met
        """
        ft = game_state.fielding_team()
        pitcher = game_state.current_pitcher()
        if pitcher is None:
            return

        pstats = ft.get_pitcher_stats(pitcher.player_id)

        # Don't pull if 3-batter minimum not met
        if ft.current_pitcher_batters_faced_this_stint < 3:
            return

        # Pitch count threshold based on stamina
        threshold = 30 + (pitcher.stamina / 100) * 85  # 30-115
        should_pull = pstats.pitches >= threshold

        # Also pull if giving up runs rapidly (5+ runs)
        if pstats.runs >= 5:
            should_pull = True

        if should_pull and ft.bullpen:
            # Find best available reliever
            available = [p for p in ft.bullpen if p.player_id not in ft.used_pitchers]
            if available:
                new_pitcher = available[0]  # Simple: take first available
                self._change_pitcher(game_state, ft, new_pitcher)

    def _change_pitcher(self, game_state: GameState, team: TeamState,
                        new_pitcher: SimPlayer) -> PlayEvent:
        """Execute a pitching change."""
        old_pitcher = team.current_pitcher
        team.current_pitcher = new_pitcher
        team.used_pitchers.append(new_pitcher.player_id)
        team.current_pitcher_batters_faced_this_stint = 0

        if old_pitcher:
            team.removed_players.append(old_pitcher.player_id)

        desc = f"Pitching change: {new_pitcher.name} replaces {old_pitcher.name if old_pitcher else 'unknown'}"
        event = PlayEvent(
            inning=game_state.inning,
            half=game_state.half,
            outs_before=game_state.outs,
            description=desc,
            event_type="decision",
            score_home=game_state.score_home,
            score_away=game_state.score_away,
        )
        game_state.play_log.append(event)
        return event

    # -------------------------------------------------------------------
    # Box score generation
    # -------------------------------------------------------------------

    def generate_box_score(self, game_state: GameState) -> dict:
        """Generate a complete box score for the game."""
        def team_box(team: TeamState, is_home: bool) -> dict:
            batting_lines = []
            for p in team.lineup:
                stats = team.get_batter_stats(p.player_id)
                batting_lines.append({
                    "name": p.name,
                    "position": p.primary_position,
                    **stats.to_dict(),
                })

            pitching_lines = []
            for pid in team.used_pitchers:
                stats = team.get_pitcher_stats(pid)
                p = team.get_player_by_id(pid)
                name = p.name if p else pid
                pitching_lines.append({
                    "name": name,
                    **stats.to_dict(),
                })

            # Pad inning runs to match total innings
            inning_runs = list(team.inning_runs)
            while len(inning_runs) < game_state.inning:
                inning_runs.append(0)

            total_hits = sum(team.get_batter_stats(p.player_id).hits for p in team.lineup)
            total_runs = game_state.score_home if is_home else game_state.score_away

            return {
                "team_name": team.name,
                "inning_runs": inning_runs,
                "total_runs": total_runs,
                "total_hits": total_hits,
                "batting": batting_lines,
                "pitching": pitching_lines,
            }

        return {
            "away": team_box(game_state.away, False),
            "home": team_box(game_state.home, True),
            "final_score": {
                "away": game_state.score_away,
                "home": game_state.score_home,
            },
            "winning_team": game_state.winning_team,
            "innings": game_state.inning,
            "seed": game_state.seed,
        }

    def print_box_score(self, game_state: GameState) -> str:
        """Generate a formatted box score string."""
        box = self.generate_box_score(game_state)
        lines = []

        lines.append("=" * 72)
        lines.append("FINAL BOX SCORE")
        lines.append("=" * 72)

        # Line score
        max_inn = game_state.inning
        header = f"{'Team':<20}"
        for i in range(1, max_inn + 1):
            header += f" {i:>3}"
        header += "  |   R   H"
        lines.append(header)
        lines.append("-" * len(header))

        for side in ("away", "home"):
            team = box[side]
            row = f"{team['team_name']:<20}"
            for r in team["inning_runs"]:
                row += f" {r:>3}"
            row += f"  | {team['total_runs']:>3} {team['total_hits']:>3}"
            lines.append(row)

        lines.append("")
        lines.append(f"Winner: {box['winning_team']}")
        lines.append(f"Seed: {box['seed']}")

        # Batting lines
        for side in ("away", "home"):
            team = box[side]
            lines.append(f"\n{team['team_name']} Batting:")
            lines.append(f"  {'Name':<20} {'Pos':<4} {'AB':>3} {'H':>3} {'R':>3} {'RBI':>4} {'BB':>3} {'K':>3}")
            lines.append(f"  {'-'*20} {'-'*4} {'-'*3} {'-'*3} {'-'*3} {'-'*4} {'-'*3} {'-'*3}")
            for b in team["batting"]:
                lines.append(
                    f"  {b['name']:<20} {b['position']:<4} {b['AB']:>3} {b['H']:>3} "
                    f"{b['R']:>3} {b['RBI']:>4} {b['BB']:>3} {b['K']:>3}"
                )

        # Pitching lines
        for side in ("away", "home"):
            team = box[side]
            lines.append(f"\n{team['team_name']} Pitching:")
            lines.append(f"  {'Name':<20} {'IP':>5} {'H':>3} {'R':>3} {'ER':>3} {'BB':>3} {'K':>3} {'P':>4}")
            lines.append(f"  {'-'*20} {'-'*5} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*4}")
            for p in team["pitching"]:
                lines.append(
                    f"  {p['name']:<20} {p['IP']:>5.1f} {p['H']:>3} {p['R']:>3} "
                    f"{p['ER']:>3} {p['BB']:>3} {p['K']:>3} {p['pitches']:>4}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialization support
# ---------------------------------------------------------------------------

def game_state_to_dict(game_state: GameState) -> dict:
    """Serialize game state to a dict for JSON persistence."""
    def team_to_dict(team: TeamState) -> dict:
        return {
            "name": team.name,
            "lineup_index": team.lineup_index,
            "used_pitchers": team.used_pitchers,
            "removed_players": team.removed_players,
            "current_pitcher_id": team.current_pitcher.player_id if team.current_pitcher else None,
            "current_pitcher_batters_faced_this_stint": team.current_pitcher_batters_faced_this_stint,
            "mound_visits_remaining": team.mound_visits_remaining,
            "challenge_available": team.challenge_available,
            "inning_runs": team.inning_runs,
            "batter_stats": {k: v.to_dict() for k, v in team.batter_stats.items()},
            "pitcher_stats": {k: v.to_dict() for k, v in team.pitcher_stats.items()},
        }

    return {
        "inning": game_state.inning,
        "half": game_state.half,
        "outs": game_state.outs,
        "score_home": game_state.score_home,
        "score_away": game_state.score_away,
        "game_over": game_state.game_over,
        "winning_team": game_state.winning_team,
        "seed": game_state.seed,
        "runners": [
            {"player_id": r.player.player_id, "name": r.player.name, "base": r.start_base}
            for r in game_state.runners
        ],
        "home": team_to_dict(game_state.home),
        "away": team_to_dict(game_state.away),
        "play_log": [e.to_dict() for e in game_state.play_log],
    }


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    rosters = load_rosters()
    engine = SimulationEngine(seed=seed)

    print(f"Simulating game with seed {engine.seed}...")
    print(f"{rosters['away']['team_name']} at {rosters['home']['team_name']}")
    print("=" * 72)

    if verbose:
        print(f"\n--- Top of the 1st ---")

    game = engine.simulate_game(rosters, verbose=verbose)

    print()
    print(engine.print_box_score(game))

    print(f"\nTotal plate appearances: {len([e for e in game.play_log if e.event_type not in ('inning_change', 'game_end', 'decision')])}")
    print(f"Total play events: {len(game.play_log)}")
