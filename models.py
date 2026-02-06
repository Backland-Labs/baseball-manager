# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Data models for the baseball manager AI agent."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Hand(str, Enum):
    L = "L"
    R = "R"
    S = "S"  # switch-hitter


class ThrowHand(str, Enum):
    L = "L"
    R = "R"


class Half(str, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class BattingTeam(str, Enum):
    HOME = "HOME"
    AWAY = "AWAY"


class BullpenRole(str, Enum):
    CLOSER = "CLOSER"
    SETUP = "SETUP"
    MIDDLE = "MIDDLE"
    LONG = "LONG"
    MOPUP = "MOPUP"


class Freshness(str, Enum):
    FRESH = "FRESH"
    MODERATE = "MODERATE"
    TIRED = "TIRED"


# ---------------------------------------------------------------------------
# Player data models
# ---------------------------------------------------------------------------

class Player(BaseModel):
    """Base player model with common attributes."""
    player_id: str
    name: str
    primary_position: str
    handedness: Hand  # bats for position players, throws for pitchers


class BatterAttributes(BaseModel):
    """Batting attributes that influence simulation outcomes."""
    contact: float = Field(ge=0.0, le=100.0, description="Contact ability (0-100)")
    power: float = Field(ge=0.0, le=100.0, description="Power rating (0-100)")
    speed: float = Field(ge=0.0, le=100.0, description="Sprint speed rating (0-100)")
    eye: float = Field(ge=0.0, le=100.0, description="Plate discipline (0-100)")
    avg_vs_l: float = Field(ge=0.0, le=1.0, description="Batting average vs LHP")
    avg_vs_r: float = Field(ge=0.0, le=1.0, description="Batting average vs RHP")


class PitcherAttributes(BaseModel):
    """Pitching attributes that influence simulation outcomes."""
    stuff: float = Field(ge=0.0, le=100.0, description="Stuff rating (0-100)")
    control: float = Field(ge=0.0, le=100.0, description="Control rating (0-100)")
    stamina: float = Field(ge=0.0, le=100.0, description="Stamina rating (0-100)")
    velocity: float = Field(ge=70.0, le=105.0, description="Fastball velocity in mph")
    era_vs_l: float = Field(ge=0.0, description="ERA vs LHB")
    era_vs_r: float = Field(ge=0.0, description="ERA vs RHB")


class FielderAttributes(BaseModel):
    """Fielding attributes for defensive evaluation."""
    range: float = Field(ge=0.0, le=100.0, description="Fielding range (0-100)")
    arm_strength: float = Field(ge=0.0, le=100.0, description="Arm strength (0-100)")
    error_rate: float = Field(ge=0.0, le=1.0, description="Error rate at position")
    positions: list[str] = Field(default_factory=list, description="Positions the player can field")


class CatcherAttributes(BaseModel):
    """Catcher-specific attributes."""
    pop_time: float = Field(ge=1.5, le=2.5, description="Pop time in seconds")
    framing: float = Field(ge=0.0, le=100.0, description="Framing ability (0-100)")


class FullPlayer(BaseModel):
    """Complete player model with all attributes."""
    player_id: str
    name: str
    primary_position: str
    bats: Hand
    throws: ThrowHand
    batter: BatterAttributes
    pitcher: Optional[PitcherAttributes] = None
    fielder: FielderAttributes
    catcher: Optional[CatcherAttributes] = None


# ---------------------------------------------------------------------------
# Runner on base
# ---------------------------------------------------------------------------

class Runner(BaseModel):
    player_id: str
    name: str
    sprint_speed: float
    sb_success_rate: float


# ---------------------------------------------------------------------------
# Game state: MatchupState
# ---------------------------------------------------------------------------

class Count(BaseModel):
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)


class Runners(BaseModel):
    first: Optional[Runner] = None
    second: Optional[Runner] = None
    third: Optional[Runner] = None


class BatterInfo(BaseModel):
    player_id: str
    name: str
    bats: Hand
    lineup_position: int = Field(ge=1, le=9)


class PitcherInfo(BaseModel):
    player_id: str
    name: str
    throws: ThrowHand
    pitch_count_today: int = 0
    batters_faced_today: int = 0
    times_through_order: int = 1
    innings_pitched_today: float = 0.0
    runs_allowed_today: int = 0
    today_line: dict = Field(default_factory=lambda: {
        "IP": 0.0, "H": 0, "R": 0, "ER": 0, "BB": 0, "K": 0
    })


class OnDeckBatter(BaseModel):
    player_id: str
    name: str
    bats: Hand


class Score(BaseModel):
    home: int = 0
    away: int = 0


class MatchupState(BaseModel):
    """Current game situation and active batter/pitcher matchup."""
    inning: int = Field(ge=1)
    half: Half
    outs: int = Field(ge=0, le=2)
    count: Count = Field(default_factory=lambda: Count(balls=0, strikes=0))
    runners: Runners = Field(default_factory=Runners)
    score: Score = Field(default_factory=Score)
    batting_team: BattingTeam
    batter: BatterInfo
    pitcher: PitcherInfo
    on_deck_batter: OnDeckBatter


# ---------------------------------------------------------------------------
# Game state: RosterState
# ---------------------------------------------------------------------------

class LineupPlayer(BaseModel):
    player_id: str
    name: str
    position: str
    bats: Hand
    in_game: bool = True


class BenchPlayer(BaseModel):
    player_id: str
    name: str
    bats: Hand
    positions: list[str] = Field(default_factory=list)
    available: bool = True


class BullpenPitcher(BaseModel):
    player_id: str
    name: str
    throws: ThrowHand
    role: BullpenRole
    available: bool = True
    freshness: Freshness = Freshness.FRESH
    pitches_last_3_days: list[int] = Field(default_factory=lambda: [0, 0, 0])
    days_since_last_appearance: int = 5
    is_warming_up: bool = False


class RosterState(BaseModel):
    """The agent's own team roster availability."""
    our_lineup: list[LineupPlayer]
    our_lineup_position: int = Field(ge=0, le=8, description="Index of who is due up next (0-8)")
    bench: list[BenchPlayer] = Field(default_factory=list)
    bullpen: list[BullpenPitcher] = Field(default_factory=list)
    mound_visits_remaining: int = 5
    challenge_available: bool = True


# ---------------------------------------------------------------------------
# Game state: OpponentRosterState
# ---------------------------------------------------------------------------

class OpponentBenchPlayer(BaseModel):
    player_id: str
    name: str
    bats: Hand
    available: bool = True


class OpponentBullpenPitcher(BaseModel):
    player_id: str
    name: str
    throws: ThrowHand
    role: BullpenRole
    available: bool = True
    freshness: Freshness = Freshness.FRESH


class OpponentRosterState(BaseModel):
    """Opposing team's roster availability."""
    their_lineup: list[LineupPlayer]
    their_lineup_position: int = Field(ge=0, le=8)
    their_bench: list[OpponentBenchPlayer] = Field(default_factory=list)
    their_bullpen: list[OpponentBullpenPitcher] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent output: ManagerDecision
# ---------------------------------------------------------------------------

class AlternativeConsidered(BaseModel):
    decision: str
    expected_wp: float
    reason_rejected: str


class ManagerDecision(BaseModel):
    """Structured output schema for the agent's managerial decision."""
    decision: str = Field(description="Decision type, e.g. PULL_STARTER, STOLEN_BASE, SWING_AWAY, PINCH_HIT, etc.")
    action_details: str = Field(description="Specific action description")
    confidence: float = Field(ge=0.0, le=1.0, description="Decision confidence")
    reasoning: str = Field(description="Full statistical justification")
    win_probability_before: Optional[float] = Field(default=None, description="WP before decision")
    win_probability_after_expected: Optional[float] = Field(default=None, description="Expected WP after decision")
    key_factors: list[str] = Field(default_factory=list, description="Top 3-5 factors that drove the decision")
    alternatives_considered: list[AlternativeConsidered] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list, description="Potential downsides")
