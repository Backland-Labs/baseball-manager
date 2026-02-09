# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Input validation layer for all tool parameters.

Provides Pydantic input models for each of the 12 agent tools, a validation
decorator that type-checks and validates parameters before execution, and
helpers for common validation patterns (player IDs, enum values, ranges).

Every tool input model enforces:
- Required vs optional parameters with correct types
- Enum parameters reject invalid values
- Numeric ranges (outs 0-2, inning >= 1, etc.)
- Player identifiers are non-empty strings

Validation errors include which parameter failed and what was expected.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Common validation helpers
# ---------------------------------------------------------------------------

def is_valid_player_id(player_id: str) -> bool:
    """Check that a player ID is a plausible MLB player identifier.

    Accepts:
    - Roster IDs like 'h_001', 'a_sp1', 'h_bp3'
    - Numeric MLB player IDs like '660271'
    - Any non-empty string (to be checked against actual roster later)
    """
    return isinstance(player_id, str) and len(player_id.strip()) > 0


class ValidationErrorDetail:
    """Container for a structured validation error."""

    def __init__(self, parameter: str, expected: str, got: Any):
        self.parameter = parameter
        self.expected = expected
        self.got = got

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "expected": self.expected,
            "got": repr(self.got),
        }

    def __str__(self) -> str:
        return f"Parameter '{self.parameter}': expected {self.expected}, got {self.got!r}"


# ---------------------------------------------------------------------------
# Enum types for tool parameters
# ---------------------------------------------------------------------------

class PitcherHand(str, Enum):
    L = "L"
    R = "R"


class HomeAway(str, Enum):
    home = "home"
    away = "away"


class RecencyWindow(str, Enum):
    last_7 = "last_7"
    last_14 = "last_14"
    last_30 = "last_30"
    season = "season"


class HalfInning(str, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class Team(str, Enum):
    home = "home"
    away = "away"


class FieldingPosition(str, Enum):
    C = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SS = "SS"
    LF = "LF"
    CF = "CF"
    RF = "RF"
    DH = "DH"
    P = "P"


# ---------------------------------------------------------------------------
# Pydantic input models for each tool
# ---------------------------------------------------------------------------


class GetBatterStatsInput(BaseModel):
    """Input schema for get_batter_stats tool."""
    player_id: str = Field(min_length=1, description="The unique identifier of the batter.")
    vs_hand: Optional[PitcherHand] = Field(default=None, description="Split by pitcher handedness ('L' or 'R').")
    home_away: Optional[HomeAway] = Field(default=None, description="Split by venue ('home' or 'away').")
    recency_window: Optional[RecencyWindow] = Field(default=None, description="Recency filter ('last_7', 'last_14', 'last_30', 'season').")

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class GetPitcherStatsInput(BaseModel):
    """Input schema for get_pitcher_stats tool."""
    player_id: str = Field(min_length=1, description="The unique identifier of the pitcher.")
    vs_hand: Optional[PitcherHand] = Field(default=None, description="Split by batter handedness ('L' or 'R').")
    home_away: Optional[HomeAway] = Field(default=None, description="Split by venue ('home' or 'away').")
    recency_window: Optional[RecencyWindow] = Field(default=None, description="Recency filter ('last_7', 'last_14', 'last_30', 'season').")

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class GetMatchupDataInput(BaseModel):
    """Input schema for get_matchup_data tool."""
    batter_id: str = Field(min_length=1, description="The unique identifier of the batter.")
    pitcher_id: str = Field(min_length=1, description="The unique identifier of the pitcher.")

    @field_validator("batter_id", "pitcher_id")
    @classmethod
    def validate_player_ids(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class GetRunExpectancyInput(BaseModel):
    """Input schema for get_run_expectancy tool."""
    runner_on_first: bool = Field(description="Whether there is a runner on first base.")
    runner_on_second: bool = Field(description="Whether there is a runner on second base.")
    runner_on_third: bool = Field(description="Whether there is a runner on third base.")
    outs: int = Field(ge=0, le=2, description="Number of outs (0, 1, or 2).")


class GetWinProbabilityInput(BaseModel):
    """Input schema for get_win_probability tool."""
    inning: int = Field(ge=1, description="Current inning (1+).")
    half: str = Field(description="Half of the inning ('TOP' or 'BOTTOM').")
    outs: int = Field(ge=0, le=2, description="Number of outs (0, 1, or 2).")
    runner_on_first: bool = Field(description="Whether there is a runner on first base.")
    runner_on_second: bool = Field(description="Whether there is a runner on second base.")
    runner_on_third: bool = Field(description="Whether there is a runner on third base.")
    score_differential: int = Field(description="Score difference from managed team's perspective.")
    managed_team_home: Optional[bool] = Field(default=None, description="Whether the managed team is the home team.")

    @field_validator("half")
    @classmethod
    def validate_half(cls, v: str) -> str:
        upper = v.upper() if isinstance(v, str) else ""
        if upper not in ("TOP", "BOTTOM"):
            raise ValueError("Must be 'TOP' or 'BOTTOM'")
        return v


class EvaluateStolenBaseInput(BaseModel):
    """Input schema for evaluate_stolen_base tool."""
    runner_id: str = Field(min_length=1, description="The unique identifier of the baserunner.")
    target_base: int = Field(ge=2, le=4, description="Base being stolen (2, 3, or 4).")
    pitcher_id: str = Field(min_length=1, description="The unique identifier of the current pitcher.")
    catcher_id: str = Field(min_length=1, description="The unique identifier of the opposing catcher.")
    runner_on_first: bool = Field(default=False, description="Whether there is a runner on first base.")
    runner_on_second: bool = Field(default=False, description="Whether there is a runner on second base.")
    runner_on_third: bool = Field(default=False, description="Whether there is a runner on third base.")
    outs: int = Field(ge=0, le=2, default=0, description="Number of outs (0, 1, or 2).")

    @field_validator("runner_id", "pitcher_id", "catcher_id")
    @classmethod
    def validate_player_ids(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class EvaluateSacrificeBuntInput(BaseModel):
    """Input schema for evaluate_sacrifice_bunt tool."""
    batter_id: str = Field(min_length=1, description="The unique identifier of the batter.")
    runner_on_first: bool = Field(description="Whether there is a runner on first base.")
    runner_on_second: bool = Field(description="Whether there is a runner on second base.")
    runner_on_third: bool = Field(description="Whether there is a runner on third base.")
    outs: int = Field(ge=0, le=2, description="Number of outs (0, 1, or 2).")
    score_differential: int = Field(description="Score difference from managed team's perspective.")
    inning: int = Field(ge=1, description="Current inning (1+).")

    @field_validator("batter_id")
    @classmethod
    def validate_player_id(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class GetBullpenStatusInput(BaseModel):
    """Input schema for get_bullpen_status tool."""
    team: Team = Field(default=Team.home, description="Which team's bullpen ('home' or 'away').")
    used_pitcher_ids: Optional[str] = Field(default=None, description="Comma-separated list of used pitcher IDs.")
    warming_pitcher_ids: Optional[str] = Field(default=None, description="Comma-separated list of warming pitcher IDs.")
    ready_pitcher_ids: Optional[str] = Field(default=None, description="Comma-separated list of ready pitcher IDs.")


class GetPitcherFatigueAssessmentInput(BaseModel):
    """Input schema for get_pitcher_fatigue_assessment tool."""
    pitcher_id: str = Field(min_length=1, description="The unique identifier of the pitcher.")
    pitch_count: int = Field(ge=0, default=0, description="Total pitches thrown so far.")
    innings_pitched: float = Field(ge=0.0, default=0.0, description="Innings pitched so far.")
    times_through_order: int = Field(ge=1, default=1, description="Times through the batting order.")
    runs_allowed: int = Field(ge=0, default=0, description="Runs allowed in the current game.")
    in_current_game: bool = Field(default=True, description="Whether the pitcher is currently in the game.")

    @field_validator("pitcher_id")
    @classmethod
    def validate_player_id(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class GetDefensivePositioningInput(BaseModel):
    """Input schema for get_defensive_positioning tool."""
    batter_id: str = Field(min_length=1, description="The unique identifier of the batter.")
    pitcher_id: str = Field(min_length=1, description="The unique identifier of the pitcher.")
    outs: int = Field(ge=0, le=2, description="Number of outs (0, 1, or 2).")
    runner_on_first: bool = Field(description="Whether there is a runner on first base.")
    runner_on_second: bool = Field(description="Whether there is a runner on second base.")
    runner_on_third: bool = Field(description="Whether there is a runner on third base.")
    score_differential: int = Field(description="Score difference from managed team's perspective.")
    inning: int = Field(ge=1, description="Current inning (1+).")

    @field_validator("batter_id", "pitcher_id")
    @classmethod
    def validate_player_ids(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


class GetDefensiveReplacementValueInput(BaseModel):
    """Input schema for get_defensive_replacement_value tool."""
    current_fielder_id: str = Field(min_length=1, description="The unique identifier of the current fielder.")
    replacement_id: str = Field(min_length=1, description="The unique identifier of the replacement player.")
    position: FieldingPosition = Field(description="Fielding position for the substitution.")
    inning: int = Field(ge=1, default=7, description="Current inning (1+).")
    half: str = Field(default="top", description="Half of the inning ('top' or 'bottom').")
    outs: int = Field(ge=0, le=2, default=0, description="Number of outs (0, 1, or 2).")

    @field_validator("current_fielder_id", "replacement_id")
    @classmethod
    def validate_player_ids(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v

    @field_validator("half")
    @classmethod
    def validate_half(cls, v: str) -> str:
        if v.lower() not in ("top", "bottom"):
            raise ValueError("Must be 'top' or 'bottom'")
        return v


class GetPlatoonComparisonInput(BaseModel):
    """Input schema for get_platoon_comparison tool."""
    current_batter_id: str = Field(min_length=1, description="The unique identifier of the current batter.")
    pinch_hitter_id: str = Field(min_length=1, description="The unique identifier of the potential pinch hitter.")
    pitcher_id: str = Field(min_length=1, description="The unique identifier of the current pitcher.")

    @field_validator("current_batter_id", "pinch_hitter_id", "pitcher_id")
    @classmethod
    def validate_player_ids(cls, v: str) -> str:
        if not is_valid_player_id(v):
            raise ValueError("Player ID must be a non-empty string")
        return v


# ---------------------------------------------------------------------------
# Mapping from tool function names to their input models
# ---------------------------------------------------------------------------

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_batter_stats": GetBatterStatsInput,
    "get_pitcher_stats": GetPitcherStatsInput,
    "get_matchup_data": GetMatchupDataInput,
    "get_run_expectancy": GetRunExpectancyInput,
    "get_win_probability": GetWinProbabilityInput,
    "evaluate_stolen_base": EvaluateStolenBaseInput,
    "evaluate_sacrifice_bunt": EvaluateSacrificeBuntInput,
    "get_bullpen_status": GetBullpenStatusInput,
    "get_pitcher_fatigue_assessment": GetPitcherFatigueAssessmentInput,
    "get_defensive_positioning": GetDefensivePositioningInput,
    "get_defensive_replacement_value": GetDefensiveReplacementValueInput,
    "get_platoon_comparison": GetPlatoonComparisonInput,
}


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------


def validate_tool_input(tool_name: str, **kwargs: Any) -> tuple[bool, Optional[str]]:
    """Validate tool input parameters against the tool's Pydantic model.

    Args:
        tool_name: The name of the tool to validate for.
        **kwargs: The input parameters to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    model_cls = TOOL_INPUT_MODELS.get(tool_name)
    if model_cls is None:
        return False, f"Unknown tool: {tool_name}"

    try:
        model_cls(**kwargs)
        return True, None
    except Exception as e:
        # Format a clear validation error message
        errors = _format_validation_error(e)
        return False, errors


def _format_validation_error(exc: Exception) -> str:
    """Format a Pydantic validation error into a human-readable message.

    Includes which parameter failed and what was expected.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            parts.append(f"Parameter '{loc}': {msg}")
        return "; ".join(parts)

    return str(exc)


# ---------------------------------------------------------------------------
# Game state validation
# ---------------------------------------------------------------------------

def validate_game_state(
    matchup_state: dict[str, Any] | None = None,
    roster_state: dict[str, Any] | None = None,
    opponent_roster_state: dict[str, Any] | None = None,
) -> tuple[bool, Optional[str]]:
    """Validate game state inputs against the Pydantic models.

    Args:
        matchup_state: Dict to validate against MatchupState schema.
        roster_state: Dict to validate against RosterState schema.
        opponent_roster_state: Dict to validate against OpponentRosterState schema.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    from models import MatchupState, RosterState, OpponentRosterState
    from pydantic import ValidationError

    errors = []

    if matchup_state is not None:
        try:
            MatchupState(**matchup_state)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                errors.append(f"MatchupState.{loc}: {error['msg']}")

    if roster_state is not None:
        try:
            RosterState(**roster_state)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                errors.append(f"RosterState.{loc}: {error['msg']}")

    if opponent_roster_state is not None:
        try:
            OpponentRosterState(**opponent_roster_state)
        except ValidationError as e:
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                errors.append(f"OpponentRosterState.{loc}: {error['msg']}")

    if errors:
        return False, "; ".join(errors)
    return True, None
