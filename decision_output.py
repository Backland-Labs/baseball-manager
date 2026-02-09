# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0"]
# ///
"""Decision output formatting for the baseball manager AI agent.

Formats the agent's decisions for output:
- Active decisions are formatted as tweet-ready text (~280 chars)
- All decisions (active and no-action) produce log entries
- Returns both tweet text (if applicable) and full reasoning
"""

from __future__ import annotations

from dataclasses import dataclass

# Decision types that represent "no action" -- the agent is not intervening
NO_ACTION_TYPES = {
    "NO_ACTION", "SWING_AWAY", "LET_HIM_HIT", "NO_CHANGE",
    "CONTINUE", "HOLD", "STANDARD_PLAY", "PITCH_TO_BATTER",
}

# Maximum tweet length
TWEET_MAX_LENGTH = 280


@dataclass
class DecisionOutput:
    """Formatted decision output.

    Attributes:
        is_active: Whether this is an active managerial decision (vs no-action).
        tweet_text: Tweet-ready text for active decisions. None for no-action.
        full_reasoning: Complete reasoning from the agent (for logging).
        log_entry: Brief log entry summarizing the decision.
        decision_type: The raw decision type string (e.g. "PITCHING_CHANGE").
        action_details: The raw action details string.
        game_context: Short description of the game situation.
    """
    is_active: bool
    tweet_text: str | None
    full_reasoning: str
    log_entry: str
    decision_type: str
    action_details: str
    game_context: str


def _build_game_context(
    inning: int,
    half: str,
    outs: int,
    score_home: int,
    score_away: int,
    runners: dict | None = None,
    home_team: str = "",
    away_team: str = "",
) -> str:
    """Build a short game context string for inclusion in tweets and logs.

    Args:
        inning: Current inning number.
        half: "TOP" or "BOTTOM".
        outs: Number of outs (0-2).
        score_home: Home team score.
        score_away: Away team score.
        runners: Dict with keys "first", "second", "third" mapping to
            runner info dicts or None.
        home_team: Home team name (optional).
        away_team: Away team name (optional).

    Returns:
        A compact situation string like "Bot 7, 1 out, runner on 2nd | Away 4, Home 3"
    """
    half_str = "Top" if half == "TOP" else "Bot"
    out_str = f"{outs} out" if outs != 1 else "1 out"

    # Runner description
    on_bases = []
    if runners:
        if runners.get("first"):
            on_bases.append("1st")
        if runners.get("second"):
            on_bases.append("2nd")
        if runners.get("third"):
            on_bases.append("3rd")

    if not on_bases:
        runners_str = "bases empty"
    elif len(on_bases) == 3:
        runners_str = "bases loaded"
    elif len(on_bases) == 1:
        runners_str = f"runner on {on_bases[0]}"
    else:
        runners_str = f"runners on {' & '.join(on_bases)}"

    situation = f"{half_str} {inning}, {out_str}, {runners_str}"

    # Score
    if home_team and away_team:
        score_str = f"{away_team} {score_away}, {home_team} {score_home}"
    else:
        score_str = f"Away {score_away}, Home {score_home}"

    return f"{situation} | {score_str}"


def _truncate_to_tweet(text: str, max_length: int = TWEET_MAX_LENGTH) -> str:
    """Truncate text to fit within tweet length, breaking at word boundaries.

    If text is already within the limit, returns it unchanged.
    Otherwise, truncates at the last word boundary before the limit
    and appends an ellipsis.

    Args:
        text: The text to truncate.
        max_length: Maximum character count (default 280).

    Returns:
        Text that fits within max_length characters.
    """
    if len(text) <= max_length:
        return text

    # Reserve space for ellipsis
    truncated = text[:max_length - 1]

    # Try to break at a word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "\u2026"


def _format_tweet_text(
    decision_type: str,
    action_details: str,
    reasoning: str,
    game_context: str,
    key_factors: list[str] | None = None,
) -> str:
    """Format an active decision as tweet-ready text.

    The tweet includes game context and the core decision. If the action
    details alone are short enough, key statistical justification from
    key_factors or reasoning is appended.

    Args:
        decision_type: The decision type (e.g. "PITCHING_CHANGE").
        action_details: Specific action description from the agent.
        reasoning: Full reasoning text.
        game_context: Short game situation string.
        key_factors: Optional list of key factors driving the decision.

    Returns:
        Tweet text, guaranteed to be <= 280 characters.
    """
    # Build the core tweet: context + action
    # Start with the action details as the primary content
    core = action_details.strip()

    # If action_details is very short or generic, try to enrich it
    if len(core) < 30 and key_factors:
        factor_text = ". ".join(key_factors[:2])
        core = f"{core}. {factor_text}"

    # Prepend game context
    tweet = f"{game_context} -- {core}"

    # If we have room, try to add a key factor
    if len(tweet) < TWEET_MAX_LENGTH - 5 and key_factors:
        for factor in key_factors:
            candidate = f"{tweet}. {factor}"
            if len(candidate) <= TWEET_MAX_LENGTH:
                tweet = candidate
                break

    return _truncate_to_tweet(tweet)


def _format_log_entry(
    decision_type: str,
    action_details: str,
    game_context: str,
    is_active: bool,
) -> str:
    """Format a brief log entry for any decision (active or no-action).

    Args:
        decision_type: The decision type string.
        action_details: The action details string.
        game_context: Short game situation string.
        is_active: Whether this was an active decision.

    Returns:
        A one-line log entry string.
    """
    if is_active:
        return f"[ACTIVE] {game_context} -> {decision_type}: {action_details}"
    else:
        return f"[NO_ACTION] {game_context} -> {decision_type}"


def format_decision_output(
    decision_dict: dict,
    inning: int,
    half: str,
    outs: int,
    score_home: int,
    score_away: int,
    runners: dict | None = None,
    home_team: str = "",
    away_team: str = "",
) -> DecisionOutput:
    """Format an agent decision for output.

    This is the main entry point for the decision_output module. It takes
    a decision dict (from ManagerDecision.model_dump() or a raw dict) and
    game state information, and produces a DecisionOutput with tweet text
    (for active decisions) and log entries (for all decisions).

    Args:
        decision_dict: The agent's decision, with at minimum a "decision"
            key. May also contain "action_details", "reasoning",
            "key_factors", "risks", "confidence".
        inning: Current inning number.
        half: "TOP" or "BOTTOM".
        outs: Number of outs.
        score_home: Home team score.
        score_away: Away team score.
        runners: Optional runner state dict with keys "first", "second",
            "third".
        home_team: Home team name (optional, improves tweet readability).
        away_team: Away team name (optional, improves tweet readability).

    Returns:
        DecisionOutput with tweet_text (None for no-action), full_reasoning,
        log_entry, and metadata.
    """
    decision_type = (decision_dict.get("decision") or "").upper().strip()
    action_details = decision_dict.get("action_details", "")
    reasoning = decision_dict.get("reasoning", "")
    key_factors = decision_dict.get("key_factors", [])
    confidence = decision_dict.get("confidence", 0.0)
    risks = decision_dict.get("risks", [])

    is_active = decision_type not in NO_ACTION_TYPES and bool(decision_type)

    game_context = _build_game_context(
        inning=inning,
        half=half,
        outs=outs,
        score_home=score_home,
        score_away=score_away,
        runners=runners,
        home_team=home_team,
        away_team=away_team,
    )

    # Build full reasoning for logging (combine all available info)
    reasoning_parts = []
    if reasoning:
        reasoning_parts.append(reasoning)
    if key_factors:
        reasoning_parts.append(f"Key factors: {'; '.join(key_factors)}")
    if risks:
        reasoning_parts.append(f"Risks: {'; '.join(risks)}")
    if confidence:
        reasoning_parts.append(f"Confidence: {confidence:.0%}")
    full_reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No reasoning provided"

    # Format tweet text for active decisions only
    tweet_text = None
    if is_active:
        tweet_text = _format_tweet_text(
            decision_type=decision_type,
            action_details=action_details,
            reasoning=reasoning,
            game_context=game_context,
            key_factors=key_factors,
        )

    log_entry = _format_log_entry(
        decision_type=decision_type,
        action_details=action_details,
        game_context=game_context,
        is_active=is_active,
    )

    return DecisionOutput(
        is_active=is_active,
        tweet_text=tweet_text,
        full_reasoning=full_reasoning,
        log_entry=log_entry,
        decision_type=decision_type,
        action_details=action_details,
        game_context=game_context,
    )


def format_decision_output_from_game_state(
    decision_dict: dict,
    game_state_dict: dict,
    home_team: str = "",
    away_team: str = "",
) -> DecisionOutput:
    """Convenience wrapper that extracts game state fields from a dict.

    Useful when you have the game state as a dict (e.g. from a log entry
    or from game_state_to_dict).

    Args:
        decision_dict: The agent's decision dict.
        game_state_dict: Dict with keys matching the game_state portion
            of a log entry (inning, half, outs, score, runners).
        home_team: Home team name.
        away_team: Away team name.

    Returns:
        DecisionOutput.
    """
    score = game_state_dict.get("score", {})
    runners_raw = game_state_dict.get("runners", {})

    return format_decision_output(
        decision_dict=decision_dict,
        inning=game_state_dict.get("inning", 1),
        half=game_state_dict.get("half", "TOP"),
        outs=game_state_dict.get("outs", 0),
        score_home=score.get("home", 0),
        score_away=score.get("away", 0),
        runners=runners_raw if runners_raw else None,
        home_team=home_team,
        away_team=away_team,
    )
