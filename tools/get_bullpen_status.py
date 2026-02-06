"""Returns detailed status of all bullpen pitchers.

Loads bullpen data from sample_rosters.json and derives platoon splits,
availability, freshness, rest days, recent pitch counts, and warm-up state
for each reliever. Accepts optional parameters to reflect in-game state
(pitchers already used and removed, pitchers currently warming up).
"""

import json
from pathlib import Path
from typing import Optional

from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Load roster data and build player lookup
# ---------------------------------------------------------------------------

_ROSTER_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_rosters.json"

_ROSTERS: dict | None = None


def _load_rosters() -> dict:
    """Load rosters from the data file. Cached after first call."""
    global _ROSTERS
    if _ROSTERS is not None:
        return _ROSTERS
    if not _ROSTER_PATH.exists():
        _ROSTERS = {}
        return _ROSTERS
    with open(_ROSTER_PATH) as f:
        _ROSTERS = json.load(f)
    return _ROSTERS


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Derive platoon splits from pitcher attributes
# ---------------------------------------------------------------------------


def _derive_era(pitcher_attrs: dict, vs_hand: str) -> float:
    """Get ERA vs a given batter hand directly from pitcher attributes."""
    if vs_hand == "L":
        return pitcher_attrs.get("era_vs_l", 4.00)
    else:
        return pitcher_attrs.get("era_vs_r", 4.00)


def _era_to_whip(era: float) -> float:
    """Approximate WHIP from ERA. Based on MLB correlation: WHIP ~ 0.667 + ERA * 0.114."""
    return round(_clamp(0.667 + era * 0.114, 0.80, 2.00), 3)


def _era_to_woba(era: float) -> float:
    """Approximate wOBA-against from ERA. Based on MLB correlation: wOBA ~ 0.210 + ERA * 0.028."""
    return round(_clamp(0.210 + era * 0.028, 0.220, 0.420), 3)


def _era_to_k_rate(pitcher_attrs: dict, era: float) -> float:
    """Approximate K% from stuff and ERA. Higher stuff and lower ERA = more strikeouts."""
    stuff = pitcher_attrs.get("stuff", 50)
    # Base K% from stuff: stuff 50 = ~20%, stuff 100 = ~35%, stuff 0 = ~10%
    base_k = 0.10 + (stuff / 100) * 0.25
    # ERA adjustment: lower ERA correlates with higher K%
    era_adj = (4.00 - era) * 0.01  # each 1.00 ERA below 4.00 = +1% K rate
    return round(_clamp(base_k + era_adj, 0.08, 0.40), 3)


def _era_to_bb_rate(pitcher_attrs: dict, era: float) -> float:
    """Approximate BB% from control and ERA. Higher control = fewer walks."""
    control = pitcher_attrs.get("control", 50)
    # Base BB% from control: control 50 = ~9%, control 100 = ~3%, control 0 = ~15%
    base_bb = 0.15 - (control / 100) * 0.12
    # ERA adjustment: higher ERA correlates with slightly higher BB%
    era_adj = (era - 4.00) * 0.005
    return round(_clamp(base_bb + era_adj, 0.02, 0.18), 3)


def _derive_platoon_splits(pitcher_attrs: dict) -> dict:
    """Derive full platoon split stats from pitcher attributes.

    Returns splits vs LHB and vs RHB with ERA, WHIP, wOBA-against, K%, and BB%.
    """
    splits = {}
    for hand in ("L", "R"):
        label = f"vs_LHB" if hand == "L" else "vs_RHB"
        era = _derive_era(pitcher_attrs, hand)
        splits[label] = {
            "ERA": round(era, 2),
            "WHIP": _era_to_whip(era),
            "wOBA_against": _era_to_woba(era),
            "K_pct": _era_to_k_rate(pitcher_attrs, era),
            "BB_pct": _era_to_bb_rate(pitcher_attrs, era),
        }
    return splits


# ---------------------------------------------------------------------------
# Derive freshness from pitch counts and days rest
# ---------------------------------------------------------------------------


def _derive_freshness(
    pitch_counts_last_3: list[int],
    days_since_last: int,
    stamina: float,
) -> str:
    """Derive freshness level from recent workload and stamina attribute.

    Freshness categories:
    - FRESH: well-rested, low recent workload
    - MODERATE: some recent usage but still effective
    - TIRED: heavy recent usage, diminished effectiveness

    Factors:
    - Total pitches in last 3 days
    - Days since last appearance
    - Stamina attribute (higher stamina = recovers faster)
    """
    total_pitches = sum(pitch_counts_last_3)

    # Stamina-adjusted thresholds:
    # Higher stamina raises the threshold before a pitcher gets tired
    # Stamina 30 (typical closer) -> tired at 30 pitches, moderate at 15
    # Stamina 55 (long reliever) -> tired at 50 pitches, moderate at 30
    stamina_factor = stamina / 100.0  # 0.0 to 1.0
    tired_threshold = 25 + stamina_factor * 40  # 25-65 pitches
    moderate_threshold = 12 + stamina_factor * 25  # 12-37 pitches

    # Days of rest reduce workload impact
    # Each day of rest "forgives" some pitch load
    rest_relief = days_since_last * 10
    effective_load = max(0, total_pitches - rest_relief)

    if effective_load >= tired_threshold:
        return "TIRED"
    elif effective_load >= moderate_threshold:
        return "MODERATE"
    else:
        return "FRESH"


def _derive_availability(
    freshness: str,
    pitch_counts_last_3: list[int],
    days_since_last: int,
    stamina: float,
) -> tuple[bool, str | None]:
    """Determine if a pitcher is available and why they might not be.

    A pitcher is unavailable if:
    - They pitched in 3 consecutive days (need mandatory rest) unless high stamina
    - Their total recent workload is extreme (3-day total > stamina-based limit)
    - They are marked as TIRED and pitched yesterday with high pitch count

    Returns (available, reason_if_unavailable).
    """
    total_pitches = sum(pitch_counts_last_3)
    days_with_appearances = sum(1 for p in pitch_counts_last_3 if p > 0)

    # Rule: 3 consecutive appearances -> unavailable unless very high stamina
    if days_with_appearances >= 3 and stamina < 50:
        return False, "Pitched 3 consecutive days, needs rest"

    # Rule: extreme workload (total > 60 pitches in 3 days for low-stamina pitchers)
    max_3_day = 40 + stamina * 0.5  # 40-90 pitch 3-day max
    if total_pitches > max_3_day:
        return False, f"Heavy 3-day workload ({total_pitches} pitches)"

    # Rule: pitched yesterday with 30+ pitches and low stamina
    if days_since_last == 0 and pitch_counts_last_3[0] >= 30 and stamina < 40:
        return False, "Pitched today with high pitch count"

    return True, None


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


@beta_tool
def get_bullpen_status(
    team: str = "home",
    used_pitcher_ids: Optional[str] = None,
    warming_pitcher_ids: Optional[str] = None,
    ready_pitcher_ids: Optional[str] = None,
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
    rosters = _load_rosters()

    # --- Validate team parameter ---
    if team not in ("home", "away"):
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PARAMETER",
            "message": f"Invalid team value: '{team}'. Must be 'home' or 'away'.",
        })

    team_data = rosters.get(team)
    if not team_data:
        return json.dumps({
            "status": "error",
            "error_code": "ROSTER_NOT_FOUND",
            "message": f"No roster data found for team '{team}'.",
        })

    bullpen_data = team_data.get("bullpen", [])
    if not bullpen_data:
        return json.dumps({
            "status": "error",
            "error_code": "NO_BULLPEN",
            "message": f"No bullpen pitchers found for team '{team}'.",
        })

    # Parse comma-separated ID lists
    used_ids = set()
    if used_pitcher_ids:
        used_ids = {pid.strip() for pid in used_pitcher_ids.split(",") if pid.strip()}

    warming_ids = set()
    if warming_pitcher_ids:
        warming_ids = {pid.strip() for pid in warming_pitcher_ids.split(",") if pid.strip()}

    ready_ids = set()
    if ready_pitcher_ids:
        ready_ids = {pid.strip() for pid in ready_pitcher_ids.split(",") if pid.strip()}

    # --- Build bullpen status for each pitcher ---
    pitchers = []
    for bp in bullpen_data:
        pid = bp["player_id"]

        # Step 9: Pitchers already used and removed in this game are excluded
        if pid in used_ids:
            continue

        pitcher_attrs = bp.get("pitcher", {})
        role = bp.get("role", "MIDDLE")
        throws = bp.get("bats", "R")  # For pitchers, 'bats' in roster may be handedness
        # Check for explicit throws field, or derive from roster structure
        if "throws" in bp:
            throws = bp["throws"]

        stamina = pitcher_attrs.get("stamina", 40)

        # For the simulation, we derive plausible recent usage from the pitcher's
        # role and stamina. In a real game, these would come from actual game logs.
        # Closers/setup: pitch less often but in shorter stints.
        # Long/mopup: pitch more innings when they appear.
        pitch_counts_last_3 = _derive_recent_pitch_counts(role, stamina, pid)
        days_since_last = _derive_days_since_last(role, pid)

        # Derive freshness from workload
        freshness = _derive_freshness(pitch_counts_last_3, days_since_last, stamina)

        # Derive availability
        available, unavailable_reason = _derive_availability(
            freshness, pitch_counts_last_3, days_since_last, stamina
        )

        # Derive platoon splits
        platoon_splits = _derive_platoon_splits(pitcher_attrs)

        # Determine warm-up state
        if pid in ready_ids:
            warmup_state = "ready"
        elif pid in warming_ids:
            warmup_state = "warming"
        else:
            warmup_state = "cold"

        pitchers.append({
            "player_id": pid,
            "name": bp.get("name", "Unknown"),
            "throws": throws,
            "role": role,
            "available": available,
            "unavailable_reason": unavailable_reason,
            "freshness": freshness,
            "days_since_last_appearance": days_since_last,
            "pitch_counts_last_3": pitch_counts_last_3,
            "platoon_splits": platoon_splits,
            "warmup_state": warmup_state,
            "velocity": pitcher_attrs.get("velocity", 90.0),
            "stuff": pitcher_attrs.get("stuff", 50),
            "control": pitcher_attrs.get("control", 50),
            "stamina": stamina,
        })

    # Sort by role priority: CLOSER > SETUP > MIDDLE > LONG > MOPUP
    role_order = {"CLOSER": 0, "SETUP": 1, "MIDDLE": 2, "LONG": 3, "MOPUP": 4}
    pitchers.sort(key=lambda p: role_order.get(p["role"], 5))

    return json.dumps({
        "status": "ok",
        "team": team,
        "team_name": team_data.get("team_name", "Unknown"),
        "bullpen_count": len(pitchers),
        "available_count": sum(1 for p in pitchers if p["available"]),
        "bullpen": pitchers,
    })


# ---------------------------------------------------------------------------
# Helpers for deriving plausible recent usage from role/stamina
# ---------------------------------------------------------------------------


def _derive_recent_pitch_counts(role: str, stamina: float, player_id: str) -> list[int]:
    """Derive plausible recent pitch counts based on role and a deterministic hash.

    In a real system these come from game logs. Here we use the player_id hash
    to create deterministic but varied usage patterns.
    """
    # Use player_id hash for deterministic variety
    h = sum(ord(c) for c in player_id)

    if role == "CLOSER":
        # Closers pitch 1 inning, 12-20 pitches when they appear
        # Appear roughly every 2-3 days
        patterns = [
            [0, 15, 0],    # pitched day before yesterday
            [14, 0, 0],    # pitched yesterday
            [0, 0, 16],    # pitched 3 days ago
            [0, 0, 0],     # well rested
            [15, 0, 18],   # pitched yesterday and 3 days ago
        ]
    elif role == "SETUP":
        # Setup men pitch 1-1.5 innings, 15-25 pitches
        patterns = [
            [0, 22, 0],
            [18, 0, 20],
            [0, 0, 0],
            [20, 0, 0],
            [0, 18, 22],
        ]
    elif role == "MIDDLE":
        # Middle relievers: 1-2 innings, 18-30 pitches
        patterns = [
            [25, 0, 0],
            [0, 0, 28],
            [0, 22, 0],
            [0, 0, 0],
            [22, 0, 25],
        ]
    elif role == "LONG":
        # Long relievers: 2-4 innings when used, but less frequently
        patterns = [
            [0, 0, 45],
            [0, 0, 0],
            [0, 40, 0],
            [0, 0, 0],
            [0, 0, 50],
        ]
    else:
        # MOPUP: varies widely
        patterns = [
            [0, 0, 0],
            [0, 35, 0],
            [0, 0, 40],
            [0, 0, 0],
            [30, 0, 0],
        ]

    return patterns[h % len(patterns)]


def _derive_days_since_last(role: str, player_id: str) -> int:
    """Derive days since last appearance from role and player ID hash."""
    h = sum(ord(c) for c in player_id)
    pitch_counts = _derive_recent_pitch_counts(role, 0, player_id)

    # Find most recent day with pitches (index 0 = today/yesterday, 1 = 2 days ago, etc.)
    for i, count in enumerate(pitch_counts):
        if count > 0:
            return i + 1  # 1-indexed: 1 = yesterday, 2 = day before, 3 = three days ago

    # No recent appearances
    if role == "CLOSER":
        return 3 + (h % 3)  # 3-5 days
    elif role in ("SETUP", "MIDDLE"):
        return 3 + (h % 4)  # 3-6 days
    else:
        return 4 + (h % 5)  # 4-8 days
