# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Returns detailed status of all bullpen pitchers.

Loads bullpen data from sample_rosters.json and derives platoon splits,
availability, freshness, rest days, recent pitch counts, and warm-up state
for each reliever. Accepts optional parameters to reflect in-game state
(pitchers already used and removed, pitchers currently warming up).
"""

import json
from pathlib import Path

from anthropic import beta_tool
from tools.response import success_response, error_response

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


def _derive_era(pitcher_attrs: dict, vs_hand: str) -> float:
    """Get ERA vs a given batter hand directly from pitcher attributes."""
    key = "era_vs_l" if vs_hand == "L" else "era_vs_r"
    return pitcher_attrs.get(key, 4.00)


def _era_to_whip(era: float) -> float:
    """Approximate WHIP from ERA. Based on MLB correlation: WHIP ~ 0.667 + ERA * 0.114."""
    return round(_clamp(0.667 + era * 0.114, 0.80, 2.00), 3)


def _era_to_woba(era: float) -> float:
    """Approximate wOBA-against from ERA. Based on MLB correlation: wOBA ~ 0.210 + ERA * 0.028."""
    return round(_clamp(0.210 + era * 0.028, 0.220, 0.420), 3)


def _era_to_k_rate(pitcher_attrs: dict, era: float) -> float:
    """Approximate K% from stuff and ERA."""
    base_k = 0.10 + (pitcher_attrs.get("stuff", 50) / 100) * 0.25
    return round(_clamp(base_k + (4.00 - era) * 0.01, 0.08, 0.40), 3)


def _era_to_bb_rate(pitcher_attrs: dict, era: float) -> float:
    """Approximate BB% from control and ERA."""
    base_bb = 0.15 - (pitcher_attrs.get("control", 50) / 100) * 0.12
    return round(_clamp(base_bb + (era - 4.00) * 0.005, 0.02, 0.18), 3)


def _derive_platoon_splits(pitcher_attrs: dict) -> dict:
    """Derive platoon split stats (ERA, WHIP, wOBA, K%, BB%) vs LHB and RHB."""
    splits = {}
    for hand in ("L", "R"):
        era = _derive_era(pitcher_attrs, hand)
        splits[f"vs_{hand}HB"] = {
            "ERA": round(era, 2),
            "WHIP": _era_to_whip(era),
            "wOBA_against": _era_to_woba(era),
            "K_pct": _era_to_k_rate(pitcher_attrs, era),
            "BB_pct": _era_to_bb_rate(pitcher_attrs, era),
        }
    return splits


def _derive_freshness(
    pitch_counts_last_3: list[int],
    days_since_last: int,
    stamina: float,
) -> str:
    """Derive freshness (FRESH/MODERATE/TIRED) from workload and stamina."""
    stamina_factor = stamina / 100.0
    tired_threshold = 25 + stamina_factor * 40
    moderate_threshold = 12 + stamina_factor * 25

    effective_load = max(0, sum(pitch_counts_last_3) - days_since_last * 10)

    if effective_load >= tired_threshold:
        return "TIRED"
    elif effective_load >= moderate_threshold:
        return "MODERATE"
    return "FRESH"


def _derive_availability(
    freshness: str,
    pitch_counts_last_3: list[int],
    days_since_last: int,
    stamina: float,
) -> tuple[bool, str | None]:
    """Determine availability and reason if unavailable."""
    total_pitches = sum(pitch_counts_last_3)
    days_with_appearances = sum(1 for p in pitch_counts_last_3 if p > 0)

    if days_with_appearances >= 3 and stamina < 50:
        return False, "Pitched 3 consecutive days, needs rest"
    if total_pitches > 40 + stamina * 0.5:
        return False, f"Heavy 3-day workload ({total_pitches} pitches)"
    if days_since_last == 0 and pitch_counts_last_3[0] >= 30 and stamina < 40:
        return False, "Pitched today with high pitch count"
    return True, None


@beta_tool
def get_bullpen_status(
    team: str = "home",
    used_pitcher_ids: str | None = None,
    warming_pitcher_ids: str | None = None,
    ready_pitcher_ids: str | None = None,
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
    TOOL_NAME = "get_bullpen_status"

    if team not in ("home", "away"):
        return error_response(TOOL_NAME, "INVALID_PARAMETER",
            f"Invalid team value: '{team}'. Must be 'home' or 'away'.")

    team_data = rosters.get(team)
    if not team_data:
        return error_response(TOOL_NAME, "ROSTER_NOT_FOUND",
            f"No roster data found for team '{team}'.")

    bullpen_data = team_data.get("bullpen", [])
    if not bullpen_data:
        return error_response(TOOL_NAME, "NO_BULLPEN",
            f"No bullpen pitchers found for team '{team}'.")

    # Parse comma-separated ID lists
    def _parse_ids(csv: str | None) -> set[str]:
        return {p.strip() for p in csv.split(",") if p.strip()} if csv else set()

    used_ids = _parse_ids(used_pitcher_ids)
    warming_ids = _parse_ids(warming_pitcher_ids)
    ready_ids = _parse_ids(ready_pitcher_ids)

    pitchers = []
    for bp in bullpen_data:
        pid = bp["player_id"]

        if pid in used_ids:
            continue

        pitcher_attrs = bp.get("pitcher", {})
        role = bp.get("role", "MIDDLE")
        throws = bp.get("throws", bp.get("bats", "R"))
        stamina = pitcher_attrs.get("stamina", 40)

        pitch_counts_last_3 = _derive_recent_pitch_counts(role, stamina, pid)
        days_since_last = _derive_days_since_last(role, pid)
        freshness = _derive_freshness(pitch_counts_last_3, days_since_last, stamina)
        available, unavailable_reason = _derive_availability(
            freshness, pitch_counts_last_3, days_since_last, stamina
        )
        platoon_splits = _derive_platoon_splits(pitcher_attrs)

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

    role_order = {"CLOSER": 0, "SETUP": 1, "MIDDLE": 2, "LONG": 3, "MOPUP": 4}
    pitchers.sort(key=lambda p: role_order.get(p["role"], 5))

    return success_response(TOOL_NAME, {
        "team": team,
        "team_name": team_data.get("team_name", "Unknown"),
        "bullpen_count": len(pitchers),
        "available_count": sum(1 for p in pitchers if p["available"]),
        "bullpen": pitchers,
    })


_PITCH_PATTERNS: dict[str, list[list[int]]] = {
    "CLOSER": [[0,15,0], [14,0,0], [0,0,16], [0,0,0], [15,0,18]],
    "SETUP":  [[0,22,0], [18,0,20], [0,0,0], [20,0,0], [0,18,22]],
    "MIDDLE": [[25,0,0], [0,0,28], [0,22,0], [0,0,0], [22,0,25]],
    "LONG":   [[0,0,45], [0,0,0], [0,40,0], [0,0,0], [0,0,50]],
    "MOPUP":  [[0,0,0], [0,35,0], [0,0,40], [0,0,0], [30,0,0]],
}


def _derive_recent_pitch_counts(role: str, stamina: float, player_id: str) -> list[int]:
    """Derive deterministic recent pitch counts from role and player_id hash."""
    h = sum(ord(c) for c in player_id)
    patterns = _PITCH_PATTERNS.get(role, _PITCH_PATTERNS["MOPUP"])
    return patterns[h % len(patterns)]


def _derive_days_since_last(role: str, player_id: str) -> int:
    """Derive days since last appearance from role and player ID hash."""
    h = sum(ord(c) for c in player_id)
    pitch_counts = _derive_recent_pitch_counts(role, 0, player_id)

    for i, count in enumerate(pitch_counts):
        if count > 0:
            return i + 1

    if role == "CLOSER":
        return 3 + (h % 3)
    elif role in ("SETUP", "MIDDLE"):
        return 3 + (h % 4)
    return 4 + (h % 5)
