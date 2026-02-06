# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Retrieves head-to-head batter vs pitcher history and projections."""

import json
from pathlib import Path
from anthropic import beta_tool

# ---------------------------------------------------------------------------
# Load roster data and build player lookup
# ---------------------------------------------------------------------------

_ROSTER_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_rosters.json"

_PLAYERS: dict[str, dict] = {}


def _load_players() -> None:
    """Load all players from the roster file into _PLAYERS keyed by player_id."""
    if _PLAYERS:
        return
    if not _ROSTER_PATH.exists():
        return
    with open(_ROSTER_PATH) as f:
        rosters = json.load(f)
    for team_key in ("home", "away"):
        team = rosters.get(team_key, {})
        for player in team.get("lineup", []):
            _PLAYERS[player["player_id"]] = player
        for player in team.get("bench", []):
            _PLAYERS[player["player_id"]] = player
        sp = team.get("starting_pitcher")
        if sp:
            _PLAYERS[sp["player_id"]] = sp
        for player in team.get("bullpen", []):
            _PLAYERS[player["player_id"]] = player


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Deterministic matchup history simulation
# ---------------------------------------------------------------------------

def _hash_pair(batter_id: str, pitcher_id: str) -> int:
    """Produce a deterministic integer hash from a batter-pitcher pair.

    Uses a simple string-based hash so that the same pair always produces
    the same simulated history. This avoids needing an external database
    while keeping results reproducible.
    """
    combined = f"{batter_id}:{pitcher_id}"
    h = 0
    for ch in combined:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _simulated_pa(batter_id: str, pitcher_id: str) -> int:
    """Return a deterministic number of career plate appearances (0-45)."""
    h = _hash_pair(batter_id, pitcher_id)
    return h % 46  # 0-45 range


def _sample_size_label(pa: int) -> str:
    """Classify sample size reliability."""
    if pa == 0:
        return "none"
    if pa < 10:
        return "small"
    if pa < 30:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Stat derivation from combined batter + pitcher attributes
# ---------------------------------------------------------------------------

def _derive_matchup_stats(batter: dict, pitcher: dict, pa: int,
                          batter_player: dict, pitcher_player: dict) -> dict:
    """Derive matchup statistics from the combination of batter and pitcher attributes.

    The approach blends batter quality vs pitcher quality to produce
    realistic matchup-level stats. When sample size is small or zero, a
    similarity-model projected wOBA is provided instead of (or alongside)
    the raw matchup numbers.

    Args:
        batter: The batter attribute dict (contact, power, speed, eye, avg_vs_l, avg_vs_r).
        pitcher: The pitcher attribute dict (stuff, control, stamina, velocity, era_vs_l, era_vs_r).
        pa: Simulated number of career plate appearances.
        batter_player: Full player dict for the batter.
        pitcher_player: Full player dict for the pitcher.
    """
    contact = batter["contact"]
    power = batter["power"]
    eye = batter["eye"]
    speed = batter["speed"]

    stuff = pitcher["stuff"]
    control = pitcher["control"]
    velocity = pitcher["velocity"]

    # Determine platoon-relevant batting average
    pitcher_hand = pitcher_player.get("throws", "R")
    if pitcher_hand == "L":
        base_avg = batter["avg_vs_l"]
    else:
        base_avg = batter["avg_vs_r"]

    # Determine pitcher quality factor from ERA
    batter_hand = batter_player.get("bats", "R")
    if batter_hand == "L":
        pitcher_era = pitcher["era_vs_l"]
    elif batter_hand == "R":
        pitcher_era = pitcher["era_vs_r"]
    else:
        # Switch hitter: use the side opposite the pitcher's hand
        if pitcher_hand == "L":
            pitcher_era = pitcher["era_vs_r"]  # switch hitter bats R vs LHP
        else:
            pitcher_era = pitcher["era_vs_l"]  # switch hitter bats L vs RHP

    # Blend batter AVG with pitcher quality. A league-average ERA is ~4.00.
    # Better pitcher (lower ERA) suppresses the batter's AVG; worse pitcher inflates it.
    era_factor = pitcher_era / 4.00  # >1 means worse pitcher, <1 means better
    matchup_avg = _clamp(round(base_avg * era_factor, 3), 0.100, 0.400)

    # Add noise based on the deterministic hash (simulates matchup-specific quirks)
    h = _hash_pair(batter_player["player_id"], pitcher_player["player_id"])
    noise = ((h % 61) - 30) / 1000  # range -0.030 to +0.030
    matchup_avg = _clamp(round(matchup_avg + noise, 3), 0.100, 0.400)

    # Slugging: derived from power vs stuff
    iso = _clamp(round(0.050 + (power / 100) * 0.250 - (stuff / 100) * 0.080, 3), 0.030, 0.350)
    matchup_slg = _clamp(round(matchup_avg + iso, 3), 0.200, 0.700)

    # Strikeout rate: stuff drives K, contact resists it
    k_rate = _clamp(round(0.10 + (stuff / 100) * 0.22 - (contact / 100) * 0.12, 3), 0.05, 0.45)

    # Outcome distribution (groundball, flyball, line drive)
    # Power hitters hit more fly balls; control pitchers induce more groundballs
    gb_rate = _clamp(round(0.45 - (power / 100) * 0.15 + (control / 100) * 0.08, 3), 0.25, 0.60)
    ld_rate = _clamp(round(0.20 + (contact / 100) * 0.05 - (stuff / 100) * 0.03, 3), 0.15, 0.28)
    fb_rate = _clamp(round(1.0 - gb_rate - ld_rate, 3), 0.15, 0.45)

    # Similarity-model projected wOBA (always computed, but most relevant when sample is small)
    # Based on batter overall quality vs pitcher quality
    batter_quality = (contact * 0.25 + power * 0.25 + eye * 0.30 + speed * 0.05) / 100
    pitcher_quality = (stuff * 0.40 + control * 0.35 + (100 - pitcher_era * 10) * 0.25) / 100
    projected_woba = _clamp(
        round(0.280 + batter_quality * 0.15 - pitcher_quality * 0.10, 3),
        0.200, 0.450,
    )

    return {
        "matchup_avg": matchup_avg,
        "matchup_slg": matchup_slg,
        "k_rate": k_rate,
        "gb_rate": gb_rate,
        "fb_rate": fb_rate,
        "ld_rate": ld_rate,
        "projected_woba": projected_woba,
    }


def _build_pitch_vulnerability(batter: dict, pitcher: dict,
                               pitcher_player: dict) -> list[dict]:
    """Build pitch-type vulnerability breakdown.

    For each pitch type the pitcher throws, estimate how the batter
    performs against it based on batter attributes and pitch characteristics.
    """
    stuff = pitcher["stuff"]
    control = pitcher["control"]
    velocity = pitcher["velocity"]
    contact = batter["contact"]
    power = batter["power"]
    eye = batter["eye"]

    pitches = []

    # Fastball (FF): contact matters most
    ff_usage = _clamp(round(0.65 - (stuff / 100) * 0.20, 2), 0.35, 0.65)
    ff_woba = _clamp(
        round(0.280 + (contact / 100) * 0.10 + (power / 100) * 0.06 - (velocity - 90) * 0.008, 3),
        0.200, 0.450,
    )
    pitches.append({"pitch_type": "FF", "usage": ff_usage, "wOBA_against": ff_woba})

    remaining = round(1.0 - ff_usage, 2)

    # Slider (SL): eye and contact help against breaking balls
    sl_usage = _clamp(round(remaining * 0.45, 2), 0.10, 0.30)
    sl_woba = _clamp(
        round(0.250 + (eye / 100) * 0.08 + (contact / 100) * 0.05 - (stuff / 100) * 0.06, 3),
        0.180, 0.400,
    )
    pitches.append({"pitch_type": "SL", "usage": sl_usage, "wOBA_against": sl_woba})

    remaining = round(remaining - sl_usage, 2)

    # Changeup (CH): eye helps most; power hitters can drive mistakes
    ch_usage = _clamp(round(remaining * 0.65, 2), 0.08, 0.25)
    ch_woba = _clamp(
        round(0.260 + (eye / 100) * 0.10 + (power / 100) * 0.04 - (stuff / 100) * 0.05, 3),
        0.190, 0.420,
    )
    pitches.append({"pitch_type": "CH", "usage": ch_usage, "wOBA_against": ch_woba})

    # Curveball (CU): only if pitcher has decent stuff
    remaining = round(remaining - ch_usage, 2)
    if remaining >= 0.05 and stuff >= 60:
        cu_woba = _clamp(
            round(0.240 + (eye / 100) * 0.07 + (contact / 100) * 0.04 - (stuff / 100) * 0.05, 3),
            0.170, 0.400,
        )
        pitches.append({"pitch_type": "CU", "usage": remaining, "wOBA_against": cu_woba})

    return pitches


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------

@beta_tool
def get_matchup_data(batter_id: str, pitcher_id: str) -> str:
    """Retrieves head-to-head batter vs pitcher history and similarity-based
    projections. Returns direct matchup results, sample-size reliability,
    similarity-model projected wOBA, and pitch-type vulnerability breakdown.

    Args:
        batter_id: The unique identifier of the batter.
        pitcher_id: The unique identifier of the pitcher.
    Returns:
        JSON string with matchup data.
    """
    _load_players()

    # Validate batter_id
    if batter_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Batter '{batter_id}' not found in any roster.",
        })

    # Validate pitcher_id
    if pitcher_id not in _PLAYERS:
        return json.dumps({
            "status": "error",
            "error_code": "INVALID_PLAYER_ID",
            "message": f"Pitcher '{pitcher_id}' not found in any roster.",
        })

    batter_player = _PLAYERS[batter_id]
    pitcher_player = _PLAYERS[pitcher_id]

    # Verify the batter has batting attributes
    if "batter" not in batter_player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_BATTER",
            "message": f"Player '{batter_id}' ({batter_player.get('name', 'unknown')}) does not have batting attributes.",
        })

    # Verify the pitcher has pitching attributes
    if "pitcher" not in pitcher_player:
        return json.dumps({
            "status": "error",
            "error_code": "NOT_A_PITCHER",
            "message": f"Player '{pitcher_id}' ({pitcher_player.get('name', 'unknown')}) does not have pitching attributes.",
        })

    batter_attrs = batter_player["batter"]
    pitcher_attrs = pitcher_player["pitcher"]

    # Determine simulated career plate appearances
    pa = _simulated_pa(batter_id, pitcher_id)
    reliability = _sample_size_label(pa)

    # Derive matchup stats
    stats = _derive_matchup_stats(
        batter_attrs, pitcher_attrs, pa, batter_player, pitcher_player,
    )

    # Build pitch-type vulnerability
    pitch_vuln = _build_pitch_vulnerability(batter_attrs, pitcher_attrs, pitcher_player)

    # Build the response
    result: dict = {
        "status": "ok",
        "batter_id": batter_id,
        "batter_name": batter_player.get("name", "Unknown"),
        "pitcher_id": pitcher_id,
        "pitcher_name": pitcher_player.get("name", "Unknown"),
        "career_pa": pa,
        "sample_size_reliability": reliability,
    }

    if pa > 0:
        result["matchup_stats"] = {
            "AVG": stats["matchup_avg"],
            "SLG": stats["matchup_slg"],
            "K_rate": stats["k_rate"],
        }
        result["outcome_distribution"] = {
            "groundball": stats["gb_rate"],
            "flyball": stats["fb_rate"],
            "line_drive": stats["ld_rate"],
        }
    else:
        # No prior matchup history: return nulls for direct stats
        result["matchup_stats"] = None
        result["outcome_distribution"] = None
        result["no_history_message"] = (
            "No prior matchup history between these players. "
            "The similarity-model projected wOBA below is the best available estimate."
        )

    # Always include similarity-model projection (most useful when sample is small/none)
    result["similarity_projected_wOBA"] = stats["projected_woba"]

    # Pitch-type vulnerability breakdown
    result["pitch_type_vulnerability"] = pitch_vuln

    return json.dumps(result)
