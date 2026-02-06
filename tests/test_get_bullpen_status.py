# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_bullpen_status tool.

Verifies all feature requirements from features.json:
1. Returns all bullpen pitchers for the managed team
2. Each pitcher includes availability status and reason if unavailable
3. Each pitcher includes role (closer, setup, middle, long, mopup)
4. Each pitcher includes freshness level (fresh, moderate, tired)
5. Each pitcher includes days since last appearance
6. Each pitcher includes pitch counts from last 3 appearances
7. Each pitcher includes platoon splits (vs LHB and vs RHB)
8. Each pitcher includes current warm-up state (cold, warming, ready)
9. Pitchers already used and removed in this game are excluded
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_bullpen_status import (
    get_bullpen_status,
    _derive_freshness,
    _derive_availability,
    _derive_platoon_splits,
    _derive_recent_pitch_counts,
    _derive_days_since_last,
    _era_to_whip,
    _era_to_woba,
    _era_to_k_rate,
    _era_to_bb_rate,
    _load_rosters,
    _clamp,
)


def parse(result: str) -> dict:
    return json.loads(result)


def parse_ok(result: str) -> dict:
    """Parse a success response, returning the data payload with status merged in."""
    raw = json.loads(result)
    assert raw["status"] == "ok"
    data = dict(raw["data"])
    data["status"] = "ok"
    return data


# -----------------------------------------------------------------------
# Step 1: Returns all bullpen pitchers for the managed team
# -----------------------------------------------------------------------

def test_step1_returns_all_home_bullpen():
    """Step 1: Returns all 8 home bullpen pitchers."""
    result = parse_ok(get_bullpen_status("home"))
    assert result["status"] == "ok"
    assert result["team"] == "home"
    assert result["team_name"] == "Rivertown Otters"
    assert result["bullpen_count"] == 8
    assert len(result["bullpen"]) == 8


def test_step1_returns_all_away_bullpen():
    """Step 1: Returns all 8 away bullpen pitchers."""
    result = parse_ok(get_bullpen_status("away"))
    assert result["status"] == "ok"
    assert result["team"] == "away"
    assert result["team_name"] == "Lakewood Falcons"
    assert result["bullpen_count"] == 8
    assert len(result["bullpen"]) == 8


def test_step1_default_team_is_home():
    """Step 1: Default team parameter is 'home'."""
    result = parse_ok(get_bullpen_status())
    assert result["team"] == "home"
    assert result["team_name"] == "Rivertown Otters"


def test_step1_invalid_team_returns_error():
    """Step 1: Invalid team parameter returns error."""
    result = parse(get_bullpen_status("west"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"
    assert "west" in result["message"]


def test_step1_each_pitcher_has_player_id_and_name():
    """Step 1: Each pitcher has player_id and name."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "player_id" in p
        assert "name" in p
        assert p["player_id"].startswith("h_bp")
        assert len(p["name"]) > 0


def test_step1_each_pitcher_has_throws():
    """Step 1: Each pitcher has throws hand (L or R)."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "throws" in p
        assert p["throws"] in ("L", "R")


def test_step1_includes_available_count():
    """Step 1: Response includes count of available pitchers."""
    result = parse_ok(get_bullpen_status("home"))
    assert "available_count" in result
    assert isinstance(result["available_count"], int)
    assert result["available_count"] <= result["bullpen_count"]


def test_step1_home_pitcher_ids_match_roster():
    """Step 1: All returned pitcher IDs match the home bullpen in sample_rosters.json."""
    result = parse_ok(get_bullpen_status("home"))
    expected_ids = {f"h_bp{i}" for i in range(1, 9)}
    actual_ids = {p["player_id"] for p in result["bullpen"]}
    assert actual_ids == expected_ids


def test_step1_away_pitcher_ids_match_roster():
    """Step 1: All returned pitcher IDs match the away bullpen in sample_rosters.json."""
    result = parse_ok(get_bullpen_status("away"))
    expected_ids = {f"a_bp{i}" for i in range(1, 9)}
    actual_ids = {p["player_id"] for p in result["bullpen"]}
    assert actual_ids == expected_ids


# -----------------------------------------------------------------------
# Step 2: Each pitcher includes availability status and reason
# -----------------------------------------------------------------------

def test_step2_each_pitcher_has_availability():
    """Step 2: Each pitcher has available boolean and unavailable_reason."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "available" in p
        assert isinstance(p["available"], bool)
        assert "unavailable_reason" in p
        # If available, reason should be None
        if p["available"]:
            assert p["unavailable_reason"] is None


def test_step2_unavailable_has_reason():
    """Step 2: If a pitcher is unavailable, a reason is provided."""
    # Test the availability derivation directly with a heavy workload
    avail, reason = _derive_availability("TIRED", [25, 25, 25], 0, 30)
    assert not avail
    assert reason is not None
    assert len(reason) > 0


def test_step2_fresh_pitcher_is_available():
    """Step 2: A well-rested pitcher should be available."""
    avail, reason = _derive_availability("FRESH", [0, 0, 0], 5, 50)
    assert avail
    assert reason is None


def test_step2_three_consecutive_days_low_stamina():
    """Step 2: Pitcher with 3 consecutive days and low stamina is unavailable."""
    avail, reason = _derive_availability("TIRED", [15, 15, 15], 0, 30)
    assert not avail
    assert "3 consecutive" in reason


def test_step2_three_consecutive_days_high_stamina():
    """Step 2: Pitcher with 3 consecutive days but high stamina remains available."""
    avail, reason = _derive_availability("TIRED", [15, 15, 15], 0, 55)
    assert avail
    assert reason is None


def test_step2_heavy_3_day_workload():
    """Step 2: Pitcher with extreme 3-day workload is unavailable."""
    # stamina 30 -> max_3_day = 40 + 30*0.5 = 55
    avail, reason = _derive_availability("TIRED", [30, 15, 15], 0, 30)
    assert not avail
    assert "workload" in reason.lower() or "consecutive" in reason.lower()


def test_step2_high_pitch_count_today_low_stamina():
    """Step 2: Pitched today with 30+ pitches and low stamina is unavailable."""
    avail, reason = _derive_availability("TIRED", [35, 0, 0], 0, 30)
    assert not avail
    assert reason is not None


# -----------------------------------------------------------------------
# Step 3: Each pitcher includes role
# -----------------------------------------------------------------------

def test_step3_each_pitcher_has_role():
    """Step 3: Each pitcher includes a valid role."""
    result = parse_ok(get_bullpen_status("home"))
    valid_roles = {"CLOSER", "SETUP", "MIDDLE", "LONG", "MOPUP"}
    for p in result["bullpen"]:
        assert "role" in p
        assert p["role"] in valid_roles


def test_step3_role_distribution():
    """Step 3: Bullpen has expected role distribution."""
    result = parse_ok(get_bullpen_status("home"))
    roles = [p["role"] for p in result["bullpen"]]
    assert roles.count("CLOSER") == 1
    assert roles.count("SETUP") == 2
    assert roles.count("MIDDLE") == 2
    assert roles.count("LONG") == 1
    assert roles.count("MOPUP") == 2


def test_step3_sorted_by_role_priority():
    """Step 3: Pitchers are sorted by role priority (CLOSER first, MOPUP last)."""
    result = parse_ok(get_bullpen_status("home"))
    role_order = {"CLOSER": 0, "SETUP": 1, "MIDDLE": 2, "LONG": 3, "MOPUP": 4}
    roles = [role_order[p["role"]] for p in result["bullpen"]]
    assert roles == sorted(roles)


# -----------------------------------------------------------------------
# Step 4: Each pitcher includes freshness level
# -----------------------------------------------------------------------

def test_step4_each_pitcher_has_freshness():
    """Step 4: Each pitcher includes a valid freshness level."""
    result = parse_ok(get_bullpen_status("home"))
    valid_freshness = {"FRESH", "MODERATE", "TIRED"}
    for p in result["bullpen"]:
        assert "freshness" in p
        assert p["freshness"] in valid_freshness


def test_step4_freshness_derivation_fresh():
    """Step 4: No recent workload = FRESH."""
    assert _derive_freshness([0, 0, 0], 5, 40) == "FRESH"


def test_step4_freshness_derivation_moderate():
    """Step 4: Moderate workload = MODERATE."""
    # stamina 40: moderate_threshold = 12 + 0.4 * 25 = 22, tired_threshold = 41
    # 30 pitches today, 0 days rest: effective_load = 30 - 0 = 30
    # 22 <= 30 < 41 -> MODERATE
    assert _derive_freshness([30, 0, 0], 0, 40) == "MODERATE"


def test_step4_freshness_derivation_tired():
    """Step 4: Heavy recent workload = TIRED."""
    # stamina 30: tired_threshold = 25 + 0.3 * 40 = 37
    # Effective load = 40 - 0 = 40 (pitched today)
    assert _derive_freshness([40, 0, 0], 0, 30) == "TIRED"


def test_step4_rest_reduces_load():
    """Step 4: Days of rest reduce effective workload."""
    # 30 pitches 3 days ago, 3 days rest: effective = 30 - 30 = 0
    assert _derive_freshness([0, 0, 30], 3, 40) == "FRESH"


def test_step4_stamina_affects_thresholds():
    """Step 4: Higher stamina means higher thresholds before getting tired."""
    # 35 pitches yesterday: stamina 30 -> tired, stamina 55 -> moderate/fresh
    low_stam = _derive_freshness([35, 0, 0], 1, 30)
    high_stam = _derive_freshness([35, 0, 0], 1, 55)
    # Low stamina pitcher should be more fatigued
    freshness_order = {"FRESH": 0, "MODERATE": 1, "TIRED": 2}
    assert freshness_order[low_stam] >= freshness_order[high_stam]


# -----------------------------------------------------------------------
# Step 5: Each pitcher includes days since last appearance
# -----------------------------------------------------------------------

def test_step5_each_pitcher_has_days_since_last():
    """Step 5: Each pitcher includes days_since_last_appearance."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "days_since_last_appearance" in p
        assert isinstance(p["days_since_last_appearance"], int)
        assert p["days_since_last_appearance"] >= 0


def test_step5_days_since_last_consistent_with_pitch_counts():
    """Step 5: Days since last appearance is consistent with pitch count history."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        counts = p["pitch_counts_last_3"]
        days = p["days_since_last_appearance"]
        # If there are recent pitches, days_since_last should reflect when
        has_recent = any(c > 0 for c in counts)
        if has_recent:
            # The first non-zero count determines days_since_last
            for i, c in enumerate(counts):
                if c > 0:
                    assert days == i + 1
                    break
        else:
            # No recent pitches -> should be 3+ days
            assert days >= 3


def test_step5_days_since_last_derivation():
    """Step 5: _derive_days_since_last returns correct values."""
    # With pitch counts [0, 15, 0] -> most recent is index 1 -> days = 2
    days = _derive_days_since_last("CLOSER", "h_bp1")
    pitch_counts = _derive_recent_pitch_counts("CLOSER", 30, "h_bp1")
    # Verify consistency
    has_recent = any(c > 0 for c in pitch_counts)
    if has_recent:
        for i, c in enumerate(pitch_counts):
            if c > 0:
                assert days == i + 1
                break


# -----------------------------------------------------------------------
# Step 6: Each pitcher includes pitch counts from last 3 appearances
# -----------------------------------------------------------------------

def test_step6_each_pitcher_has_pitch_counts():
    """Step 6: Each pitcher includes pitch_counts_last_3."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "pitch_counts_last_3" in p
        counts = p["pitch_counts_last_3"]
        assert isinstance(counts, list)
        assert len(counts) == 3
        for c in counts:
            assert isinstance(c, int)
            assert c >= 0


def test_step6_pitch_counts_are_realistic():
    """Step 6: Pitch counts are in realistic ranges for each role."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        counts = p["pitch_counts_last_3"]
        for c in counts:
            if p["role"] == "CLOSER":
                assert c <= 25, f"Closer {p['name']} had {c} pitches (max ~25)"
            elif p["role"] == "SETUP":
                assert c <= 30, f"Setup {p['name']} had {c} pitches (max ~30)"
            elif p["role"] in ("LONG", "MOPUP"):
                assert c <= 60, f"{p['role']} {p['name']} had {c} pitches (max ~60)"


def test_step6_pitch_counts_deterministic():
    """Step 6: Pitch counts are deterministic across calls."""
    result1 = parse_ok(get_bullpen_status("home"))
    result2 = parse_ok(get_bullpen_status("home"))
    for p1, p2 in zip(result1["bullpen"], result2["bullpen"]):
        assert p1["pitch_counts_last_3"] == p2["pitch_counts_last_3"]


# -----------------------------------------------------------------------
# Step 7: Each pitcher includes platoon splits (vs LHB and vs RHB)
# -----------------------------------------------------------------------

def test_step7_each_pitcher_has_platoon_splits():
    """Step 7: Each pitcher includes platoon_splits with vs_LHB and vs_RHB."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "platoon_splits" in p
        splits = p["platoon_splits"]
        assert "vs_LHB" in splits
        assert "vs_RHB" in splits


def test_step7_splits_contain_era():
    """Step 7: Platoon splits contain ERA."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "ERA" in p["platoon_splits"]["vs_LHB"]
        assert "ERA" in p["platoon_splits"]["vs_RHB"]
        # ERA should be a reasonable value (0-10)
        assert 0 < p["platoon_splits"]["vs_LHB"]["ERA"] < 10
        assert 0 < p["platoon_splits"]["vs_RHB"]["ERA"] < 10


def test_step7_splits_contain_whip():
    """Step 7: Platoon splits contain WHIP."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "WHIP" in p["platoon_splits"]["vs_LHB"]
        assert "WHIP" in p["platoon_splits"]["vs_RHB"]


def test_step7_splits_contain_woba_against():
    """Step 7: Platoon splits contain wOBA-against."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert "wOBA_against" in p["platoon_splits"]["vs_LHB"]
        assert "wOBA_against" in p["platoon_splits"]["vs_RHB"]


def test_step7_splits_contain_k_and_bb_rate():
    """Step 7: Platoon splits contain K% and BB%."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        for side in ("vs_LHB", "vs_RHB"):
            assert "K_pct" in p["platoon_splits"][side]
            assert "BB_pct" in p["platoon_splits"][side]
            assert 0 < p["platoon_splits"][side]["K_pct"] < 0.50
            assert 0 < p["platoon_splits"][side]["BB_pct"] < 0.20


def test_step7_splits_match_roster_era():
    """Step 7: Platoon split ERA values match the pitcher's roster attributes."""
    rosters = _load_rosters()
    home_bp = rosters["home"]["bullpen"]
    result = parse_ok(get_bullpen_status("home"))

    for bp_roster, bp_result in zip(home_bp, result["bullpen"][:len(home_bp)]):
        pitcher_attrs = bp_roster.get("pitcher", {})
        era_vs_l = pitcher_attrs.get("era_vs_l")
        era_vs_r = pitcher_attrs.get("era_vs_r")
        if era_vs_l is not None:
            assert abs(bp_result["platoon_splits"]["vs_LHB"]["ERA"] - era_vs_l) < 0.01
        if era_vs_r is not None:
            assert abs(bp_result["platoon_splits"]["vs_RHB"]["ERA"] - era_vs_r) < 0.01


def test_step7_derive_platoon_splits_basic():
    """Step 7: _derive_platoon_splits produces valid output."""
    attrs = {"stuff": 75, "control": 70, "era_vs_l": 3.20, "era_vs_r": 2.80}
    splits = _derive_platoon_splits(attrs)
    assert splits["vs_LHB"]["ERA"] == 3.20
    assert splits["vs_RHB"]["ERA"] == 2.80
    # Higher ERA vs LHB means worse performance, so wOBA against should be higher
    assert splits["vs_LHB"]["wOBA_against"] > splits["vs_RHB"]["wOBA_against"]


def test_step7_era_to_stat_correlations():
    """Step 7: Derived stats correlate correctly with ERA."""
    # Lower ERA should mean better peripheral stats
    low_era = 2.50
    high_era = 4.50
    assert _era_to_whip(low_era) < _era_to_whip(high_era)
    assert _era_to_woba(low_era) < _era_to_woba(high_era)

    attrs = {"stuff": 70, "control": 70}
    assert _era_to_k_rate(attrs, low_era) > _era_to_k_rate(attrs, high_era)


# -----------------------------------------------------------------------
# Step 8: Each pitcher includes current warm-up state
# -----------------------------------------------------------------------

def test_step8_each_pitcher_has_warmup_state():
    """Step 8: Each pitcher includes warmup_state."""
    result = parse_ok(get_bullpen_status("home"))
    valid_states = {"cold", "warming", "ready"}
    for p in result["bullpen"]:
        assert "warmup_state" in p
        assert p["warmup_state"] in valid_states


def test_step8_default_warmup_state_is_cold():
    """Step 8: Default warmup state is cold when no warming/ready IDs provided."""
    result = parse_ok(get_bullpen_status("home"))
    for p in result["bullpen"]:
        assert p["warmup_state"] == "cold"


def test_step8_warming_pitcher():
    """Step 8: Pitcher in warming_pitcher_ids has warmup_state 'warming'."""
    result = parse_ok(get_bullpen_status("home", warming_pitcher_ids="h_bp3"))
    for p in result["bullpen"]:
        if p["player_id"] == "h_bp3":
            assert p["warmup_state"] == "warming"
        else:
            assert p["warmup_state"] in ("cold", "ready")


def test_step8_ready_pitcher():
    """Step 8: Pitcher in ready_pitcher_ids has warmup_state 'ready'."""
    result = parse_ok(get_bullpen_status("home", ready_pitcher_ids="h_bp1"))
    for p in result["bullpen"]:
        if p["player_id"] == "h_bp1":
            assert p["warmup_state"] == "ready"
        else:
            assert p["warmup_state"] in ("cold", "warming")


def test_step8_multiple_warming_and_ready():
    """Step 8: Multiple pitchers can be warming or ready simultaneously."""
    result = parse_ok(get_bullpen_status(
        "home",
        warming_pitcher_ids="h_bp3,h_bp4",
        ready_pitcher_ids="h_bp1,h_bp2",
    ))
    states = {p["player_id"]: p["warmup_state"] for p in result["bullpen"]}
    assert states["h_bp1"] == "ready"
    assert states["h_bp2"] == "ready"
    assert states["h_bp3"] == "warming"
    assert states["h_bp4"] == "warming"
    assert states["h_bp5"] == "cold"


def test_step8_ready_overrides_warming():
    """Step 8: If a pitcher is in both warming and ready, ready takes precedence."""
    result = parse_ok(get_bullpen_status(
        "home",
        warming_pitcher_ids="h_bp1",
        ready_pitcher_ids="h_bp1",
    ))
    for p in result["bullpen"]:
        if p["player_id"] == "h_bp1":
            assert p["warmup_state"] == "ready"


# -----------------------------------------------------------------------
# Step 9: Pitchers already used and removed are excluded
# -----------------------------------------------------------------------

def test_step9_used_pitchers_excluded():
    """Step 9: Pitchers in used_pitcher_ids are excluded from results."""
    result = parse_ok(get_bullpen_status("home", used_pitcher_ids="h_bp1"))
    ids = {p["player_id"] for p in result["bullpen"]}
    assert "h_bp1" not in ids
    assert result["bullpen_count"] == 7


def test_step9_multiple_used_pitchers_excluded():
    """Step 9: Multiple used pitchers are all excluded."""
    result = parse_ok(get_bullpen_status("home", used_pitcher_ids="h_bp1,h_bp2,h_bp3"))
    ids = {p["player_id"] for p in result["bullpen"]}
    assert "h_bp1" not in ids
    assert "h_bp2" not in ids
    assert "h_bp3" not in ids
    assert result["bullpen_count"] == 5


def test_step9_available_count_updated():
    """Step 9: available_count reflects the reduced bullpen."""
    result_full = parse_ok(get_bullpen_status("home"))
    result_reduced = parse_ok(get_bullpen_status("home", used_pitcher_ids="h_bp1"))
    assert result_reduced["available_count"] <= result_full["available_count"]
    assert result_reduced["bullpen_count"] == result_full["bullpen_count"] - 1


def test_step9_all_used_returns_empty():
    """Step 9: If all pitchers are used, bullpen is empty."""
    all_ids = ",".join(f"h_bp{i}" for i in range(1, 9))
    result = parse_ok(get_bullpen_status("home", used_pitcher_ids=all_ids))
    assert result["status"] == "ok"
    assert result["bullpen_count"] == 0
    assert result["available_count"] == 0
    assert result["bullpen"] == []


def test_step9_unknown_used_id_ignored():
    """Step 9: Unknown IDs in used_pitcher_ids are silently ignored."""
    result = parse_ok(get_bullpen_status("home", used_pitcher_ids="unknown_001"))
    assert result["bullpen_count"] == 8  # All pitchers still present


def test_step9_whitespace_in_ids_handled():
    """Step 9: Whitespace around IDs in used_pitcher_ids is trimmed."""
    result = parse_ok(get_bullpen_status("home", used_pitcher_ids=" h_bp1 , h_bp2 "))
    ids = {p["player_id"] for p in result["bullpen"]}
    assert "h_bp1" not in ids
    assert "h_bp2" not in ids


# -----------------------------------------------------------------------
# Additional integration tests
# -----------------------------------------------------------------------

def test_integration_closer_stats():
    """Integration: Home closer (Greg Foster) has expected attributes."""
    result = parse_ok(get_bullpen_status("home"))
    closer = next(p for p in result["bullpen"] if p["role"] == "CLOSER")
    assert closer["player_id"] == "h_bp1"
    assert closer["name"] == "Greg Foster"
    assert closer["throws"] == "R"
    assert closer["velocity"] == 97.0
    assert closer["stuff"] == 82
    assert closer["control"] == 75
    assert closer["stamina"] == 30


def test_integration_lefty_setup():
    """Integration: Home lefty setup man (Luis Herrera) is identified."""
    result = parse_ok(get_bullpen_status("home"))
    lefties = [p for p in result["bullpen"] if p["throws"] == "L"]
    assert len(lefties) >= 1
    herrera = next((p for p in lefties if p["name"] == "Luis Herrera"), None)
    assert herrera is not None
    assert herrera["role"] == "SETUP"


def test_integration_away_closer():
    """Integration: Away closer (Zach Miller) has expected attributes."""
    result = parse_ok(get_bullpen_status("away"))
    closer = next(p for p in result["bullpen"] if p["role"] == "CLOSER")
    assert closer["player_id"] == "a_bp1"
    assert closer["name"] == "Zach Miller"
    assert closer["throws"] == "R"
    assert closer["velocity"] == 98.0


def test_integration_combined_parameters():
    """Integration: Using all parameters together works correctly."""
    result = parse_ok(get_bullpen_status(
        team="home",
        used_pitcher_ids="h_bp7,h_bp8",
        warming_pitcher_ids="h_bp3",
        ready_pitcher_ids="h_bp1",
    ))
    assert result["status"] == "ok"
    assert result["bullpen_count"] == 6
    ids = {p["player_id"] for p in result["bullpen"]}
    assert "h_bp7" not in ids
    assert "h_bp8" not in ids
    states = {p["player_id"]: p["warmup_state"] for p in result["bullpen"]}
    assert states["h_bp1"] == "ready"
    assert states["h_bp3"] == "warming"
    assert states["h_bp4"] == "cold"


def test_integration_response_structure():
    """Integration: Full response has expected top-level structure."""
    result = parse(get_bullpen_status("home"))
    assert result["status"] == "ok"
    assert result["tool"] == "get_bullpen_status"
    data = result["data"]
    assert "team" in data
    assert "team_name" in data
    assert "bullpen_count" in data
    assert "available_count" in data
    assert "bullpen" in data
    assert isinstance(data["bullpen"], list)


def test_helper_clamp():
    """Helper: _clamp works correctly."""
    assert _clamp(5, 0, 10) == 5
    assert _clamp(-1, 0, 10) == 0
    assert _clamp(15, 0, 10) == 10


def test_helper_era_to_whip_bounds():
    """Helper: _era_to_whip stays in realistic bounds."""
    assert 0.80 <= _era_to_whip(0.0) <= 2.00
    assert 0.80 <= _era_to_whip(5.0) <= 2.00
    assert 0.80 <= _era_to_whip(10.0) <= 2.00


def test_helper_era_to_woba_bounds():
    """Helper: _era_to_woba stays in realistic bounds."""
    assert 0.220 <= _era_to_woba(0.0) <= 0.420
    assert 0.220 <= _era_to_woba(5.0) <= 0.420


def test_helper_era_to_bb_rate_control_effect():
    """Helper: Higher control = lower BB%."""
    high_ctrl = {"stuff": 70, "control": 85}
    low_ctrl = {"stuff": 70, "control": 50}
    assert _era_to_bb_rate(high_ctrl, 3.50) < _era_to_bb_rate(low_ctrl, 3.50)


def test_helper_era_to_k_rate_stuff_effect():
    """Helper: Higher stuff = higher K%."""
    high_stuff = {"stuff": 85, "control": 70}
    low_stuff = {"stuff": 50, "control": 70}
    assert _era_to_k_rate(high_stuff, 3.50) > _era_to_k_rate(low_stuff, 3.50)


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    test_funcs = [
        obj for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    errors = []
    for func in test_funcs:
        try:
            func()
            passed += 1
            print(f"  PASS: {func.__name__}")
        except Exception as e:
            failed += 1
            errors.append((func.__name__, e))
            print(f"  FAIL: {func.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
