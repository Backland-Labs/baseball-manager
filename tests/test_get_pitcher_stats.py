# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_pitcher_stats tool.

Verifies all feature requirements from features.json:
1. Accepts player identifier and optional split parameters (vs_hand: L/R, home_away, recency_window)
2. Returns ERA, FIP, xFIP, SIERA
3. Returns K% and BB%
4. Returns ground ball rate and batted ball distribution
5. Returns pitch mix with per-pitch velocity, spin rate, and whiff rate
6. Returns times-through-order wOBA splits (1st, 2nd, 3rd+)
7. Returns current game performance (today's line: IP, H, R, ER, BB, K)
8. Returns an error if the player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_pitcher_stats import get_pitcher_stats, set_today_line, reset_today_lines, _PLAYERS, _load_players


def parse(result: str) -> dict:
    """Parse JSON response. For success responses, merge data into top level for easy access."""
    d = json.loads(result)
    if d.get("status") == "ok" and "data" in d:
        return {"status": "ok", "tool": d.get("tool"), **d["data"]}
    return d


def test_step1_accepts_player_id_and_splits():
    """Step 1: Accepts a player identifier and optional split parameters."""
    # Basic call with just player_id (starting pitcher)
    result = parse(get_pitcher_stats("h_sp1"))
    assert result["status"] == "ok"
    assert result["player_id"] == "h_sp1"
    assert result["player_name"] == "Brandon Cole"

    # Call with all optional splits
    result = parse(get_pitcher_stats("h_sp1", vs_hand="L", home_away="home", recency_window="last_7"))
    assert result["status"] == "ok"
    assert result["splits"]["vs_hand"] == "L"
    assert result["splits"]["home_away"] == "home"
    assert result["splits"]["recency_window"] == "last_7"


def test_step2_era_fip_xfip_siera():
    """Step 2: Returns ERA, FIP, xFIP, SIERA."""
    result = parse(get_pitcher_stats("h_sp1"))
    trad = result["traditional"]
    assert "ERA" in trad
    assert "FIP" in trad
    assert "xFIP" in trad
    assert "SIERA" in trad
    # ERA should be in reasonable range
    assert 1.50 <= trad["ERA"] <= 7.00
    assert 1.50 <= trad["FIP"] <= 6.50
    assert 1.80 <= trad["xFIP"] <= 6.00
    assert 1.80 <= trad["SIERA"] <= 6.00


def test_step3_k_pct_and_bb_pct():
    """Step 3: Returns K% and BB%."""
    result = parse(get_pitcher_stats("h_bp1"))  # Greg Foster - stuff=82
    rates = result["rates"]
    assert "K_pct" in rates
    assert "BB_pct" in rates
    # K% should be a proportion between 0 and 1
    assert 0.08 <= rates["K_pct"] <= 0.40
    # BB% should be a proportion between 0 and 1
    assert 0.03 <= rates["BB_pct"] <= 0.15
    # High-stuff pitcher should have above-average K%
    assert rates["K_pct"] > 0.22  # above MLB average


def test_step4_batted_ball_distribution():
    """Step 4: Returns ground ball rate and batted ball distribution."""
    result = parse(get_pitcher_stats("h_sp1"))
    bb = result["batted_ball"]
    assert "GB_pct" in bb
    assert "FB_pct" in bb
    assert "LD_pct" in bb
    # All should be proportions
    assert 0.25 <= bb["GB_pct"] <= 0.60
    assert 0.20 <= bb["FB_pct"] <= 0.45
    assert 0.15 <= bb["LD_pct"] <= 0.25
    # Should roughly sum to 1.0 (within rounding tolerance)
    total = bb["GB_pct"] + bb["FB_pct"] + bb["LD_pct"]
    assert 0.95 <= total <= 1.05


def test_step5_pitch_mix():
    """Step 5: Returns pitch mix with per-pitch velocity, spin rate, and whiff rate."""
    result = parse(get_pitcher_stats("h_sp1"))  # Brandon Cole - velocity=94.5, stuff=75
    pm = result["pitch_mix"]
    assert isinstance(pm, list)
    assert len(pm) >= 3  # at minimum FF, SL, CH

    for pitch in pm:
        assert "pitch_type" in pitch
        assert "usage" in pitch
        assert "velocity" in pitch
        assert "spin_rate" in pitch
        assert "whiff_rate" in pitch
        # Usage should be a proportion
        assert 0.0 < pitch["usage"] <= 1.0
        # Velocity should be realistic
        assert 70.0 <= pitch["velocity"] <= 105.0
        # Spin rate should be realistic
        assert 1000 <= pitch["spin_rate"] <= 3500
        # Whiff rate should be a proportion
        assert 0.05 <= pitch["whiff_rate"] <= 0.50

    # Total usage should sum close to 1.0
    total_usage = sum(p["usage"] for p in pm)
    assert 0.90 <= total_usage <= 1.10

    # First pitch should be fastball
    assert pm[0]["pitch_type"] == "FF"
    # Fastball velocity should match pitcher's velocity attribute
    assert abs(pm[0]["velocity"] - 94.5) < 0.5


def test_step5_pitch_mix_high_stuff_has_curveball():
    """High-stuff pitchers should get a curveball in their mix."""
    result = parse(get_pitcher_stats("h_bp1"))  # Greg Foster - stuff=82
    pm = result["pitch_mix"]
    pitch_types = [p["pitch_type"] for p in pm]
    assert "CU" in pitch_types  # stuff >= 60 should produce curveball


def test_step5_pitch_mix_low_stuff_no_curveball():
    """Low-stuff pitchers should not get a curveball."""
    result = parse(get_pitcher_stats("h_bp8"))  # Sam Rodriguez - stuff=55
    pm = result["pitch_mix"]
    pitch_types = [p["pitch_type"] for p in pm]
    assert "CU" not in pitch_types  # stuff < 60 should not produce curveball


def test_step6_times_through_order():
    """Step 6: Returns times-through-order wOBA splits (1st, 2nd, 3rd+)."""
    result = parse(get_pitcher_stats("h_sp1"))
    tto = result["times_through_order"]
    assert "1st" in tto
    assert "2nd" in tto
    assert "3rd_plus" in tto
    # wOBA should increase each time through the order
    assert tto["2nd"] >= tto["1st"]
    assert tto["3rd_plus"] >= tto["2nd"]
    # All values should be realistic wOBA
    assert 0.250 <= tto["1st"] <= 0.400
    assert 0.250 <= tto["2nd"] <= 0.420
    assert 0.260 <= tto["3rd_plus"] <= 0.450


def test_step6_stamina_affects_tto():
    """Higher stamina pitchers should degrade less through the order."""
    # Brandon Cole: stamina=70 (higher)
    result_high = parse(get_pitcher_stats("h_sp1"))
    tto_high = result_high["times_through_order"]
    degradation_high = tto_high["3rd_plus"] - tto_high["1st"]

    # Greg Foster: stamina=30 (lower, closer)
    result_low = parse(get_pitcher_stats("h_bp1"))
    tto_low = result_low["times_through_order"]
    degradation_low = tto_low["3rd_plus"] - tto_low["1st"]

    # Lower stamina should show more TTO degradation
    assert degradation_low >= degradation_high


def test_step7_today_line():
    """Step 7: Returns current game performance (today's line)."""
    reset_today_lines()

    # Default today line (no game data yet)
    result = parse(get_pitcher_stats("h_sp1"))
    assert "today" in result
    assert result["today"]["IP"] == 0.0
    assert result["today"]["H"] == 0
    assert result["today"]["R"] == 0
    assert result["today"]["ER"] == 0
    assert result["today"]["BB"] == 0
    assert result["today"]["K"] == 0

    # Set a today line and verify it's returned
    set_today_line("h_sp1", {"IP": 6.1, "H": 5, "R": 2, "ER": 2, "BB": 1, "K": 7})
    result = parse(get_pitcher_stats("h_sp1"))
    assert result["today"]["IP"] == 6.1
    assert result["today"]["H"] == 5
    assert result["today"]["R"] == 2
    assert result["today"]["ER"] == 2
    assert result["today"]["BB"] == 1
    assert result["today"]["K"] == 7

    reset_today_lines()


def test_step8_invalid_player_id():
    """Step 8: Returns an error if the player identifier is invalid."""
    result = parse(get_pitcher_stats("nonexistent_player"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "not found" in result["message"]


def test_splits_vs_hand():
    """Verify vs_hand splits produce different ERA for L vs R."""
    # Brandon Cole: era_vs_l=3.60, era_vs_r=3.20
    result_vs_l = parse(get_pitcher_stats("h_sp1", vs_hand="L"))
    result_vs_r = parse(get_pitcher_stats("h_sp1", vs_hand="R"))
    # ERA vs L should be higher than ERA vs R for this pitcher
    assert result_vs_l["traditional"]["ERA"] > result_vs_r["traditional"]["ERA"]


def test_splits_home_away():
    """Verify home/away splits produce different ERA."""
    result_home = parse(get_pitcher_stats("h_sp1", home_away="home"))
    result_away = parse(get_pitcher_stats("h_sp1", home_away="away"))
    # Home ERA should be lower (pitchers do better at home)
    assert result_home["traditional"]["ERA"] <= result_away["traditional"]["ERA"]


def test_splits_recency_window():
    """Verify recency window affects stats."""
    result_season = parse(get_pitcher_stats("h_sp1", recency_window="season"))
    result_last7 = parse(get_pitcher_stats("h_sp1", recency_window="last_7"))
    # last_7 should show slightly better ERA (recency bias)
    assert result_last7["traditional"]["ERA"] <= result_season["traditional"]["ERA"]


def test_invalid_parameters():
    """Verify invalid parameter values return errors."""
    # Invalid vs_hand
    result = parse(get_pitcher_stats("h_sp1", vs_hand="X"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"

    # Invalid home_away
    result = parse(get_pitcher_stats("h_sp1", home_away="neutral"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"

    # Invalid recency_window
    result = parse(get_pitcher_stats("h_sp1", recency_window="last_3"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_position_player_as_pitcher():
    """Position players without pitcher attributes should return error."""
    result = parse(get_pitcher_stats("h_001"))  # Marcus Chen (CF) - no pitcher attributes
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_PITCHER"


def test_different_pitchers_different_stats():
    """Different pitchers should produce different statistics based on their attributes."""
    # Ace closer vs mopup reliever
    foster = parse(get_pitcher_stats("h_bp1"))   # stuff=82, velocity=97.0
    rodriguez = parse(get_pitcher_stats("h_bp8"))  # stuff=55, velocity=88.0

    # Foster should have higher K%
    assert foster["rates"]["K_pct"] > rodriguez["rates"]["K_pct"]
    # Foster should have lower ERA
    assert foster["traditional"]["ERA"] < rodriguez["traditional"]["ERA"]
    # Foster's fastball velocity should be higher
    foster_ff = next(p for p in foster["pitch_mix"] if p["pitch_type"] == "FF")
    rodriguez_ff = next(p for p in rodriguez["pitch_mix"] if p["pitch_type"] == "FF")
    assert foster_ff["velocity"] > rodriguez_ff["velocity"]


def test_away_team_pitchers():
    """Verify away team pitchers are accessible."""
    result = parse(get_pitcher_stats("a_sp1"))  # Matt Henderson
    assert result["status"] == "ok"
    assert result["player_name"] == "Matt Henderson"
    assert "traditional" in result


def test_bullpen_pitchers():
    """Verify bullpen pitchers from both teams are accessible."""
    # Home bullpen
    result = parse(get_pitcher_stats("h_bp3"))  # Marcus Webb
    assert result["status"] == "ok"
    assert result["player_name"] == "Marcus Webb"

    # Away bullpen
    result = parse(get_pitcher_stats("a_bp2"))  # Omar Hassan
    assert result["status"] == "ok"
    assert result["player_name"] == "Omar Hassan"


def test_throws_hand_returned():
    """Verify the pitcher's throwing hand is returned."""
    # Right-hander
    result = parse(get_pitcher_stats("h_sp1"))
    assert result["throws"] == "R"

    # Left-hander
    result = parse(get_pitcher_stats("h_bp2"))  # Luis Herrera - LHP
    assert result["throws"] == "L"


def test_response_structure_completeness():
    """Verify the full response structure has all required fields."""
    result = parse(get_pitcher_stats("h_sp1"))
    assert result["status"] == "ok"
    assert "player_id" in result
    assert "player_name" in result
    assert "throws" in result
    assert "splits" in result
    assert "traditional" in result
    assert "rates" in result
    assert "batted_ball" in result
    assert "pitch_mix" in result
    assert "times_through_order" in result
    assert "today" in result


def test_high_control_low_bb():
    """Pitchers with high control should have low BB%."""
    # Greg Foster: control=75
    result = parse(get_pitcher_stats("h_bp1"))
    assert result["rates"]["BB_pct"] < 0.08  # below MLB average

    # Sam Rodriguez: control=62
    result_low = parse(get_pitcher_stats("h_bp8"))
    assert result_low["rates"]["BB_pct"] > result["rates"]["BB_pct"]


def test_secondary_pitch_velocities_lower_than_fastball():
    """All secondary pitches should have lower velocity than the fastball."""
    result = parse(get_pitcher_stats("h_sp1"))
    pm = result["pitch_mix"]
    ff_velo = next(p["velocity"] for p in pm if p["pitch_type"] == "FF")
    for pitch in pm:
        if pitch["pitch_type"] != "FF":
            assert pitch["velocity"] < ff_velo, (
                f"{pitch['pitch_type']} velocity {pitch['velocity']} >= FF velocity {ff_velo}"
            )


if __name__ == "__main__":
    tests = [
        test_step1_accepts_player_id_and_splits,
        test_step2_era_fip_xfip_siera,
        test_step3_k_pct_and_bb_pct,
        test_step4_batted_ball_distribution,
        test_step5_pitch_mix,
        test_step5_pitch_mix_high_stuff_has_curveball,
        test_step5_pitch_mix_low_stuff_no_curveball,
        test_step6_times_through_order,
        test_step6_stamina_affects_tto,
        test_step7_today_line,
        test_step8_invalid_player_id,
        test_splits_vs_hand,
        test_splits_home_away,
        test_splits_recency_window,
        test_invalid_parameters,
        test_position_player_as_pitcher,
        test_different_pitchers_different_stats,
        test_away_team_pitchers,
        test_bullpen_pitchers,
        test_throws_hand_returned,
        test_response_structure_completeness,
        test_high_control_low_bb,
        test_secondary_pitch_velocities_lower_than_fastball,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
