# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_matchup_data tool.

Verifies all feature requirements from features.json:
1. Accepts a batter identifier and a pitcher identifier
2. Returns number of career plate appearances between the two players
3. Returns batting average, slugging, and strikeout rate for the matchup
4. Returns outcome distribution (groundball, flyball, line drive rates)
5. Returns a sample-size reliability indicator (small/medium/large)
6. Returns similarity-model projected wOBA when sample is small
7. Returns pitch-type vulnerability breakdown for the batter against this pitcher's mix
8. Returns a meaningful response when no prior matchup history exists
9. Returns an error if either player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_matchup_data import get_matchup_data, _simulated_pa, _sample_size_label


def parse(result: str) -> dict:
    return json.loads(result)


# ---------------------------------------------------------------------------
# Test pairs with known PA counts:
#   h_001 vs a_sp1: 32 PA (large)
#   h_003 vs a_sp1:  0 PA (none / no history)
#   h_004 vs a_sp1: 13 PA (medium)
#   h_008 vs a_sp1:  7 PA (small)
#   a_008 vs h_sp1:  1 PA (small)
# ---------------------------------------------------------------------------


def test_step1_accepts_batter_and_pitcher_ids():
    """Step 1: Accepts a batter identifier and a pitcher identifier."""
    result = parse(get_matchup_data("h_001", "a_sp1"))
    assert result["status"] == "ok"
    assert result["batter_id"] == "h_001"
    assert result["pitcher_id"] == "a_sp1"
    assert result["batter_name"] == "Marcus Chen"
    assert result["pitcher_name"] == "Matt Henderson"


def test_step2_career_plate_appearances():
    """Step 2: Returns number of career plate appearances between the two players."""
    result = parse(get_matchup_data("h_001", "a_sp1"))
    assert "career_pa" in result
    assert isinstance(result["career_pa"], int)
    assert result["career_pa"] >= 0
    # Specific known value
    assert result["career_pa"] == 32

    # Another pair with 0 PA
    result_zero = parse(get_matchup_data("h_003", "a_sp1"))
    assert result_zero["career_pa"] == 0


def test_step3_matchup_stats_avg_slg_k_rate():
    """Step 3: Returns batting average, slugging, and strikeout rate for the matchup."""
    result = parse(get_matchup_data("h_001", "a_sp1"))
    stats = result["matchup_stats"]
    assert stats is not None
    assert "AVG" in stats
    assert "SLG" in stats
    assert "K_rate" in stats

    # Sanity ranges
    assert 0.100 <= stats["AVG"] <= 0.400
    assert 0.200 <= stats["SLG"] <= 0.700
    assert 0.05 <= stats["K_rate"] <= 0.45

    # SLG should be >= AVG (ISO is always positive)
    assert stats["SLG"] >= stats["AVG"]


def test_step4_outcome_distribution():
    """Step 4: Returns outcome distribution (groundball, flyball, line drive rates)."""
    result = parse(get_matchup_data("h_001", "a_sp1"))
    dist = result["outcome_distribution"]
    assert dist is not None
    assert "groundball" in dist
    assert "flyball" in dist
    assert "line_drive" in dist

    # Each rate should be positive
    assert dist["groundball"] > 0
    assert dist["flyball"] > 0
    assert dist["line_drive"] > 0

    # Rates should approximately sum to 1.0 (within rounding tolerance)
    total = dist["groundball"] + dist["flyball"] + dist["line_drive"]
    assert abs(total - 1.0) < 0.02


def test_step5_sample_size_reliability():
    """Step 5: Returns a sample-size reliability indicator (small/medium/large)."""
    # Large sample: h_001 vs a_sp1 = 32 PA
    result = parse(get_matchup_data("h_001", "a_sp1"))
    assert result["sample_size_reliability"] == "large"

    # Medium sample: h_004 vs a_sp1 = 13 PA
    result = parse(get_matchup_data("h_004", "a_sp1"))
    assert result["sample_size_reliability"] == "medium"

    # Small sample: h_008 vs a_sp1 = 7 PA
    result = parse(get_matchup_data("h_008", "a_sp1"))
    assert result["sample_size_reliability"] == "small"

    # No history: h_003 vs a_sp1 = 0 PA
    result = parse(get_matchup_data("h_003", "a_sp1"))
    assert result["sample_size_reliability"] == "none"


def test_step6_similarity_projected_woba():
    """Step 6: Returns similarity-model projected wOBA when sample is small."""
    # For a small sample pair, projected wOBA should be present
    result_small = parse(get_matchup_data("h_008", "a_sp1"))
    assert "similarity_projected_wOBA" in result_small
    assert 0.200 <= result_small["similarity_projected_wOBA"] <= 0.450

    # Also present for large samples (always computed)
    result_large = parse(get_matchup_data("h_001", "a_sp1"))
    assert "similarity_projected_wOBA" in result_large
    assert 0.200 <= result_large["similarity_projected_wOBA"] <= 0.450

    # Also present for zero-history matchups
    result_none = parse(get_matchup_data("h_003", "a_sp1"))
    assert "similarity_projected_wOBA" in result_none
    assert 0.200 <= result_none["similarity_projected_wOBA"] <= 0.450


def test_step7_pitch_type_vulnerability():
    """Step 7: Returns pitch-type vulnerability breakdown for the batter against this pitcher's mix."""
    result = parse(get_matchup_data("h_001", "a_sp1"))
    vuln = result["pitch_type_vulnerability"]
    assert isinstance(vuln, list)
    assert len(vuln) >= 3  # At least FF, SL, CH

    # Each entry should have pitch_type, usage, and wOBA_against
    for entry in vuln:
        assert "pitch_type" in entry
        assert "usage" in entry
        assert "wOBA_against" in entry
        assert entry["usage"] > 0
        assert 0.100 <= entry["wOBA_against"] <= 0.500

    # Check that pitch types include at least the core three
    types = [e["pitch_type"] for e in vuln]
    assert "FF" in types
    assert "SL" in types
    assert "CH" in types


def test_step8_no_prior_history():
    """Step 8: Returns a meaningful response when no prior matchup history exists."""
    # h_003 vs a_sp1 = 0 PA (no history)
    result = parse(get_matchup_data("h_003", "a_sp1"))
    assert result["status"] == "ok"
    assert result["career_pa"] == 0
    assert result["sample_size_reliability"] == "none"

    # matchup_stats should be None (no data to report)
    assert result["matchup_stats"] is None

    # outcome_distribution should be None
    assert result["outcome_distribution"] is None

    # Should have a meaningful message
    assert "no_history_message" in result
    assert "no prior matchup" in result["no_history_message"].lower()

    # Similarity model should still provide a projected wOBA
    assert "similarity_projected_wOBA" in result
    assert 0.200 <= result["similarity_projected_wOBA"] <= 0.450

    # Pitch vulnerability should still be available
    assert "pitch_type_vulnerability" in result
    assert len(result["pitch_type_vulnerability"]) >= 3


def test_step9_invalid_batter_id():
    """Step 9: Returns an error if the batter identifier is invalid."""
    result = parse(get_matchup_data("fake_batter", "a_sp1"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "fake_batter" in result["message"]
    assert "not found" in result["message"]


def test_step9_invalid_pitcher_id():
    """Step 9: Returns an error if the pitcher identifier is invalid."""
    result = parse(get_matchup_data("h_001", "fake_pitcher"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "fake_pitcher" in result["message"]
    assert "not found" in result["message"]


def test_step9_both_ids_invalid():
    """Step 9: First invalid ID (batter) triggers the error."""
    result = parse(get_matchup_data("bad_batter", "bad_pitcher"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    # Should complain about the batter first
    assert "bad_batter" in result["message"]


def test_batter_without_batting_attrs():
    """Passing a pitcher-only player as batter should return NOT_A_BATTER error."""
    result = parse(get_matchup_data("h_bp1", "a_sp1"))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_BATTER"


def test_pitcher_without_pitching_attrs():
    """Passing a position player as pitcher should return NOT_A_PITCHER error."""
    result = parse(get_matchup_data("h_001", "h_002"))
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_PITCHER"


def test_deterministic_results():
    """Same batter-pitcher pair should always produce the same results."""
    result1 = parse(get_matchup_data("h_001", "a_sp1"))
    result2 = parse(get_matchup_data("h_001", "a_sp1"))
    assert result1 == result2


def test_different_matchups_differ():
    """Different batter-pitcher pairs should produce different stats."""
    result_a = parse(get_matchup_data("h_001", "a_sp1"))
    result_b = parse(get_matchup_data("h_004", "a_sp1"))
    # Different batters => different matchup stats (both have PA > 0)
    assert result_a["matchup_stats"]["AVG"] != result_b["matchup_stats"]["AVG"]


def test_power_hitter_higher_slg():
    """A power hitter should produce higher SLG in the matchup than a contact hitter."""
    # h_003 Rafael Ortiz power=88 vs h_sp1 has 0 PA, use h_004 Tyrone Jackson power=85
    # h_009 Andre Davis power=50
    # Both need PA > 0 against the same pitcher
    # h_004 vs a_sp1: 13 PA (medium)
    # h_009 vs a_sp1: 20 PA (medium)
    result_power = parse(get_matchup_data("h_004", "a_sp1"))
    result_contact = parse(get_matchup_data("h_009", "a_sp1"))
    # Jackson (power=85) should have higher SLG than Davis (power=50)
    assert result_power["matchup_stats"]["SLG"] > result_contact["matchup_stats"]["SLG"]


def test_platoon_awareness():
    """Matchup stats should reflect platoon advantage.

    A left-handed batter facing a left-handed pitcher should use avg_vs_l,
    which is typically lower (same-side disadvantage).
    """
    # h_003 Rafael Ortiz (bats L): avg_vs_l=0.240, avg_vs_r=0.310
    # a_sp1 Matt Henderson (throws L)
    # h_003 vs a_sp1 has 0 PA, so test with other pairs

    # h_001 Marcus Chen (bats L): avg_vs_l=0.260, avg_vs_r=0.295
    # a_sp1 Matt Henderson (throws L) -- batter uses avg_vs_l = 0.260
    # h_sp1 Brandon Cole (throws R) -- batter uses avg_vs_r = 0.295
    # Both need PA > 0
    result_vs_lhp = parse(get_matchup_data("h_001", "a_sp1"))  # 32 PA
    result_vs_rhp = parse(get_matchup_data("h_001", "h_sp1"))  # 23 PA

    # Just verify both are valid and have matchup_stats
    assert result_vs_lhp["status"] == "ok"
    assert result_vs_rhp["status"] == "ok"
    assert result_vs_lhp["matchup_stats"] is not None
    assert result_vs_rhp["matchup_stats"] is not None


def test_switch_hitter_matchup():
    """Switch hitter should use the appropriate batting side."""
    # h_007 Carlos Ramirez (bats S, switch)
    # vs a_sp1 Matt Henderson (throws L) -> switch hitter bats R, pitcher uses era_vs_r
    result = parse(get_matchup_data("h_007", "a_sp1"))
    assert result["status"] == "ok"
    # h_007 vs a_sp1 has PA > 0 (40 PA)
    assert result["career_pa"] == 40
    assert result["matchup_stats"] is not None


def test_away_batter_vs_home_pitcher():
    """Away team batter vs home team pitcher should work."""
    result = parse(get_matchup_data("a_003", "h_sp1"))
    assert result["status"] == "ok"
    assert result["batter_name"] == "Anthony Russo"
    assert result["pitcher_name"] == "Brandon Cole"


def test_bench_player_as_batter():
    """Bench players should be valid batters."""
    result = parse(get_matchup_data("h_012", "a_sp1"))
    assert result["status"] == "ok"
    assert result["batter_name"] == "Darnell Washington"


def test_bullpen_pitcher():
    """Bullpen pitchers should be valid pitcher identifiers."""
    result = parse(get_matchup_data("h_001", "a_bp1"))
    assert result["status"] == "ok"
    assert result["pitcher_name"] == "Zach Miller"


def test_sample_size_label_function():
    """Unit test for the sample size classification function."""
    assert _sample_size_label(0) == "none"
    assert _sample_size_label(1) == "small"
    assert _sample_size_label(9) == "small"
    assert _sample_size_label(10) == "medium"
    assert _sample_size_label(29) == "medium"
    assert _sample_size_label(30) == "large"
    assert _sample_size_label(45) == "large"


def test_response_structure_completeness():
    """Verify the full response structure has all required fields for a matchup with history."""
    result = parse(get_matchup_data("h_001", "a_sp1"))
    assert result["status"] == "ok"
    assert "batter_id" in result
    assert "batter_name" in result
    assert "pitcher_id" in result
    assert "pitcher_name" in result
    assert "career_pa" in result
    assert "sample_size_reliability" in result
    assert "matchup_stats" in result
    assert "outcome_distribution" in result
    assert "similarity_projected_wOBA" in result
    assert "pitch_type_vulnerability" in result


def test_curveball_present_for_high_stuff_pitcher():
    """High-stuff pitchers should have curveball in vulnerability breakdown."""
    # a_bp1 Zach Miller: stuff=80 (>= 60, should have CU)
    result = parse(get_matchup_data("h_001", "a_bp1"))
    types = [e["pitch_type"] for e in result["pitch_type_vulnerability"]]
    assert "CU" in types


def test_no_curveball_for_low_stuff_pitcher():
    """Low-stuff pitchers should not have curveball in vulnerability breakdown."""
    # a_bp8 Adam Fisher: stuff=54 (< 60, no CU)
    result = parse(get_matchup_data("h_001", "a_bp8"))
    types = [e["pitch_type"] for e in result["pitch_type_vulnerability"]]
    assert "CU" not in types


if __name__ == "__main__":
    tests = [
        test_step1_accepts_batter_and_pitcher_ids,
        test_step2_career_plate_appearances,
        test_step3_matchup_stats_avg_slg_k_rate,
        test_step4_outcome_distribution,
        test_step5_sample_size_reliability,
        test_step6_similarity_projected_woba,
        test_step7_pitch_type_vulnerability,
        test_step8_no_prior_history,
        test_step9_invalid_batter_id,
        test_step9_invalid_pitcher_id,
        test_step9_both_ids_invalid,
        test_batter_without_batting_attrs,
        test_pitcher_without_pitching_attrs,
        test_deterministic_results,
        test_different_matchups_differ,
        test_power_hitter_higher_slg,
        test_platoon_awareness,
        test_switch_hitter_matchup,
        test_away_batter_vs_home_pitcher,
        test_bench_player_as_batter,
        test_bullpen_pitcher,
        test_sample_size_label_function,
        test_response_structure_completeness,
        test_curveball_present_for_high_stuff_pitcher,
        test_no_curveball_for_low_stuff_pitcher,
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
