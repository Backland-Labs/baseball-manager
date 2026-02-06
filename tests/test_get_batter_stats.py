# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the get_batter_stats tool.

Verifies all feature requirements from features.json:
1. Accepts player identifier and optional split parameters
2. Returns traditional stats: AVG, OBP, SLG, OPS
3. Returns advanced metrics: wOBA, wRC+, barrel rate, xwOBA
4. Returns plate discipline: K%, BB%, chase rate, whiff rate
5. Returns batted ball profile: GB%, pull%, exit velocity, launch angle
6. Returns sprint speed
7. Returns situational stats: RISP, high leverage, late and close
8. Returns current game performance (today's line)
9. Returns an error if the player identifier is invalid
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.get_batter_stats import get_batter_stats, set_today_line, reset_today_lines, _PLAYERS, _load_players


def parse(result: str) -> dict:
    return json.loads(result)


def test_step1_accepts_player_id_and_splits():
    """Step 1: Accepts a player identifier and optional split parameters."""
    # Basic call with just player_id
    result = parse(get_batter_stats("h_001"))
    assert result["status"] == "ok"
    assert result["player_id"] == "h_001"
    assert result["player_name"] == "Marcus Chen"

    # Call with all optional splits
    result = parse(get_batter_stats("h_001", vs_hand="L", home_away="home", recency_window="last_7"))
    assert result["status"] == "ok"
    assert result["splits"]["vs_hand"] == "L"
    assert result["splits"]["home_away"] == "home"
    assert result["splits"]["recency_window"] == "last_7"


def test_step2_traditional_stats():
    """Step 2: Returns traditional stats: AVG, OBP, SLG, OPS."""
    result = parse(get_batter_stats("h_001"))
    trad = result["traditional"]
    assert "AVG" in trad
    assert "OBP" in trad
    assert "SLG" in trad
    assert "OPS" in trad
    # Sanity: AVG should be between .100 and .400
    assert 0.100 <= trad["AVG"] <= 0.400
    # OBP >= AVG always
    assert trad["OBP"] >= trad["AVG"]
    # OPS = OBP + SLG
    assert abs(trad["OPS"] - (trad["OBP"] + trad["SLG"])) < 0.002


def test_step3_advanced_metrics():
    """Step 3: Returns advanced metrics: wOBA, wRC+, barrel rate, xwOBA."""
    result = parse(get_batter_stats("h_003"))  # Rafael Ortiz - power hitter
    adv = result["advanced"]
    assert "wOBA" in adv
    assert "wRC_plus" in adv
    assert "barrel_rate" in adv
    assert "xwOBA" in adv
    # wOBA should be between .200 and .450
    assert 0.200 <= adv["wOBA"] <= 0.450
    # wRC+ should be an integer
    assert isinstance(adv["wRC_plus"], int)
    # Barrel rate should be a proportion
    assert 0.01 <= adv["barrel_rate"] <= 0.18


def test_step4_plate_discipline():
    """Step 4: Returns plate discipline: K%, BB%, chase rate, whiff rate."""
    result = parse(get_batter_stats("h_006"))  # Shin-Soo Park - good eye
    disc = result["plate_discipline"]
    assert "K_pct" in disc
    assert "BB_pct" in disc
    assert "chase_rate" in disc
    assert "whiff_rate" in disc
    # Park has eye=82, should have above-average BB%
    assert disc["BB_pct"] > 0.08  # above MLB avg


def test_step5_batted_ball_profile():
    """Step 5: Returns batted ball profile: GB%, pull%, exit velocity, launch angle."""
    result = parse(get_batter_stats("h_003"))  # Rafael Ortiz - power=88
    bb = result["batted_ball"]
    assert "GB_pct" in bb
    assert "pull_pct" in bb
    assert "exit_velocity" in bb
    assert "launch_angle" in bb
    # Power hitter should have higher exit velocity and launch angle
    assert bb["exit_velocity"] > 90.0
    assert bb["launch_angle"] > 10.0
    # Power hitter should pull more
    assert bb["pull_pct"] > 0.40


def test_step6_sprint_speed():
    """Step 6: Returns sprint speed."""
    # Fast player
    result_fast = parse(get_batter_stats("h_001"))  # Marcus Chen - speed=85
    assert "sprint_speed" in result_fast
    assert result_fast["sprint_speed"] > 28.0  # above average

    # Slow player
    result_slow = parse(get_batter_stats("h_008"))  # Tommy Sullivan - speed=35
    assert result_slow["sprint_speed"] < 26.5  # below average

    # Speed hierarchy should hold
    assert result_fast["sprint_speed"] > result_slow["sprint_speed"]


def test_step7_situational_stats():
    """Step 7: Returns situational stats: RISP, high leverage, late and close."""
    result = parse(get_batter_stats("h_005"))
    sit = result["situational"]
    assert "RISP_avg" in sit
    assert "high_leverage_ops" in sit
    assert "late_and_close_ops" in sit
    # All should be positive numbers
    assert sit["RISP_avg"] > 0
    assert sit["high_leverage_ops"] > 0
    assert sit["late_and_close_ops"] > 0


def test_step8_today_line():
    """Step 8: Returns current game performance (today's line)."""
    reset_today_lines()

    # Default today line (no game data yet)
    result = parse(get_batter_stats("h_001"))
    assert "today" in result
    assert result["today"]["AB"] == 0
    assert result["today"]["H"] == 0

    # Set a today line and verify it's returned
    set_today_line("h_001", {"AB": 3, "H": 2, "BB": 1, "K": 0, "RBI": 1})
    result = parse(get_batter_stats("h_001"))
    assert result["today"]["AB"] == 3
    assert result["today"]["H"] == 2
    assert result["today"]["RBI"] == 1

    reset_today_lines()


def test_step9_invalid_player_id():
    """Step 9: Returns an error if the player identifier is invalid."""
    result = parse(get_batter_stats("nonexistent_player"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PLAYER_ID"
    assert "not found" in result["message"]


def test_splits_vs_hand():
    """Verify vs_hand splits produce different stats for L vs R."""
    # Rafael Ortiz: avg_vs_l=0.240, avg_vs_r=0.310 -- big platoon split
    result_vs_l = parse(get_batter_stats("h_003", vs_hand="L"))
    result_vs_r = parse(get_batter_stats("h_003", vs_hand="R"))
    # AVG vs R should be higher than vs L for this player
    assert result_vs_r["traditional"]["AVG"] > result_vs_l["traditional"]["AVG"]


def test_splits_home_away():
    """Verify home/away splits produce different stats."""
    result_home = parse(get_batter_stats("h_001", home_away="home"))
    result_away = parse(get_batter_stats("h_001", home_away="away"))
    # Home should have slightly higher AVG
    assert result_home["traditional"]["AVG"] >= result_away["traditional"]["AVG"]


def test_splits_recency_window():
    """Verify recency window affects stats."""
    result_season = parse(get_batter_stats("h_001", recency_window="season"))
    result_last7 = parse(get_batter_stats("h_001", recency_window="last_7"))
    # last_7 should show a slight increase (hot streak bias)
    assert result_last7["traditional"]["AVG"] >= result_season["traditional"]["AVG"]


def test_invalid_parameters():
    """Verify invalid parameter values return errors."""
    # Invalid vs_hand
    result = parse(get_batter_stats("h_001", vs_hand="X"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"

    # Invalid home_away
    result = parse(get_batter_stats("h_001", home_away="neutral"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"

    # Invalid recency_window
    result = parse(get_batter_stats("h_001", recency_window="last_3"))
    assert result["status"] == "error"
    assert result["error_code"] == "INVALID_PARAMETER"


def test_pitcher_as_batter():
    """Pitchers without batter attributes should still work if they have batter data."""
    # Bullpen pitchers in our roster don't have batter attributes
    result = parse(get_batter_stats("h_bp1"))
    # Greg Foster (RP) has no "batter" key in the roster
    assert result["status"] == "error"
    assert result["error_code"] == "NOT_A_BATTER"


def test_different_players_different_stats():
    """Different players should produce different statistics based on their attributes."""
    # Power hitter vs contact hitter
    ortiz = parse(get_batter_stats("h_003"))   # power=88
    chen = parse(get_batter_stats("h_001"))    # contact=78, power=55

    # Ortiz should have higher SLG and barrel rate
    assert ortiz["traditional"]["SLG"] > chen["traditional"]["SLG"]
    assert ortiz["advanced"]["barrel_rate"] > chen["advanced"]["barrel_rate"]
    assert ortiz["batted_ball"]["exit_velocity"] > chen["batted_ball"]["exit_velocity"]

    # Chen should have higher sprint speed (speed=85 vs 40)
    assert chen["sprint_speed"] > ortiz["sprint_speed"]


def test_away_team_players():
    """Verify away team players are also accessible."""
    result = parse(get_batter_stats("a_001"))  # Jordan Bell
    assert result["status"] == "ok"
    assert result["player_name"] == "Jordan Bell"
    assert "traditional" in result


def test_bench_players():
    """Verify bench players are accessible."""
    result = parse(get_batter_stats("h_012"))  # Darnell Washington (bench)
    assert result["status"] == "ok"
    assert result["player_name"] == "Darnell Washington"


def test_response_structure_completeness():
    """Verify the full response structure has all required fields."""
    result = parse(get_batter_stats("h_001"))
    assert result["status"] == "ok"
    assert "player_id" in result
    assert "player_name" in result
    assert "bats" in result
    assert "splits" in result
    assert "traditional" in result
    assert "advanced" in result
    assert "plate_discipline" in result
    assert "batted_ball" in result
    assert "sprint_speed" in result
    assert "situational" in result
    assert "today" in result


if __name__ == "__main__":
    tests = [
        test_step1_accepts_player_id_and_splits,
        test_step2_traditional_stats,
        test_step3_advanced_metrics,
        test_step4_plate_discipline,
        test_step5_batted_ball_profile,
        test_step6_sprint_speed,
        test_step7_situational_stats,
        test_step8_today_line,
        test_step9_invalid_player_id,
        test_splits_vs_hand,
        test_splits_home_away,
        test_splits_recency_window,
        test_invalid_parameters,
        test_pitcher_as_batter,
        test_different_players_different_stats,
        test_away_team_players,
        test_bench_players,
        test_response_structure_completeness,
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
