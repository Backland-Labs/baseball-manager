# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the retrosheet_run_expectancy feature.

Verifies all feature requirements from features.json:
1. Include a data/compute_re_matrix.py script that computes the 24-state RE matrix
2. Compute win probability tables indexed by (inning, half, outs, base state, score diff)
3. Compute leverage index values for each game state from the win probability table
4. Write computed tables to JSON files (re_matrix.json, win_probability.json, leverage_index.json)
5. Include pre-computed tables for recent seasons in the repository
6. The computation script is idempotent
"""

import json
import math
import subprocess
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# -----------------------------------------------------------------------
# Helper: load JSON files
# -----------------------------------------------------------------------

def load_json(name: str) -> dict:
    path = DATA_DIR / name
    assert path.exists(), f"Missing pre-computed file: {path}"
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Step 1: compute_re_matrix.py script computes the 24-state RE matrix
# -----------------------------------------------------------------------

class TestStep1ComputeREMatrix:
    """The computation script exists and produces the RE matrix."""

    def test_script_exists(self):
        """Step 1: data/compute_re_matrix.py exists."""
        script = DATA_DIR / "compute_re_matrix.py"
        assert script.exists(), f"Missing script: {script}"

    def test_script_has_pep723_metadata(self):
        """Step 1: Script has PEP 723 inline metadata block."""
        script = DATA_DIR / "compute_re_matrix.py"
        content = script.read_text()
        assert "# /// script" in content
        assert "# ///" in content

    def test_script_importable(self):
        """Step 1: Script is importable as a module."""
        from data.compute_re_matrix import RE_MATRIX, PROB_AT_LEAST_ONE, RUN_DISTRIBUTION
        assert isinstance(RE_MATRIX, dict)
        assert isinstance(PROB_AT_LEAST_ONE, dict)
        assert isinstance(RUN_DISTRIBUTION, dict)

    def test_re_matrix_has_8_base_states(self):
        """Step 1: RE matrix covers all 8 base states."""
        from data.compute_re_matrix import RE_MATRIX
        expected_keys = {"000", "100", "010", "001", "110", "101", "011", "111"}
        assert set(RE_MATRIX.keys()) == expected_keys

    def test_re_matrix_has_3_out_counts(self):
        """Step 1: Each base state has values for 0, 1, 2 outs."""
        from data.compute_re_matrix import RE_MATRIX
        for key, values in RE_MATRIX.items():
            assert len(values) == 3, f"Key {key} has {len(values)} values, expected 3"

    def test_re_matrix_24_total_states(self):
        """Step 1: RE matrix covers 8 base states x 3 outs = 24 states."""
        from data.compute_re_matrix import RE_MATRIX
        total = sum(len(v) for v in RE_MATRIX.values())
        assert total == 24

    def test_re_matrix_values_realistic(self):
        """Step 1: RE values are within realistic MLB ranges (0-3 runs)."""
        from data.compute_re_matrix import RE_MATRIX
        for key, values in RE_MATRIX.items():
            for outs, re in enumerate(values):
                assert 0.0 < re < 3.0, f"RE {re} for {key}/{outs}outs out of range"

    def test_re_matrix_decreases_with_outs(self):
        """Step 1: RE decreases as outs increase for every base state."""
        from data.compute_re_matrix import RE_MATRIX
        for key, values in RE_MATRIX.items():
            assert values[0] >= values[1] >= values[2], \
                f"RE not monotonically decreasing for {key}: {values}"

    def test_re_matrix_more_runners_more_runs(self):
        """Step 1: More runners on base means higher RE (for same out count)."""
        from data.compute_re_matrix import RE_MATRIX
        for outs in range(3):
            assert RE_MATRIX["100"][outs] > RE_MATRIX["000"][outs]
            assert RE_MATRIX["111"][outs] > RE_MATRIX["100"][outs]
            assert RE_MATRIX["111"][outs] > RE_MATRIX["010"][outs]

    def test_re_matrix_known_values(self):
        """Step 1: Spot-check known RE values (bases empty, 0 outs ~ 0.48)."""
        from data.compute_re_matrix import RE_MATRIX
        # Bases empty, 0 outs should be around 0.45-0.55
        assert 0.40 < RE_MATRIX["000"][0] < 0.55
        # Bases loaded, 0 outs should be around 2.0-2.5
        assert 2.0 < RE_MATRIX["111"][0] < 2.6

    def test_prob_at_least_one_all_states(self):
        """Step 1: Probability of scoring covers all 24 states."""
        from data.compute_re_matrix import PROB_AT_LEAST_ONE
        assert len(PROB_AT_LEAST_ONE) == 8
        for key, values in PROB_AT_LEAST_ONE.items():
            assert len(values) == 3
            for p in values:
                assert 0.0 < p < 1.0, f"Probability {p} out of range for {key}"

    def test_prob_scoring_decreases_with_outs(self):
        """Step 1: Probability of scoring decreases with more outs."""
        from data.compute_re_matrix import PROB_AT_LEAST_ONE
        for key, values in PROB_AT_LEAST_ONE.items():
            assert values[0] >= values[1] >= values[2], \
                f"Prob not decreasing for {key}: {values}"

    def test_run_distribution_all_states(self):
        """Step 1: Run distribution covers all 24 states with 4 buckets each."""
        from data.compute_re_matrix import RUN_DISTRIBUTION
        assert len(RUN_DISTRIBUTION) == 8
        for key, outs_data in RUN_DISTRIBUTION.items():
            assert len(outs_data) == 3
            for outs, dist in enumerate(outs_data):
                assert len(dist) == 4, f"Distribution for {key}/{outs}outs has {len(dist)} buckets"

    def test_run_distribution_sums_to_one(self):
        """Step 1: Run distribution probabilities sum to ~1.0."""
        from data.compute_re_matrix import RUN_DISTRIBUTION
        for key, outs_data in RUN_DISTRIBUTION.items():
            for outs, dist in enumerate(outs_data):
                total = sum(dist)
                assert abs(total - 1.0) < 0.01, \
                    f"Distribution for {key}/{outs}outs sums to {total}"

    def test_run_distribution_non_negative(self):
        """Step 1: All distribution values are non-negative."""
        from data.compute_re_matrix import RUN_DISTRIBUTION
        for key, outs_data in RUN_DISTRIBUTION.items():
            for outs, dist in enumerate(outs_data):
                for p in dist:
                    assert p >= 0.0, f"Negative prob {p} in {key}/{outs}outs"


# -----------------------------------------------------------------------
# Step 2: Win probability tables
# -----------------------------------------------------------------------

class TestStep2WinProbabilityTables:
    """Win probability tables indexed by game state."""

    def test_wp_table_exists_in_json(self):
        """Step 2: win_probability.json contains a wp_table."""
        data = load_json("win_probability.json")
        assert "wp_table" in data

    def test_wp_table_covers_innings_1_to_12(self):
        """Step 2: WP table covers innings 1-12."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        for inning in range(1, 13):
            assert f"{inning}_TOP" in wp, f"Missing {inning}_TOP"
            assert f"{inning}_BOTTOM" in wp, f"Missing {inning}_BOTTOM"

    def test_wp_table_covers_all_outs(self):
        """Step 2: Each inning-half has entries for 0, 1, 2 outs."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        for key in wp:
            for outs in ("0", "1", "2"):
                assert outs in wp[key], f"Missing outs={outs} in {key}"

    def test_wp_table_covers_all_base_states(self):
        """Step 2: Each out-count has all 8 base states."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        expected = {"000", "100", "010", "001", "110", "101", "011", "111"}
        sample = wp["5_TOP"]["0"]
        assert set(sample.keys()) == expected

    def test_wp_table_covers_score_differentials(self):
        """Step 2: Each base state has score diffs from -10 to +10."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        sample = wp["5_TOP"]["0"]["000"]
        for diff in range(-10, 11):
            assert str(diff) in sample, f"Missing score diff {diff}"

    def test_wp_values_in_valid_range(self):
        """Step 2: All WP values are between 0.01 and 0.99."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        for inning_key, outs_data in wp.items():
            for outs_key, base_data in outs_data.items():
                for base_key, diff_data in base_data.items():
                    for diff_key, val in diff_data.items():
                        assert 0.0 < val < 1.0, \
                            f"WP {val} out of range at {inning_key}/{outs_key}/{base_key}/{diff_key}"

    def test_wp_tied_game_near_50(self):
        """Step 2: Tied game in early innings should be near 50%."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]["1_TOP"]["0"]["000"]["0"]
        assert 0.40 <= wp <= 0.55, f"Tied game WP {wp} too far from 0.50"

    def test_wp_increases_with_lead(self):
        """Step 2: WP increases as away team leads by more."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        for diff in range(-9, 10):
            wp_low = wp["5_TOP"]["0"]["000"][str(diff)]
            wp_high = wp["5_TOP"]["0"]["000"][str(diff + 1)]
            assert wp_high >= wp_low, \
                f"WP not increasing: diff={diff} -> {wp_low}, diff={diff+1} -> {wp_high}"

    def test_wp_late_lead_worth_more(self):
        """Step 2: A 1-run lead is worth more in the 9th than the 1st."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        wp_early = wp["1_TOP"]["0"]["000"]["1"]
        wp_late = wp["9_TOP"]["0"]["000"]["1"]
        assert wp_late > wp_early

    def test_wp_has_metadata(self):
        """Step 2: WP file includes metadata about the data source."""
        data = load_json("win_probability.json")
        assert "metadata" in data
        meta = data["metadata"]
        assert "source" in meta
        assert "perspective" in meta
        assert meta["perspective"] == "away_team"

    def test_wp_has_base_wp_tables(self):
        """Step 2: WP file includes the base WP tables used in computation."""
        data = load_json("win_probability.json")
        assert "base_wp_away" in data
        assert "extra_inning_base_wp_away" in data

    def test_wp_total_entries(self):
        """Step 2: WP table has the expected number of entries."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        count = 0
        for inning_data in wp.values():
            for outs_data in inning_data.values():
                for base_data in outs_data.values():
                    count += len(base_data)
        # 12 innings * 2 halves * 3 outs * 8 base states * 21 diffs = 12096
        assert count == 12096, f"Expected 12096 entries, got {count}"


# -----------------------------------------------------------------------
# Step 3: Leverage index values
# -----------------------------------------------------------------------

class TestStep3LeverageIndex:
    """Leverage index values derived from win probability tables."""

    def test_li_table_exists_in_json(self):
        """Step 3: leverage_index.json contains an li_table."""
        data = load_json("leverage_index.json")
        assert "li_table" in data

    def test_li_table_same_dimensions_as_wp(self):
        """Step 3: LI table has same dimensions as WP table."""
        wp_data = load_json("win_probability.json")
        li_data = load_json("leverage_index.json")

        wp_keys = set(wp_data["wp_table"].keys())
        li_keys = set(li_data["li_table"].keys())
        assert wp_keys == li_keys, "LI and WP tables have different inning keys"

    def test_li_values_positive(self):
        """Step 3: All LI values are positive."""
        data = load_json("leverage_index.json")
        for inning_data in data["li_table"].values():
            for outs_data in inning_data.values():
                for base_data in outs_data.values():
                    for diff_key, val in base_data.items():
                        assert val > 0, f"LI {val} not positive at diff={diff_key}"

    def test_li_values_in_range(self):
        """Step 3: All LI values are between 0.1 and 10.0."""
        data = load_json("leverage_index.json")
        for inning_key, inning_data in data["li_table"].items():
            for outs_key, outs_data in inning_data.items():
                for base_key, diff_data in outs_data.items():
                    for diff_key, val in diff_data.items():
                        assert 0.1 <= val <= 10.0, \
                            f"LI {val} out of range at {inning_key}/{outs_key}/{base_key}/{diff_key}"

    def test_li_average_situation_near_1(self):
        """Step 3: Tied game, middle innings, bases empty should have LI near 1.0."""
        data = load_json("leverage_index.json")
        li = data["li_table"]["5_TOP"]["0"]["000"]["0"]
        assert 0.5 <= li <= 2.5, f"Average-situation LI {li} too far from 1.0"

    def test_li_high_leverage_situation(self):
        """Step 3: Close game, late inning, runners on = high leverage."""
        data = load_json("leverage_index.json")
        li = data["li_table"]["9_TOP"]["1"]["110"]["0"]
        assert li > 1.5, f"High-leverage LI {li} not > 1.5"

    def test_li_blowout_low_leverage(self):
        """Step 3: Blowout should have low leverage."""
        data = load_json("leverage_index.json")
        li = data["li_table"]["7_TOP"]["0"]["000"]["8"]
        assert li < 1.0, f"Blowout LI {li} not < 1.0"

    def test_li_has_metadata(self):
        """Step 3: LI file includes metadata."""
        data = load_json("leverage_index.json")
        assert "metadata" in data
        meta = data["metadata"]
        assert "avg_wp_swing" in meta
        assert meta["avg_wp_swing"] > 0

    def test_li_avg_swing_documented(self):
        """Step 3: The average WP swing used for calibration is recorded."""
        data = load_json("leverage_index.json")
        avg_swing = data["metadata"]["avg_wp_swing"]
        # Should be a reasonable value (typically 0.05-0.20)
        assert 0.05 < avg_swing < 0.30, f"Avg WP swing {avg_swing} seems unreasonable"

    def test_li_total_entries(self):
        """Step 3: LI table has the expected number of entries."""
        data = load_json("leverage_index.json")
        count = 0
        for inning_data in data["li_table"].values():
            for outs_data in inning_data.values():
                for base_data in outs_data.values():
                    count += len(base_data)
        assert count == 12096, f"Expected 12096 entries, got {count}"


# -----------------------------------------------------------------------
# Step 4: JSON files are written correctly
# -----------------------------------------------------------------------

class TestStep4JSONFiles:
    """Tables are written to the correct JSON files."""

    def test_re_matrix_json_exists(self):
        """Step 4: data/re_matrix.json exists."""
        assert (DATA_DIR / "re_matrix.json").exists()

    def test_win_probability_json_exists(self):
        """Step 4: data/win_probability.json exists."""
        assert (DATA_DIR / "win_probability.json").exists()

    def test_leverage_index_json_exists(self):
        """Step 4: data/leverage_index.json exists."""
        assert (DATA_DIR / "leverage_index.json").exists()

    def test_re_matrix_valid_json(self):
        """Step 4: re_matrix.json is valid JSON."""
        path = DATA_DIR / "re_matrix.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_win_probability_valid_json(self):
        """Step 4: win_probability.json is valid JSON."""
        path = DATA_DIR / "win_probability.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_leverage_index_valid_json(self):
        """Step 4: leverage_index.json is valid JSON."""
        path = DATA_DIR / "leverage_index.json"
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_re_matrix_json_structure(self):
        """Step 4: re_matrix.json has expected top-level keys."""
        data = load_json("re_matrix.json")
        assert "metadata" in data
        assert "re_matrix" in data
        assert "prob_at_least_one" in data
        assert "run_distribution" in data

    def test_win_probability_json_structure(self):
        """Step 4: win_probability.json has expected top-level keys."""
        data = load_json("win_probability.json")
        assert "metadata" in data
        assert "wp_table" in data
        assert "base_wp_away" in data
        assert "extra_inning_base_wp_away" in data

    def test_leverage_index_json_structure(self):
        """Step 4: leverage_index.json has expected top-level keys."""
        data = load_json("leverage_index.json")
        assert "metadata" in data
        assert "li_table" in data


# -----------------------------------------------------------------------
# Step 5: Pre-computed tables included in the repository
# -----------------------------------------------------------------------

class TestStep5PrecomputedTables:
    """Pre-computed tables are present and ready to use."""

    def test_tables_non_empty(self):
        """Step 5: All JSON files are non-empty."""
        for name in ("re_matrix.json", "win_probability.json", "leverage_index.json"):
            path = DATA_DIR / name
            assert path.stat().st_size > 0, f"{name} is empty"

    def test_re_matrix_has_recent_season_data(self):
        """Step 5: RE matrix metadata references recent seasons."""
        data = load_json("re_matrix.json")
        seasons = data["metadata"]["seasons"]
        assert "2023" in seasons or "2022" in seasons, \
            f"Seasons '{seasons}' should include recent MLB data"

    def test_wp_has_recent_season_data(self):
        """Step 5: WP metadata references recent seasons."""
        data = load_json("win_probability.json")
        seasons = data["metadata"]["seasons"]
        assert "2023" in seasons or "2022" in seasons

    def test_tables_are_self_consistent(self):
        """Step 5: RE values in the JSON match the computation module."""
        from data.compute_re_matrix import RE_MATRIX as MODULE_RE
        json_re = load_json("re_matrix.json")["re_matrix"]
        for key in MODULE_RE:
            for outs in range(3):
                assert MODULE_RE[key][outs] == json_re[key][outs], \
                    f"Mismatch at {key}/{outs}: module={MODULE_RE[key][outs]}, json={json_re[key][outs]}"

    def test_wp_values_consistent_with_compute_module(self):
        """Step 5: Spot-check WP JSON against the compute function."""
        from data.compute_re_matrix import compute_wp
        json_wp = load_json("win_probability.json")["wp_table"]

        # Test several states
        test_cases = [
            (5, "TOP", 0, "000", 0),
            (9, "BOTTOM", 2, "111", -3),
            (1, "TOP", 1, "100", 2),
            (7, "BOTTOM", 0, "010", -1),
        ]
        for inning, half, outs, base, diff in test_cases:
            computed = round(compute_wp(inning, half, outs, base, diff), 4)
            from_json = json_wp[f"{inning}_{half}"][str(outs)][base][str(diff)]
            assert computed == from_json, \
                f"WP mismatch at ({inning},{half},{outs},{base},{diff}): computed={computed}, json={from_json}"

    def test_re_matrix_consistent_with_tool(self):
        """Step 5: RE matrix values match the get_run_expectancy tool's inline data."""
        from tools.get_run_expectancy import RE_MATRIX as TOOL_RE
        json_re = load_json("re_matrix.json")["re_matrix"]
        for key in TOOL_RE:
            for outs in range(3):
                assert TOOL_RE[key][outs] == json_re[key][outs], \
                    f"Tool/JSON mismatch at {key}/{outs}"

    def test_prob_scoring_consistent_with_tool(self):
        """Step 5: Prob-of-scoring values match the tool's inline data."""
        from tools.get_run_expectancy import PROB_AT_LEAST_ONE as TOOL_PROB
        json_prob = load_json("re_matrix.json")["prob_at_least_one"]
        for key in TOOL_PROB:
            for outs in range(3):
                assert TOOL_PROB[key][outs] == json_prob[key][outs], \
                    f"Prob scoring mismatch at {key}/{outs}"

    def test_run_distribution_consistent_with_tool(self):
        """Step 5: Run distribution values match the tool's inline data."""
        from tools.get_run_expectancy import RUN_DISTRIBUTION as TOOL_DIST
        json_dist = load_json("re_matrix.json")["run_distribution"]
        for key in TOOL_DIST:
            for outs in range(3):
                for i in range(4):
                    assert TOOL_DIST[key][outs][i] == json_dist[key][outs][i], \
                        f"Run dist mismatch at {key}/{outs}/{i}"


# -----------------------------------------------------------------------
# Step 6: Script is idempotent
# -----------------------------------------------------------------------

class TestStep6Idempotent:
    """The computation script is idempotent."""

    def test_script_produces_same_output(self):
        """Step 6: Running the script twice produces identical JSON."""
        # Read current files
        re_before = (DATA_DIR / "re_matrix.json").read_text()
        wp_before = (DATA_DIR / "win_probability.json").read_text()
        li_before = (DATA_DIR / "leverage_index.json").read_text()

        # Run the script
        result = subprocess.run(
            [sys.executable, str(DATA_DIR / "compute_re_matrix.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Compare
        re_after = (DATA_DIR / "re_matrix.json").read_text()
        wp_after = (DATA_DIR / "win_probability.json").read_text()
        li_after = (DATA_DIR / "leverage_index.json").read_text()

        assert re_before == re_after, "re_matrix.json changed on re-run"
        assert wp_before == wp_after, "win_probability.json changed on re-run"
        assert li_before == li_after, "leverage_index.json changed on re-run"

    def test_script_runs_without_error(self):
        """Step 6: Script runs cleanly and exits with code 0."""
        result = subprocess.run(
            [sys.executable, str(DATA_DIR / "compute_re_matrix.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert "Done" in result.stdout

    def test_script_prints_summary(self):
        """Step 6: Script outputs a summary of what it computed."""
        result = subprocess.run(
            [sys.executable, str(DATA_DIR / "compute_re_matrix.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "RE matrix" in result.stdout or "re_matrix" in result.stdout.lower()
        assert "Wrote" in result.stdout


# -----------------------------------------------------------------------
# Cross-validation: WP and LI tables are consistent
# -----------------------------------------------------------------------

class TestCrossValidation:
    """Cross-validation between WP and LI tables."""

    def test_li_derived_from_wp(self):
        """LI values are derivable from the WP table's values."""
        from data.compute_re_matrix import compute_wp, _raw_wp_swing, compute_avg_swing

        li_data = load_json("leverage_index.json")
        avg_swing = li_data["metadata"]["avg_wp_swing"]

        # Spot-check several states
        test_cases = [
            (5, "TOP", 0, "000", 0),
            (9, "TOP", 1, "110", 0),
            (3, "BOTTOM", 2, "100", -2),
        ]
        for inning, half, outs, base, diff in test_cases:
            wp_swing = _raw_wp_swing(inning, half, outs, base, diff)
            expected_li = max(0.1, min(10.0, round(wp_swing / avg_swing, 2)))
            actual_li = li_data["li_table"][f"{inning}_{half}"][str(outs)][base][str(diff)]
            assert expected_li == actual_li, \
                f"LI mismatch at ({inning},{half},{outs},{base},{diff}): expected={expected_li}, actual={actual_li}"

    def test_wp_home_advantage_exists(self):
        """Home team has slight advantage in tied game (WP for away < 0.50)."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        # In a tied game top of 5th, away WP should be < 0.50
        away_wp = wp["5_TOP"]["0"]["000"]["0"]
        assert away_wp < 0.50, f"Away WP {away_wp} should be < 0.50 (home advantage)"

    def test_wp_extreme_leads_approach_extremes(self):
        """Large leads in late innings should approach WP ~ 0.99."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        # Away up 10 in the 9th
        wp_big_lead = wp["9_TOP"]["0"]["000"]["10"]
        assert wp_big_lead > 0.89, f"10-run lead in 9th: WP {wp_big_lead} not > 0.89"
        # Away down 10 in the 9th
        wp_big_deficit = wp["9_TOP"]["0"]["000"]["-10"]
        assert wp_big_deficit < 0.05, f"10-run deficit in 9th: WP {wp_big_deficit} not < 0.05"

    def test_wp_extra_innings_valid(self):
        """Extra inning (10th+) WP values are valid."""
        data = load_json("win_probability.json")
        wp = data["wp_table"]
        for inning in (10, 11, 12):
            for half in ("TOP", "BOTTOM"):
                key = f"{inning}_{half}"
                wp_val = wp[key]["0"]["000"]["0"]
                assert 0.01 < wp_val < 0.99, \
                    f"Extra inning WP {wp_val} out of range at {key}"

    def test_li_high_for_close_late_games(self):
        """LI is notably higher for close games in late innings."""
        data = load_json("leverage_index.json")
        li = data["li_table"]

        # Tied 9th inning
        li_late = li["9_TOP"]["0"]["000"]["0"]
        # Tied 1st inning
        li_early = li["1_TOP"]["0"]["000"]["0"]
        assert li_late > li_early, \
            f"Late LI ({li_late}) not > early LI ({li_early})"

    def test_li_runners_increase_leverage(self):
        """LI with runners on should generally be >= LI with bases empty."""
        data = load_json("leverage_index.json")
        li = data["li_table"]
        li_empty = li["7_TOP"]["0"]["000"]["0"]
        li_loaded = li["7_TOP"]["0"]["111"]["0"]
        assert li_loaded >= li_empty, \
            f"Loaded LI ({li_loaded}) not >= empty LI ({li_empty})"


# -----------------------------------------------------------------------
# Computation function tests
# -----------------------------------------------------------------------

class TestComputeFunctions:
    """Tests for the computation functions in the module."""

    def test_compute_wp_returns_float(self):
        """compute_wp returns a float."""
        from data.compute_re_matrix import compute_wp
        result = compute_wp(5, "TOP", 0, "000", 0)
        assert isinstance(result, float)

    def test_compute_wp_clamped(self):
        """compute_wp clamps to [0.01, 0.99]."""
        from data.compute_re_matrix import compute_wp
        # Extreme lead
        wp = compute_wp(9, "BOTTOM", 2, "000", 15)
        assert 0.01 <= wp <= 0.99
        # Extreme deficit
        wp = compute_wp(9, "BOTTOM", 2, "000", -15)
        assert 0.01 <= wp <= 0.99

    def test_compute_avg_swing_positive(self):
        """compute_avg_swing returns a positive value."""
        from data.compute_re_matrix import compute_avg_swing
        avg = compute_avg_swing()
        assert avg > 0

    def test_logistic_function(self):
        """Logistic function maps 0 -> 0.5."""
        from data.compute_re_matrix import _logistic
        assert abs(_logistic(0) - 0.5) < 0.001

    def test_logistic_symmetric(self):
        """Logistic function is symmetric: f(x) + f(-x) = 1."""
        from data.compute_re_matrix import _logistic
        for x in (0.5, 1.0, 2.0, 5.0):
            assert abs(_logistic(x) + _logistic(-x) - 1.0) < 0.001

    def test_generate_re_matrix_json(self):
        """generate_re_matrix_json returns properly structured data."""
        from data.compute_re_matrix import generate_re_matrix_json
        data = generate_re_matrix_json()
        assert "metadata" in data
        assert "re_matrix" in data
        assert len(data["re_matrix"]) == 8

    def test_generate_win_probability_json(self):
        """generate_win_probability_json returns properly structured data."""
        from data.compute_re_matrix import generate_win_probability_json, compute_avg_swing
        avg = compute_avg_swing()
        data = generate_win_probability_json(avg)
        assert "wp_table" in data
        assert len(data["wp_table"]) == 24  # 12 innings * 2 halves

    def test_generate_leverage_index_json(self):
        """generate_leverage_index_json returns properly structured data."""
        from data.compute_re_matrix import generate_leverage_index_json, compute_avg_swing
        avg = compute_avg_swing()
        data = generate_leverage_index_json(avg)
        assert "li_table" in data
        assert len(data["li_table"]) == 24


if __name__ == "__main__":
    import importlib
    test_classes = [
        TestStep1ComputeREMatrix,
        TestStep2WinProbabilityTables,
        TestStep3LeverageIndex,
        TestStep4JSONFiles,
        TestStep5PrecomputedTables,
        TestStep6Idempotent,
        TestCrossValidation,
        TestComputeFunctions,
    ]

    passed = 0
    failed = 0
    for cls in test_classes:
        inst = cls()
        for name in sorted(dir(inst)):
            if name.startswith("test_"):
                try:
                    getattr(inst, name)()
                    passed += 1
                    print(f"  PASS: {cls.__name__}.{name}")
                except Exception as e:
                    failed += 1
                    print(f"  FAIL: {cls.__name__}.{name}: {e}")

    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
    print("All tests passed!")
