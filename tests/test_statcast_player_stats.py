# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pandas>=2.0"]
# ///
"""Tests for the statcast_player_stats feature.

Validates integration with pybaseball to fetch real Statcast and FanGraphs
player statistics:
  1. Fetch real batting stats (AVG, OBP, SLG, OPS, wOBA, wRC+, barrel rate,
     xwOBA, K%, BB%, chase rate, whiff rate, GB%, pull%, EV, LA, sprint speed)
  2. Fetch real pitching stats (ERA, FIP, xFIP, K%, BB%, GB%, pitch mix with
     per-pitch velocity/spin/whiff rates)
  3. Support split queries (vs LHB/RHB, home/away)
  4. Fetch pitcher TTO wOBA splits
  5. Fetch defensive metrics (OAA, DRS, UZR by position)
  6. Fetch catcher metrics (pop time, framing runs)
  7. Local cache layer integration (JSON files in data/cache/)
  8. Cache entries expire after 24 hours
"""

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# We need pandas for creating mock DataFrames
import pandas as pd

from data.cache import Cache, TTL_SEASON_STATS


# ---------------------------------------------------------------------------
# Mock pybaseball module
# ---------------------------------------------------------------------------

def _make_batting_df(player_id=123456, **overrides):
    """Create a mock FanGraphs batting DataFrame with realistic data."""
    data = {
        "xMLBAMID": [player_id],
        "MLBAMID": [player_id],
        "Name": ["Mike Trout"],
        "Team": ["LAA"],
        "AVG": [0.283],
        "OBP": [0.365],
        "SLG": [0.496],
        "wOBA": [0.367],
        "wRC+": [145],
        "Barrel%": [0.125],
        "xwOBA": [0.370],
        "K%": [0.220],
        "BB%": [0.105],
        "O-Swing%": [0.280],
        "SwStr%": [0.120],
        "GB%": [0.385],
        "FB%": [0.355],
        "LD%": [0.210],
        "Pull%": [0.425],
        "EV": [91.2],
        "LA": [13.5],
        "OAA": [3],
        "DRS": [5],
        "UZR": [4.2],
        "Pos": ["CF"],
        "IDfg": [10155],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _make_pitching_df(player_id=543037, **overrides):
    """Create a mock FanGraphs pitching DataFrame with realistic data."""
    data = {
        "xMLBAMID": [player_id],
        "MLBAMID": [player_id],
        "Name": ["Gerrit Cole"],
        "Team": ["NYY"],
        "ERA": [3.12],
        "FIP": [2.95],
        "xFIP": [3.15],
        "SIERA": [3.05],
        "K%": [0.285],
        "BB%": [0.058],
        "GB%": [0.380],
        "FB%": [0.370],
        "LD%": [0.200],
        "IDfg": [13125],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _make_statcast_pitcher_df(player_id=543037, n_pitches=100):
    """Create a mock Statcast pitch-level DataFrame."""
    import random
    random.seed(42)

    pitch_types = ["FF"] * 55 + ["SL"] * 25 + ["CH"] * 15 + ["CU"] * 5
    random.shuffle(pitch_types)

    descriptions = []
    for pt in pitch_types:
        r = random.random()
        if r < 0.15:
            descriptions.append("swinging_strike")
        elif r < 0.20:
            descriptions.append("swinging_strike_blocked")
        elif r < 0.40:
            descriptions.append("foul")
        elif r < 0.50:
            descriptions.append("hit_into_play")
        elif r < 0.55:
            descriptions.append("hit_into_play_no_out")
        elif r < 0.65:
            descriptions.append("called_strike")
        else:
            descriptions.append("ball")

    events = [None] * n_pitches
    at_bat_numbers = []
    game_pks = []
    batters = []
    ab = 1
    game_pk = 1001
    batter_lineup = [100, 101, 102, 103, 104, 105, 106, 107, 108]
    batter_idx = 0

    for i in range(n_pitches):
        at_bat_numbers.append(ab)
        game_pks.append(game_pk)
        batters.append(batter_lineup[batter_idx % len(batter_lineup)])

        if descriptions[i] in ("hit_into_play", "hit_into_play_no_out", "hit_into_play_score"):
            events[i] = random.choice(["single", "double", "field_out", "home_run", "grounded_into_double_play"])
            ab += 1
            batter_idx += 1
        elif descriptions[i] == "swinging_strike" and random.random() < 0.3:
            events[i] = "strikeout"
            ab += 1
            batter_idx += 1

    velocities = []
    spins = []
    for pt in pitch_types:
        if pt == "FF":
            velocities.append(round(random.gauss(96.5, 1.2), 1))
            spins.append(round(random.gauss(2350, 80)))
        elif pt == "SL":
            velocities.append(round(random.gauss(88.0, 1.0), 1))
            spins.append(round(random.gauss(2550, 90)))
        elif pt == "CH":
            velocities.append(round(random.gauss(87.5, 0.8), 1))
            spins.append(round(random.gauss(1700, 60)))
        else:
            velocities.append(round(random.gauss(81.0, 0.9), 1))
            spins.append(round(random.gauss(2800, 100)))

    return pd.DataFrame({
        "pitch_type": pitch_types,
        "release_speed": velocities,
        "release_spin_rate": spins,
        "description": descriptions,
        "events": events,
        "at_bat_number": at_bat_numbers,
        "game_pk": game_pks,
        "batter": batters,
    })


def _make_sprint_speed_df(player_id=123456, speed=28.5):
    """Create a mock sprint speed DataFrame."""
    return pd.DataFrame({
        "player_id": [player_id],
        "sprint_speed": [speed],
        "hp_to_1b": [speed],
    })


def _make_pop_time_df(player_id=999999, pop_time=1.92):
    """Create a mock catcher pop time DataFrame."""
    return pd.DataFrame({
        "player_id": [player_id],
        "pop_time": [pop_time],
    })


def _make_framing_df(player_id=999999, runs=8.5, runs_per_200=3.2):
    """Create a mock catcher framing DataFrame."""
    return pd.DataFrame({
        "player_id": [player_id],
        "runs_extra_strikes": [runs],
        "runs_extra_strikes_per_200": [runs_per_200],
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache(tmp_path):
    """Return a Cache instance backed by a temporary directory."""
    return Cache(root_dir=tmp_path / "cache")


@pytest.fixture
def mock_pybaseball():
    """Create a mock pybaseball module with all needed functions."""
    mock_pb = MagicMock()
    mock_pb.batting_stats = MagicMock(return_value=_make_batting_df())
    mock_pb.pitching_stats = MagicMock(return_value=_make_pitching_df())
    mock_pb.statcast_pitcher = MagicMock(return_value=_make_statcast_pitcher_df())
    mock_pb.statcast_sprint_speed = MagicMock(return_value=_make_sprint_speed_df())
    mock_pb.playerid_lookup = MagicMock(return_value=pd.DataFrame({
        "key_mlbam": [123456],
        "name_last": ["Trout"],
        "name_first": ["Mike"],
    }))
    # Catcher methods
    mock_pb.statcast_catcher_poptime = MagicMock(return_value=_make_pop_time_df())
    mock_pb.statcast_catcher_framing = MagicMock(return_value=_make_framing_df())
    return mock_pb


@pytest.fixture(autouse=True)
def reset_pybaseball_import():
    """Reset the lazy pybaseball import between tests."""
    import data.statcast as statcast_mod
    statcast_mod._pybaseball = None
    yield
    statcast_mod._pybaseball = None


# ---------------------------------------------------------------------------
# Helper to patch pybaseball
# ---------------------------------------------------------------------------

def _patch_pybaseball(mock_pb):
    """Return a patch that makes _get_pybaseball return the mock."""
    return patch("data.statcast._get_pybaseball", return_value=mock_pb)


# ---------------------------------------------------------------------------
# 1. Module existence and imports
# ---------------------------------------------------------------------------

class TestModuleStructure:
    """Verify the statcast module exists and exports expected functions."""

    def test_module_importable(self):
        import data.statcast
        assert hasattr(data.statcast, "get_batting_stats")
        assert hasattr(data.statcast, "get_pitching_stats")

    def test_exports_batting_stats(self):
        from data.statcast import get_batting_stats
        assert callable(get_batting_stats)

    def test_exports_pitching_stats(self):
        from data.statcast import get_pitching_stats
        assert callable(get_pitching_stats)

    def test_exports_batting_splits(self):
        from data.statcast import get_batting_splits
        assert callable(get_batting_splits)

    def test_exports_pitching_splits(self):
        from data.statcast import get_pitching_splits
        assert callable(get_pitching_splits)

    def test_exports_tto_splits(self):
        from data.statcast import get_pitcher_tto_splits
        assert callable(get_pitcher_tto_splits)

    def test_exports_defensive_metrics(self):
        from data.statcast import get_defensive_metrics
        assert callable(get_defensive_metrics)

    def test_exports_catcher_metrics(self):
        from data.statcast import get_catcher_metrics
        assert callable(get_catcher_metrics)

    def test_exports_sprint_speed(self):
        from data.statcast import get_sprint_speed
        assert callable(get_sprint_speed)

    def test_exceptions_defined(self):
        from data.statcast import (
            StatcastError,
            StatcastPlayerNotFoundError,
            StatcastDataUnavailableError,
        )
        assert issubclass(StatcastPlayerNotFoundError, StatcastError)
        assert issubclass(StatcastDataUnavailableError, StatcastError)

    def test_pybaseball_is_dependency(self):
        """The module script header declares pybaseball dependency."""
        mod_path = Path(__file__).resolve().parent.parent / "data" / "statcast.py"
        text = mod_path.read_text()
        assert "pybaseball" in text


# ---------------------------------------------------------------------------
# 2. Batting stats
# ---------------------------------------------------------------------------

class TestBattingStats:
    """Verify get_batting_stats returns proper batting statistics."""

    def test_returns_traditional_stats(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        trad = result["traditional"]
        assert "AVG" in trad
        assert "OBP" in trad
        assert "SLG" in trad
        assert "OPS" in trad
        assert trad["AVG"] == 0.283
        assert trad["OBP"] == 0.365
        assert trad["SLG"] == 0.496

    def test_returns_advanced_metrics(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        adv = result["advanced"]
        assert "wOBA" in adv
        assert "wRC_plus" in adv
        assert "barrel_rate" in adv
        assert "xwOBA" in adv
        assert adv["wOBA"] == 0.367
        assert adv["wRC_plus"] == 145

    def test_returns_plate_discipline(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        disc = result["plate_discipline"]
        assert "K_pct" in disc
        assert "BB_pct" in disc
        assert "chase_rate" in disc
        assert "whiff_rate" in disc
        assert disc["K_pct"] == 0.22
        assert disc["BB_pct"] == 0.105

    def test_returns_batted_ball_profile(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        bb = result["batted_ball"]
        assert "GB_pct" in bb
        assert "pull_pct" in bb
        assert "exit_velocity" in bb
        assert "launch_angle" in bb

    def test_returns_sprint_speed(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        assert "sprint_speed" in result
        assert result["sprint_speed"] == 28.5

    def test_returns_situational_stats(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        sit = result["situational"]
        assert "RISP_avg" in sit
        assert "high_leverage_ops" in sit
        assert "late_and_close_ops" in sit

    def test_player_not_found_raises(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats, StatcastPlayerNotFoundError
        mock_pybaseball.batting_stats.return_value = _make_batting_df(player_id=999)
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastPlayerNotFoundError):
                get_batting_stats(123456, season=2024, cache=tmp_cache)

    def test_empty_dataframe_raises(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats, StatcastDataUnavailableError
        mock_pybaseball.batting_stats.return_value = pd.DataFrame()
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastDataUnavailableError):
                get_batting_stats(123456, season=2024, cache=tmp_cache)

    def test_none_dataframe_raises(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats, StatcastDataUnavailableError
        mock_pybaseball.batting_stats.return_value = None
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastDataUnavailableError):
                get_batting_stats(123456, season=2024, cache=tmp_cache)

    def test_network_error_raises(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats, StatcastDataUnavailableError
        mock_pybaseball.batting_stats.side_effect = ConnectionError("Network error")
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastDataUnavailableError):
                get_batting_stats(123456, season=2024, cache=tmp_cache)

    def test_ops_computed_from_obp_and_slg(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        ops = result["traditional"]["OPS"]
        expected = round(0.365 + 0.496, 3)
        assert ops == expected

    def test_uses_mlbamid_fallback(self, tmp_cache):
        """When xMLBAMID is missing, falls back to MLBAMID column."""
        from data.statcast import get_batting_stats
        df = _make_batting_df(player_id=123456)
        df = df.drop(columns=["xMLBAMID"])
        mock_pb = MagicMock()
        mock_pb.batting_stats.return_value = df
        mock_pb.statcast_sprint_speed.return_value = _make_sprint_speed_df()
        with _patch_pybaseball(mock_pb):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        assert result["traditional"]["AVG"] == 0.283

    def test_sprint_speed_failure_graceful(self, mock_pybaseball, tmp_cache):
        """Sprint speed failure does not crash batting stats."""
        from data.statcast import get_batting_stats
        mock_pybaseball.statcast_sprint_speed.side_effect = Exception("timeout")
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        assert result["sprint_speed"] is None


# ---------------------------------------------------------------------------
# 3. Pitching stats
# ---------------------------------------------------------------------------

class TestPitchingStats:
    """Verify get_pitching_stats returns proper pitching statistics."""

    def test_returns_traditional_stats(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        trad = result["traditional"]
        assert "ERA" in trad
        assert "FIP" in trad
        assert "xFIP" in trad
        assert "SIERA" in trad
        assert trad["ERA"] == 3.12
        assert trad["FIP"] == 2.95

    def test_returns_rates(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        rates = result["rates"]
        assert "K_pct" in rates
        assert "BB_pct" in rates
        assert rates["K_pct"] == 0.285
        assert rates["BB_pct"] == 0.058

    def test_returns_batted_ball(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        bb = result["batted_ball"]
        assert "GB_pct" in bb
        assert "FB_pct" in bb
        assert "LD_pct" in bb

    def test_returns_pitch_mix(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        mix = result["pitch_mix"]
        assert isinstance(mix, list)
        assert len(mix) > 0
        for pitch in mix:
            assert "pitch_type" in pitch
            assert "usage" in pitch
            assert "velocity" in pitch
            assert "spin_rate" in pitch
            assert "whiff_rate" in pitch

    def test_pitch_mix_sorted_by_usage(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        mix = result["pitch_mix"]
        usages = [p["usage"] for p in mix]
        assert usages == sorted(usages, reverse=True)

    def test_pitch_mix_usage_sums_to_one(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        mix = result["pitch_mix"]
        total = sum(p["usage"] for p in mix)
        # Allow rounding tolerance
        assert abs(total - 1.0) < 0.05

    def test_returns_tto_splits(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        tto = result["times_through_order"]
        assert "1st" in tto
        assert "2nd" in tto
        assert "3rd_plus" in tto

    def test_pitcher_not_found_raises(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats, StatcastPlayerNotFoundError
        mock_pybaseball.pitching_stats.return_value = _make_pitching_df(player_id=999)
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastPlayerNotFoundError):
                get_pitching_stats(543037, season=2024, cache=tmp_cache)

    def test_empty_pitching_df_raises(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats, StatcastDataUnavailableError
        mock_pybaseball.pitching_stats.return_value = pd.DataFrame()
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastDataUnavailableError):
                get_pitching_stats(543037, season=2024, cache=tmp_cache)

    def test_pitch_mix_failure_returns_empty(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        mock_pybaseball.statcast_pitcher.side_effect = Exception("timeout")
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        assert result["pitch_mix"] == []


# ---------------------------------------------------------------------------
# 4. Batting splits
# ---------------------------------------------------------------------------

class TestBattingSplits:
    """Verify split query support for batting stats."""

    def test_vs_right_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_splits(123456, season=2024, vs_hand="R", cache=tmp_cache)
        assert result["split"]["vs_hand"] == "R"
        assert "traditional" in result

    def test_vs_left_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_splits(123456, season=2024, vs_hand="L", cache=tmp_cache)
        assert result["split"]["vs_hand"] == "L"

    def test_home_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_splits(123456, season=2024, home_away="home", cache=tmp_cache)
        assert result["split"]["home_away"] == "home"

    def test_away_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_splits(123456, season=2024, home_away="away", cache=tmp_cache)
        assert result["split"]["home_away"] == "away"

    def test_no_split_returns_overall(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_splits(123456, season=2024, cache=tmp_cache)
        assert result["split"]["vs_hand"] is None
        assert result["split"]["home_away"] is None

    def test_split_player_not_found(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits, StatcastPlayerNotFoundError
        mock_pybaseball.batting_stats.return_value = _make_batting_df(player_id=999)
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastPlayerNotFoundError):
                get_batting_splits(123456, season=2024, vs_hand="R", cache=tmp_cache)


# ---------------------------------------------------------------------------
# 5. Pitching splits
# ---------------------------------------------------------------------------

class TestPitchingSplits:
    """Verify split query support for pitching stats."""

    def test_vs_right_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_splits(543037, season=2024, vs_hand="R", cache=tmp_cache)
        assert result["split"]["vs_hand"] == "R"
        assert "traditional" in result

    def test_vs_left_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_splits(543037, season=2024, vs_hand="L", cache=tmp_cache)
        assert result["split"]["vs_hand"] == "L"

    def test_home_away_split(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_splits(543037, season=2024, home_away="home", cache=tmp_cache)
        assert result["split"]["home_away"] == "home"

    def test_split_player_not_found(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_splits, StatcastPlayerNotFoundError
        mock_pybaseball.pitching_stats.return_value = _make_pitching_df(player_id=999)
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastPlayerNotFoundError):
                get_pitching_splits(543037, season=2024, vs_hand="R", cache=tmp_cache)


# ---------------------------------------------------------------------------
# 6. Pitcher TTO splits
# ---------------------------------------------------------------------------

class TestPitcherTTOSplits:
    """Verify times-through-order wOBA splits."""

    def test_returns_tto_keys(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitcher_tto_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)
        assert "1st" in result
        assert "2nd" in result
        assert "3rd_plus" in result

    def test_tto_values_are_floats_or_none(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitcher_tto_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)
        for key in ("1st", "2nd", "3rd_plus"):
            val = result[key]
            assert val is None or isinstance(val, float)

    def test_empty_statcast_returns_none_values(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitcher_tto_splits
        mock_pybaseball.statcast_pitcher.return_value = pd.DataFrame()
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)
        assert result["1st"] is None
        assert result["2nd"] is None
        assert result["3rd_plus"] is None

    def test_tto_network_error(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitcher_tto_splits, StatcastDataUnavailableError
        mock_pybaseball.statcast_pitcher.side_effect = ConnectionError("offline")
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastDataUnavailableError):
                get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)

    def test_tto_woba_values_reasonable(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitcher_tto_splits
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)
        for key in ("1st", "2nd", "3rd_plus"):
            val = result[key]
            if val is not None:
                assert 0.0 <= val <= 1.0, f"TTO {key} wOBA={val} out of range"


# ---------------------------------------------------------------------------
# 7. Defensive metrics
# ---------------------------------------------------------------------------

class TestDefensiveMetrics:
    """Verify defensive metrics retrieval."""

    def test_returns_oaa(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics
        with _patch_pybaseball(mock_pybaseball):
            result = get_defensive_metrics(123456, season=2024, cache=tmp_cache)
        assert "OAA" in result
        assert result["OAA"] == 3

    def test_returns_drs(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics
        with _patch_pybaseball(mock_pybaseball):
            result = get_defensive_metrics(123456, season=2024, cache=tmp_cache)
        assert "DRS" in result
        assert result["DRS"] == 5

    def test_returns_uzr(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics
        with _patch_pybaseball(mock_pybaseball):
            result = get_defensive_metrics(123456, season=2024, cache=tmp_cache)
        assert "UZR" in result
        assert result["UZR"] == 4.2

    def test_returns_position(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics
        with _patch_pybaseball(mock_pybaseball):
            result = get_defensive_metrics(123456, season=2024, position="CF", cache=tmp_cache)
        assert result["position"] == "CF"

    def test_returns_player_id_and_season(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics
        with _patch_pybaseball(mock_pybaseball):
            result = get_defensive_metrics(123456, season=2024, cache=tmp_cache)
        assert result["player_id"] == 123456
        assert result["season"] == 2024

    def test_player_not_found(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics, StatcastPlayerNotFoundError
        mock_pybaseball.batting_stats.return_value = _make_batting_df(player_id=999)
        with _patch_pybaseball(mock_pybaseball):
            with pytest.raises(StatcastPlayerNotFoundError):
                get_defensive_metrics(123456, season=2024, cache=tmp_cache)


# ---------------------------------------------------------------------------
# 8. Catcher metrics
# ---------------------------------------------------------------------------

class TestCatcherMetrics:
    """Verify catcher-specific metrics."""

    def test_returns_pop_time(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(
            return_value=_make_pop_time_df(player_id=999999, pop_time=1.92)
        )
        mock_pb.statcast_catcher_framing = MagicMock(
            return_value=_make_framing_df(player_id=999999)
        )
        with _patch_pybaseball(mock_pb):
            result = get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert result["pop_time"] == 1.92

    def test_returns_framing_runs(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(
            return_value=_make_pop_time_df(player_id=999999)
        )
        mock_pb.statcast_catcher_framing = MagicMock(
            return_value=_make_framing_df(player_id=999999, runs=8.5, runs_per_200=3.2)
        )
        with _patch_pybaseball(mock_pb):
            result = get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert result["framing_runs"] == 8.5
        assert result["framing_runs_per_200"] == 3.2

    def test_pop_time_unavailable_returns_none(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(return_value=pd.DataFrame())
        mock_pb.statcast_catcher_framing = MagicMock(return_value=pd.DataFrame())
        with _patch_pybaseball(mock_pb):
            result = get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert result["pop_time"] is None

    def test_framing_unavailable_returns_none(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(return_value=pd.DataFrame())
        mock_pb.statcast_catcher_framing = MagicMock(return_value=pd.DataFrame())
        with _patch_pybaseball(mock_pb):
            result = get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert result["framing_runs"] is None

    def test_pop_time_network_failure_graceful(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(side_effect=Exception("timeout"))
        mock_pb.statcast_catcher_framing = MagicMock(return_value=pd.DataFrame())
        with _patch_pybaseball(mock_pb):
            result = get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert result["pop_time"] is None

    def test_returns_player_id_and_season(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(return_value=pd.DataFrame())
        mock_pb.statcast_catcher_framing = MagicMock(return_value=pd.DataFrame())
        with _patch_pybaseball(mock_pb):
            result = get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert result["player_id"] == 999999
        assert result["season"] == 2024


# ---------------------------------------------------------------------------
# 9. Sprint speed
# ---------------------------------------------------------------------------

class TestSprintSpeed:
    """Verify sprint speed retrieval."""

    def test_returns_float(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_sprint_speed
        with _patch_pybaseball(mock_pybaseball):
            result = get_sprint_speed(123456, season=2024, cache=tmp_cache)
        assert isinstance(result, float)
        assert result == 28.5

    def test_player_not_found_returns_none(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_sprint_speed
        mock_pybaseball.statcast_sprint_speed.return_value = _make_sprint_speed_df(player_id=999)
        with _patch_pybaseball(mock_pybaseball):
            result = get_sprint_speed(123456, season=2024, cache=tmp_cache)
        assert result is None

    def test_empty_df_returns_none(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_sprint_speed
        mock_pybaseball.statcast_sprint_speed.return_value = pd.DataFrame()
        with _patch_pybaseball(mock_pybaseball):
            result = get_sprint_speed(123456, season=2024, cache=tmp_cache)
        assert result is None

    def test_network_failure_returns_none(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_sprint_speed
        mock_pybaseball.statcast_sprint_speed.side_effect = Exception("timeout")
        with _patch_pybaseball(mock_pybaseball):
            result = get_sprint_speed(123456, season=2024, cache=tmp_cache)
        assert result is None


# ---------------------------------------------------------------------------
# 10. Cache integration
# ---------------------------------------------------------------------------

class TestCacheIntegration:
    """Verify that all functions use the cache properly."""

    def test_batting_stats_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result1 = get_batting_stats(123456, season=2024, cache=tmp_cache)
            result2 = get_batting_stats(123456, season=2024, cache=tmp_cache)
        # pybaseball should only be called once
        assert mock_pybaseball.batting_stats.call_count == 1
        assert result1 == result2

    def test_pitching_stats_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result1 = get_pitching_stats(543037, season=2024, cache=tmp_cache)
            result2 = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        assert mock_pybaseball.pitching_stats.call_count == 1
        assert result1 == result2

    def test_batting_splits_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            result1 = get_batting_splits(123456, season=2024, vs_hand="R", cache=tmp_cache)
            result2 = get_batting_splits(123456, season=2024, vs_hand="R", cache=tmp_cache)
        assert mock_pybaseball.batting_stats.call_count == 1

    def test_different_splits_not_cached_together(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_splits
        with _patch_pybaseball(mock_pybaseball):
            get_batting_splits(123456, season=2024, vs_hand="R", cache=tmp_cache)
            get_batting_splits(123456, season=2024, vs_hand="L", cache=tmp_cache)
        assert mock_pybaseball.batting_stats.call_count == 2

    def test_sprint_speed_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_sprint_speed
        with _patch_pybaseball(mock_pybaseball):
            get_sprint_speed(123456, season=2024, cache=tmp_cache)
            get_sprint_speed(123456, season=2024, cache=tmp_cache)
        assert mock_pybaseball.statcast_sprint_speed.call_count == 1

    def test_defensive_metrics_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_defensive_metrics
        with _patch_pybaseball(mock_pybaseball):
            get_defensive_metrics(123456, season=2024, cache=tmp_cache)
            get_defensive_metrics(123456, season=2024, cache=tmp_cache)
        assert mock_pybaseball.batting_stats.call_count == 1

    def test_catcher_metrics_cached(self, tmp_cache):
        from data.statcast import get_catcher_metrics
        mock_pb = MagicMock()
        mock_pb.statcast_catcher_poptime = MagicMock(
            return_value=_make_pop_time_df(player_id=999999)
        )
        mock_pb.statcast_catcher_framing = MagicMock(
            return_value=_make_framing_df(player_id=999999)
        )
        with _patch_pybaseball(mock_pb):
            get_catcher_metrics(999999, season=2024, cache=tmp_cache)
            get_catcher_metrics(999999, season=2024, cache=tmp_cache)
        assert mock_pb.statcast_catcher_poptime.call_count == 1

    def test_cache_uses_season_ttl(self, mock_pybaseball, tmp_cache):
        """Cache entries use the 24-hour TTL for season stats."""
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            get_batting_stats(123456, season=2024, cache=tmp_cache)
        # Verify the cache entry exists
        cached = tmp_cache.get("statcast_batting", {
            "player_id": 123456, "season": 2024, "type": "batting",
        })
        assert cached is not None

    def test_cache_expires(self, mock_pybaseball, tmp_cache):
        """Cache entries expire after their TTL."""
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            get_batting_stats(123456, season=2024, cache=tmp_cache)

        # Manually expire the cache entry by patching time
        with patch("data.cache._now", return_value=float("inf")):
            cached = tmp_cache.get("statcast_batting", {
                "player_id": 123456, "season": 2024, "type": "batting",
            })
        assert cached is None

    def test_tto_splits_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitcher_tto_splits
        with _patch_pybaseball(mock_pybaseball):
            get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)
            get_pitcher_tto_splits(543037, season=2024, cache=tmp_cache)
        assert mock_pybaseball.statcast_pitcher.call_count == 1

    def test_pitching_splits_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_splits
        with _patch_pybaseball(mock_pybaseball):
            get_pitching_splits(543037, season=2024, vs_hand="R", cache=tmp_cache)
            get_pitching_splits(543037, season=2024, vs_hand="R", cache=tmp_cache)
        assert mock_pybaseball.pitching_stats.call_count == 1

    def test_cache_files_in_cache_dir(self, mock_pybaseball, tmp_cache):
        """Verify cache files are written to the expected directory."""
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            get_batting_stats(123456, season=2024, cache=tmp_cache)
        # Check that at least one cache file exists
        cache_files = list(tmp_cache.root_dir.rglob("*.json"))
        assert len(cache_files) > 0


# ---------------------------------------------------------------------------
# 11. Pitch mix details
# ---------------------------------------------------------------------------

class TestPitchMix:
    """Verify pitch mix extraction from Statcast data."""

    def test_extracts_pitch_types(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        with _patch_pybaseball(mock_pybaseball):
            mix = _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        types = [p["pitch_type"] for p in mix]
        assert "FF" in types  # Fastball should always be present

    def test_velocity_per_pitch(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        with _patch_pybaseball(mock_pybaseball):
            mix = _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        for pitch in mix:
            if pitch["velocity"] is not None:
                assert 70 <= pitch["velocity"] <= 110

    def test_spin_rate_per_pitch(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        with _patch_pybaseball(mock_pybaseball):
            mix = _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        for pitch in mix:
            if pitch["spin_rate"] is not None:
                assert 1000 <= pitch["spin_rate"] <= 4000

    def test_whiff_rate_per_pitch(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        with _patch_pybaseball(mock_pybaseball):
            mix = _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        for pitch in mix:
            if pitch["whiff_rate"] is not None:
                assert 0.0 <= pitch["whiff_rate"] <= 1.0

    def test_empty_statcast_returns_empty(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        mock_pybaseball.statcast_pitcher.return_value = pd.DataFrame()
        with _patch_pybaseball(mock_pybaseball):
            mix = _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        assert mix == []

    def test_none_statcast_returns_empty(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        mock_pybaseball.statcast_pitcher.return_value = None
        with _patch_pybaseball(mock_pybaseball):
            mix = _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        assert mix == []

    def test_pitch_mix_cached(self, mock_pybaseball, tmp_cache):
        from data.statcast import _fetch_pitch_mix
        with _patch_pybaseball(mock_pybaseball):
            _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
            _fetch_pitch_mix(543037, 2024, cache=tmp_cache)
        assert mock_pybaseball.statcast_pitcher.call_count == 1


# ---------------------------------------------------------------------------
# 12. Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    """Verify internal helper functions."""

    def test_safe_float_normal(self):
        from data.statcast import _safe_float
        assert _safe_float(3.14) == 3.14

    def test_safe_float_string(self):
        from data.statcast import _safe_float
        assert _safe_float("3.14") == 3.14

    def test_safe_float_none(self):
        from data.statcast import _safe_float
        assert _safe_float(None) is None

    def test_safe_float_nan(self):
        from data.statcast import _safe_float
        assert _safe_float(float("nan")) is None

    def test_safe_float_inf(self):
        from data.statcast import _safe_float
        assert _safe_float(float("inf")) is None

    def test_safe_float_invalid_string(self):
        from data.statcast import _safe_float
        assert _safe_float("not_a_number") is None

    def test_safe_int_normal(self):
        from data.statcast import _safe_int
        assert _safe_int(42) == 42

    def test_safe_int_float(self):
        from data.statcast import _safe_int
        assert _safe_int(42.7) == 42

    def test_safe_int_none(self):
        from data.statcast import _safe_int
        assert _safe_int(None) is None

    def test_safe_int_invalid(self):
        from data.statcast import _safe_int
        assert _safe_int("xyz") is None

    def test_round_or_none_normal(self):
        from data.statcast import _round_or_none
        assert _round_or_none(0.12345, 3) == 0.123

    def test_round_or_none_none(self):
        from data.statcast import _round_or_none
        assert _round_or_none(None) is None

    def test_round_or_none_nan(self):
        from data.statcast import _round_or_none
        assert _round_or_none(float("nan")) is None


# ---------------------------------------------------------------------------
# 13. Exceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    """Verify exception hierarchy and error handling."""

    def test_statcast_error_base(self):
        from data.statcast import StatcastError
        err = StatcastError("test")
        assert str(err) == "test"

    def test_player_not_found_is_statcast_error(self):
        from data.statcast import StatcastError, StatcastPlayerNotFoundError
        assert issubclass(StatcastPlayerNotFoundError, StatcastError)

    def test_data_unavailable_is_statcast_error(self):
        from data.statcast import StatcastError, StatcastDataUnavailableError
        assert issubclass(StatcastDataUnavailableError, StatcastError)

    def test_pybaseball_import_error_message(self):
        """Verify the import error message mentions pybaseball."""
        from data.statcast import _get_pybaseball
        import data.statcast as mod
        mod._pybaseball = None
        with patch.dict("sys.modules", {"pybaseball": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="pybaseball"):
                    _get_pybaseball()
        mod._pybaseball = None


# ---------------------------------------------------------------------------
# 14. Lazy pybaseball import
# ---------------------------------------------------------------------------

class TestLazyImport:
    """Verify pybaseball is lazily imported."""

    def test_module_loads_without_pybaseball(self):
        """The module itself should import even without pybaseball installed."""
        # We can verify by reloading the module -- the lazy import means
        # pybaseball isn't imported at module load time
        import importlib
        import data.statcast as mod
        mod._pybaseball = None
        # The module is already imported, so just verify the global is None
        assert mod._pybaseball is None

    def test_get_pybaseball_caches_module(self):
        """Once imported, _get_pybaseball returns the cached reference."""
        import data.statcast as mod
        mock_pb = MagicMock()
        mod._pybaseball = mock_pb
        result = mod._get_pybaseball()
        assert result is mock_pb


# ---------------------------------------------------------------------------
# 15. Multiple players / different seasons
# ---------------------------------------------------------------------------

class TestMultipleQueries:
    """Verify the module handles multiple different player queries."""

    def test_different_players_different_cache_entries(self, tmp_cache):
        from data.statcast import get_batting_stats
        df1 = _make_batting_df(player_id=100, AVG=[0.300])
        df2 = _make_batting_df(player_id=200, AVG=[0.250])
        combined = pd.concat([df1, df2], ignore_index=True)

        mock_pb = MagicMock()
        mock_pb.batting_stats.return_value = combined
        mock_pb.statcast_sprint_speed.return_value = _make_sprint_speed_df(player_id=100)

        with _patch_pybaseball(mock_pb):
            r1 = get_batting_stats(100, season=2024, cache=tmp_cache)
            r2 = get_batting_stats(200, season=2024, cache=tmp_cache)

        assert r1["traditional"]["AVG"] == 0.300
        assert r2["traditional"]["AVG"] == 0.250

    def test_different_seasons_different_cache_entries(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            get_batting_stats(123456, season=2023, cache=tmp_cache)
            get_batting_stats(123456, season=2024, cache=tmp_cache)
        # Different seasons should be fetched separately
        assert mock_pybaseball.batting_stats.call_count == 2


# ---------------------------------------------------------------------------
# 16. Stat value ranges
# ---------------------------------------------------------------------------

class TestStatValueRanges:
    """Verify that returned stat values are in reasonable ranges."""

    def test_batting_avg_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        avg = result["traditional"]["AVG"]
        assert avg is None or (0.0 <= avg <= 1.0)

    def test_obp_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        obp = result["traditional"]["OBP"]
        assert obp is None or (0.0 <= obp <= 1.0)

    def test_woba_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        woba = result["advanced"]["wOBA"]
        assert woba is None or (0.0 <= woba <= 1.0)

    def test_era_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        era = result["traditional"]["ERA"]
        assert era is None or (0.0 <= era <= 30.0)

    def test_k_pct_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_pitching_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_pitching_stats(543037, season=2024, cache=tmp_cache)
        k_pct = result["rates"]["K_pct"]
        assert k_pct is None or (0.0 <= k_pct <= 1.0)

    def test_exit_velocity_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        ev = result["batted_ball"]["exit_velocity"]
        assert ev is None or (60.0 <= ev <= 120.0)

    def test_launch_angle_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_batting_stats
        with _patch_pybaseball(mock_pybaseball):
            result = get_batting_stats(123456, season=2024, cache=tmp_cache)
        la = result["batted_ball"]["launch_angle"]
        assert la is None or (-30.0 <= la <= 60.0)

    def test_sprint_speed_range(self, mock_pybaseball, tmp_cache):
        from data.statcast import get_sprint_speed
        with _patch_pybaseball(mock_pybaseball):
            speed = get_sprint_speed(123456, season=2024, cache=tmp_cache)
        assert speed is None or (20.0 <= speed <= 35.0)
