# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0"]
# ///
"""Tests for the batter_vs_pitcher_history feature.

Validates the data layer that fetches real batter-vs-pitcher matchup
history from the MLB Stats API:
  1. Use the MLB Stats API vsPlayer endpoint to fetch career matchup data
  2. Retrieve career PA, AB, H, 2B, 3B, HR, BB, K, AVG, SLG, OPS
  3. When the matchup has fewer than 10 plate appearances, flag as small sample
  4. Cache matchup lookups in data/cache/matchups/ to avoid redundant API calls
  5. Handle cases where one or both players have no MLB history (rookies)
  6. Return a consistent JSON schema regardless of data availability
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from data.cache import Cache, TTL_MATCHUP
from data.bvp_history import (
    CACHE_ENDPOINT,
    SMALL_SAMPLE_THRESHOLD,
    BvPHistoryError,
    BvPPlayerNotFoundError,
    BvPDataUnavailableError,
    MatchupResult,
    get_matchup_history,
    get_matchup_history_safe,
    _safe_int,
    _safe_float,
)
from data.mlb_api import MLBApiError, MLBApiNotFoundError, MLBApiConnectionError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache(tmp_path):
    """Return a Cache instance backed by a temporary directory."""
    return Cache(root_dir=tmp_path / "cache")


@pytest.fixture
def sample_api_response():
    """Sample MLB Stats API BvP response with career totals and seasons."""
    return {
        "career": {
            "plateAppearances": "25",
            "atBats": "22",
            "hits": "7",
            "doubles": "2",
            "triples": "0",
            "homeRuns": "1",
            "baseOnBalls": "3",
            "strikeOuts": "6",
            "avg": ".318",
            "obp": ".400",
            "slg": ".500",
            "ops": ".900",
        },
        "seasons": [
            {
                "season": "2022",
                "stats": {
                    "plateAppearances": "10",
                    "atBats": "8",
                    "hits": "3",
                    "avg": ".375",
                    "slg": ".625",
                    "ops": "1.000",
                },
                "team": {"name": "Los Angeles Dodgers"},
                "opponent": {"name": "San Diego Padres"},
            },
            {
                "season": "2023",
                "stats": {
                    "plateAppearances": "15",
                    "atBats": "14",
                    "hits": "4",
                    "avg": ".286",
                    "slg": ".429",
                    "ops": ".786",
                },
                "team": {"name": "Los Angeles Dodgers"},
                "opponent": {"name": "San Diego Padres"},
            },
        ],
        "plate_appearances": 25,
        "small_sample": False,
        "batter": {"id": 545361, "fullName": "Mike Trout"},
        "pitcher": {"id": 434378, "fullName": "Clayton Kershaw"},
    }


@pytest.fixture
def sample_small_sample_response():
    """BvP response with fewer than 10 PA (small sample)."""
    return {
        "career": {
            "plateAppearances": "5",
            "atBats": "4",
            "hits": "1",
            "doubles": "0",
            "triples": "0",
            "homeRuns": "0",
            "baseOnBalls": "1",
            "strikeOuts": "2",
            "avg": ".250",
            "obp": ".400",
            "slg": ".250",
            "ops": ".650",
        },
        "seasons": [
            {
                "season": "2024",
                "stats": {
                    "plateAppearances": "5",
                    "atBats": "4",
                    "hits": "1",
                    "avg": ".250",
                    "slg": ".250",
                    "ops": ".650",
                },
                "team": {},
                "opponent": {},
            },
        ],
        "plate_appearances": 5,
        "small_sample": True,
        "batter": {"id": 100, "fullName": "Rookie Batter"},
        "pitcher": {"id": 200, "fullName": "Veteran Pitcher"},
    }


@pytest.fixture
def sample_no_history_response():
    """BvP response when no matchup history exists (rookies)."""
    return {
        "career": None,
        "seasons": [],
        "plate_appearances": 0,
        "small_sample": True,
        "batter": {"id": 999, "fullName": "New Rookie"},
        "pitcher": {"id": 888, "fullName": "Other Pitcher"},
    }


@pytest.fixture
def sample_exact_threshold_response():
    """BvP response with exactly 10 PA (at the threshold)."""
    return {
        "career": {
            "plateAppearances": "10",
            "atBats": "9",
            "hits": "3",
            "doubles": "1",
            "triples": "0",
            "homeRuns": "0",
            "baseOnBalls": "1",
            "strikeOuts": "2",
            "avg": ".333",
            "obp": ".400",
            "slg": ".444",
            "ops": ".844",
        },
        "seasons": [
            {
                "season": "2024",
                "stats": {
                    "plateAppearances": "10",
                    "atBats": "9",
                    "hits": "3",
                    "avg": ".333",
                    "slg": ".444",
                    "ops": ".844",
                },
                "team": {},
                "opponent": {},
            },
        ],
        "plate_appearances": 10,
        "small_sample": False,
        "batter": {"id": 300, "fullName": "Threshold Batter"},
        "pitcher": {"id": 400, "fullName": "Threshold Pitcher"},
    }


# ===========================================================================
# Step 1: Use MLB Stats API vsPlayer endpoint to fetch career matchup data
# ===========================================================================

class TestStep1FetchCareerMatchupData:
    """Tests for fetching career matchup data via the MLB Stats API."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_calls_mlb_api_get_batter_vs_pitcher(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """get_matchup_history delegates to mlb_api.get_batter_vs_pitcher."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        mock_get_bvp.assert_called_once_with(
            batter_id=545361,
            pitcher_id=434378,
            season=None,
            cache=tmp_cache,
        )

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_passes_season_filter(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Season parameter is forwarded to the API call."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, season=2023, cache=tmp_cache)
        mock_get_bvp.assert_called_once_with(
            batter_id=545361,
            pitcher_id=434378,
            season=2023,
            cache=tmp_cache,
        )

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_returns_matchup_result(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Returns a MatchupResult object."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert isinstance(result, MatchupResult)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_batter_and_pitcher_ids_stored(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Batter and pitcher IDs are stored on the result."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.batter_id == 545361
        assert result.pitcher_id == 434378

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_player_names_extracted(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Player names are extracted from the API response."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.batter_name == "Mike Trout"
        assert result.pitcher_name == "Clayton Kershaw"


# ===========================================================================
# Step 2: Retrieve career stats
# ===========================================================================

class TestStep2CareerStats:
    """Tests for career stat fields (PA, AB, H, 2B, 3B, HR, BB, K, AVG, SLG, OPS)."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_plate_appearances(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.plate_appearances == 25

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_at_bats(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.at_bats == 22

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_hits(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.hits == 7

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_doubles(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.doubles == 2

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_triples(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.triples == 0

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_home_runs(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.home_runs == 1

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_walks(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.walks == 3

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_strikeouts(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.strikeouts == 6

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_batting_average(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.batting_average == pytest.approx(0.318, abs=0.001)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_slugging(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.slugging == pytest.approx(0.500, abs=0.001)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_ops(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.ops == pytest.approx(0.900, abs=0.001)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_obp(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.obp == pytest.approx(0.400, abs=0.001)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_has_history_true_with_data(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.has_history is True

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_season_splits_present(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Season-level splits are included in the result."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert len(result.seasons) == 2
        assert result.seasons[0]["season"] == "2022"
        assert result.seasons[1]["season"] == "2023"

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_season_splits_contain_stats(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Each season split contains PA, AB, H, AVG, SLG, OPS."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        season = result.seasons[0]
        assert "plate_appearances" in season
        assert "at_bats" in season
        assert "hits" in season
        assert "batting_average" in season
        assert "slugging" in season
        assert "ops" in season


# ===========================================================================
# Step 3: Small sample flagging
# ===========================================================================

class TestStep3SmallSample:
    """Tests for small sample flagging when PA < 10."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_large_sample_not_flagged(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """25 PA is not flagged as small sample."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert result.small_sample is False

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_small_sample_flagged(
        self, mock_get_bvp, sample_small_sample_response, tmp_cache
    ):
        """5 PA is flagged as small sample."""
        mock_get_bvp.return_value = sample_small_sample_response
        result = get_matchup_history(100, 200, cache=tmp_cache)
        assert result.small_sample is True

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_zero_pa_is_small_sample(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """0 PA is flagged as small sample."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert result.small_sample is True

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_exactly_10_pa_not_small(
        self, mock_get_bvp, sample_exact_threshold_response, tmp_cache
    ):
        """Exactly 10 PA is NOT flagged as small sample (threshold is <10)."""
        mock_get_bvp.return_value = sample_exact_threshold_response
        result = get_matchup_history(300, 400, cache=tmp_cache)
        assert result.small_sample is False

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_nine_pa_is_small_sample(self, mock_get_bvp, tmp_cache):
        """9 PA is flagged as small sample."""
        response = {
            "career": {
                "plateAppearances": "9",
                "atBats": "8",
                "hits": "2",
                "doubles": "0",
                "triples": "0",
                "homeRuns": "0",
                "baseOnBalls": "1",
                "strikeOuts": "3",
                "avg": ".250",
                "obp": ".333",
                "slg": ".250",
                "ops": ".583",
            },
            "seasons": [],
            "plate_appearances": 9,
            "small_sample": True,
            "batter": {"id": 500, "fullName": "Nine PA Batter"},
            "pitcher": {"id": 600, "fullName": "Nine PA Pitcher"},
        }
        mock_get_bvp.return_value = response
        result = get_matchup_history(500, 600, cache=tmp_cache)
        assert result.small_sample is True

    def test_threshold_constant_is_10(self):
        """The small sample threshold constant is 10."""
        assert SMALL_SAMPLE_THRESHOLD == 10


# ===========================================================================
# Step 4: Cache matchup lookups in data/cache/matchups/
# ===========================================================================

class TestStep4Caching:
    """Tests for caching matchup lookups."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_result_cached_after_fetch(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """First call caches the result."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        # Verify cache has an entry under the matchups endpoint
        cached = tmp_cache.get(
            CACHE_ENDPOINT,
            {"batter_id": 545361, "pitcher_id": 434378, "season": None},
        )
        assert cached is not None
        assert cached["plate_appearances"] == 25

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_second_call_uses_cache(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Second call uses the cache, not the API."""
        mock_get_bvp.return_value = sample_api_response
        result1 = get_matchup_history(545361, 434378, cache=tmp_cache)
        result2 = get_matchup_history(545361, 434378, cache=tmp_cache)
        # API should be called only once
        mock_get_bvp.assert_called_once()
        # Both results should be equivalent
        assert result1.plate_appearances == result2.plate_appearances
        assert result1.batter_name == result2.batter_name

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_different_matchups_cached_separately(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Different batter-pitcher pairs get separate cache entries."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        get_matchup_history(123, 456, cache=tmp_cache)
        assert mock_get_bvp.call_count == 2

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_cache_uses_matchups_endpoint(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Cache entries are stored under the 'matchups' endpoint."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        # The cache directory should have a 'matchups' subdirectory
        matchups_dir = tmp_cache.root_dir / "matchups"
        assert matchups_dir.exists()
        # Should have at least one JSON file
        json_files = list(matchups_dir.glob("*.json"))
        assert len(json_files) >= 1

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_cache_entry_has_24h_ttl(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Cached entries use the 24-hour TTL."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        # Read the raw cache file to check TTL
        matchups_dir = tmp_cache.root_dir / "matchups"
        json_files = list(matchups_dir.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            entry = json.load(f)
        assert entry["ttl"] == TTL_MATCHUP
        assert TTL_MATCHUP == 86400  # 24 hours

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_season_filter_changes_cache_key(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Different season filters produce different cache entries."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        get_matchup_history(545361, 434378, season=2023, cache=tmp_cache)
        # Both calls should hit the API (different cache keys)
        assert mock_get_bvp.call_count == 2

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_cached_result_is_matchup_result(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Cached result deserializes back to a MatchupResult."""
        mock_get_bvp.return_value = sample_api_response
        get_matchup_history(545361, 434378, cache=tmp_cache)
        # Second call uses cache
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert isinstance(result, MatchupResult)
        assert result.batter_id == 545361
        assert result.pitcher_id == 434378


# ===========================================================================
# Step 5: Handle rookies with no MLB history
# ===========================================================================

class TestStep5RookieHandling:
    """Tests for handling players with no MLB matchup history."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_returns_empty_result(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """No matchup history returns an empty MatchupResult (not an error)."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert isinstance(result, MatchupResult)
        assert result.plate_appearances == 0
        assert result.has_history is False

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_rate_stats_are_none(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Rate stats are None when no history exists."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert result.batting_average is None
        assert result.slugging is None
        assert result.ops is None
        assert result.obp is None

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_counting_stats_are_zero(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Counting stats are 0 when no history exists."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert result.at_bats == 0
        assert result.hits == 0
        assert result.doubles == 0
        assert result.triples == 0
        assert result.home_runs == 0
        assert result.walks == 0
        assert result.strikeouts == 0

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_small_sample_true(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Small sample is True when no history exists."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert result.small_sample is True

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_seasons_empty(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Seasons list is empty when no history exists."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert result.seasons == []

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_player_names_preserved(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Player names are preserved even when no matchup data exists."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        assert result.batter_name == "New Rookie"
        assert result.pitcher_name == "Other Pitcher"

    def test_empty_factory_creates_empty_result(self):
        """MatchupResult.empty() creates a result with defaults."""
        result = MatchupResult.empty(111, 222, "Test Batter", "Test Pitcher")
        assert result.batter_id == 111
        assert result.pitcher_id == 222
        assert result.plate_appearances == 0
        assert result.has_history is False
        assert result.small_sample is True
        assert result.batting_average is None

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_404_raises_player_not_found(self, mock_get_bvp, tmp_cache):
        """404 from the API raises BvPPlayerNotFoundError."""
        mock_get_bvp.side_effect = MLBApiNotFoundError(
            "Not found", status_code=404
        )
        with pytest.raises(BvPPlayerNotFoundError):
            get_matchup_history(999999, 888888, cache=tmp_cache)


# ===========================================================================
# Step 6: Consistent JSON schema regardless of data availability
# ===========================================================================

class TestStep6ConsistentSchema:
    """Tests for consistent JSON schema output."""

    EXPECTED_KEYS = {
        "batter_id", "pitcher_id", "batter_name", "pitcher_name",
        "plate_appearances", "at_bats", "hits", "doubles", "triples",
        "home_runs", "walks", "strikeouts", "batting_average", "slugging",
        "ops", "obp", "small_sample", "has_history", "seasons",
    }

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_to_dict_with_history(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """to_dict() returns all expected keys with history."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        d = result.to_dict()
        assert set(d.keys()) == self.EXPECTED_KEYS

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_to_dict_without_history(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """to_dict() returns all expected keys without history."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        d = result.to_dict()
        assert set(d.keys()) == self.EXPECTED_KEYS

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_to_dict_small_sample(
        self, mock_get_bvp, sample_small_sample_response, tmp_cache
    ):
        """to_dict() returns all expected keys for small samples."""
        mock_get_bvp.return_value = sample_small_sample_response
        result = get_matchup_history(100, 200, cache=tmp_cache)
        d = result.to_dict()
        assert set(d.keys()) == self.EXPECTED_KEYS

    def test_empty_result_same_schema(self):
        """MatchupResult.empty().to_dict() has same keys."""
        d = MatchupResult.empty(1, 2).to_dict()
        assert set(d.keys()) == self.EXPECTED_KEYS

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_to_dict_is_json_serializable(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """to_dict() output can be serialized to JSON."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history(545361, 434378, cache=tmp_cache)
        # Should not raise
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)
        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["plate_appearances"] == 25

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_no_history_to_dict_is_json_serializable(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Empty result to_dict() can be serialized to JSON."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        json_str = json.dumps(result.to_dict())
        parsed = json.loads(json_str)
        assert parsed["plate_appearances"] == 0
        assert parsed["batting_average"] is None

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_none_values_present_not_omitted(
        self, mock_get_bvp, sample_no_history_response, tmp_cache
    ):
        """Rate stat fields are present with None value, not omitted."""
        mock_get_bvp.return_value = sample_no_history_response
        result = get_matchup_history(999, 888, cache=tmp_cache)
        d = result.to_dict()
        assert "batting_average" in d
        assert d["batting_average"] is None
        assert "slugging" in d
        assert d["slugging"] is None
        assert "ops" in d
        assert d["ops"] is None
        assert "obp" in d
        assert d["obp"] is None


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:
    """Tests for error handling and the safe variant."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_api_connection_error_raises(self, mock_get_bvp, tmp_cache):
        """Connection errors raise BvPDataUnavailableError."""
        mock_get_bvp.side_effect = MLBApiConnectionError("Connection refused")
        with pytest.raises(BvPDataUnavailableError):
            get_matchup_history(1, 2, cache=tmp_cache)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_generic_api_error_raises(self, mock_get_bvp, tmp_cache):
        """Generic MLB API errors raise BvPDataUnavailableError."""
        mock_get_bvp.side_effect = MLBApiError("Server error", status_code=500)
        with pytest.raises(BvPDataUnavailableError):
            get_matchup_history(1, 2, cache=tmp_cache)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_404_raises_player_not_found(self, mock_get_bvp, tmp_cache):
        """404 errors raise BvPPlayerNotFoundError."""
        mock_get_bvp.side_effect = MLBApiNotFoundError(
            "Not found", status_code=404
        )
        with pytest.raises(BvPPlayerNotFoundError):
            get_matchup_history(999, 888, cache=tmp_cache)

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_safe_variant_returns_empty_on_error(
        self, mock_get_bvp, tmp_cache
    ):
        """get_matchup_history_safe returns empty on error."""
        mock_get_bvp.side_effect = MLBApiConnectionError("Connection refused")
        result = get_matchup_history_safe(1, 2, cache=tmp_cache)
        assert isinstance(result, MatchupResult)
        assert result.plate_appearances == 0
        assert result.has_history is False

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_safe_variant_returns_data_on_success(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """get_matchup_history_safe returns data normally on success."""
        mock_get_bvp.return_value = sample_api_response
        result = get_matchup_history_safe(545361, 434378, cache=tmp_cache)
        assert result.plate_appearances == 25
        assert result.has_history is True

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_safe_variant_on_404(self, mock_get_bvp, tmp_cache):
        """get_matchup_history_safe returns empty on 404."""
        mock_get_bvp.side_effect = MLBApiNotFoundError(
            "Not found", status_code=404
        )
        result = get_matchup_history_safe(999, 888, cache=tmp_cache)
        assert isinstance(result, MatchupResult)
        assert result.plate_appearances == 0

    def test_exception_hierarchy(self):
        """All BvP exceptions inherit from BvPHistoryError."""
        assert issubclass(BvPPlayerNotFoundError, BvPHistoryError)
        assert issubclass(BvPDataUnavailableError, BvPHistoryError)


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:
    """Tests for internal helper functions."""

    def test_safe_int_normal(self):
        assert _safe_int("25") == 25

    def test_safe_int_zero(self):
        assert _safe_int("0") == 0

    def test_safe_int_none(self):
        assert _safe_int(None) == 0

    def test_safe_int_empty_string(self):
        assert _safe_int("") == 0

    def test_safe_int_invalid(self):
        assert _safe_int("abc") == 0

    def test_safe_float_normal(self):
        assert _safe_float(".318") == pytest.approx(0.318, abs=0.001)

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_dash(self):
        """'-.--' from API (no qualifying stats) returns None."""
        assert _safe_float("-.--") is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None


# ===========================================================================
# MatchupResult data class
# ===========================================================================

class TestMatchupResult:
    """Tests for the MatchupResult data class."""

    def test_from_api_response_with_history(self, sample_api_response):
        """from_api_response creates a populated MatchupResult."""
        result = MatchupResult.from_api_response(
            545361, 434378, sample_api_response
        )
        assert result.batter_id == 545361
        assert result.pitcher_id == 434378
        assert result.plate_appearances == 25
        assert result.has_history is True
        assert result.small_sample is False
        assert result.batting_average == pytest.approx(0.318, abs=0.001)

    def test_from_api_response_no_history(self, sample_no_history_response):
        """from_api_response handles empty history."""
        result = MatchupResult.from_api_response(
            999, 888, sample_no_history_response
        )
        assert result.plate_appearances == 0
        assert result.has_history is False
        assert result.small_sample is True
        assert result.batting_average is None

    def test_from_api_response_small_sample(
        self, sample_small_sample_response
    ):
        """from_api_response flags small samples."""
        result = MatchupResult.from_api_response(
            100, 200, sample_small_sample_response
        )
        assert result.plate_appearances == 5
        assert result.small_sample is True
        assert result.has_history is True

    def test_to_dict_round_trips(self, sample_api_response):
        """to_dict output can reconstruct MatchupResult via **kwargs."""
        result = MatchupResult.from_api_response(
            545361, 434378, sample_api_response
        )
        d = result.to_dict()
        reconstructed = MatchupResult(**d)
        assert reconstructed.plate_appearances == result.plate_appearances
        assert reconstructed.batter_name == result.batter_name
        assert reconstructed.has_history == result.has_history

    def test_empty_with_names(self):
        """empty() stores provided names."""
        result = MatchupResult.empty(1, 2, "Batter X", "Pitcher Y")
        assert result.batter_name == "Batter X"
        assert result.pitcher_name == "Pitcher Y"

    def test_empty_without_names(self):
        """empty() defaults names to empty strings."""
        result = MatchupResult.empty(1, 2)
        assert result.batter_name == ""
        assert result.pitcher_name == ""


# ===========================================================================
# Integration-style tests
# ===========================================================================

class TestIntegration:
    """Integration-style tests combining multiple aspects."""

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_full_workflow_fetch_then_cache(
        self, mock_get_bvp, sample_api_response, tmp_cache
    ):
        """Full workflow: fetch from API, cache, then serve from cache."""
        mock_get_bvp.return_value = sample_api_response

        # First call: fetches from API
        result1 = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert mock_get_bvp.call_count == 1
        assert result1.plate_appearances == 25
        assert result1.has_history is True

        # Second call: from cache
        result2 = get_matchup_history(545361, 434378, cache=tmp_cache)
        assert mock_get_bvp.call_count == 1  # No additional API call
        assert result2.plate_appearances == 25

        # Convert to dict for consistent schema
        d = result2.to_dict()
        assert "batter_id" in d
        assert "pitcher_id" in d
        assert d["small_sample"] is False

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_multiple_matchups_independent(
        self, mock_get_bvp, sample_api_response, sample_no_history_response,
        tmp_cache
    ):
        """Multiple different matchups are independent in the cache."""
        mock_get_bvp.side_effect = [
            sample_api_response,
            sample_no_history_response,
        ]

        result1 = get_matchup_history(545361, 434378, cache=tmp_cache)
        result2 = get_matchup_history(999, 888, cache=tmp_cache)

        assert result1.has_history is True
        assert result2.has_history is False
        assert mock_get_bvp.call_count == 2

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_error_does_not_cache(self, mock_get_bvp, tmp_cache):
        """Failed API calls are not cached."""
        mock_get_bvp.side_effect = MLBApiConnectionError("Connection failed")

        with pytest.raises(BvPDataUnavailableError):
            get_matchup_history(1, 2, cache=tmp_cache)

        # Cache should be empty
        cached = tmp_cache.get(
            CACHE_ENDPOINT,
            {"batter_id": 1, "pitcher_id": 2, "season": None},
        )
        assert cached is None

    @patch("data.bvp_history.get_batter_vs_pitcher")
    def test_safe_variant_does_not_cache_on_error(
        self, mock_get_bvp, tmp_cache
    ):
        """Safe variant does not cache error responses."""
        mock_get_bvp.side_effect = MLBApiConnectionError("Connection failed")
        result = get_matchup_history_safe(1, 2, cache=tmp_cache)
        assert result.plate_appearances == 0

        # Cache should be empty
        cached = tmp_cache.get(
            CACHE_ENDPOINT,
            {"batter_id": 1, "pitcher_id": 2, "season": None},
        )
        assert cached is None
