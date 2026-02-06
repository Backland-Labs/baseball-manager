# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0"]
# ///
"""Tests for the data_caching feature.

Validates the file-based caching layer:
  1. data/cache.py module with get/set/invalidate functions
  2. Cache is file-based (JSON files in data/cache/)
  3. Cache keys are derived from API endpoint and parameters
  4. Default TTL is 24 hours for season stats, 1 hour for live game data
  5. Cache can be cleared by deleting the cache directory
  6. All data-fetching functions check cache before making network requests
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from data.cache import (
    Cache,
    TTL_DEFAULT,
    TTL_LIVE_GAME,
    TTL_MATCHUP,
    TTL_SEASON_STATS,
    cache_clear,
    cache_get,
    cache_invalidate,
    cache_set,
    get_default_cache,
    make_key,
    set_default_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache(tmp_path):
    """Return a Cache instance backed by a temporary directory."""
    return Cache(root_dir=tmp_path / "cache")


@pytest.fixture
def populated_cache(tmp_cache):
    """Return a cache pre-populated with a few entries."""
    tmp_cache.set("season_stats", {"player_id": "p1", "season": 2024}, {"AVG": 0.300}, ttl=TTL_SEASON_STATS)
    tmp_cache.set("season_stats", {"player_id": "p2", "season": 2024}, {"AVG": 0.250}, ttl=TTL_SEASON_STATS)
    tmp_cache.set("live_game", {"game_pk": 12345}, {"inning": 5, "score": "3-2"}, ttl=TTL_LIVE_GAME)
    tmp_cache.set("matchups", {"batter": "p1", "pitcher": "p3"}, {"PA": 15, "AVG": 0.267}, ttl=TTL_MATCHUP)
    return tmp_cache


# ---------------------------------------------------------------------------
# 1. Module structure and imports
# ---------------------------------------------------------------------------

class TestModuleStructure:
    """Verify the data/cache.py module exists and exports expected symbols."""

    def test_cache_module_importable(self):
        """data.cache is importable."""
        import data.cache
        assert hasattr(data.cache, "Cache")

    def test_cache_class_exists(self):
        """Cache class exists with expected methods."""
        assert callable(Cache)
        c = Cache.__new__(Cache)
        for method in ("get", "set", "invalidate", "clear", "has", "stats"):
            assert hasattr(c, method), f"Cache missing method: {method}"

    def test_ttl_constants_defined(self):
        """TTL constants are defined with expected values."""
        assert TTL_SEASON_STATS == 86_400
        assert TTL_LIVE_GAME == 3_600
        assert TTL_MATCHUP == 86_400
        assert TTL_DEFAULT == 86_400

    def test_convenience_functions_exist(self):
        """Module-level convenience functions are importable."""
        from data.cache import cache_get, cache_set, cache_invalidate, cache_clear
        assert callable(cache_get)
        assert callable(cache_set)
        assert callable(cache_invalidate)
        assert callable(cache_clear)


# ---------------------------------------------------------------------------
# 2. Cache key generation
# ---------------------------------------------------------------------------

class TestMakeKey:
    """Verify deterministic cache key generation."""

    def test_key_is_hex_string(self):
        key = make_key("season_stats", {"player_id": "123"})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_inputs_same_key(self):
        """Identical endpoint+params always produce the same key."""
        k1 = make_key("season_stats", {"player_id": "123", "season": 2024})
        k2 = make_key("season_stats", {"player_id": "123", "season": 2024})
        assert k1 == k2

    def test_different_params_different_key(self):
        k1 = make_key("season_stats", {"player_id": "123"})
        k2 = make_key("season_stats", {"player_id": "456"})
        assert k1 != k2

    def test_different_endpoints_different_key(self):
        k1 = make_key("season_stats", {"player_id": "123"})
        k2 = make_key("live_game", {"player_id": "123"})
        assert k1 != k2

    def test_param_order_does_not_matter(self):
        """Dict ordering should not affect the cache key."""
        k1 = make_key("x", {"a": 1, "b": 2})
        k2 = make_key("x", {"b": 2, "a": 1})
        assert k1 == k2

    def test_none_params_gives_consistent_key(self):
        k1 = make_key("endpoint")
        k2 = make_key("endpoint", None)
        k3 = make_key("endpoint", {})
        assert k1 == k2 == k3


# ---------------------------------------------------------------------------
# 3. Basic get / set / invalidate
# ---------------------------------------------------------------------------

class TestGetSetInvalidate:
    """Core cache operations."""

    def test_get_returns_none_on_miss(self, tmp_cache):
        assert tmp_cache.get("nonexistent", {"x": 1}) is None

    def test_set_and_get_round_trip(self, tmp_cache):
        payload = {"AVG": 0.300, "OBP": 0.380}
        tmp_cache.set("season_stats", {"player_id": "p1"}, payload)
        result = tmp_cache.get("season_stats", {"player_id": "p1"})
        assert result == payload

    def test_set_returns_path(self, tmp_cache):
        path = tmp_cache.set("ep", {"k": "v"}, {"data": 1})
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_set_creates_directory_structure(self, tmp_cache):
        tmp_cache.set("season_stats", {"player_id": "p1"}, {"x": 1})
        ep_dir = tmp_cache.root_dir / "season_stats"
        assert ep_dir.is_dir()

    def test_set_overwrites_existing_entry(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, {"old": True})
        tmp_cache.set("ep", {"k": 1}, {"new": True})
        result = tmp_cache.get("ep", {"k": 1})
        assert result == {"new": True}

    def test_invalidate_removes_entry(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, {"data": 1})
        assert tmp_cache.get("ep", {"k": 1}) is not None
        removed = tmp_cache.invalidate("ep", {"k": 1})
        assert removed is True
        assert tmp_cache.get("ep", {"k": 1}) is None

    def test_invalidate_nonexistent_returns_false(self, tmp_cache):
        assert tmp_cache.invalidate("nonexistent", {"k": 1}) is False

    def test_get_with_complex_payload(self, tmp_cache):
        """Cache handles nested dicts, lists, numbers, nulls, booleans."""
        payload = {
            "stats": [1, 2, 3],
            "nested": {"deep": {"value": None}},
            "flag": True,
            "rate": 0.123,
        }
        tmp_cache.set("ep", {"k": 1}, payload)
        assert tmp_cache.get("ep", {"k": 1}) == payload

    def test_set_with_none_data(self, tmp_cache):
        """Caching None as the data value works."""
        tmp_cache.set("ep", {"k": 1}, None)
        # get returns None for both miss and None-data, so use has()
        assert tmp_cache.has("ep", {"k": 1}) is True

    def test_set_with_string_data(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, "hello world")
        assert tmp_cache.get("ep", {"k": 1}) == "hello world"

    def test_set_with_list_data(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, [1, 2, 3])
        assert tmp_cache.get("ep", {"k": 1}) == [1, 2, 3]


# ---------------------------------------------------------------------------
# 4. TTL and expiry
# ---------------------------------------------------------------------------

class TestTTLAndExpiry:
    """Verify that cache entries expire after their TTL."""

    def test_entry_available_before_ttl(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, {"alive": True}, ttl=60)
        assert tmp_cache.get("ep", {"k": 1}) == {"alive": True}

    def test_entry_expires_after_ttl(self, tmp_cache):
        """Entry with a very short TTL should be treated as expired."""
        tmp_cache.set("ep", {"k": 1}, {"alive": True}, ttl=1)
        # Patch _now to simulate time passing
        with patch("data.cache._now", return_value=time.time() + 2):
            result = tmp_cache.get("ep", {"k": 1})
        assert result is None

    def test_expired_entry_is_cleaned_up(self, tmp_cache):
        """Expired entry file is deleted on access."""
        path = tmp_cache.set("ep", {"k": 1}, {"data": 1}, ttl=1)
        assert path.exists()
        with patch("data.cache._now", return_value=time.time() + 2):
            tmp_cache.get("ep", {"k": 1})
        assert not path.exists()

    def test_has_returns_false_for_expired(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, {"data": 1}, ttl=1)
        with patch("data.cache._now", return_value=time.time() + 2):
            assert tmp_cache.has("ep", {"k": 1}) is False

    def test_season_stats_ttl_is_24h(self):
        assert TTL_SEASON_STATS == 24 * 60 * 60

    def test_live_game_ttl_is_1h(self):
        assert TTL_LIVE_GAME == 1 * 60 * 60

    def test_default_ttl_applied(self, tmp_cache):
        """When no TTL is specified, the default (24h) is used."""
        path = tmp_cache.set("ep", {"k": 1}, {"data": 1})
        with open(path) as f:
            entry = json.load(f)
        assert entry["ttl"] == TTL_DEFAULT

    def test_custom_ttl_stored(self, tmp_cache):
        path = tmp_cache.set("ep", {"k": 1}, {"data": 1}, ttl=120)
        with open(path) as f:
            entry = json.load(f)
        assert entry["ttl"] == 120

    def test_entry_just_before_expiry_still_valid(self, tmp_cache):
        """Entry accessed at exactly TTL-1 second should still be valid."""
        now = time.time()
        with patch("data.cache._now", return_value=now):
            tmp_cache.set("ep", {"k": 1}, {"fresh": True}, ttl=100)
        with patch("data.cache._now", return_value=now + 99):
            assert tmp_cache.get("ep", {"k": 1}) == {"fresh": True}

    def test_entry_at_exact_ttl_is_expired(self, tmp_cache):
        """Entry accessed at exactly TTL+epsilon should be expired."""
        now = time.time()
        with patch("data.cache._now", return_value=now):
            tmp_cache.set("ep", {"k": 1}, {"stale": True}, ttl=100)
        with patch("data.cache._now", return_value=now + 100.001):
            assert tmp_cache.get("ep", {"k": 1}) is None


# ---------------------------------------------------------------------------
# 5. Clear (wipe entire cache)
# ---------------------------------------------------------------------------

class TestClear:
    """Verify clearing the entire cache."""

    def test_clear_removes_all_entries(self, populated_cache):
        count = populated_cache.clear()
        assert count == 4  # 4 entries were populated
        assert not populated_cache.root_dir.exists()

    def test_clear_on_empty_cache(self, tmp_cache):
        count = tmp_cache.clear()
        assert count == 0

    def test_clear_returns_file_count(self, tmp_cache):
        for i in range(5):
            tmp_cache.set("ep", {"i": i}, {"n": i})
        count = tmp_cache.clear()
        assert count == 5

    def test_cache_usable_after_clear(self, populated_cache):
        populated_cache.clear()
        populated_cache.set("new_ep", {"k": 1}, {"data": "after_clear"})
        assert populated_cache.get("new_ep", {"k": 1}) == {"data": "after_clear"}


# ---------------------------------------------------------------------------
# 6. File-based storage verification
# ---------------------------------------------------------------------------

class TestFileBasedStorage:
    """Verify cache entries are stored as JSON files."""

    def test_cache_file_is_valid_json(self, tmp_cache):
        path = tmp_cache.set("ep", {"k": 1}, {"value": 42})
        with open(path) as f:
            entry = json.load(f)
        assert "created" in entry
        assert "ttl" in entry
        assert "data" in entry
        assert entry["data"] == {"value": 42}

    def test_cache_files_in_endpoint_subdirectory(self, tmp_cache):
        tmp_cache.set("season_stats", {"p": 1}, {"x": 1})
        tmp_cache.set("live_game", {"g": 1}, {"y": 2})
        assert (tmp_cache.root_dir / "season_stats").is_dir()
        assert (tmp_cache.root_dir / "live_game").is_dir()

    def test_cache_file_name_is_sha256(self, tmp_cache):
        path = tmp_cache.set("ep", {"k": 1}, {"data": 1})
        stem = path.stem
        assert len(stem) == 64
        assert all(c in "0123456789abcdef" for c in stem)

    def test_corrupted_file_treated_as_miss(self, tmp_cache):
        """If a cache file contains invalid JSON, treat as miss."""
        path = tmp_cache.set("ep", {"k": 1}, {"data": 1})
        with open(path, "w") as f:
            f.write("NOT JSON{{{")
        assert tmp_cache.get("ep", {"k": 1}) is None
        # File should be cleaned up
        assert not path.exists()

    def test_corrupted_file_cleaned_by_has(self, tmp_cache):
        path = tmp_cache.set("ep", {"k": 1}, {"data": 1})
        with open(path, "w") as f:
            f.write("garbage")
        assert tmp_cache.has("ep", {"k": 1}) is False
        assert not path.exists()


# ---------------------------------------------------------------------------
# 7. has() method
# ---------------------------------------------------------------------------

class TestHas:
    """Verify the has() convenience method."""

    def test_has_returns_true_for_valid_entry(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, {"data": 1})
        assert tmp_cache.has("ep", {"k": 1}) is True

    def test_has_returns_false_for_missing(self, tmp_cache):
        assert tmp_cache.has("ep", {"k": 1}) is False

    def test_has_returns_false_for_expired(self, tmp_cache):
        tmp_cache.set("ep", {"k": 1}, {"data": 1}, ttl=1)
        with patch("data.cache._now", return_value=time.time() + 2):
            assert tmp_cache.has("ep", {"k": 1}) is False


# ---------------------------------------------------------------------------
# 8. stats() method
# ---------------------------------------------------------------------------

class TestStats:
    """Verify cache statistics."""

    def test_stats_empty_cache(self, tmp_cache):
        s = tmp_cache.stats()
        assert s["files"] == 0
        assert s["size_bytes"] == 0

    def test_stats_counts_files(self, populated_cache):
        s = populated_cache.stats()
        assert s["files"] == 4

    def test_stats_size_positive(self, populated_cache):
        s = populated_cache.stats()
        assert s["size_bytes"] > 0


# ---------------------------------------------------------------------------
# 9. Default cache and convenience functions
# ---------------------------------------------------------------------------

class TestDefaultCacheAndConvenience:
    """Verify module-level default cache and convenience functions."""

    def test_get_default_cache_returns_cache_instance(self):
        c = get_default_cache()
        assert isinstance(c, Cache)

    def test_set_default_cache_overrides(self, tmp_cache):
        old = get_default_cache()
        try:
            set_default_cache(tmp_cache)
            assert get_default_cache() is tmp_cache
        finally:
            set_default_cache(old)

    def test_convenience_set_and_get(self, tmp_cache):
        old = get_default_cache()
        try:
            set_default_cache(tmp_cache)
            cache_set("ep", {"k": 1}, {"val": 99})
            assert cache_get("ep", {"k": 1}) == {"val": 99}
        finally:
            set_default_cache(old)

    def test_convenience_invalidate(self, tmp_cache):
        old = get_default_cache()
        try:
            set_default_cache(tmp_cache)
            cache_set("ep", {"k": 1}, {"val": 99})
            assert cache_invalidate("ep", {"k": 1}) is True
            assert cache_get("ep", {"k": 1}) is None
        finally:
            set_default_cache(old)

    def test_convenience_clear(self, tmp_cache):
        old = get_default_cache()
        try:
            set_default_cache(tmp_cache)
            cache_set("ep", {"k": 1}, {"v": 1})
            cache_set("ep", {"k": 2}, {"v": 2})
            count = cache_clear()
            assert count == 2
        finally:
            set_default_cache(old)


# ---------------------------------------------------------------------------
# 10. Cache key derivation from endpoint + params
# ---------------------------------------------------------------------------

class TestKeyDerivation:
    """Verify that cache keys are derived from endpoint and parameters."""

    def test_player_id_in_key(self, tmp_cache):
        """Different player IDs produce different cache entries."""
        tmp_cache.set("season_stats", {"player_id": "p1"}, {"AVG": 0.300})
        tmp_cache.set("season_stats", {"player_id": "p2"}, {"AVG": 0.250})
        assert tmp_cache.get("season_stats", {"player_id": "p1"}) == {"AVG": 0.300}
        assert tmp_cache.get("season_stats", {"player_id": "p2"}) == {"AVG": 0.250}

    def test_season_in_key(self, tmp_cache):
        """Different seasons produce different cache entries."""
        tmp_cache.set("season_stats", {"player_id": "p1", "season": 2023}, {"AVG": 0.280})
        tmp_cache.set("season_stats", {"player_id": "p1", "season": 2024}, {"AVG": 0.300})
        assert tmp_cache.get("season_stats", {"player_id": "p1", "season": 2023}) == {"AVG": 0.280}
        assert tmp_cache.get("season_stats", {"player_id": "p1", "season": 2024}) == {"AVG": 0.300}

    def test_split_type_in_key(self, tmp_cache):
        """Different split types produce different cache entries."""
        tmp_cache.set("season_stats", {"player_id": "p1", "vs_hand": "L"}, {"AVG": 0.310})
        tmp_cache.set("season_stats", {"player_id": "p1", "vs_hand": "R"}, {"AVG": 0.280})
        assert tmp_cache.get("season_stats", {"player_id": "p1", "vs_hand": "L"}) == {"AVG": 0.310}
        assert tmp_cache.get("season_stats", {"player_id": "p1", "vs_hand": "R"}) == {"AVG": 0.280}


# ---------------------------------------------------------------------------
# 11. Multiple endpoints coexist
# ---------------------------------------------------------------------------

class TestMultipleEndpoints:
    """Verify that entries from different endpoints are isolated."""

    def test_different_endpoints_same_params(self, tmp_cache):
        tmp_cache.set("season_stats", {"id": 1}, {"type": "season"})
        tmp_cache.set("live_game", {"id": 1}, {"type": "live"})
        assert tmp_cache.get("season_stats", {"id": 1}) == {"type": "season"}
        assert tmp_cache.get("live_game", {"id": 1}) == {"type": "live"}

    def test_invalidate_one_endpoint_does_not_affect_another(self, tmp_cache):
        tmp_cache.set("season_stats", {"id": 1}, {"type": "season"})
        tmp_cache.set("live_game", {"id": 1}, {"type": "live"})
        tmp_cache.invalidate("season_stats", {"id": 1})
        assert tmp_cache.get("season_stats", {"id": 1}) is None
        assert tmp_cache.get("live_game", {"id": 1}) == {"type": "live"}


# ---------------------------------------------------------------------------
# 12. Custom cache directory
# ---------------------------------------------------------------------------

class TestCustomDirectory:
    """Verify cache works with custom root directories."""

    def test_custom_root_dir(self, tmp_path):
        custom_dir = tmp_path / "custom_cache"
        cache = Cache(root_dir=custom_dir)
        cache.set("ep", {"k": 1}, {"data": 1})
        assert custom_dir.exists()
        assert cache.get("ep", {"k": 1}) == {"data": 1}

    def test_string_root_dir(self, tmp_path):
        cache = Cache(root_dir=str(tmp_path / "str_cache"))
        cache.set("ep", {"k": 1}, {"data": 1})
        assert cache.get("ep", {"k": 1}) == {"data": 1}

    def test_default_root_dir_is_data_cache(self):
        cache = Cache()
        expected = Path(__file__).resolve().parent.parent / "data" / "cache"
        assert cache.root_dir == expected


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and robustness."""

    def test_empty_endpoint_name(self, tmp_cache):
        """Empty string as endpoint still works."""
        tmp_cache.set("", {"k": 1}, {"data": 1})
        assert tmp_cache.get("", {"k": 1}) == {"data": 1}

    def test_large_payload(self, tmp_cache):
        """Cache handles large payloads."""
        big = {"items": list(range(10_000))}
        tmp_cache.set("ep", {"k": 1}, big)
        assert tmp_cache.get("ep", {"k": 1}) == big

    def test_unicode_in_params(self, tmp_cache):
        tmp_cache.set("ep", {"name": "Shohei Ohtani"}, {"data": 1})
        assert tmp_cache.get("ep", {"name": "Shohei Ohtani"}) == {"data": 1}

    def test_numeric_params(self, tmp_cache):
        tmp_cache.set("ep", {"id": 12345, "rate": 0.95}, {"ok": True})
        assert tmp_cache.get("ep", {"id": 12345, "rate": 0.95}) == {"ok": True}

    def test_nested_params(self, tmp_cache):
        params = {"player": {"id": 1, "team": "NYY"}}
        tmp_cache.set("ep", params, {"result": "ok"})
        assert tmp_cache.get("ep", params) == {"result": "ok"}

    def test_concurrent_set_same_key(self, tmp_cache):
        """Multiple sets to the same key -- last write wins."""
        tmp_cache.set("ep", {"k": 1}, {"v": 1})
        tmp_cache.set("ep", {"k": 1}, {"v": 2})
        tmp_cache.set("ep", {"k": 1}, {"v": 3})
        assert tmp_cache.get("ep", {"k": 1}) == {"v": 3}


# ---------------------------------------------------------------------------
# 14. Integration: cache keys match real usage patterns
# ---------------------------------------------------------------------------

class TestRealUsagePatterns:
    """Verify cache works for the data-fetching patterns described in DESIGN.md."""

    def test_season_stats_caching(self, tmp_cache):
        """Season stats cached with 24-hour TTL."""
        stats = {
            "AVG": 0.300, "OBP": 0.380, "SLG": 0.520, "OPS": 0.900,
            "wOBA": 0.370, "wRC_plus": 145, "barrel_rate": 0.12,
        }
        tmp_cache.set("season_stats", {"player_id": "ohtani17", "season": 2024}, stats, ttl=TTL_SEASON_STATS)
        cached = tmp_cache.get("season_stats", {"player_id": "ohtani17", "season": 2024})
        assert cached == stats

    def test_live_game_caching_with_short_ttl(self, tmp_cache):
        """Live game data cached with 1-hour TTL."""
        game_data = {"inning": 7, "score": {"home": 3, "away": 2}, "outs": 1}
        tmp_cache.set("live_game", {"game_pk": 718765}, game_data, ttl=TTL_LIVE_GAME)
        cached = tmp_cache.get("live_game", {"game_pk": 718765})
        assert cached == game_data

    def test_matchup_caching(self, tmp_cache):
        """Batter-vs-pitcher matchup cached for 24 hours."""
        matchup = {"PA": 25, "AVG": 0.320, "HR": 2, "K": 5}
        tmp_cache.set("matchups", {"batter": "judge99", "pitcher": "cole45"}, matchup, ttl=TTL_MATCHUP)
        cached = tmp_cache.get("matchups", {"batter": "judge99", "pitcher": "cole45"})
        assert cached == matchup

    def test_roster_caching(self, tmp_cache):
        """Team roster info cached."""
        roster = {"team": "NYY", "players": [{"id": "p1", "name": "Judge"}]}
        tmp_cache.set("roster", {"team_id": 147}, roster, ttl=TTL_SEASON_STATS)
        assert tmp_cache.get("roster", {"team_id": 147}) == roster

    def test_cache_miss_triggers_fetch_pattern(self, tmp_cache):
        """Demonstrate the check-cache-first pattern."""
        result = tmp_cache.get("season_stats", {"player_id": "new_player"})
        assert result is None  # cache miss
        # In real code, this would trigger a network fetch
        fetched_data = {"AVG": 0.275}
        tmp_cache.set("season_stats", {"player_id": "new_player"}, fetched_data)
        # Now it's cached
        assert tmp_cache.get("season_stats", {"player_id": "new_player"}) == fetched_data
