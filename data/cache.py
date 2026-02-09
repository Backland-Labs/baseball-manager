# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""File-based caching layer for API responses.

Provides get/set/invalidate functions backed by JSON files in a cache
directory.  Cache keys are derived from the API endpoint and parameters.
Each entry has a TTL (time-to-live) that defaults to 24 hours for season
stats and 1 hour for live game data.

Usage::

    from data.cache import Cache

    cache = Cache()                        # uses default data/cache/ dir
    cache = Cache("/tmp/my_cache")         # custom directory

    cache.set("season_stats", {"player": 123, "season": 2024}, payload, ttl=86400)
    result = cache.get("season_stats", {"player": 123, "season": 2024})
    cache.invalidate("season_stats", {"player": 123, "season": 2024})
    cache.clear()                          # wipe entire cache

The module also exposes convenience constants for common TTL values and a
``make_key`` helper for deterministic cache key generation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------

TTL_SEASON_STATS: int = 86_400       # 24 hours
TTL_LIVE_GAME: int = 3_600           # 1 hour
TTL_MATCHUP: int = 86_400            # 24 hours
TTL_DEFAULT: int = 86_400            # 24 hours


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def make_key(endpoint: str, params: dict[str, Any] | None = None) -> str:
    """Build a deterministic cache key from an endpoint name and parameters.

    The key is a SHA-256 hex digest of the canonicalised JSON representation
    of ``(endpoint, sorted-params)``.  This guarantees the same inputs always
    map to the same file regardless of dict ordering.

    Args:
        endpoint: Logical name of the data source (e.g. ``"season_stats"``).
        params: Optional dict of query parameters (player ID, season, etc.).

    Returns:
        A 64-character hex string suitable for use as a filename.
    """
    canonical = json.dumps(
        {"endpoint": endpoint, "params": params or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Cache entry on-disk format
# ---------------------------------------------------------------------------

def _now() -> float:
    """Return current UTC timestamp as a float."""
    return time.time()


# ---------------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------------

class Cache:
    """File-based JSON cache with per-entry TTL.

    Each cached value is stored as a JSON file inside *root_dir*.  The file
    contains ``{"created": <timestamp>, "ttl": <seconds>, "data": <payload>}``.
    A value is considered expired when ``now - created > ttl``.

    Args:
        root_dir: Path to the cache directory.  Created on first write.
            Defaults to ``data/cache/`` relative to this file's parent.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            root_dir = Path(__file__).resolve().parent / "cache"
        self._root = Path(root_dir)

    # -- public API --------------------------------------------------------

    @property
    def root_dir(self) -> Path:
        """Return the root cache directory path."""
        return self._root

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any | None:
        """Retrieve a cached value, or ``None`` if missing / expired.

        Args:
            endpoint: Logical data-source name.
            params: Query parameters used to build the cache key.

        Returns:
            The cached payload (deserialised from JSON), or ``None``.
        """
        path = self._path_for(endpoint, params)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupted entry -- treat as cache miss and clean up.
            path.unlink(missing_ok=True)
            return None

        created: float = entry.get("created", 0)
        ttl: float = entry.get("ttl", TTL_DEFAULT)
        if _now() - created > ttl:
            # Expired -- remove stale file and return miss.
            path.unlink(missing_ok=True)
            return None

        return entry.get("data")

    def set(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: Any = None,
        ttl: int = TTL_DEFAULT,
    ) -> Path:
        """Store *data* in the cache.

        Args:
            endpoint: Logical data-source name.
            params: Query parameters used to build the cache key.
            data: Arbitrary JSON-serialisable payload.
            ttl: Time-to-live in seconds.  Defaults to :data:`TTL_DEFAULT`.

        Returns:
            The :class:`Path` to the written cache file.
        """
        path = self._path_for(endpoint, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "created": _now(),
            "ttl": ttl,
            "data": data,
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(entry, f, separators=(",", ":"))
        tmp_path.replace(path)  # atomic rename
        return path

    def invalidate(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Remove a single cached entry.

        Args:
            endpoint: Logical data-source name.
            params: Query parameters used to build the cache key.

        Returns:
            ``True`` if an entry was removed, ``False`` if it did not exist.
        """
        path = self._path_for(endpoint, params)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Delete **all** cached entries by removing the cache directory.

        Returns:
            The number of cache files that were removed.
        """
        if not self._root.exists():
            return 0
        count = sum(1 for _ in self._root.rglob("*.json"))
        shutil.rmtree(self._root)
        return count

    def has(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Check whether a non-expired entry exists for the given key.

        This performs the same expiry check as :meth:`get` but does not
        deserialise the payload.
        """
        path = self._path_for(endpoint, params)
        if not path.exists():
            return False
        try:
            with open(path) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
            return False
        created: float = entry.get("created", 0)
        ttl: float = entry.get("ttl", TTL_DEFAULT)
        if _now() - created > ttl:
            path.unlink(missing_ok=True)
            return False
        return True

    def stats(self) -> dict[str, int]:
        """Return basic statistics about the cache directory.

        Returns:
            A dict with ``"files"`` (total JSON files) and ``"size_bytes"``
            (total size on disk).
        """
        if not self._root.exists():
            return {"files": 0, "size_bytes": 0}
        files = list(self._root.rglob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {"files": len(files), "size_bytes": total_size}

    # -- helpers -----------------------------------------------------------

    def _path_for(
        self,
        endpoint: str,
        params: dict[str, Any] | None,
    ) -> Path:
        """Return the filesystem path for a cache entry.

        Entries are stored under a subdirectory named after the endpoint to
        keep the cache directory organised::

            <root>/season_stats/<sha256>.json
            <root>/live_game/<sha256>.json
        """
        key = make_key(endpoint, params)
        return self._root / endpoint / f"{key}.json"


# ---------------------------------------------------------------------------
# Module-level default cache instance
# ---------------------------------------------------------------------------

_default_cache: Cache | None = None


def get_default_cache() -> Cache:
    """Return (and lazily create) the module-level default :class:`Cache`."""
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache()
    return _default_cache


def set_default_cache(cache: Cache) -> None:
    """Override the module-level default cache (useful for testing)."""
    global _default_cache
    _default_cache = cache


# ---------------------------------------------------------------------------
# Convenience functions using the default cache
# ---------------------------------------------------------------------------

def cache_get(endpoint: str, params: dict[str, Any] | None = None) -> Any | None:
    """Shortcut for ``get_default_cache().get(endpoint, params)``."""
    return get_default_cache().get(endpoint, params)


def cache_set(
    endpoint: str,
    params: dict[str, Any] | None = None,
    data: Any = None,
    ttl: int = TTL_DEFAULT,
) -> Path:
    """Shortcut for ``get_default_cache().set(endpoint, params, data, ttl)``."""
    return get_default_cache().set(endpoint, params, data, ttl)


def cache_invalidate(endpoint: str, params: dict[str, Any] | None = None) -> bool:
    """Shortcut for ``get_default_cache().invalidate(endpoint, params)``."""
    return get_default_cache().invalidate(endpoint, params)


def cache_clear() -> int:
    """Shortcut for ``get_default_cache().clear()``."""
    return get_default_cache().clear()
