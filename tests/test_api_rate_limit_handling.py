# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.0", "anthropic>=0.78.0"]
# ///
"""Tests for the api_rate_limit_handling feature.

Verifies all feature requirements from features.json:
1. Claude API rate limit (429) responses trigger exponential backoff retry
   with jitter
2. MLB Stats API rate limits are handled with backoff and reduced polling
   frequency
3. pybaseball/FanGraphs rate limits are mitigated by the data caching layer
4. A maximum retry count is enforced to avoid infinite retry loops
5. Rate limit events are logged with timestamps
6. The system respects Retry-After headers when provided
"""

import http.client
import io
import json
import logging
import sys
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===========================================================================
# 1. MLB Stats API 429 rate limit handling
# ===========================================================================

def test_mlb_api_fetch_json_429_retries():
    """_fetch_json should retry on 429 responses with backoff."""
    from data.mlb_api import _fetch_json, MLBApiRateLimitError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            # Simulate 429
            exc = urllib.error.HTTPError(
                url="http://test.com",
                code=429,
                msg="Too Many Requests",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b"rate limited"),
            )
            # Mock headers
            exc.headers = {"Retry-After": "1"}
            raise exc
        # Third attempt succeeds
        resp = MagicMock()
        resp.read.return_value = b'{"data": "ok"}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep") as mock_sleep:
        result = _fetch_json("http://test.com/api", max_retries=5)

    assert result == {"data": "ok"}
    assert call_count == 3
    # Should have slept twice (after attempt 1 and 2)
    assert mock_sleep.call_count == 2
    print("  test_mlb_api_fetch_json_429_retries: PASSED")


def test_mlb_api_429_raises_after_max_retries():
    """_fetch_json should raise MLBApiRateLimitError after exhausting retries."""
    from data.mlb_api import _fetch_json, MLBApiRateLimitError

    def mock_urlopen(req, timeout=None):
        exc = urllib.error.HTTPError(
            url="http://test.com",
            code=429,
            msg="Too Many Requests",
            hdrs=http.client.HTTPResponse,
            fp=io.BytesIO(b"rate limited"),
        )
        exc.headers = {}
        raise exc

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=3)
            assert False, "Should have raised MLBApiRateLimitError"
        except MLBApiRateLimitError as e:
            assert e.status_code == 429
            assert "429" in str(e)

    print("  test_mlb_api_429_raises_after_max_retries: PASSED")


def test_mlb_api_429_respects_retry_after_header():
    """_fetch_json should sleep at least as long as Retry-After header says."""
    from data.mlb_api import _fetch_json

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            exc = urllib.error.HTTPError(
                url="http://test.com",
                code=429,
                msg="Too Many Requests",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b"rate limited"),
            )
            exc.headers = {"Retry-After": "30"}
            raise exc
        resp = MagicMock()
        resp.read.return_value = b'{"data": "ok"}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep") as mock_sleep:
        result = _fetch_json("http://test.com/api", max_retries=3)

    assert result == {"data": "ok"}
    # The sleep should be at least 30 seconds (Retry-After value)
    sleep_arg = mock_sleep.call_args[0][0]
    assert sleep_arg >= 30.0, f"Expected sleep >= 30, got {sleep_arg}"
    print("  test_mlb_api_429_respects_retry_after_header: PASSED")


def test_mlb_api_429_retry_after_not_set():
    """When Retry-After header is missing, backoff uses computed delay."""
    from data.mlb_api import _fetch_json

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            exc = urllib.error.HTTPError(
                url="http://test.com",
                code=429,
                msg="Too Many Requests",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b"rate limited"),
            )
            exc.headers = {}  # No Retry-After
            raise exc
        resp = MagicMock()
        resp.read.return_value = b'{"data": "ok"}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep") as mock_sleep:
        result = _fetch_json("http://test.com/api", max_retries=3)

    assert result == {"data": "ok"}
    # Backoff should be computed (RETRY_BACKOFF_BASE * 2^0 + jitter)
    sleep_arg = mock_sleep.call_args[0][0]
    assert sleep_arg > 0, f"Expected positive sleep, got {sleep_arg}"
    print("  test_mlb_api_429_retry_after_not_set: PASSED")


def test_parse_retry_after_integer():
    """_parse_retry_after should parse integer Retry-After values."""
    from data.mlb_api import _parse_retry_after

    exc = urllib.error.HTTPError(
        url="http://test.com",
        code=429,
        msg="Too Many Requests",
        hdrs=http.client.HTTPResponse,
        fp=io.BytesIO(b""),
    )
    exc.headers = {"Retry-After": "42"}
    assert _parse_retry_after(exc) == 42.0
    print("  test_parse_retry_after_integer: PASSED")


def test_parse_retry_after_float():
    """_parse_retry_after should parse float Retry-After values."""
    from data.mlb_api import _parse_retry_after

    exc = urllib.error.HTTPError(
        url="http://test.com",
        code=429,
        msg="Too Many Requests",
        hdrs=http.client.HTTPResponse,
        fp=io.BytesIO(b""),
    )
    exc.headers = {"Retry-After": "2.5"}
    assert _parse_retry_after(exc) == 2.5
    print("  test_parse_retry_after_float: PASSED")


def test_parse_retry_after_missing():
    """_parse_retry_after should return None when header is absent."""
    from data.mlb_api import _parse_retry_after

    exc = urllib.error.HTTPError(
        url="http://test.com",
        code=429,
        msg="Too Many Requests",
        hdrs=http.client.HTTPResponse,
        fp=io.BytesIO(b""),
    )
    exc.headers = {}
    assert _parse_retry_after(exc) is None
    print("  test_parse_retry_after_missing: PASSED")


def test_parse_retry_after_invalid():
    """_parse_retry_after should return None for non-numeric values."""
    from data.mlb_api import _parse_retry_after

    exc = urllib.error.HTTPError(
        url="http://test.com",
        code=429,
        msg="Too Many Requests",
        hdrs=http.client.HTTPResponse,
        fp=io.BytesIO(b""),
    )
    exc.headers = {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"}
    assert _parse_retry_after(exc) is None
    print("  test_parse_retry_after_invalid: PASSED")


def test_parse_retry_after_negative_clamped():
    """_parse_retry_after should clamp negative values to 0."""
    from data.mlb_api import _parse_retry_after

    exc = urllib.error.HTTPError(
        url="http://test.com",
        code=429,
        msg="Too Many Requests",
        hdrs=http.client.HTTPResponse,
        fp=io.BytesIO(b""),
    )
    exc.headers = {"Retry-After": "-5"}
    assert _parse_retry_after(exc) == 0.0
    print("  test_parse_retry_after_negative_clamped: PASSED")


def test_mlb_api_rate_limit_error_attributes():
    """MLBApiRateLimitError should carry retry_after and status_code."""
    from data.mlb_api import MLBApiRateLimitError

    err = MLBApiRateLimitError(
        "Rate limited", retry_after=30.0, url="http://example.com"
    )
    assert err.status_code == 429
    assert err.retry_after == 30.0
    assert err.url == "http://example.com"
    assert "Rate limited" in str(err)
    print("  test_mlb_api_rate_limit_error_attributes: PASSED")


def test_mlb_api_rate_limit_error_no_retry_after():
    """MLBApiRateLimitError should work with retry_after=None."""
    from data.mlb_api import MLBApiRateLimitError

    err = MLBApiRateLimitError("Rate limited")
    assert err.status_code == 429
    assert err.retry_after is None
    print("  test_mlb_api_rate_limit_error_no_retry_after: PASSED")


# ===========================================================================
# 2. MLB API backoff with jitter
# ===========================================================================

def test_backoff_sleep_includes_jitter():
    """_backoff_sleep should add jitter to the base delay."""
    from data.mlb_api import _backoff_sleep, RETRY_BACKOFF_BASE

    delays = []
    with patch("data.mlb_api.time.sleep") as mock_sleep:
        for _ in range(20):
            _backoff_sleep(0)
            delays.append(mock_sleep.call_args[0][0])

    # With jitter, not all delays should be identical
    unique_delays = set(round(d, 6) for d in delays)
    assert len(unique_delays) > 1, "Expected jitter to produce varying delays"
    # All delays should be positive and within expected range
    for d in delays:
        assert d >= RETRY_BACKOFF_BASE, f"Delay {d} < base {RETRY_BACKOFF_BASE}"
        # Max is base + jitter (up to base), so max = 2*base
        assert d <= RETRY_BACKOFF_BASE * 2 + 0.001
    print("  test_backoff_sleep_includes_jitter: PASSED")


def test_backoff_sleep_exponential():
    """_backoff_sleep should increase delay exponentially with attempt."""
    from data.mlb_api import _backoff_sleep, RETRY_BACKOFF_BASE

    delays_by_attempt = {}
    with patch("data.mlb_api.time.sleep") as mock_sleep:
        for attempt in range(4):
            # Average over multiple calls to smooth out jitter
            total = 0.0
            n = 50
            for _ in range(n):
                _backoff_sleep(attempt)
                total += mock_sleep.call_args[0][0]
            delays_by_attempt[attempt] = total / n

    # Each attempt should roughly double the delay
    for a in range(1, 4):
        ratio = delays_by_attempt[a] / delays_by_attempt[a - 1]
        assert 1.5 < ratio < 2.5, (
            f"Expected ~2x increase from attempt {a-1} to {a}, got {ratio:.2f}"
        )
    print("  test_backoff_sleep_exponential: PASSED")


def test_backoff_sleep_respects_retry_after():
    """_backoff_sleep should use retry_after when it exceeds computed delay."""
    from data.mlb_api import _backoff_sleep

    with patch("data.mlb_api.time.sleep") as mock_sleep:
        _backoff_sleep(0, retry_after=60.0)
        sleep_val = mock_sleep.call_args[0][0]

    assert sleep_val >= 60.0, f"Expected >= 60, got {sleep_val}"
    print("  test_backoff_sleep_respects_retry_after: PASSED")


def test_backoff_sleep_ignores_small_retry_after():
    """_backoff_sleep should use computed delay when it exceeds retry_after."""
    from data.mlb_api import _backoff_sleep, RETRY_BACKOFF_BASE

    with patch("data.mlb_api.time.sleep") as mock_sleep:
        # attempt=3 gives base = 1.0 * 8 = 8.0 + jitter
        _backoff_sleep(3, retry_after=0.1)
        sleep_val = mock_sleep.call_args[0][0]

    # Should use the computed delay which is much larger than 0.1
    assert sleep_val >= RETRY_BACKOFF_BASE * (2 ** 3), \
        f"Expected >= {RETRY_BACKOFF_BASE * 8}, got {sleep_val}"
    print("  test_backoff_sleep_ignores_small_retry_after: PASSED")


# ===========================================================================
# 3. Claude API rate limit handling
# ===========================================================================

def test_claude_is_rate_limit_error_status_code():
    """_is_rate_limit_error should detect 429 via status_code attribute."""
    from game import _is_rate_limit_error

    exc = Exception("rate limited")
    exc.status_code = 429
    assert _is_rate_limit_error(exc) is True

    exc2 = Exception("server error")
    exc2.status_code = 500
    assert _is_rate_limit_error(exc2) is False
    print("  test_claude_is_rate_limit_error_status_code: PASSED")


def test_claude_is_rate_limit_error_status():
    """_is_rate_limit_error should detect 429 via status attribute."""
    from game import _is_rate_limit_error

    exc = Exception("rate limited")
    exc.status = 429
    assert _is_rate_limit_error(exc) is True
    print("  test_claude_is_rate_limit_error_status: PASSED")


def test_claude_is_rate_limit_error_response():
    """_is_rate_limit_error should detect 429 via response.status_code."""
    from game import _is_rate_limit_error

    exc = Exception("rate limited")
    exc.response = MagicMock()
    exc.response.status_code = 429
    assert _is_rate_limit_error(exc) is True
    print("  test_claude_is_rate_limit_error_response: PASSED")


def test_claude_is_rate_limit_error_not_rate_limit():
    """_is_rate_limit_error should return False for non-429 errors."""
    from game import _is_rate_limit_error

    exc = ValueError("something else")
    assert _is_rate_limit_error(exc) is False
    print("  test_claude_is_rate_limit_error_not_rate_limit: PASSED")


def test_claude_extract_retry_after_from_headers():
    """_extract_retry_after should parse retry-after from response headers."""
    from game import _extract_retry_after

    exc = Exception("rate limited")
    exc.response = MagicMock()
    exc.response.headers = {"retry-after": "10"}
    assert _extract_retry_after(exc) == 10.0
    print("  test_claude_extract_retry_after_from_headers: PASSED")


def test_claude_extract_retry_after_x_header():
    """_extract_retry_after should parse x-retry-after header."""
    from game import _extract_retry_after

    exc = Exception("rate limited")
    exc.response = MagicMock()
    exc.response.headers = {"x-retry-after": "5.5"}
    assert _extract_retry_after(exc) == 5.5
    print("  test_claude_extract_retry_after_x_header: PASSED")


def test_claude_extract_retry_after_no_response():
    """_extract_retry_after should return None when no response."""
    from game import _extract_retry_after

    exc = Exception("generic error")
    assert _extract_retry_after(exc) is None
    print("  test_claude_extract_retry_after_no_response: PASSED")


def test_claude_extract_retry_after_no_headers():
    """_extract_retry_after should return None when no headers."""
    from game import _extract_retry_after

    exc = Exception("error")
    exc.response = MagicMock()
    exc.response.headers = None
    assert _extract_retry_after(exc) is None
    print("  test_claude_extract_retry_after_no_headers: PASSED")


def test_claude_extract_retry_after_invalid_value():
    """_extract_retry_after should return None for non-numeric values."""
    from game import _extract_retry_after

    exc = Exception("error")
    exc.response = MagicMock()
    exc.response.headers = {"retry-after": "not-a-number"}
    assert _extract_retry_after(exc) is None
    print("  test_claude_extract_retry_after_invalid_value: PASSED")


def test_claude_backoff_sleep_with_jitter():
    """_claude_backoff_sleep should apply exponential backoff with jitter."""
    from game import _claude_backoff_sleep, CLAUDE_BACKOFF_BASE

    delays = []
    with patch("game.time.sleep") as mock_sleep:
        for _ in range(20):
            _claude_backoff_sleep(0)
            delays.append(mock_sleep.call_args[0][0])

    unique_delays = set(round(d, 6) for d in delays)
    assert len(unique_delays) > 1, "Expected jitter to produce varying delays"
    for d in delays:
        assert d >= CLAUDE_BACKOFF_BASE
    print("  test_claude_backoff_sleep_with_jitter: PASSED")


def test_claude_backoff_sleep_respects_retry_after():
    """_claude_backoff_sleep should respect Retry-After when it exceeds backoff."""
    from game import _claude_backoff_sleep

    with patch("game.time.sleep") as mock_sleep:
        _claude_backoff_sleep(0, retry_after=120.0)
        sleep_val = mock_sleep.call_args[0][0]

    assert sleep_val >= 120.0, f"Expected >= 120, got {sleep_val}"
    print("  test_claude_backoff_sleep_respects_retry_after: PASSED")


def test_call_agent_retries_on_429():
    """_call_agent should retry when Claude API returns 429."""
    from game import _call_agent, CLAUDE_MAX_RETRIES

    call_count = 0

    def mock_tool_runner(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            exc = Exception("rate limited")
            exc.status_code = 429
            exc.response = MagicMock()
            exc.response.headers = {"retry-after": "1"}
            raise exc
        # Return a mock runner that yields one message
        message = MagicMock()
        message.usage = MagicMock()
        message.usage.input_tokens = 100
        message.usage.output_tokens = 50
        block = MagicMock()
        block.type = "text"
        block.text = "NO_ACTION"
        message.content = [block]
        message.parsed = None
        return iter([message])

    client = MagicMock()
    client.beta.messages.tool_runner = mock_tool_runner

    with patch("game.time.sleep"), \
         patch("game.SYSTEM_PROMPT", "test prompt"), \
         patch("game.ALL_TOOLS", []), \
         patch("game.ManagerDecision"):
        decision, msg, meta = _call_agent(client, [], verbose=False)

    assert call_count == 3
    assert meta["rate_limit_retries"] == 2
    print("  test_call_agent_retries_on_429: PASSED")


def test_call_agent_raises_after_max_429_retries():
    """_call_agent should raise after CLAUDE_MAX_RETRIES 429 responses."""
    from game import _call_agent, CLAUDE_MAX_RETRIES

    def mock_tool_runner(**kwargs):
        exc = Exception("rate limited")
        exc.status_code = 429
        exc.response = MagicMock()
        exc.response.headers = {}
        raise exc

    client = MagicMock()
    client.beta.messages.tool_runner = mock_tool_runner

    with patch("game.time.sleep"), \
         patch("game.SYSTEM_PROMPT", "test prompt"), \
         patch("game.ALL_TOOLS", []), \
         patch("game.ManagerDecision"):
        try:
            _call_agent(client, [], verbose=False)
            assert False, "Should have raised"
        except Exception as e:
            assert "rate limited" in str(e).lower()

    print("  test_call_agent_raises_after_max_429_retries: PASSED")


def test_call_agent_non_429_error_not_retried():
    """_call_agent should not retry non-429 errors."""
    from game import _call_agent

    call_count = 0

    def mock_tool_runner(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("Not a rate limit error")

    client = MagicMock()
    client.beta.messages.tool_runner = mock_tool_runner

    with patch("game.time.sleep"), \
         patch("game.SYSTEM_PROMPT", "test prompt"), \
         patch("game.ALL_TOOLS", []), \
         patch("game.ManagerDecision"):
        try:
            _call_agent(client, [], verbose=False)
            assert False, "Should have raised"
        except ValueError:
            pass

    assert call_count == 1, f"Expected 1 call, got {call_count}"
    print("  test_call_agent_non_429_error_not_retried: PASSED")


# ===========================================================================
# 4. Maximum retry count enforcement
# ===========================================================================

def test_mlb_api_max_retries_enforced():
    """_fetch_json should not exceed max_retries attempts."""
    from data.mlb_api import _fetch_json, MLBApiRateLimitError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        exc = urllib.error.HTTPError(
            url="http://test.com",
            code=429,
            msg="Too Many Requests",
            hdrs=http.client.HTTPResponse,
            fp=io.BytesIO(b""),
        )
        exc.headers = {}
        raise exc

    max_retries = 4
    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=max_retries)
        except MLBApiRateLimitError:
            pass

    assert call_count == max_retries, \
        f"Expected exactly {max_retries} attempts, got {call_count}"
    print("  test_mlb_api_max_retries_enforced: PASSED")


def test_claude_max_retries_enforced():
    """_call_agent should not exceed CLAUDE_MAX_RETRIES attempts."""
    from game import _call_agent, CLAUDE_MAX_RETRIES

    call_count = 0

    def mock_tool_runner(**kwargs):
        nonlocal call_count
        call_count += 1
        exc = Exception("rate limited")
        exc.status_code = 429
        exc.response = MagicMock()
        exc.response.headers = {}
        raise exc

    client = MagicMock()
    client.beta.messages.tool_runner = mock_tool_runner

    with patch("game.time.sleep"), \
         patch("game.SYSTEM_PROMPT", "test prompt"), \
         patch("game.ALL_TOOLS", []), \
         patch("game.ManagerDecision"):
        try:
            _call_agent(client, [], verbose=False)
        except Exception:
            pass

    assert call_count == CLAUDE_MAX_RETRIES, \
        f"Expected exactly {CLAUDE_MAX_RETRIES} attempts, got {call_count}"
    print("  test_claude_max_retries_enforced: PASSED")


def test_mlb_api_5xx_max_retries():
    """_fetch_json should also respect max_retries for 5xx errors."""
    from data.mlb_api import _fetch_json, MLBApiConnectionError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="http://test.com",
            code=500,
            msg="Internal Server Error",
            hdrs=http.client.HTTPResponse,
            fp=io.BytesIO(b""),
        )

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=3)
        except MLBApiConnectionError:
            pass

    assert call_count == 3
    print("  test_mlb_api_5xx_max_retries: PASSED")


# ===========================================================================
# 5. Rate limit logging
# ===========================================================================

def test_mlb_api_429_logs_warning():
    """MLB API rate limit events should be logged with timestamps."""
    from data.mlb_api import _fetch_json

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            exc = urllib.error.HTTPError(
                url="http://test.com",
                code=429,
                msg="Too Many Requests",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b""),
            )
            exc.headers = {"Retry-After": "5"}
            raise exc
        resp = MagicMock()
        resp.read.return_value = b'{"ok": true}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"), \
         patch("data.mlb_api.logger") as mock_logger:
        _fetch_json("http://test.com/api", max_retries=3)

    # Verify a warning was logged
    assert mock_logger.warning.called, "Expected warning log for 429"
    logged_msg = mock_logger.warning.call_args[0][0]
    assert "429" in logged_msg or "rate limit" in logged_msg.lower()
    print("  test_mlb_api_429_logs_warning: PASSED")


def test_claude_429_logs_warning():
    """Claude API rate limit events should be logged."""
    from game import _call_agent

    call_count = 0

    def mock_tool_runner(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            exc = Exception("rate limited")
            exc.status_code = 429
            exc.response = MagicMock()
            exc.response.headers = {"retry-after": "2"}
            raise exc
        message = MagicMock()
        message.usage = MagicMock()
        message.usage.input_tokens = 10
        message.usage.output_tokens = 5
        block = MagicMock()
        block.type = "text"
        block.text = "test"
        message.content = [block]
        message.parsed = None
        return iter([message])

    client = MagicMock()
    client.beta.messages.tool_runner = mock_tool_runner

    with patch("game.time.sleep"), \
         patch("game.SYSTEM_PROMPT", "test prompt"), \
         patch("game.ALL_TOOLS", []), \
         patch("game.ManagerDecision"), \
         patch("game.logger") as mock_logger:
        _call_agent(client, [], verbose=False)

    assert mock_logger.warning.called, "Expected warning log for Claude 429"
    logged_msg = mock_logger.warning.call_args[0][0]
    assert "429" in logged_msg or "rate limit" in logged_msg.lower()
    print("  test_claude_429_logs_warning: PASSED")


def test_live_game_feed_rate_limit_logged():
    """Rate limit errors during live game polling should be logged."""
    from live_game_feed import poll_game_feed, GameFeedState
    from data.mlb_api import MLBApiRateLimitError

    state = GameFeedState(
        game_pk=12345,
        team="Red Sox",
        managed_team_side="home",
    )

    def mock_fetch(game_pk):
        raise MLBApiRateLimitError("Rate limited", retry_after=5.0)

    result = poll_game_feed(12345, state, fetch_fn=mock_fetch)

    assert result.error is not None
    assert state.consecutive_errors == 1
    assert len(state.error_log) == 1
    assert state.error_log[0]["error_type"] == "rate_limit_error"
    assert "Rate limited" in state.error_log[0]["error"]
    assert "timestamp" in state.error_log[0]
    print("  test_live_game_feed_rate_limit_logged: PASSED")


# ===========================================================================
# 6. pybaseball/FanGraphs rate limit mitigation via caching
# ===========================================================================

def test_cache_prevents_redundant_api_calls():
    """The cache layer should prevent repeated API calls for the same data."""
    from data.cache import Cache
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)

        # First call: cache miss
        result = cache.get("test_endpoint", {"player_id": 123})
        assert result is None

        # Store data
        cache.set("test_endpoint", {"player_id": 123}, {"stats": "data"}, ttl=86400)

        # Second call: cache hit
        result = cache.get("test_endpoint", {"player_id": 123})
        assert result == {"stats": "data"}

    print("  test_cache_prevents_redundant_api_calls: PASSED")


def test_cache_mitigates_rate_limits_for_statcast():
    """Statcast data requests should be served from cache on repeat."""
    from data.cache import Cache
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)

        # Simulate a cached statcast batting response
        batting_data = {
            "traditional": {"AVG": 0.287, "OBP": 0.350},
            "advanced": {"wOBA": 0.340},
        }
        cache.set("statcast_batting", {"player_id": 660271, "season": 2024, "type": "batting"},
                  batting_data, ttl=86400)

        # Verify it's retrievable without network call
        result = cache.get("statcast_batting", {"player_id": 660271, "season": 2024, "type": "batting"})
        assert result is not None
        assert result["traditional"]["AVG"] == 0.287

    print("  test_cache_mitigates_rate_limits_for_statcast: PASSED")


def test_cache_ttl_expiry():
    """Expired cache entries should trigger fresh API requests."""
    from data.cache import Cache
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)

        # Store with TTL=0 (already expired)
        cache.set("test_ep", {"id": 1}, {"data": "old"}, ttl=0)

        # Immediate retrieval should be a miss (expired)
        # Need to wait a tiny bit or patch time
        with patch("data.cache._now", return_value=time.time() + 1):
            result = cache.get("test_ep", {"id": 1})
        assert result is None

    print("  test_cache_ttl_expiry: PASSED")


# ===========================================================================
# 7. MLB API non-429 errors still work correctly
# ===========================================================================

def test_mlb_api_404_not_retried():
    """404 errors should raise immediately, not be retried."""
    from data.mlb_api import _fetch_json, MLBApiNotFoundError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        raise urllib.error.HTTPError(
            url="http://test.com",
            code=404,
            msg="Not Found",
            hdrs=http.client.HTTPResponse,
            fp=io.BytesIO(b""),
        )

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=3)
            assert False, "Should have raised"
        except MLBApiNotFoundError:
            pass

    assert call_count == 1, "404 should not be retried"
    print("  test_mlb_api_404_not_retried: PASSED")


def test_mlb_api_400_not_retried():
    """400 errors should raise immediately, not be retried."""
    from data.mlb_api import _fetch_json, MLBApiError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        exc = urllib.error.HTTPError(
            url="http://test.com",
            code=400,
            msg="Bad Request",
            hdrs=http.client.HTTPResponse,
            fp=io.BytesIO(b"bad"),
        )
        raise exc

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=3)
            assert False, "Should have raised"
        except MLBApiError as e:
            assert e.status_code == 400

    assert call_count == 1, "400 should not be retried"
    print("  test_mlb_api_400_not_retried: PASSED")


def test_mlb_api_mixed_errors_then_success():
    """Fetch should handle a mix of error types before succeeding."""
    from data.mlb_api import _fetch_json

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 429 rate limit
            exc = urllib.error.HTTPError(
                url="http://test.com",
                code=429,
                msg="Too Many Requests",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b""),
            )
            exc.headers = {"Retry-After": "1"}
            raise exc
        if call_count == 2:
            # 500 server error
            raise urllib.error.HTTPError(
                url="http://test.com",
                code=500,
                msg="Internal Server Error",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b""),
            )
        # Third attempt succeeds
        resp = MagicMock()
        resp.read.return_value = b'{"success": true}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        result = _fetch_json("http://test.com/api", max_retries=5)

    assert result == {"success": True}
    assert call_count == 3
    print("  test_mlb_api_mixed_errors_then_success: PASSED")


# ===========================================================================
# 8. Integration: rate limiting in broader context
# ===========================================================================

def test_run_agent_decision_handles_rate_limit_in_call_agent():
    """run_agent_decision should catch rate limit errors from _call_agent
    and fall back to NO_ACTION."""
    from game import run_agent_decision

    # Create a mock game state
    from simulation import SimulationEngine, load_rosters
    engine = SimulationEngine(seed=42)
    rosters = load_rosters()
    game = engine.initialize_game(rosters)

    # Mock client that always rate-limits
    client = MagicMock()
    rate_error = Exception("rate limited")
    rate_error.status_code = 429
    rate_error.response = MagicMock()
    rate_error.response.headers = {}
    client.beta.messages.tool_runner = MagicMock(side_effect=rate_error)

    messages = []
    with patch("game.time.sleep"):
        decision_dict, msgs, meta = run_agent_decision(
            client, game, "home", messages, verbose=False, max_retries=2
        )

    # Should fall back to NO_ACTION after _call_agent raises
    assert decision_dict["decision"] == "NO_ACTION"
    assert "error" in decision_dict["action_details"].lower() or "agent" in decision_dict["action_details"].lower()
    print("  test_run_agent_decision_handles_rate_limit_in_call_agent: PASSED")


def test_call_agent_metadata_tracks_rate_limit_retries():
    """_call_agent metadata should include rate_limit_retries count."""
    from game import _call_agent

    call_count = 0

    def mock_tool_runner(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            exc = Exception("rate limited")
            exc.status_code = 429
            exc.response = MagicMock()
            exc.response.headers = {}
            raise exc
        message = MagicMock()
        message.usage = MagicMock()
        message.usage.input_tokens = 50
        message.usage.output_tokens = 25
        block = MagicMock()
        block.type = "text"
        block.text = "decision"
        message.content = [block]
        message.parsed = None
        return iter([message])

    client = MagicMock()
    client.beta.messages.tool_runner = mock_tool_runner

    with patch("game.time.sleep"), \
         patch("game.SYSTEM_PROMPT", "test"), \
         patch("game.ALL_TOOLS", []), \
         patch("game.ManagerDecision"):
        _, _, meta = _call_agent(client, [], verbose=False)

    assert meta["rate_limit_retries"] == 2
    print("  test_call_agent_metadata_tracks_rate_limit_retries: PASSED")


def test_mlb_api_429_and_connection_error_sequence():
    """Fetch should handle 429 followed by connection error correctly."""
    from data.mlb_api import _fetch_json, MLBApiConnectionError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            exc = urllib.error.HTTPError(
                url="http://test.com",
                code=429,
                msg="Too Many Requests",
                hdrs=http.client.HTTPResponse,
                fp=io.BytesIO(b""),
            )
            exc.headers = {}
            raise exc
        # Connection error on subsequent attempts
        raise urllib.error.URLError("Connection refused")

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=3)
            assert False, "Should have raised"
        except MLBApiConnectionError:
            pass

    assert call_count == 3
    print("  test_mlb_api_429_and_connection_error_sequence: PASSED")


# ===========================================================================
# 9. Edge cases
# ===========================================================================

def test_mlb_api_fetch_json_single_retry():
    """_fetch_json with max_retries=1 should try once and fail."""
    from data.mlb_api import _fetch_json, MLBApiRateLimitError

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        exc = urllib.error.HTTPError(
            url="http://test.com",
            code=429,
            msg="Too Many Requests",
            hdrs=http.client.HTTPResponse,
            fp=io.BytesIO(b""),
        )
        exc.headers = {}
        raise exc

    with patch("data.mlb_api.urllib.request.urlopen", side_effect=mock_urlopen), \
         patch("data.mlb_api.time.sleep"):
        try:
            _fetch_json("http://test.com/api", max_retries=1)
            assert False, "Should have raised"
        except MLBApiRateLimitError:
            pass

    assert call_count == 1
    print("  test_mlb_api_fetch_json_single_retry: PASSED")


def test_parse_retry_after_no_headers_attr():
    """_parse_retry_after should handle exceptions with no headers attr."""
    from data.mlb_api import _parse_retry_after

    exc = Exception("generic error")
    # No .headers attribute
    result = _parse_retry_after(exc)
    assert result is None
    print("  test_parse_retry_after_no_headers_attr: PASSED")


def test_mlb_api_rate_limit_error_is_mlb_api_error():
    """MLBApiRateLimitError should be a subclass of MLBApiError."""
    from data.mlb_api import MLBApiRateLimitError, MLBApiError

    err = MLBApiRateLimitError("test")
    assert isinstance(err, MLBApiError)
    print("  test_mlb_api_rate_limit_error_is_mlb_api_error: PASSED")


def test_claude_max_retries_constant():
    """CLAUDE_MAX_RETRIES should be a positive integer."""
    from game import CLAUDE_MAX_RETRIES
    assert isinstance(CLAUDE_MAX_RETRIES, int)
    assert CLAUDE_MAX_RETRIES > 0
    print("  test_claude_max_retries_constant: PASSED")


def test_claude_backoff_base_constant():
    """CLAUDE_BACKOFF_BASE should be a positive number."""
    from game import CLAUDE_BACKOFF_BASE
    assert isinstance(CLAUDE_BACKOFF_BASE, (int, float))
    assert CLAUDE_BACKOFF_BASE > 0
    print("  test_claude_backoff_base_constant: PASSED")


def test_mlb_api_retry_backoff_base_constant():
    """RETRY_BACKOFF_BASE should be a positive number."""
    from data.mlb_api import RETRY_BACKOFF_BASE
    assert isinstance(RETRY_BACKOFF_BASE, (int, float))
    assert RETRY_BACKOFF_BASE > 0
    print("  test_mlb_api_retry_backoff_base_constant: PASSED")


def test_mlb_api_max_retries_constant():
    """MAX_RETRIES should be a positive integer."""
    from data.mlb_api import MAX_RETRIES
    assert isinstance(MAX_RETRIES, int)
    assert MAX_RETRIES > 0
    print("  test_mlb_api_max_retries_constant: PASSED")


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    # 1. MLB API 429 handling
    test_mlb_api_fetch_json_429_retries,
    test_mlb_api_429_raises_after_max_retries,
    test_mlb_api_429_respects_retry_after_header,
    test_mlb_api_429_retry_after_not_set,
    test_parse_retry_after_integer,
    test_parse_retry_after_float,
    test_parse_retry_after_missing,
    test_parse_retry_after_invalid,
    test_parse_retry_after_negative_clamped,
    test_mlb_api_rate_limit_error_attributes,
    test_mlb_api_rate_limit_error_no_retry_after,
    # 2. Backoff with jitter
    test_backoff_sleep_includes_jitter,
    test_backoff_sleep_exponential,
    test_backoff_sleep_respects_retry_after,
    test_backoff_sleep_ignores_small_retry_after,
    # 3. Claude API rate limit handling
    test_claude_is_rate_limit_error_status_code,
    test_claude_is_rate_limit_error_status,
    test_claude_is_rate_limit_error_response,
    test_claude_is_rate_limit_error_not_rate_limit,
    test_claude_extract_retry_after_from_headers,
    test_claude_extract_retry_after_x_header,
    test_claude_extract_retry_after_no_response,
    test_claude_extract_retry_after_no_headers,
    test_claude_extract_retry_after_invalid_value,
    test_claude_backoff_sleep_with_jitter,
    test_claude_backoff_sleep_respects_retry_after,
    test_call_agent_retries_on_429,
    test_call_agent_raises_after_max_429_retries,
    test_call_agent_non_429_error_not_retried,
    # 4. Max retry enforcement
    test_mlb_api_max_retries_enforced,
    test_claude_max_retries_enforced,
    test_mlb_api_5xx_max_retries,
    # 5. Rate limit logging
    test_mlb_api_429_logs_warning,
    test_claude_429_logs_warning,
    test_live_game_feed_rate_limit_logged,
    # 6. Cache mitigates rate limits
    test_cache_prevents_redundant_api_calls,
    test_cache_mitigates_rate_limits_for_statcast,
    test_cache_ttl_expiry,
    # 7. Non-429 errors still work
    test_mlb_api_404_not_retried,
    test_mlb_api_400_not_retried,
    test_mlb_api_mixed_errors_then_success,
    # 8. Integration
    test_run_agent_decision_handles_rate_limit_in_call_agent,
    test_call_agent_metadata_tracks_rate_limit_retries,
    test_mlb_api_429_and_connection_error_sequence,
    # 9. Edge cases
    test_mlb_api_fetch_json_single_retry,
    test_parse_retry_after_no_headers_attr,
    test_mlb_api_rate_limit_error_is_mlb_api_error,
    test_claude_max_retries_constant,
    test_claude_backoff_base_constant,
    test_mlb_api_retry_backoff_base_constant,
    test_mlb_api_max_retries_constant,
]


if __name__ == "__main__":
    print("=" * 72)
    print("API Rate Limit Handling Tests")
    print("=" * 72)

    passed = 0
    failed = 0
    errors = []

    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  {test_fn.__name__}: FAILED -- {e}")

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
