# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the tweet_integration feature.

Validates:
  1. Accept tweet-ready text from the decision output module
  2. Post the tweet via the Twitter/X API (using tweepy)
  3. Include relevant hashtags or game context (team names, inning)
  4. Handle Twitter API rate limits and errors gracefully
  5. Log all posted tweets with timestamps and tweet IDs
  6. Support a dry-run mode that logs tweets without posting them
"""

import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tweet_integration import (
    TweetConfig,
    TweetResult,
    TweetLog,
    TweetPoster,
    TwitterApiError,
    TwitterRateLimitError,
    TwitterAuthError,
    get_team_hashtag,
    append_hashtags,
    TWEET_MAX_LENGTH,
    TEAM_HASHTAGS,
    ABBREV_HASHTAGS,
    MAX_RETRIES,
    _create_tweepy_client,
    _post_tweet_via_api,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config():
    """A fully configured TweetConfig."""
    return TweetConfig(
        api_key="test-api-key",
        api_secret="test-api-secret",
        access_token="test-access-token",
        access_token_secret="test-access-token-secret",
        bearer_token="test-bearer-token",
    )


@pytest.fixture
def empty_config():
    """An empty TweetConfig with no credentials."""
    return TweetConfig()


@pytest.fixture
def sample_tweet_text():
    """A sample active decision tweet text."""
    return "Bot 7, 1 out, runner on 2nd | Away 4, Home 3 -- Pulling Smith off the mound for Rivera"


@pytest.fixture
def long_tweet_text():
    """A tweet that's already at the limit."""
    return "X" * TWEET_MAX_LENGTH


@pytest.fixture
def tweet_log():
    """A fresh TweetLog."""
    return TweetLog(game_pk=716463, team="Red Sox")


def _make_success_post_fn(tweet_id="1234567890123456789"):
    """Create a post_fn that always succeeds."""
    def post_fn(client, text):
        return tweet_id
    return post_fn


@pytest.fixture
def poster_dry_run(valid_config):
    """A TweetPoster in dry-run mode."""
    return TweetPoster(config=valid_config, dry_run=True)


@pytest.fixture
def poster_live(valid_config):
    """A TweetPoster in live mode with a mock post_fn."""
    return TweetPoster(
        config=valid_config,
        dry_run=False,
        client_factory=lambda cfg: MagicMock(),
        post_fn=_make_success_post_fn(),
    )


# ---------------------------------------------------------------------------
# TweetConfig tests
# ---------------------------------------------------------------------------

class TestTweetConfig:
    """Tests for TweetConfig data class."""

    def test_from_env_reads_environment(self, monkeypatch):
        """TweetConfig.from_env() reads TWITTER_* environment variables."""
        monkeypatch.setenv("TWITTER_API_KEY", "key1")
        monkeypatch.setenv("TWITTER_API_SECRET", "secret1")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "token1")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN_SECRET", "tokensecret1")
        monkeypatch.setenv("TWITTER_BEARER_TOKEN", "bearer1")

        config = TweetConfig.from_env()

        assert config.api_key == "key1"
        assert config.api_secret == "secret1"
        assert config.access_token == "token1"
        assert config.access_token_secret == "tokensecret1"
        assert config.bearer_token == "bearer1"

    def test_from_env_defaults_to_empty(self, monkeypatch):
        """TweetConfig.from_env() returns empty strings when vars not set."""
        monkeypatch.delenv("TWITTER_API_KEY", raising=False)
        monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("TWITTER_ACCESS_TOKEN_SECRET", raising=False)
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)

        config = TweetConfig.from_env()

        assert config.api_key == ""
        assert config.api_secret == ""
        assert not config.is_configured()

    def test_is_configured_with_oauth1(self, valid_config):
        """is_configured() returns True when OAuth 1.0a credentials are present."""
        assert valid_config.is_configured()

    def test_is_configured_without_credentials(self, empty_config):
        """is_configured() returns False when credentials are missing."""
        assert not empty_config.is_configured()

    def test_is_configured_partial_credentials(self):
        """is_configured() returns False with partial credentials."""
        config = TweetConfig(api_key="key", api_secret="secret")
        assert not config.is_configured()

    def test_validate_reports_missing_fields(self, empty_config):
        """validate() returns a list of missing credential fields."""
        missing = empty_config.validate()
        assert "TWITTER_API_KEY" in missing
        assert "TWITTER_API_SECRET" in missing
        assert "TWITTER_ACCESS_TOKEN" in missing
        assert "TWITTER_ACCESS_TOKEN_SECRET" in missing

    def test_validate_no_missing_when_configured(self, valid_config):
        """validate() returns an empty list when all credentials present."""
        assert valid_config.validate() == []


# ---------------------------------------------------------------------------
# TweetResult tests
# ---------------------------------------------------------------------------

class TestTweetResult:
    """Tests for TweetResult data class."""

    def test_successful_result_to_dict(self):
        """Successful TweetResult serializes correctly."""
        result = TweetResult(
            success=True,
            tweet_id="123456",
            tweet_text="Test tweet",
            timestamp=1000.0,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["tweet_id"] == "123456"
        assert d["tweet_text"] == "Test tweet"
        assert d["timestamp"] == 1000.0
        assert d["error"] is None
        assert d["dry_run"] is False

    def test_failed_result_to_dict(self):
        """Failed TweetResult includes error info."""
        result = TweetResult(
            success=False,
            tweet_id=None,
            tweet_text="Test tweet",
            timestamp=1000.0,
            error="Rate limit exceeded",
            error_code=429,
            retries=3,
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["tweet_id"] is None
        assert d["error"] == "Rate limit exceeded"
        assert d["error_code"] == 429
        assert d["retries"] == 3

    def test_dry_run_result_to_dict(self):
        """Dry-run TweetResult has dry_run flag set."""
        result = TweetResult(
            success=True,
            tweet_id=None,
            tweet_text="Test tweet",
            timestamp=1000.0,
            dry_run=True,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["tweet_id"] is None
        assert d["dry_run"] is True


# ---------------------------------------------------------------------------
# TweetLog tests
# ---------------------------------------------------------------------------

class TestTweetLog:
    """Tests for TweetLog data class."""

    def test_add_and_count(self, tweet_log):
        """Adding entries updates counts correctly."""
        tweet_log.add(TweetResult(
            success=True, tweet_id="1", tweet_text="t1",
            timestamp=1.0,
        ))
        tweet_log.add(TweetResult(
            success=False, tweet_id=None, tweet_text="t2",
            timestamp=2.0, error="fail",
        ))
        tweet_log.add(TweetResult(
            success=True, tweet_id=None, tweet_text="t3",
            timestamp=3.0, dry_run=True,
        ))

        assert tweet_log.total_attempts == 3
        assert tweet_log.successful_posts == 1
        assert tweet_log.failed_posts == 1
        assert tweet_log.dry_run_posts == 1

    def test_to_dict(self, tweet_log):
        """to_dict() serializes the full log."""
        tweet_log.add(TweetResult(
            success=True, tweet_id="1", tweet_text="t1",
            timestamp=1.0,
        ))
        d = tweet_log.to_dict()
        assert d["game_pk"] == 716463
        assert d["team"] == "Red Sox"
        assert d["summary"]["total_attempts"] == 1
        assert d["summary"]["successful_posts"] == 1
        assert len(d["tweets"]) == 1

    def test_save_creates_file(self, tweet_log, tmp_path):
        """save() writes the log to a JSON file."""
        tweet_log.add(TweetResult(
            success=True, tweet_id="1", tweet_text="t1",
            timestamp=1.0,
        ))
        log_path = tweet_log.save(log_dir=tmp_path)

        assert log_path.exists()
        assert "716463" in log_path.name
        data = json.loads(log_path.read_text())
        assert data["game_pk"] == 716463
        assert len(data["tweets"]) == 1

    def test_save_creates_directory(self, tweet_log, tmp_path):
        """save() creates the log directory if it doesn't exist."""
        log_dir = tmp_path / "nested" / "dir"
        tweet_log.add(TweetResult(
            success=True, tweet_id="1", tweet_text="t1",
            timestamp=1.0,
        ))
        log_path = tweet_log.save(log_dir=log_dir)
        assert log_path.exists()

    def test_empty_log(self, tweet_log):
        """An empty log has zero counts."""
        assert tweet_log.total_attempts == 0
        assert tweet_log.successful_posts == 0
        assert tweet_log.failed_posts == 0
        assert tweet_log.dry_run_posts == 0


# ---------------------------------------------------------------------------
# Hashtag tests
# ---------------------------------------------------------------------------

class TestHashtags:
    """Tests for hashtag lookup and append functions."""

    def test_get_team_hashtag_by_abbreviation(self):
        """get_team_hashtag looks up by abbreviation."""
        assert get_team_hashtag("BOS") == "#RedSox"
        assert get_team_hashtag("NYY") == "#Yankees"
        assert get_team_hashtag("LAD") == "#Dodgers"

    def test_get_team_hashtag_by_full_name(self):
        """get_team_hashtag looks up by full team name."""
        assert get_team_hashtag("Boston Red Sox") == "#RedSox"
        assert get_team_hashtag("new york yankees") == "#Yankees"

    def test_get_team_hashtag_by_partial_name(self):
        """get_team_hashtag matches partial team names."""
        assert get_team_hashtag("Red Sox") == "#RedSox"
        assert get_team_hashtag("Yankees") == "#Yankees"

    def test_get_team_hashtag_not_found(self):
        """get_team_hashtag returns None for unknown teams."""
        assert get_team_hashtag("Unknown Team") is None
        assert get_team_hashtag("") is None

    def test_append_hashtags_adds_team_tags(self, sample_tweet_text):
        """append_hashtags adds team hashtags when space allows."""
        result = append_hashtags(
            sample_tweet_text,
            home_team="Boston Red Sox",
            away_team="New York Yankees",
        )
        assert "#RedSox" in result or "#Yankees" in result
        assert len(result) <= TWEET_MAX_LENGTH

    def test_append_hashtags_respects_max_length(self, long_tweet_text):
        """append_hashtags doesn't exceed max length."""
        result = append_hashtags(
            long_tweet_text,
            home_team="BOS",
            away_team="NYY",
        )
        assert len(result) <= TWEET_MAX_LENGTH
        # Should not append hashtags when already at limit
        assert result == long_tweet_text

    def test_append_hashtags_no_teams(self, sample_tweet_text):
        """append_hashtags returns unchanged text when no teams provided."""
        result = append_hashtags(sample_tweet_text)
        assert result == sample_tweet_text

    def test_append_hashtags_empty_text(self):
        """append_hashtags handles empty tweet text."""
        result = append_hashtags("", home_team="BOS")
        assert result == ""

    def test_append_hashtags_adds_both_teams(self):
        """append_hashtags includes both team hashtags if space allows."""
        short = "Test tweet"
        result = append_hashtags(short, home_team="BOS", away_team="NYY")
        assert "#RedSox" in result
        assert "#Yankees" in result

    def test_append_hashtags_adds_single_when_tight(self):
        """append_hashtags adds only one hashtag when space is tight."""
        # Create text that has room for exactly one hashtag
        text = "X" * (TWEET_MAX_LENGTH - 10)
        result = append_hashtags(text, home_team="BOS", away_team="NYY")
        assert len(result) <= TWEET_MAX_LENGTH
        # Should have at most one hashtag
        hashtag_count = result.count("#")
        assert hashtag_count <= 1

    def test_all_30_teams_have_hashtags(self):
        """Every team abbreviation has a corresponding hashtag."""
        expected_teams = [
            "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE",
            "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
            "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
            "SEA", "STL", "TB", "TEX", "TOR", "WSH",
        ]
        for abbrev in expected_teams:
            assert abbrev in ABBREV_HASHTAGS, f"Missing hashtag for {abbrev}"


# ---------------------------------------------------------------------------
# TweetPoster dry-run tests
# ---------------------------------------------------------------------------

class TestTweetPosterDryRun:
    """Tests for TweetPoster in dry-run mode."""

    def test_dry_run_does_not_post(self, poster_dry_run, sample_tweet_text):
        """Dry-run mode logs the tweet without posting."""
        result = poster_dry_run.post_decision(sample_tweet_text)

        assert result.success is True
        assert result.dry_run is True
        assert result.tweet_id is None
        assert result.tweet_text  # Should have text
        assert result.error is None

    def test_dry_run_logs_entry(self, poster_dry_run, sample_tweet_text):
        """Dry-run mode adds an entry to the tweet log."""
        poster_dry_run.post_decision(sample_tweet_text)

        assert poster_dry_run.tweet_log.total_attempts == 1
        assert poster_dry_run.tweet_log.dry_run_posts == 1
        assert poster_dry_run.tweet_log.successful_posts == 0

    def test_dry_run_includes_hashtags(self, poster_dry_run):
        """Dry-run mode still appends hashtags to tweet text."""
        result = poster_dry_run.post_decision(
            "Short test tweet",
            home_team="BOS",
            away_team="NYY",
        )
        assert "#RedSox" in result.tweet_text or "#Yankees" in result.tweet_text

    def test_dry_run_records_timestamp(self, poster_dry_run, sample_tweet_text):
        """Dry-run result includes a valid timestamp."""
        before = time.time()
        result = poster_dry_run.post_decision(sample_tweet_text)
        after = time.time()

        assert before <= result.timestamp <= after

    def test_dry_run_empty_text(self, poster_dry_run):
        """Dry-run with empty text returns an error."""
        result = poster_dry_run.post_decision("")
        assert result.success is False
        assert "Empty tweet text" in result.error


# ---------------------------------------------------------------------------
# TweetPoster live mode tests
# ---------------------------------------------------------------------------

class TestTweetPosterLive:
    """Tests for TweetPoster in live mode (with mock post_fn)."""

    def test_successful_post(self, poster_live, sample_tweet_text):
        """Successful tweet posting returns the tweet ID."""
        result = poster_live.post_decision(sample_tweet_text)

        assert result.success is True
        assert result.tweet_id == "1234567890123456789"
        assert result.dry_run is False
        assert result.error is None

    def test_post_logs_entry(self, poster_live, sample_tweet_text):
        """Successful posting adds an entry to the tweet log."""
        poster_live.post_decision(sample_tweet_text)

        assert poster_live.tweet_log.total_attempts == 1
        assert poster_live.tweet_log.successful_posts == 1

    def test_post_includes_hashtags(self, poster_live):
        """Live posting appends hashtags to tweet text."""
        result = poster_live.post_decision(
            "Short test tweet",
            home_team="BOS",
            away_team="NYY",
        )
        assert "#RedSox" in result.tweet_text or "#Yankees" in result.tweet_text

    def test_post_with_game_context(self, poster_live):
        """post_decision extracts team names from game_context dict."""
        result = poster_live.post_decision(
            "Short test tweet",
            game_context={
                "home_team": "Boston Red Sox",
                "away_team": "New York Yankees",
                "inning": 7,
            },
        )
        assert result.success is True
        assert "#RedSox" in result.tweet_text or "#Yankees" in result.tweet_text

    def test_post_empty_text_rejected(self, poster_live):
        """Posting empty text returns an error without calling the API."""
        result = poster_live.post_decision("")
        assert result.success is False
        assert "Empty tweet text" in result.error

    def test_post_enforces_max_length(self, valid_config):
        """Excessively long text is truncated before posting."""
        poster = TweetPoster(
            config=valid_config,
            client_factory=lambda cfg: MagicMock(),
            post_fn=_make_success_post_fn(),
        )
        long_text = "A" * 500
        result = poster.post_decision(long_text)

        assert result.success is True
        assert len(result.tweet_text) <= TWEET_MAX_LENGTH

    def test_multiple_posts_logged(self, poster_live):
        """Multiple posts are all logged."""
        poster_live.post_decision("Tweet 1")
        poster_live.post_decision("Tweet 2")
        poster_live.post_decision("Tweet 3")

        assert poster_live.tweet_log.total_attempts == 3
        assert poster_live.tweet_log.successful_posts == 3


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestTweetPosterErrorHandling:
    """Tests for Twitter API error handling and retries."""

    def _make_poster(self, valid_config, post_fn, max_retries=MAX_RETRIES):
        """Helper to create a poster with a custom post_fn."""
        return TweetPoster(
            config=valid_config,
            client_factory=lambda cfg: MagicMock(),
            post_fn=post_fn,
            max_retries=max_retries,
        )

    def test_rate_limit_retries(self, valid_config):
        """Rate limit errors trigger retries with backoff."""
        call_count = 0

        def post_fn(client, text):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TwitterRateLimitError(
                    "Rate limit exceeded",
                    retry_after=0.01,
                )
            return "success_after_retry"

        poster = self._make_poster(valid_config, post_fn, max_retries=3)
        result = poster._post_with_retries("Test tweet", time.time())

        assert result.success is True
        assert result.tweet_id == "success_after_retry"
        assert result.retries == 2

    def test_auth_error_no_retry(self, valid_config):
        """Auth errors are not retried."""
        def post_fn(client, text):
            raise TwitterAuthError("Invalid credentials")

        poster = self._make_poster(valid_config, post_fn)
        result = poster._post_with_retries("Test tweet", time.time())

        assert result.success is False
        assert result.error_code == 401
        assert result.retries == 0

    def test_server_error_retries(self, valid_config):
        """Server errors trigger retries."""
        call_count = 0

        def post_fn(client, text):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TwitterApiError("Server error", error_code=500)
            return "recovered"

        poster = self._make_poster(valid_config, post_fn, max_retries=3)
        result = poster._post_with_retries("Test tweet", time.time())

        assert result.success is True
        assert result.tweet_id == "recovered"
        assert result.retries == 1

    def test_all_retries_exhausted(self, valid_config):
        """When all retries fail, the result indicates failure."""
        def post_fn(client, text):
            raise TwitterApiError("Persistent error", error_code=500)

        poster = self._make_poster(valid_config, post_fn, max_retries=2)
        result = poster._post_with_retries("Test tweet", time.time())

        assert result.success is False
        assert result.retries == 3  # initial + 2 retries
        assert result.error is not None

    def test_unexpected_error_no_retry(self, valid_config):
        """Unexpected errors are not retried."""
        def post_fn(client, text):
            raise RuntimeError("Unexpected!")

        poster = self._make_poster(valid_config, post_fn)
        result = poster._post_with_retries("Test tweet", time.time())

        assert result.success is False
        assert "Unexpected" in result.error

    def test_failed_posts_logged(self, valid_config):
        """Failed posts are recorded in the tweet log."""
        def post_fn(client, text):
            raise TwitterAuthError("Bad creds")

        poster = self._make_poster(valid_config, post_fn)
        poster.post_decision("Test tweet")

        assert poster.tweet_log.total_attempts == 1
        assert poster.tweet_log.failed_posts == 1
        assert poster.tweet_log.successful_posts == 0

    def test_rate_limit_uses_retry_after(self, valid_config):
        """Rate limit errors use the retry_after value for delay."""
        calls = []

        def post_fn(client, text):
            calls.append(time.time())
            if len(calls) < 2:
                raise TwitterRateLimitError("Rate limited", retry_after=0.01)
            return "42"

        poster = self._make_poster(valid_config, post_fn, max_retries=2)
        result = poster._post_with_retries("Test", time.time())
        assert result.success is True
        assert result.tweet_id == "42"


# ---------------------------------------------------------------------------
# Twitter API wrapper tests (unit tests for _create_tweepy_client and
# _post_tweet_via_api using proper mocking)
# ---------------------------------------------------------------------------

class TestTwitterApiWrappers:
    """Tests for the low-level Twitter API wrapper functions."""

    def test_create_tweepy_client_validates_config(self, empty_config):
        """_create_tweepy_client raises on missing credentials."""
        # _create_tweepy_client imports tweepy, so we mock that import
        with patch.dict("sys.modules", {"tweepy": MagicMock()}):
            with pytest.raises(TwitterAuthError, match="Missing Twitter credentials"):
                _create_tweepy_client(empty_config)

    def test_post_tweet_via_api_success(self):
        """_post_tweet_via_api returns tweet ID on success."""
        mock_tweepy = MagicMock()
        # Make exception classes that behave like real exception types
        mock_tweepy.TooManyRequests = type("TooManyRequests", (Exception,), {})
        mock_tweepy.Unauthorized = type("Unauthorized", (Exception,), {})
        mock_tweepy.Forbidden = type("Forbidden", (Exception,), {})
        mock_tweepy.TwitterServerError = type("TwitterServerError", (Exception,), {})
        mock_tweepy.TweepyException = type("TweepyException", (Exception,), {})

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = MagicMock()
            response = MagicMock()
            response.data = {"id": "999"}
            client.create_tweet.return_value = response

            tweet_id = _post_tweet_via_api(client, "Hello")
            assert tweet_id == "999"
            client.create_tweet.assert_called_once_with(text="Hello")

    def test_post_tweet_via_api_rate_limit(self):
        """_post_tweet_via_api raises TwitterRateLimitError on 429."""
        mock_tweepy = MagicMock()
        TooManyRequests = type("TooManyRequests", (Exception,), {})
        mock_tweepy.TooManyRequests = TooManyRequests
        mock_tweepy.Unauthorized = type("Unauthorized", (Exception,), {})
        mock_tweepy.Forbidden = type("Forbidden", (Exception,), {})
        mock_tweepy.TwitterServerError = type("TwitterServerError", (Exception,), {})
        mock_tweepy.TweepyException = type("TweepyException", (Exception,), {})

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = MagicMock()
            exc = TooManyRequests("rate limited")
            exc.response = MagicMock()
            exc.response.headers = {"retry-after": "30"}
            client.create_tweet.side_effect = exc

            with pytest.raises(TwitterRateLimitError):
                _post_tweet_via_api(client, "Hello")

    def test_post_tweet_via_api_unauthorized(self):
        """_post_tweet_via_api raises TwitterAuthError on 401."""
        mock_tweepy = MagicMock()
        mock_tweepy.TooManyRequests = type("TooManyRequests", (Exception,), {})
        Unauthorized = type("Unauthorized", (Exception,), {})
        mock_tweepy.Unauthorized = Unauthorized
        mock_tweepy.Forbidden = type("Forbidden", (Exception,), {})
        mock_tweepy.TwitterServerError = type("TwitterServerError", (Exception,), {})
        mock_tweepy.TweepyException = type("TweepyException", (Exception,), {})

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = MagicMock()
            client.create_tweet.side_effect = Unauthorized("unauthorized")

            with pytest.raises(TwitterAuthError):
                _post_tweet_via_api(client, "Hello")

    def test_post_tweet_via_api_forbidden(self):
        """_post_tweet_via_api raises TwitterAuthError on 403."""
        mock_tweepy = MagicMock()
        mock_tweepy.TooManyRequests = type("TooManyRequests", (Exception,), {})
        mock_tweepy.Unauthorized = type("Unauthorized", (Exception,), {})
        Forbidden = type("Forbidden", (Exception,), {})
        mock_tweepy.Forbidden = Forbidden
        mock_tweepy.TwitterServerError = type("TwitterServerError", (Exception,), {})
        mock_tweepy.TweepyException = type("TweepyException", (Exception,), {})

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = MagicMock()
            client.create_tweet.side_effect = Forbidden("forbidden")

            with pytest.raises(TwitterAuthError):
                _post_tweet_via_api(client, "Hello")

    def test_post_tweet_via_api_server_error(self):
        """_post_tweet_via_api raises TwitterApiError on 5xx."""
        mock_tweepy = MagicMock()
        mock_tweepy.TooManyRequests = type("TooManyRequests", (Exception,), {})
        mock_tweepy.Unauthorized = type("Unauthorized", (Exception,), {})
        mock_tweepy.Forbidden = type("Forbidden", (Exception,), {})
        TwitterServerError = type("TwitterServerError", (Exception,), {})
        mock_tweepy.TwitterServerError = TwitterServerError
        mock_tweepy.TweepyException = type("TweepyException", (Exception,), {})

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = MagicMock()
            client.create_tweet.side_effect = TwitterServerError("server error")

            with pytest.raises(TwitterApiError):
                _post_tweet_via_api(client, "Hello")

    def test_post_tweet_via_api_generic_tweepy_error(self):
        """_post_tweet_via_api raises TwitterApiError on generic TweepyException."""
        mock_tweepy = MagicMock()
        mock_tweepy.TooManyRequests = type("TooManyRequests", (Exception,), {})
        mock_tweepy.Unauthorized = type("Unauthorized", (Exception,), {})
        mock_tweepy.Forbidden = type("Forbidden", (Exception,), {})
        mock_tweepy.TwitterServerError = type("TwitterServerError", (Exception,), {})
        TweepyException = type("TweepyException", (Exception,), {})
        mock_tweepy.TweepyException = TweepyException

        with patch.dict("sys.modules", {"tweepy": mock_tweepy}):
            client = MagicMock()
            client.create_tweet.side_effect = TweepyException("generic error")

            with pytest.raises(TwitterApiError):
                _post_tweet_via_api(client, "Hello")


# ---------------------------------------------------------------------------
# Integration with decision output
# ---------------------------------------------------------------------------

class TestDecisionOutputIntegration:
    """Tests for integration with the decision_output module."""

    def test_accepts_tweet_text_from_decision_output(self, poster_dry_run):
        """TweetPoster accepts tweet text from DecisionOutput.tweet_text."""
        from decision_output import format_decision_output

        output = format_decision_output(
            decision_dict={
                "decision": "PITCHING_CHANGE",
                "action_details": "Bringing in Rivera for Smith",
                "reasoning": "TTO penalty, velocity decline",
                "key_factors": ["3rd TTO", "Velo down 2mph"],
            },
            inning=7,
            half="BOTTOM",
            outs=1,
            score_home=3,
            score_away=4,
        )

        assert output.is_active
        assert output.tweet_text is not None

        result = poster_dry_run.post_decision(output.tweet_text)
        assert result.success is True
        assert result.tweet_text  # Non-empty

    def test_no_action_not_posted(self, poster_dry_run):
        """No-action decisions produce None tweet_text and should not be posted."""
        from decision_output import format_decision_output

        output = format_decision_output(
            decision_dict={
                "decision": "NO_ACTION",
                "action_details": "Let the batter hit",
                "reasoning": "Standard situation",
            },
            inning=3,
            half="TOP",
            outs=0,
            score_home=0,
            score_away=0,
        )

        assert not output.is_active
        assert output.tweet_text is None


# ---------------------------------------------------------------------------
# Integration with live game feed
# ---------------------------------------------------------------------------

def _make_live_feed(play_index=0, at_bat_index=0, inning=1, is_top=True):
    """Create a minimal live game feed dict for testing."""
    return {
        "gameData": {
            "status": {"abstractGameState": "Live"},
            "teams": {
                "home": {"id": 111, "name": "Boston Red Sox", "abbreviation": "BOS"},
                "away": {"id": 147, "name": "New York Yankees", "abbreviation": "NYY"},
            },
        },
        "liveData": {
            "plays": {
                "currentPlayIndex": play_index,
                "currentPlay": {
                    "atBatIndex": at_bat_index,
                    "matchup": {
                        "batter": {"fullName": "Test Batter"},
                        "pitcher": {"fullName": "Test Pitcher"},
                    },
                },
                "allPlays": [{}] * (play_index + 1),
            },
            "linescore": {
                "currentInning": inning,
                "isTopInning": is_top,
                "teams": {
                    "home": {"runs": 3},
                    "away": {"runs": 4},
                },
            },
        },
    }


def _make_final_feed():
    """Create a minimal final game feed dict for testing."""
    return {
        "gameData": {
            "status": {"abstractGameState": "Final"},
            "teams": {
                "home": {"id": 111, "name": "Boston Red Sox", "abbreviation": "BOS"},
                "away": {"id": 147, "name": "New York Yankees", "abbreviation": "NYY"},
            },
        },
        "liveData": {
            "plays": {
                "currentPlayIndex": 0,
                "currentPlay": {"atBatIndex": 0},
                "allPlays": [{}],
            },
            "linescore": {
                "currentInning": 9,
                "isTopInning": False,
                "teams": {
                    "home": {"runs": 5},
                    "away": {"runs": 4},
                },
            },
        },
    }


class TestLiveGameFeedIntegration:
    """Tests for integration with live_game_feed.py."""

    def test_run_live_game_accepts_tweet_poster(self):
        """run_live_game accepts a tweet_poster parameter."""
        from live_game_feed import run_live_game

        import inspect
        sig = inspect.signature(run_live_game)
        assert "tweet_poster" in sig.parameters

    def _run_with_mock_agent(self, mock_output, poster=None, team="BOS"):
        """Helper: run live game with mocked feed and agent."""
        from live_game_feed import run_live_game

        call_count = 0
        def mock_fetch(game_pk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_live_feed(play_index=0, at_bat_index=0)
            return _make_final_feed()

        decision_dict = {"decision": "PITCHING_CHANGE"}
        mock_result = {
            "decision": decision_dict,
            "decision_output": mock_output,
            "metadata": {"tool_calls": [], "token_usage": {}, "latency_ms": 0},
            "timestamp": time.time(),
        }

        def mock_invoke_agent(feed, state, client=None, dry_run=False):
            """Mock invoke_agent that also populates state.decision_log."""
            state.total_agent_calls += 1
            if mock_output.is_active:
                state.active_decisions += 1
            log_entry = {
                "turn": state.total_agent_calls,
                "timestamp": time.time(),
                "game_state": {"inning": 1, "half": "TOP", "outs": 0},
                "decision": decision_dict,
                "is_active_decision": mock_output.is_active,
                "tweet_text": mock_output.tweet_text,
                "log_entry": mock_output.log_entry,
                "tool_calls": [],
                "token_usage": {},
                "latency_ms": 0,
            }
            state.decision_log.append(log_entry)
            return mock_result

        with patch("live_game_feed.invoke_agent", side_effect=mock_invoke_agent):
            with patch("live_game_feed.time.sleep"):
                with patch("live_game_feed.determine_managed_team_side", return_value="home"):
                    state = run_live_game(
                        game_pk=999,
                        team=team,
                        dry_run=True,
                        verbose=False,
                        fetch_fn=mock_fetch,
                        tweet_poster=poster,
                    )
        return state

    def test_tweet_poster_called_for_active_decisions(self, valid_config):
        """The tweet poster is called when the agent makes an active decision."""
        poster = TweetPoster(config=valid_config, dry_run=True)

        mock_output = MagicMock()
        mock_output.is_active = True
        mock_output.tweet_text = "Test active decision tweet"
        mock_output.log_entry = "test"

        self._run_with_mock_agent(mock_output, poster=poster)

        assert poster.tweet_log.total_attempts >= 1

    def test_tweet_result_in_decision_log(self, valid_config):
        """Tweet result is attached to the decision log entry."""
        poster = TweetPoster(config=valid_config, dry_run=True)

        mock_output = MagicMock()
        mock_output.is_active = True
        mock_output.tweet_text = "Test tweet"
        mock_output.log_entry = "test"

        state = self._run_with_mock_agent(mock_output, poster=poster)

        # Check that decision log entries have tweet_result
        active_entries = [d for d in state.decision_log if d.get("tweet_result")]
        assert len(active_entries) >= 1
        tweet_result = active_entries[0]["tweet_result"]
        assert tweet_result["dry_run"] is True
        assert tweet_result["success"] is True

    def test_no_tweet_when_no_poster(self):
        """When tweet_poster is None, no tweet posting occurs."""
        mock_output = MagicMock()
        mock_output.is_active = True
        mock_output.tweet_text = "Test tweet"
        mock_output.log_entry = "test"

        state = self._run_with_mock_agent(mock_output, poster=None)

        # No tweet_result in log entries
        for entry in state.decision_log:
            assert "tweet_result" not in entry

    def test_no_action_not_tweeted(self, valid_config):
        """No-action decisions are not tweeted."""
        poster = TweetPoster(config=valid_config, dry_run=True)

        mock_output = MagicMock()
        mock_output.is_active = False
        mock_output.tweet_text = None
        mock_output.log_entry = "[NO_ACTION] test"

        self._run_with_mock_agent(mock_output, poster=poster)

        assert poster.tweet_log.total_attempts == 0


# ---------------------------------------------------------------------------
# Exception class tests
# ---------------------------------------------------------------------------

class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_twitter_api_error(self):
        exc = TwitterApiError("Test error", error_code=500)
        assert str(exc) == "Test error"
        assert exc.error_code == 500

    def test_twitter_rate_limit_error(self):
        exc = TwitterRateLimitError("Rate limited", retry_after=30.0)
        assert str(exc) == "Rate limited"
        assert exc.error_code == 429
        assert exc.retry_after == 30.0

    def test_twitter_auth_error(self):
        exc = TwitterAuthError("Bad auth")
        assert str(exc) == "Bad auth"
        assert exc.error_code == 401

    def test_rate_limit_inherits_api_error(self):
        exc = TwitterRateLimitError("rl")
        assert isinstance(exc, TwitterApiError)

    def test_auth_error_inherits_api_error(self):
        exc = TwitterAuthError("auth")
        assert isinstance(exc, TwitterApiError)


# ---------------------------------------------------------------------------
# Tweet log save/load roundtrip
# ---------------------------------------------------------------------------

class TestTweetLogPersistence:
    """Tests for tweet log file persistence."""

    def test_roundtrip_json(self, tmp_path):
        """Tweet log can be saved and loaded back as JSON."""
        log = TweetLog(game_pk=12345, team="Yankees")
        log.add(TweetResult(
            success=True, tweet_id="111", tweet_text="Go Yanks",
            timestamp=1000.0,
        ))
        log.add(TweetResult(
            success=False, tweet_id=None, tweet_text="Failed tweet",
            timestamp=1001.0, error="Rate limit", error_code=429,
        ))

        path = log.save(log_dir=tmp_path)
        data = json.loads(path.read_text())

        assert data["game_pk"] == 12345
        assert data["team"] == "Yankees"
        assert data["summary"]["total_attempts"] == 2
        assert data["summary"]["successful_posts"] == 1
        assert data["summary"]["failed_posts"] == 1
        assert len(data["tweets"]) == 2
        assert data["tweets"][0]["tweet_id"] == "111"
        assert data["tweets"][1]["error"] == "Rate limit"

    def test_log_filename_includes_game_pk(self, tmp_path):
        """Log filename includes the game PK."""
        log = TweetLog(game_pk=716463)
        log.add(TweetResult(
            success=True, tweet_id="1", tweet_text="t",
            timestamp=1.0, dry_run=True,
        ))
        path = log.save(log_dir=tmp_path)
        assert "716463" in path.name

    def test_log_unknown_game_pk(self, tmp_path):
        """Log with no game_pk uses 'unknown' in filename."""
        log = TweetLog()
        log.add(TweetResult(
            success=True, tweet_id=None, tweet_text="t",
            timestamp=1.0, dry_run=True,
        ))
        path = log.save(log_dir=tmp_path)
        assert "unknown" in path.name
