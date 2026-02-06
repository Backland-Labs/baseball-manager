# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tweepy>=4.14",
# ]
# ///
"""Tweet integration for the baseball manager AI agent.

Posts the agent's active decisions to Twitter/X during live games.
Connects the decision output to the Twitter API via tweepy.

Usage::

    from tweet_integration import TweetPoster, TweetConfig, TweetResult

    config = TweetConfig.from_env()            # reads TWITTER_* env vars
    poster = TweetPoster(config)
    result = poster.post_decision(
        tweet_text="Bot 7, 1 out | Away 4, Home 3 -- Pulling Smith...",
        game_context={"home_team": "Red Sox", "away_team": "Yankees", "inning": 7},
    )

    # Dry-run mode (no actual posting)
    poster = TweetPoster(config, dry_run=True)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("tweet_integration")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TWEET_MAX_LENGTH = 280
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2.0  # seconds
MAX_HASHTAG_LENGTH = 30  # max chars to devote to hashtags

# Common MLB team hashtags (subset -- covers all 30 teams)
TEAM_HASHTAGS: dict[str, str] = {
    "arizona diamondbacks": "#Dbacks",
    "atlanta braves": "#ForTheA",
    "baltimore orioles": "#Birdland",
    "boston red sox": "#RedSox",
    "chicago cubs": "#Cubs",
    "chicago white sox": "#WhiteSox",
    "cincinnati reds": "#Reds",
    "cleveland guardians": "#Guardians",
    "colorado rockies": "#Rockies",
    "detroit tigers": "#Tigers",
    "houston astros": "#Astros",
    "kansas city royals": "#Royals",
    "los angeles angels": "#Angels",
    "los angeles dodgers": "#Dodgers",
    "miami marlins": "#Marlins",
    "milwaukee brewers": "#Brewers",
    "minnesota twins": "#Twins",
    "new york mets": "#Mets",
    "new york yankees": "#Yankees",
    "oakland athletics": "#Athletics",
    "philadelphia phillies": "#Phillies",
    "pittsburgh pirates": "#Pirates",
    "san diego padres": "#Padres",
    "san francisco giants": "#SFGiants",
    "seattle mariners": "#Mariners",
    "st. louis cardinals": "#Cardinals",
    "tampa bay rays": "#Rays",
    "texas rangers": "#Rangers",
    "toronto blue jays": "#BlueJays",
    "washington nationals": "#Nationals",
}

# Abbreviation-to-hashtag fallback
ABBREV_HASHTAGS: dict[str, str] = {
    "ARI": "#Dbacks", "ATL": "#ForTheA", "BAL": "#Birdland",
    "BOS": "#RedSox", "CHC": "#Cubs", "CWS": "#WhiteSox",
    "CIN": "#Reds", "CLE": "#Guardians", "COL": "#Rockies",
    "DET": "#Tigers", "HOU": "#Astros", "KC": "#Royals",
    "LAA": "#Angels", "LAD": "#Dodgers", "MIA": "#Marlins",
    "MIL": "#Brewers", "MIN": "#Twins", "NYM": "#Mets",
    "NYY": "#Yankees", "OAK": "#Athletics", "PHI": "#Phillies",
    "PIT": "#Pirates", "SD": "#Padres", "SF": "#SFGiants",
    "SEA": "#Mariners", "STL": "#Cardinals", "TB": "#Rays",
    "TEX": "#Rangers", "TOR": "#BlueJays", "WSH": "#Nationals",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TweetConfig:
    """Configuration for the Twitter/X API client.

    Attributes:
        api_key: Twitter API key (consumer key).
        api_secret: Twitter API secret (consumer secret).
        access_token: OAuth 1.0a access token.
        access_token_secret: OAuth 1.0a access token secret.
        bearer_token: OAuth 2.0 bearer token (optional, for v2 API).
    """
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""
    bearer_token: str = ""

    @classmethod
    def from_env(cls) -> TweetConfig:
        """Create a TweetConfig from environment variables.

        Reads:
            TWITTER_API_KEY, TWITTER_API_SECRET,
            TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
            TWITTER_BEARER_TOKEN
        """
        return cls(
            api_key=os.environ.get("TWITTER_API_KEY", ""),
            api_secret=os.environ.get("TWITTER_API_SECRET", ""),
            access_token=os.environ.get("TWITTER_ACCESS_TOKEN", ""),
            access_token_secret=os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", ""),
            bearer_token=os.environ.get("TWITTER_BEARER_TOKEN", ""),
        )

    def is_configured(self) -> bool:
        """Return True if enough credentials are present for posting."""
        # Need either OAuth 1.0a (all four) or bearer token
        has_oauth1 = all([
            self.api_key, self.api_secret,
            self.access_token, self.access_token_secret,
        ])
        return has_oauth1

    def validate(self) -> list[str]:
        """Return a list of missing credential fields."""
        missing = []
        if not self.api_key:
            missing.append("TWITTER_API_KEY")
        if not self.api_secret:
            missing.append("TWITTER_API_SECRET")
        if not self.access_token:
            missing.append("TWITTER_ACCESS_TOKEN")
        if not self.access_token_secret:
            missing.append("TWITTER_ACCESS_TOKEN_SECRET")
        return missing


@dataclass
class TweetResult:
    """Result of a tweet posting attempt.

    Attributes:
        success: Whether the tweet was posted successfully.
        tweet_id: The tweet ID if successful, None otherwise.
        tweet_text: The text that was posted (or would have been in dry-run).
        timestamp: Unix timestamp of the posting attempt.
        error: Error message if the posting failed, None otherwise.
        error_code: Twitter API error code if applicable.
        dry_run: True if this was a dry-run (no actual posting).
        retries: Number of retries before success or failure.
    """
    success: bool
    tweet_id: str | None
    tweet_text: str
    timestamp: float
    error: str | None = None
    error_code: int | None = None
    dry_run: bool = False
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict for logging."""
        return {
            "success": self.success,
            "tweet_id": self.tweet_id,
            "tweet_text": self.tweet_text,
            "timestamp": self.timestamp,
            "error": self.error,
            "error_code": self.error_code,
            "dry_run": self.dry_run,
            "retries": self.retries,
        }


@dataclass
class TweetLog:
    """Maintains a log of all tweet posting attempts during a game.

    Attributes:
        entries: List of TweetResult objects for all posting attempts.
        game_pk: MLB game identifier.
        team: Team the agent is managing.
    """
    entries: list[TweetResult] = field(default_factory=list)
    game_pk: int | None = None
    team: str = ""

    def add(self, result: TweetResult) -> None:
        """Add a tweet result to the log."""
        self.entries.append(result)

    @property
    def total_attempts(self) -> int:
        return len(self.entries)

    @property
    def successful_posts(self) -> int:
        return sum(1 for e in self.entries if e.success and not e.dry_run)

    @property
    def failed_posts(self) -> int:
        return sum(1 for e in self.entries if not e.success)

    @property
    def dry_run_posts(self) -> int:
        return sum(1 for e in self.entries if e.dry_run)

    def to_dict(self) -> dict[str, Any]:
        """Convert the full log to a JSON-serializable dict."""
        return {
            "game_pk": self.game_pk,
            "team": self.team,
            "summary": {
                "total_attempts": self.total_attempts,
                "successful_posts": self.successful_posts,
                "failed_posts": self.failed_posts,
                "dry_run_posts": self.dry_run_posts,
            },
            "tweets": [e.to_dict() for e in self.entries],
        }

    def save(self, log_dir: Path | None = None) -> Path:
        """Save the tweet log to a JSON file.

        Args:
            log_dir: Directory for log files. Defaults to data/tweet_logs/.

        Returns:
            Path to the saved log file.
        """
        if log_dir is None:
            log_dir = Path(__file__).parent / "data" / "tweet_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        filename = f"tweets_{self.game_pk or 'unknown'}.json"
        log_path = log_dir / filename
        with open(log_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return log_path


# ---------------------------------------------------------------------------
# Hashtag helpers
# ---------------------------------------------------------------------------

def get_team_hashtag(team: str) -> str | None:
    """Look up a hashtag for a team by name or abbreviation.

    Args:
        team: Team full name or abbreviation (e.g. "Boston Red Sox" or "BOS").

    Returns:
        Hashtag string (e.g. "#RedSox") or None if not found.
    """
    if not team:
        return None

    # Try abbreviation first
    abbrev = team.upper().strip()
    if abbrev in ABBREV_HASHTAGS:
        return ABBREV_HASHTAGS[abbrev]

    # Try full name match
    team_lower = team.lower().strip()
    for name, hashtag in TEAM_HASHTAGS.items():
        if team_lower in name or name in team_lower:
            return hashtag

    return None


def append_hashtags(
    tweet_text: str,
    home_team: str = "",
    away_team: str = "",
    max_length: int = TWEET_MAX_LENGTH,
) -> str:
    """Append relevant team hashtags to tweet text if space allows.

    Args:
        tweet_text: The current tweet text.
        home_team: Home team name or abbreviation.
        away_team: Away team name or abbreviation.
        max_length: Maximum tweet length.

    Returns:
        Tweet text with hashtags appended if space permits.
    """
    if not tweet_text:
        return tweet_text

    hashtags = []
    # Add hashtags for both teams
    for team in [home_team, away_team]:
        tag = get_team_hashtag(team)
        if tag and tag not in hashtags:
            hashtags.append(tag)

    if not hashtags:
        return tweet_text

    # Try to append hashtags
    tag_str = " " + " ".join(hashtags)
    if len(tweet_text) + len(tag_str) <= max_length:
        return tweet_text + tag_str

    # Try just the first hashtag
    if hashtags:
        tag_str = " " + hashtags[0]
        if len(tweet_text) + len(tag_str) <= max_length:
            return tweet_text + tag_str

    return tweet_text


# ---------------------------------------------------------------------------
# Twitter API interaction
# ---------------------------------------------------------------------------

class TwitterApiError(Exception):
    """Raised when the Twitter API returns an error."""

    def __init__(self, message: str, error_code: int | None = None):
        super().__init__(message)
        self.error_code = error_code


class TwitterRateLimitError(TwitterApiError):
    """Raised when the Twitter API rate limit is hit."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message, error_code=429)
        self.retry_after = retry_after


class TwitterAuthError(TwitterApiError):
    """Raised when Twitter API authentication fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code=401)


def _create_tweepy_client(config: TweetConfig) -> Any:
    """Create a tweepy Client for the Twitter v2 API.

    Args:
        config: Twitter API credentials.

    Returns:
        tweepy.Client instance.

    Raises:
        ImportError: If tweepy is not installed.
        TwitterAuthError: If credentials are missing.
    """
    try:
        import tweepy
    except ImportError:
        raise ImportError(
            "tweepy is required for tweet posting. "
            "Install it with: pip install tweepy>=4.14"
        )

    if not config.is_configured():
        missing = config.validate()
        raise TwitterAuthError(
            f"Missing Twitter credentials: {', '.join(missing)}"
        )

    client = tweepy.Client(
        consumer_key=config.api_key,
        consumer_secret=config.api_secret,
        access_token=config.access_token,
        access_token_secret=config.access_token_secret,
        bearer_token=config.bearer_token or None,
    )
    return client


def _post_tweet_via_api(client: Any, text: str) -> str:
    """Post a tweet using tweepy and return the tweet ID.

    Args:
        client: tweepy.Client instance.
        text: Tweet text to post.

    Returns:
        Tweet ID string.

    Raises:
        TwitterRateLimitError: On 429 responses.
        TwitterAuthError: On 401/403 responses.
        TwitterApiError: On other API errors.
    """
    import tweepy

    try:
        response = client.create_tweet(text=text)
        tweet_id = str(response.data["id"])
        return tweet_id
    except tweepy.TooManyRequests as exc:
        retry_after = None
        if hasattr(exc, "response") and exc.response is not None:
            retry_after_header = exc.response.headers.get("retry-after")
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    pass
        raise TwitterRateLimitError(
            f"Twitter rate limit exceeded: {exc}",
            retry_after=retry_after,
        ) from exc
    except tweepy.Unauthorized as exc:
        raise TwitterAuthError(f"Twitter authentication failed: {exc}") from exc
    except tweepy.Forbidden as exc:
        raise TwitterAuthError(
            f"Twitter authorization failed (forbidden): {exc}"
        ) from exc
    except tweepy.TwitterServerError as exc:
        raise TwitterApiError(
            f"Twitter server error: {exc}", error_code=500
        ) from exc
    except tweepy.TweepyException as exc:
        raise TwitterApiError(f"Twitter API error: {exc}") from exc


# ---------------------------------------------------------------------------
# TweetPoster -- main interface
# ---------------------------------------------------------------------------

class TweetPoster:
    """Posts agent decisions to Twitter/X.

    Handles tweet formatting (hashtags), rate limit retries, error handling,
    and logging.  Supports dry-run mode for testing.

    Args:
        config: Twitter API credentials.
        dry_run: If True, log tweets without posting them.
        max_retries: Max retry attempts on transient failures.
        tweet_log: Optional TweetLog to record all attempts.
        client_factory: Optional override for creating the tweepy client
            (for testing).
    """

    def __init__(
        self,
        config: TweetConfig | None = None,
        dry_run: bool = False,
        max_retries: int = MAX_RETRIES,
        tweet_log: TweetLog | None = None,
        client_factory: Any | None = None,
        post_fn: Any | None = None,
    ):
        self.config = config or TweetConfig()
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.tweet_log = tweet_log or TweetLog()
        self._client_factory = client_factory or _create_tweepy_client
        self._post_fn = post_fn or _post_tweet_via_api
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily create and cache the tweepy client."""
        if self._client is None:
            self._client = self._client_factory(self.config)
        return self._client

    def post_decision(
        self,
        tweet_text: str,
        game_context: dict[str, Any] | None = None,
        home_team: str = "",
        away_team: str = "",
    ) -> TweetResult:
        """Post an agent decision to Twitter/X.

        Args:
            tweet_text: The tweet-ready text from the decision output module.
            game_context: Optional dict with game metadata (home_team,
                away_team, inning, etc.) for hashtag enrichment.
            home_team: Home team name or abbreviation (for hashtags).
            away_team: Away team name or abbreviation (for hashtags).

        Returns:
            TweetResult with success status, tweet ID, and metadata.
        """
        timestamp = time.time()

        if not tweet_text:
            result = TweetResult(
                success=False,
                tweet_id=None,
                tweet_text="",
                timestamp=timestamp,
                error="Empty tweet text",
            )
            self.tweet_log.add(result)
            return result

        # Extract team names from game_context if provided
        if game_context:
            home_team = home_team or game_context.get("home_team", "")
            away_team = away_team or game_context.get("away_team", "")

        # Append hashtags if space allows
        enriched_text = append_hashtags(
            tweet_text, home_team=home_team, away_team=away_team,
        )

        # Enforce max length
        if len(enriched_text) > TWEET_MAX_LENGTH:
            enriched_text = tweet_text  # Fall back to original without hashtags
            if len(enriched_text) > TWEET_MAX_LENGTH:
                enriched_text = enriched_text[:TWEET_MAX_LENGTH - 1] + "\u2026"

        # Dry-run mode
        if self.dry_run:
            logger.info("[DRY-RUN] Would post tweet: %s", enriched_text)
            result = TweetResult(
                success=True,
                tweet_id=None,
                tweet_text=enriched_text,
                timestamp=timestamp,
                dry_run=True,
            )
            self.tweet_log.add(result)
            return result

        # Post with retries
        return self._post_with_retries(enriched_text, timestamp)

    def _post_with_retries(
        self,
        text: str,
        timestamp: float,
    ) -> TweetResult:
        """Post a tweet with exponential backoff retry on transient errors.

        Args:
            text: Tweet text to post.
            timestamp: Timestamp of the posting attempt.

        Returns:
            TweetResult with success/failure info.
        """
        last_error: str | None = None
        last_error_code: int | None = None
        retries = 0

        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                tweet_id = self._post_fn(client, text)

                result = TweetResult(
                    success=True,
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    retries=attempt,
                )
                self.tweet_log.add(result)
                logger.info(
                    "Tweet posted successfully (id=%s, retries=%d)",
                    tweet_id, attempt,
                )
                return result

            except TwitterRateLimitError as exc:
                last_error = str(exc)
                last_error_code = 429
                retries = attempt + 1
                delay = exc.retry_after or (BASE_RETRY_DELAY * (2 ** attempt))
                logger.warning(
                    "Twitter rate limit hit (attempt %d/%d). "
                    "Retrying in %.1fs: %s",
                    attempt + 1, self.max_retries + 1, delay, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)

            except TwitterAuthError as exc:
                # Auth errors are not retryable
                last_error = str(exc)
                last_error_code = exc.error_code
                logger.error("Twitter auth error (not retrying): %s", exc)
                break

            except TwitterApiError as exc:
                last_error = str(exc)
                last_error_code = exc.error_code
                retries = attempt + 1
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Twitter API error (attempt %d/%d). "
                    "Retrying in %.1fs: %s",
                    attempt + 1, self.max_retries + 1, delay, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)

            except Exception as exc:
                # Unexpected error -- don't retry
                last_error = f"Unexpected error: {exc}"
                last_error_code = None
                logger.error("Unexpected tweet posting error: %s", exc)
                break

        # All retries exhausted or non-retryable error
        result = TweetResult(
            success=False,
            tweet_id=None,
            tweet_text=text,
            timestamp=timestamp,
            error=last_error,
            error_code=last_error_code,
            retries=retries,
        )
        self.tweet_log.add(result)
        return result
