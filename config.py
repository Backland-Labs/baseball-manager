"""Centralized configuration for environment variables."""

import os

from anthropic import Anthropic

ANTHROPIC_KEY_ENV = "ANTHROPIC_KEY"


def get_api_key() -> str:
    """Return the Anthropic API key, or empty string if not set."""
    return os.environ.get(ANTHROPIC_KEY_ENV, "")


def require_api_key(message: str = "") -> str:
    """Return the API key or exit with an error."""
    key = get_api_key()
    if not key:
        import sys

        msg = message or f"{ANTHROPIC_KEY_ENV} environment variable not set."
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    return key


def create_anthropic_client() -> Anthropic:
    """Create an Anthropic client using the configured API key."""
    return Anthropic(api_key=require_api_key())
