# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Structured tool response helpers.

Provides consistent JSON response formatting for all 12 agent tools.
Every tool response follows the same top-level structure:

  Success:
    {
      "status": "ok",
      "tool": "<tool_name>",
      "data": { ... }
    }

  Error:
    {
      "status": "error",
      "tool": "<tool_name>",
      "error_code": "<ERROR_CODE>",
      "message": "Human-readable error description"
    }

Conventions enforced by this module:
- Numeric values use consistent formats (e.g., batting average as 0.300)
- Player references include both player_id and player_name
- Unavailable fields are present with null value and explanation, not omitted
- Error responses always include error_code and message
"""

import json
from typing import Any


def success_response(tool: str, data: dict[str, Any]) -> str:
    """Build a structured success response.

    Args:
        tool: The name of the tool producing this response.
        data: The tool-specific data payload.

    Returns:
        JSON string with consistent top-level structure.
    """
    return json.dumps({
        "status": "ok",
        "tool": tool,
        "data": data,
    })


def error_response(tool: str, error_code: str, message: str) -> str:
    """Build a structured error response.

    Args:
        tool: The name of the tool producing this response.
        error_code: Machine-readable error code (e.g., INVALID_PLAYER_ID).
        message: Human-readable error description.

    Returns:
        JSON string with consistent error structure.
    """
    return json.dumps({
        "status": "error",
        "tool": tool,
        "error_code": error_code,
        "message": message,
    })


def player_ref(player_id: str, player_name: str) -> dict[str, str]:
    """Build a consistent player reference with both ID and display name.

    Args:
        player_id: The MLB player ID.
        player_name: The display name.

    Returns:
        Dict with player_id and player_name.
    """
    return {
        "player_id": player_id,
        "player_name": player_name,
    }


def unavailable(explanation: str) -> dict[str, Any]:
    """Mark a field as unavailable with an explanation.

    Use this when a data field cannot be computed or is not applicable.
    This ensures the field is present in the response as null with context,
    rather than being omitted entirely.

    Args:
        explanation: Why the data is unavailable.

    Returns:
        Dict with value=None and explanation string.
    """
    return {
        "value": None,
        "explanation": explanation,
    }
