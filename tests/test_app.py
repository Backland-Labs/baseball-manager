# /// script
# requires-python = ">=3.12"
# dependencies = ["flask>=3.0", "pytest>=7.0", "pydantic>=2.0"]
# ///
"""Tests for the web UI Flask app.

Validates:
  1. Dashboard and page routes render correctly
  2. Game log API lists and retrieves files
  3. Path traversal is blocked
  4. Simulation API starts games and returns state
  5. SSE endpoint returns correct content type
  6. Simulation thread runs to completion
  7. Static files are served
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app import app, GAMES, _run_simulation, GAME_LOGS_DIR


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def sample_log(tmp_path, monkeypatch):
    """Create a temporary game log file."""
    log_dir = tmp_path / "game_logs"
    log_dir.mkdir()
    log_data = {
        "game_info": {
            "home_team": "Test Home",
            "away_team": "Test Away",
            "final_score": {"home": 5, "away": 3},
        },
        "summary": {
            "total_polls": 10,
            "active_decisions": 2,
        },
        "decisions": [
            {
                "turn": 1,
                "game_state": {"inning": 1, "half": "TOP", "outs": 0},
                "decision": {"decision": "NO_ACTION"},
                "is_active_decision": False,
            }
        ],
    }
    log_file = log_dir / "test_game_42.json"
    log_file.write_text(json.dumps(log_data))

    import app as app_module
    monkeypatch.setattr(app_module, "GAME_LOGS_DIR", log_dir)

    return log_file


# -----------------------------------------------------------------------
# Dashboard & Pages
# -----------------------------------------------------------------------


class TestDashboardAndPages:
    def test_dashboard_renders(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Baseball AI Manager" in resp.data

    def test_game_page_not_found(self, client):
        resp = client.get("/game/nonexistent")
        assert resp.status_code == 404

    def test_backtest_page_renders(self, client):
        resp = client.get("/backtest")
        assert resp.status_code == 200
        assert b"Backtest" in resp.data


# -----------------------------------------------------------------------
# Game Logs API
# -----------------------------------------------------------------------


class TestGameLogsAPI:
    def test_list_logs(self, client, sample_log):
        resp = client.get("/api/logs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["filename"] == "test_game_42.json"

    def test_get_log_file(self, client, sample_log):
        resp = client.get("/api/logs/test_game_42.json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["game_info"]["home_team"] == "Test Home"

    def test_get_log_not_found(self, client, sample_log):
        resp = client.get("/api/logs/nonexistent.json")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self, client):
        resp = client.get("/api/logs/..secret.json")
        assert resp.status_code == 400

    def test_path_traversal_page_blocked(self, client):
        resp = client.get("/logs/..secret.json")
        assert resp.status_code == 400


# -----------------------------------------------------------------------
# Simulation API
# -----------------------------------------------------------------------


class TestSimulationAPI:
    def test_start_simulation(self, client):
        resp = client.post(
            "/api/simulate",
            json={"seed": 42},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "game_id" in data
        assert data["seed"] == 42
        # Clean up
        game_id = data["game_id"]
        # Wait briefly for thread to start
        time.sleep(0.1)
        assert game_id in GAMES

    def test_game_state_endpoint(self, client):
        resp = client.post(
            "/api/simulate",
            json={"seed": 99},
            content_type="application/json",
        )
        game_id = resp.get_json()["game_id"]
        # Wait for simulation to start
        time.sleep(0.5)

        resp = client.get(f"/api/game/{game_id}/state")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] in ("pending", "running", "completed")

    def test_game_state_not_found(self, client):
        resp = client.get("/api/game/nonexistent/state")
        assert resp.status_code == 404

    def test_game_events_content_type(self, client):
        resp = client.post(
            "/api/simulate",
            json={"seed": 77},
            content_type="application/json",
        )
        game_id = resp.get_json()["game_id"]
        time.sleep(0.1)

        resp = client.get(f"/api/game/{game_id}/events")
        assert resp.content_type.startswith("text/event-stream")


# -----------------------------------------------------------------------
# Simulation Integration
# -----------------------------------------------------------------------


class TestSimulationIntegration:
    def test_simulation_thread_completes(self):
        """Run the simulation thread function directly and verify events."""
        import queue

        game_id = "test_integration"
        q = queue.Queue()
        GAMES[game_id] = {
            "queue": q,
            "status": "pending",
            "seed": 42,
            "managed_team": "home",
            "created_at": time.time(),
        }

        # Run simulation (this blocks until complete)
        _run_simulation(game_id, seed=42, managed_team="home")

        # Collect all events
        events = []
        while not q.empty():
            msg = q.get_nowait()
            if msg is not None:
                events.append(msg)

        # Verify key events
        event_types = [e["event"] for e in events]
        assert "game_start" in event_types
        assert "game_end" in event_types
        assert "play" in event_types

        # Verify game completed
        assert GAMES[game_id]["status"] == "completed"
        assert GAMES[game_id].get("box_score") is not None

        # Verify box score structure
        box = GAMES[game_id]["box_score"]
        assert "away" in box
        assert "home" in box
        assert "final_score" in box

        # Clean up
        del GAMES[game_id]


# -----------------------------------------------------------------------
# Static Files
# -----------------------------------------------------------------------


class TestStaticFiles:
    def test_css_served(self, client):
        resp = client.get("/static/css/style.css")
        assert resp.status_code == 200

    def test_js_served(self, client):
        resp = client.get("/static/js/game.js")
        assert resp.status_code == 200

    def test_charts_js_served(self, client):
        resp = client.get("/static/js/charts.js")
        assert resp.status_code == 200

    def test_htmx_served(self, client):
        resp = client.get("/static/js/htmx.min.js")
        assert resp.status_code == 200
