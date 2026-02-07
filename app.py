# /// script
# requires-python = ">=3.12"
# dependencies = ["flask>=3.0", "pydantic>=2.0"]
# ///
"""Web UI for the Baseball AI Manager.

Provides a browser interface for running simulations, viewing game logs
with WPA charts, and running backtests against historical games.

Usage:
    uv run app.py
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from simulation import (
    SimulationEngine,
    game_state_to_dict,
    load_rosters,
)
from decision_quality_wpa import compute_wp_from_game_state

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

GAME_LOGS_DIR = Path(__file__).resolve().parent / "data" / "game_logs"

# In-memory store for active/completed simulations
GAMES: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Simulation thread
# ---------------------------------------------------------------------------


def _run_simulation(game_id: str, seed: int, managed_team: str = "home") -> None:
    """Run a simulation in a background thread, posting events to a queue."""
    entry = GAMES[game_id]
    q: queue.Queue = entry["queue"]
    entry["status"] = "running"

    try:
        rosters = load_rosters()
        engine = SimulationEngine(seed=seed)
        game_state = engine.initialize_game(rosters)

        entry["seed"] = engine.seed
        entry["home_team"] = rosters["home"]["team_name"]
        entry["away_team"] = rosters["away"]["team_name"]

        # Send game_start event
        wp = compute_wp_from_game_state(game_state, managed_team)
        state_dict = game_state_to_dict(game_state)
        q.put({
            "event": "game_start",
            "data": {
                "state": state_dict,
                "wp": wp,
                "home_team": rosters["home"]["team_name"],
                "away_team": rosters["away"]["team_name"],
                "seed": engine.seed,
            },
        })

        pa_count = 0
        while not game_state.game_over:
            if game_state.inning > 15:
                game_state.game_over = True
                if game_state.score_home == game_state.score_away:
                    game_state.winning_team = "TIE (innings limit)"
                else:
                    game_state.winning_team = (
                        game_state.home.name
                        if game_state.score_home > game_state.score_away
                        else game_state.away.name
                    )
                break

            # Automated pitcher management
            engine._auto_manage_pitcher(game_state)

            # Simulate plate appearance
            pa_result = engine.simulate_plate_appearance(game_state)

            # Apply result
            events = engine.apply_pa_result(game_state, pa_result)

            pa_count += 1
            wp = compute_wp_from_game_state(game_state, managed_team)

            # Build play description
            description = pa_result.get("description", "")
            detail = pa_result.get("detail_descriptions", [])
            detail_text = "; ".join(d for d in detail if d)

            state_dict = game_state_to_dict(game_state)

            q.put({
                "event": "play",
                "data": {
                    "pa_count": pa_count,
                    "description": description,
                    "detail": detail_text,
                    "state": state_dict,
                    "wp": wp,
                    "runs_scored": sum(e.runs_scored for e in events),
                },
            })

            # Small delay for streaming effect
            time.sleep(0.02)

        # Generate box score
        box_score = engine.generate_box_score(game_state)
        entry["box_score"] = box_score
        entry["final_state"] = game_state_to_dict(game_state)

        q.put({
            "event": "game_end",
            "data": {
                "box_score": box_score,
                "state": entry["final_state"],
                "winning_team": game_state.winning_team,
            },
        })

    except Exception as e:
        q.put({
            "event": "error",
            "data": {"message": str(e)},
        })

    finally:
        entry["status"] = "completed"
        # Sentinel so SSE readers know to stop
        q.put(None)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/game/<game_id>")
def game_page(game_id: str):
    if game_id not in GAMES:
        return render_template("dashboard.html", error="Game not found"), 404
    return render_template("game.html", game_id=game_id, game=GAMES[game_id])


@app.route("/logs/<filename>")
def game_log_page(filename: str):
    # Validate filename (no path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Invalid filename", 400
    filepath = GAME_LOGS_DIR / filename
    if not filepath.exists():
        return render_template("dashboard.html", error="Log not found"), 404
    return render_template("game_log.html", filename=filename)


@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json(silent=True) or {}
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)
    managed_team = data.get("managed_team", "home")

    game_id = uuid.uuid4().hex[:12]
    GAMES[game_id] = {
        "queue": queue.Queue(),
        "status": "pending",
        "seed": seed,
        "managed_team": managed_team,
        "created_at": time.time(),
    }

    thread = threading.Thread(
        target=_run_simulation,
        args=(game_id, seed, managed_team),
        daemon=True,
    )
    thread.start()

    return jsonify({"game_id": game_id, "seed": seed})


@app.route("/api/game/<game_id>/state")
def api_game_state(game_id: str):
    if game_id not in GAMES:
        return jsonify({"error": "Game not found"}), 404
    entry = GAMES[game_id]
    result = {
        "status": entry["status"],
        "seed": entry.get("seed"),
        "home_team": entry.get("home_team"),
        "away_team": entry.get("away_team"),
    }
    if entry.get("final_state"):
        result["state"] = entry["final_state"]
    if entry.get("box_score"):
        result["box_score"] = entry["box_score"]
    return jsonify(result)


@app.route("/api/game/<game_id>/events")
def api_game_events(game_id: str):
    if game_id not in GAMES:
        return jsonify({"error": "Game not found"}), 404

    entry = GAMES[game_id]
    q: queue.Queue = entry["queue"]

    def generate():
        while True:
            try:
                msg = q.get(timeout=30)
            except queue.Empty:
                # Send keepalive
                yield ":\n\n"
                continue

            if msg is None:
                # Stream complete
                break

            event_type = msg["event"]
            data = json.dumps(msg["data"])
            yield f"event: {event_type}\ndata: {data}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/logs")
def api_list_logs():
    GAME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    for f in sorted(GAME_LOGS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
            info = data.get("game_info", {})
            summary = data.get("summary", {})
            logs.append({
                "filename": f.name,
                "home_team": info.get("home_team", info.get("home", "")),
                "away_team": info.get("away_team", info.get("away", "")),
                "final_score": info.get("final_score", {}),
                "decisions_count": summary.get("total_polls", summary.get("total_decisions", 0)),
                "managed_team": info.get("managed_team", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return jsonify(logs)


@app.route("/api/logs/<filename>")
def api_get_log(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    filepath = GAME_LOGS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Not found"}), 404
    data = json.loads(filepath.read_text())
    return jsonify(data)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GAME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port, threaded=True)
