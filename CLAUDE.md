# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An AI-powered baseball manager that watches live MLB games and makes real-time managerial decisions using Claude. The agent receives game state before each at-bat, analyzes the situation using 12 specialized tools backed by real MLB data, and outputs plain-text decisions suitable for tweeting. The agent is stateless per at-bat invocation.

## Commands

```bash
# Run simulated game (no API key needed)
uv run game.py --sim

# Run agent-managed game
ANTHROPIC_KEY=sk-... uv run game.py

# Find today's games (or any date)
uv run find_games.py
uv run find_games.py --date 2025-07-04 --team "Red Sox"
uv run find_games.py --live  # only in-progress games

# Live MLB game monitoring (with explicit game PK)
ANTHROPIC_KEY=sk-... uv run live_game_feed.py --game-pk 716463 --team "Red Sox"

# Live monitoring (auto-discovers today's game for team)
ANTHROPIC_KEY=sk-... uv run live_game_feed.py --team "Red Sox"

# Live monitoring with tweet output
ANTHROPIC_KEY=sk-... uv run live_game_feed.py --team "Red Sox" --tweet-dry-run
ANTHROPIC_KEY=sk-... uv run live_game_feed.py --team "Red Sox" --tweet  # real posting

# Web dashboard
uv run app.py

# Backtesting
uv run backtest.py --game-pk 746865 --team CHC

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_simulation.py -v

# Run tests matching a keyword
pytest -k "stolen_base" -v

# Regenerate run expectancy matrix
uv run data/compute_re_matrix.py
```

## Architecture

```
MLB Stats API --> External Service (polls ~10s) --> Agent (Claude + 12 tools) --> Decision Output (tweets/logs)
```

**Layered structure:** Pydantic models --> data layer --> tools --> agent --> output

- **Entry points:** `game.py` (agent game runner), `live_game_feed.py` (live polling), `find_games.py` (game discovery), `app.py` (Flask web UI), `backtest.py` (backtesting CLI)
- **Agent input:** Three Pydantic models per at-bat: `MatchupState`, `RosterState`, `OpponentRosterState` (defined in `models.py`)
- **Tools** (`tools/`): 12 read-only analytical tools that query MLB Stats API, Statcast, and FanGraphs. Tools gather context; the agent's decision is expressed through its plain-text response, not via tool calls.
- **Data layer** (`data/`): `mlb_api.py` (MLB Stats API client), `statcast.py` (pybaseball/Statcast/FanGraphs), `bvp_history.py` (batter-vs-pitcher), `cache.py` (file-based JSON cache, 24h TTL)
- **Pre-computed tables** (`data/`): `re_matrix.json`, `win_probability.json`, `leverage_index.json` -- lookup tables derived from Retrosheet play-by-play data
- **Simulation** (`simulation.py`): Full pitch-level game simulation engine with fatigue modeling, platoon factors, and MLB rules enforcement
- **WPA scoring** (`decision_quality_wpa.py`): Win Probability Added computation for evaluating decision quality
- **Backtest framework** (`backtest/`): Evaluates agent against historical games -- `extractor.py` walks game feeds, `runner.py` builds scenarios and runs backtests

## Key Conventions

- **PEP 723 inline metadata:** Every Python file declares dependencies in a `/// script` header. Run files with `uv run script.py` -- there is no requirements.txt or pyproject.toml.
- **Centralized API key access:** Use `config.py` (`get_api_key()`, `require_api_key()`, `create_anthropic_client()`) rather than reading `ANTHROPIC_KEY` directly from `os.environ`.
- **Tool registration:** Tools are defined as JSON schema in `tools/` and registered via `ALL_TOOLS`. The agent calls them through the Claude tool-use protocol.
- **Error codes:** Use `UPPER_SNAKE_CASE` for structured error responses (e.g., in `tools/response.py`).
- **Enums:** `Half`, `BattingTeam`, `BullpenRole` etc. are string enums in `models.py`.
