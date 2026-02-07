# Baseball Manager AI Agent

An AI-powered baseball manager that watches MLB games and makes real-time managerial decisions using Claude. The agent receives game state before each at-bat, analyzes the situation using real player data via 12 specialized tools, and outputs plain-text decisions suitable for tweeting.

## How It Works

```
MLB Stats API (live game feed)
        |
        v
External Service (polls ~10s)
  - Detects new at-bats
  - Converts API data to agent input
        |
        v
Agent (Claude + 12 analytical tools)
  - Analyzes matchup, roster, situation
  - Decides: pitching change, pinch hit,
    stolen base, defensive shift, or no action
        |
        v
Decision Output
  - Tweets active decisions to Twitter/X
  - Logs all decisions with full reasoning
  - Scores decisions via Win Probability Added (WPA)
```

The agent is stateless per invocation -- each at-bat is an independent decision. The external service maintains game context (substitution history, bullpen usage, etc.) and passes it as input.

## Features

- **Live game monitoring** via the MLB Stats API (free, no key required)
- **12 analytical tools** backed by Statcast/FanGraphs data: batter stats, pitcher stats, matchup history, run expectancy, win probability, stolen base evaluation, sacrifice bunt evaluation, bullpen status, pitcher fatigue, defensive positioning, defensive replacement value, platoon comparison
- **Full game simulation engine** for testing without live games -- pitch-level outcome resolution, fatigue modeling, platoon factors, MLB rules enforcement
- **Decision quality scoring** using Win Probability Added (WPA)
- **Twitter/X integration** with rate limit handling, dry-run mode, and team hashtag enrichment
- **Structured game logs** saved as JSON for post-game analysis

## Prerequisites

- Python 3.12+
- [uv](https://astral.sh/uv) package runner
- `ANTHROPIC_API_KEY` environment variable (not needed for `--sim` mode)

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run a simulated game (no API key needed)
uv run game.py --sim

# Run with the AI agent managing the home team
ANTHROPIC_API_KEY=sk-... uv run game.py

# Run against a live MLB game
ANTHROPIC_API_KEY=sk-... uv run live_game_feed.py --game-pk 716463 --team "Red Sox"
```

## Usage

### Simulated Game (no API key)

```bash
uv run game.py --sim
uv run game.py --sim --seed 42    # deterministic replay
```

### Agent Game

```bash
uv run game.py                    # agent manages home team
uv run game.py --away             # agent manages away team
uv run game.py --seed 42          # set random seed
uv run game.py --dry-run          # validate setup without API calls
```

### Live Game Feed

```bash
uv run live_game_feed.py --game-pk 716463 --team "Red Sox"
uv run live_game_feed.py --game-pk 716463 --team BOS --interval 15
uv run live_game_feed.py --game-pk 716463 --team BOS --dry-run
uv run live_game_feed.py --game-pk 716463 --team BOS --tweet
uv run live_game_feed.py --game-pk 716463 --team BOS --tweet-dry-run
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (except `--sim`) | Claude API key |
| `TWITTER_API_KEY` | No | Twitter API v2 key |
| `TWITTER_API_SECRET` | No | Twitter API secret |
| `TWITTER_ACCESS_TOKEN` | No | Twitter access token |
| `TWITTER_ACCESS_TOKEN_SECRET` | No | Twitter access token secret |

Twitter integration is optional -- the agent runs without it.

## Project Structure

```
game.py                     Main entry point & agent game runner
live_game_feed.py           Live game polling service
simulation.py               Game simulation engine
models.py                   Pydantic data models
game_state_ingestion.py     Game state parsing & validation
decision_output.py          Decision formatting for tweets & logs
decision_quality_wpa.py     WPA scoring & analysis
tweet_integration.py        Twitter/X API integration

tools/                      12 analytical tools
  get_batter_stats.py         Batter stats (AVG/OBP/SLG/wOBA/wRC+)
  get_pitcher_stats.py        Pitcher stats (ERA/FIP/K%/pitch mix)
  get_matchup_data.py         Batter-vs-pitcher history
  get_run_expectancy.py       Run expectancy matrix (24 base-out states)
  get_win_probability.py      Win probability & leverage index
  evaluate_stolen_base.py     SB success probability & breakeven
  evaluate_sacrifice_bunt.py  Bunt proficiency & RE comparison
  get_bullpen_status.py       Bullpen availability & freshness
  get_pitcher_fatigue_assessment.py  Velocity/spin/batted ball trends
  get_defensive_positioning.py      Shift recommendations
  get_defensive_replacement_value.py  Defensive upgrade vs offensive cost
  get_platoon_comparison.py   Pinch hitter evaluation

data/                       Data layer
  mlb_api.py                  MLB Stats API client
  statcast.py                 Statcast/pybaseball integration
  bvp_history.py              Batter-vs-pitcher lookups
  cache.py                    File-based JSON cache (24h TTL)
  re_matrix.json              Pre-computed run expectancy matrix
  win_probability.json        Pre-computed WP tables
  leverage_index.json         Pre-computed leverage index values
  sample_rosters.json         Sample rosters for testing

tests/                      35+ test files
```

## Testing

```bash
python -m pytest tests/ -v                    # all tests
python -m pytest tests/test_simulation.py -v  # specific file
pytest -k "test_stolen_base" -v               # filter by name
```

## Pre-computed Data

The `data/` directory contains pre-computed statistical tables used for win probability, run expectancy, and leverage index lookups. These can be regenerated:

```bash
uv run data/compute_re_matrix.py
```
