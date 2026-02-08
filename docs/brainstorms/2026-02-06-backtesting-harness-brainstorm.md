# Backtesting Harness for Historical MLB Games

**Date:** 2026-02-06
**Status:** Brainstorm complete

## What We're Building

A backtesting harness that replays real historical MLB games through the baseball manager agent to evaluate its tactical decision-making against what real managers actually did. The harness extracts decision points from MLB Stats API game feeds, builds scenarios directly from real player stats, presents them to the agent, and compares the agent's decision to the real manager's choice -- scored by actual WPA.

### Core Requirements

- **Real MLB games** as the data source, not simulated games
- **Tactical decision evaluation** -- pitching changes, stolen bases, bunts, intentional walks, pinch hitters, defensive positioning
- **Strict temporal data cutoff** -- the agent's analytical tools must only return stats available before the game date (no future data leakage)
- **Decision comparison only** (no counterfactual simulation) -- compare agent vs. real manager decisions, score both by what actually happened
- **Single-game scope** to start, with the interface designed so multi-game carryover (bullpen fatigue, roster changes) can be added later
- **Real player stats directly** -- no synthetic 0-100 attribute mapping; the agent sees actual MLB stat lines

## Why This Approach

**Decision Point Extraction Pipeline** was chosen over full game replay or scenario library approaches because:

1. **Rigorous and unbiased** -- extracts all meaningful decision points from a game rather than cherry-picking scenarios or wasting API calls on routine PAs
2. **Naturally parallelizable** -- each decision point is independent, so they can be evaluated concurrently or in any order
3. **Clean separation of concerns** -- data ingestion, decision point identification, agent evaluation, and scoring are distinct stages that can be tested and improved independently
4. **Reuses existing infrastructure** -- WPA scoring uses the same `decision_quality_wpa.py` module; the agent's prompt and decision format are unchanged

## Key Decisions

1. **Data source: MLB Stats API** -- the codebase already has `data/mlb_api.py` integration. The API provides pitch-by-pitch game feeds for recent seasons. The existing `game_state_ingestion.py` module already parses raw MLB Stats API game feed JSON into internal models.

2. **No SimulationEngine, no synthetic attributes** -- the harness bypasses the simulation engine entirely. Since we're doing decision comparison (not counterfactual simulation), we never need to simulate outcomes. The agent receives real player stats (AVG, OBP, SLG, ERA, K%, etc.) directly from the MLB Stats API rather than synthetic 0-100 scale attributes. This eliminates the entire player attribute mapping problem.

3. **Backtesting-specific scenario formatter** -- instead of routing through `GameState` -> `game_state_to_scenario()`, the harness builds scenarios directly from MLB Stats API game feed data. The scenario format presented to the agent remains similar (matchup state, roster state, decision prompt), but populated with real stats from the API.

4. **Decision point identification** -- a "decision point" is any moment where a manager could make a tactical choice: before each plate appearance (pitching change, pinch hit, defensive shift, stolen base), and at specific game events (mound visits, challenges). The harness must distinguish between active decisions (manager did something) and passive decisions (let the game play out).

5. **Temporal data isolation** -- the agent's 12 analytical tools must be wrapped or replaced with versions that filter data to a cutoff date. This is the hardest part of the system. Options:
   - Mock tool layer that returns pre-fetched, date-filtered stats
   - Stats snapshot cache keyed by (player_id, date) that's built ahead of time
   - The `data/cache.py` module could be extended with temporal awareness

6. **Scoring** -- WPA is computed from the existing `win_probability.json` lookup tables. For each decision point: `wp_before` is computed from the game state, `wp_after` is the WP after the play resolved (from the actual game outcome). Both the agent's decision and the real manager's decision are scored by what actually happened.

7. **Output format** -- a backtest report per game containing: list of decision points with agent vs. manager comparison, per-decision WPA scores, aggregate metrics (agreement rate, average WPA delta, best/worst agent calls), and the agent's reasoning at each point.

## Open Questions

1. **How far back does the MLB Stats API provide pitch-by-pitch data?** Need to verify coverage for the seasons we want to backtest against. Recent seasons (2020+) should have good coverage.

2. **What counts as a "decision point" precisely?** Need to define the heuristic: every PA? Only PAs where a managerial action is plausible (e.g., when bullpen arms are available, when runners are on base for steal opportunities)? Only moments where the real manager actually made an active decision?

3. **API cost for temporal stat lookups** -- building date-filtered stat snapshots requires many API calls. Should this be a one-time batch job that caches results, or on-demand with caching?

4. **Scenario format convergence** -- the backtesting scenario format should stay close enough to the existing `game_state_to_scenario()` output that the agent prompt doesn't need significant changes. Need to verify what the agent actually keys on in its reasoning.

## Architecture Sketch

```
Historical Game Feed (MLB Stats API)
        |
        v
  DecisionPointExtractor
  - Parses play-by-play from game feed
  - Identifies decision points (managerial action opportunities)
  - Captures real manager's actual decision at each point
  - Extracts game context: score, inning, outs, runners, pitcher pitch count
        |
        v
  [DecisionPoint, DecisionPoint, ...]
  Each contains: game_context, real_manager_decision, real_outcome
        |
        v
  ScenarioBuilder
  - Builds agent-readable scenario from game context + real player stats
  - Uses MLB Stats API for player stats (with temporal cutoff)
  - Produces format compatible with agent's expected input
        |
        v
  BacktestRunner
  - For each DecisionPoint:
    1. Build scenario via ScenarioBuilder
    2. Present to agent (run_agent_decision with temporal tool layer)
    3. Capture agent decision
    4. Compare to real manager decision
    5. Score both by actual WPA from game outcome
        |
        v
  BacktestReport
  - Per-decision comparison table
  - Agreement rate
  - WPA delta analysis
  - Agent reasoning log
```

```
TemporalToolLayer
- Wraps the 12 analytical tools
- Filters all stat lookups to cutoff_date (game date)
- Uses pre-cached stat snapshots or date-filtered API calls
- Returns real MLB stats, not synthetic attributes
- Ensures no future data leakage
```
