---
title: "feat: Historical Game Backtesting Harness"
type: feat
date: 2026-02-06
---

# Historical Game Backtesting Harness

## Overview

A backtesting harness that replays real historical MLB games through the baseball manager agent to evaluate its tactical decision-making against what real managers actually did. The harness extracts decision points from MLB Stats API game feeds, builds scenarios from real player stats with strict temporal cutoffs, presents them to the agent, and compares the agent's decisions to the real manager's -- scoring by agreement, leverage, and qualitative analysis.

## Problem Statement

The agent currently can only be evaluated in two modes: simulated games with fictional teams (SimulationEngine) or live games as they happen (live_game_feed.py). There is no way to evaluate how the agent would have managed real historical games against known outcomes and real managerial decisions. Backtesting against history is the standard method for evaluating decision-making systems.

## Proposed Solution

A **Decision Point Extraction Pipeline** that:
1. Fetches completed game feeds from the MLB Stats API
2. Walks the play-by-play to reconstruct game state at each decision point
3. Presents the state to the agent with temporally-filtered real player stats
4. Compares agent decisions to real manager decisions
5. Produces a comparison report

### Key Design Decisions

**D1: Decision point = every at-bat where the managed team is fielding or batting.** This matches the live agent model where the agent is consulted before every plate appearance. Actual managerial actions (pitching changes, pinch hits, etc.) are flagged distinctly from no-action decisions. This provides the richest evaluation: we see both when the agent agrees with inaction AND when it would have intervened.

**D2: Scoring is qualitative comparison + leverage context, not counterfactual WPA.** Since we don't simulate counterfactuals, we cannot assign a WPA to the agent's alternative decision. Instead:
- **Agreement/disagreement classification** at category level (both chose pitching change) and action level (both chose the same reliever)
- **Leverage index** at each decision point (how important was this moment)
- **Real outcome WPA** as context (what actually happened after the real manager's decision)
- **High-leverage disagreements** are the primary signal -- moments where the agent would have done something different in a critical situation
- The agent's reasoning is logged for qualitative review

**D3: MVP tool set -- 5 real-stat tools, 2 static, 5 pass-through.** The existing 12 tools derive stats from `sample_rosters.json` synthetic attributes. For backtesting, the 5 tools that matter most for managerial decisions get real-stat implementations with temporal filtering: `get_batter_stats`, `get_pitcher_stats`, `get_matchup_data`, `get_bullpen_status`, `get_pitcher_fatigue_assessment`. The 2 table-lookup tools (`get_run_expectancy`, `get_win_probability`) work unchanged. The remaining 5 tools (`evaluate_stolen_base`, `evaluate_sacrifice_bunt`, `get_defensive_positioning`, `get_defensive_replacement_value`, `get_platoon_comparison`) pass through to existing implementations with a logged warning. Statcast-backed tools (spray charts, OAA, sprint speed) are deferred -- they require a different data pipeline.

**D4: Game state reconstruction via play-by-play walk.** A `walk_game_feed()` generator function walks the `allPlays[]` array from the MLB Stats API game feed, maintaining running state (lineup, substitutions, bullpen usage, pitch counts, score, base-out state). At each at-bat boundary, it yields a DecisionPoint. State reconstruction is verified against boxscore data at each half-inning -- drift causes a hard abort, not a soft warning.

**D5: One managed team per invocation, specified via `--team`.** Consistent with live game agent pattern.

**D6: Each decision point is an independent Claude invocation.** Fresh context per decision, no accumulated game history. Simpler, cheaper, reproducible. Matches the "stateless per invocation" design.

**D7: Scope is 2022+ games (Universal DH era).** The agent prompt assumes Universal DH. Pre-2022 NL games are out of scope initially.

**D8: File-based caching only.** The existing `data/cache.py` handles temporal stat caching with a permanent TTL for historical data (completed stats never change). Backtest results are saved as JSON files to `data/game_logs/`. No new infrastructure dependencies.

**D9: Bullpen freshness approximated from game feed only.** For single-game scope, bullpen freshness data from prior games is not fetched. Pitchers who appear in the game feed's bullpen are marked available; their in-game pitch count is tracked from the play-by-play. Multi-game bullpen history is a future enhancement.

**D10: Defensive shifts deferred.** Not reliably detectable from the MLB Stats API play-by-play. Dropped from initial scope.

**D11: Refactor `_call_agent()` to accept tools and model as parameters.** The current `_call_agent()` in `game.py` hardcodes `ALL_TOOLS`, `SYSTEM_PROMPT`, and the model string. Extract a shared `call_agent(client, messages, tools, model, system_prompt, verbose)` function that both `game.py` and the backtest can import. Default parameters preserve existing behavior.

**D12: Backtest tools use `@beta_tool`-decorated module-level functions with a configurable data provider.** The 5 real-stat tools are defined at module level with `@beta_tool` (matching existing tool pattern). A `set_backtest_context(game_date, game_feed)` function configures the data source before agent calls. This avoids re-creating tool closures per decision point and works with `client.beta.messages.tool_runner(tools=...)`.

## Technical Approach

### Architecture

```
backtest.py                     # CLI entry point (argparse)
backtest/
  __init__.py
  extractor.py                  # walk_game_feed() + decision point extraction
  tools.py                      # Backtest tool implementations (real stats)
  runner.py                     # Orchestration, scenario building, comparison, reporting
```

### Data Flow

```
backtest.py --game-pk 716463 --team "Red Sox"
  |
  v
1. Fetch game feed: mlb_api.get_live_game_feed(716463)
  |
  v
2. Validate: game is Final, 2022+ season, teams identified
  |
  v
3. walk_game_feed(game_feed, managed_team)
   - Iterates allPlays[] chronologically
   - Tracks: lineup, substitutions, bullpen usage, pitch counts,
     score, outs, runners, mound visits
   - Verifies state against boxscore at each half-inning (hard abort on drift)
   - At each at-bat boundary, yields DecisionPoint with:
     - inning, half, outs, runners, score (game context fields inlined)
     - current_lineup (who is playing where, after all subs to this point)
     - bullpen_state (who has pitched, pitch counts, who is available)
     - batter_info, pitcher_info
     - real_manager_action (substitution/tactical play that occurred, or None)
     - real_outcome (what happened in this PA)
     - play_index (position in allPlays[])
  |
  v
4. For each DecisionPoint:
   a. build_scenario(decision_point, game_date)
      - Populates MatchupState from game context + real batter/pitcher stats
      - Populates RosterState from current lineup/bench/bullpen state
      - Populates OpponentRosterState similarly
      - Stats fetched from file-based temporal cache or MLB API (with cutoff)
      - Returns scenario dict matching agent's expected input format
   |
   b. evaluate_decision(scenario, decision_point)
      - Calls agent via call_agent() with backtest tools
      - Captures ManagerDecision
      - Compares to real_manager_action:
        - Category match (both: pitching change? both: no action?)
        - Action match (same reliever? same pinch hitter?)
      - Computes leverage_index from game state
      - Records real_outcome_wpa from actual play result
      - Builds ComparisonEntry
  |
  v
5. format_report(comparison_entries, game_info)
   - Per-decision comparison table
   - Agreement rate (category + action level)
   - Disagreement breakdown by type and leverage
   - High-leverage disagreement highlights
   - Agent reasoning excerpts for key moments
   - Summary statistics
  |
  v
6. Output: stdout (human-readable) + JSON file
```

### Implementation Phases

#### Phase 1: Extract + Wire Up

Build the full pipeline: decision point extraction, real-stat tools, scenario building, agent invocation, comparison, and report output. Ship a working single-game backtest.

**Files:**
- `backtest/extractor.py` -- `walk_game_feed()` generator, `DecisionPoint` and `RealManagerAction` Pydantic models
- `backtest/tools.py` -- 5 real-stat tool implementations + pass-through wrappers for remaining 5
- `backtest/runner.py` -- `build_scenario()`, `run_backtest()`, `compare_decisions()`, `format_report()`, `ComparisonEntry` model
- `backtest.py` -- CLI entry point
- Refactor in `game.py` -- extract `call_agent()` with configurable tools/model parameters

**Tasks:**

*Extraction (the hard part -- most time goes here):*
- [ ] Define `DecisionPoint` Pydantic model: inning, half, outs, runners, score, current_lineup, bullpen_state, batter_info, pitcher_info, real_manager_action (or None), real_outcome, play_index, leverage_index
- [ ] Define `RealManagerAction` Pydantic model with `ActionType(str, Enum)`: PITCHING_CHANGE, PINCH_HIT, STOLEN_BASE, BUNT, IBB, NO_ACTION; plus player_in, player_out, details
- [ ] Define `ComparisonEntry` Pydantic model: decision_point, agent_decision, real_manager_action, category_match, action_match, disagreement_type, leverage_index, real_outcome_wpa, agent_reasoning, tool_calls, token_usage
- [ ] Implement `walk_game_feed(feed, managed_team) -> Iterator[DecisionPoint]`: walks `allPlays[]` maintaining lineup, substitutions, bullpen usage, pitch counts per pitcher, score, base-out state
- [ ] Parse substitution events from play-by-play to detect real pitching changes, pinch hitters, pinch runners, double switches
- [ ] Parse play result events to detect stolen base attempts, sacrifice bunts, intentional walks
- [ ] Handle edge cases: ejections (mark as non-tactical, exclude from evaluation), injuries (flag but include), extra innings (ghost runner rule for 2022+)
- [ ] Verify reconstructed state against boxscore at each half-inning boundary -- hard assert on mismatch (abort game, not soft warning)
- [ ] Compute leverage_index at each decision point using existing `lookup_li()` from `decision_quality_wpa.py`
- [ ] Pick 3 specific historical games with complex substitution patterns for test fixtures: (a) game with a double switch, (b) game with pinch-hit-then-defensive-replacement sequence, (c) extra-inning game. Write tests for these first, then build the extractor to pass them.

*Tools:*
- [ ] Implement `set_backtest_context(game_date, game_feed)` to configure the temporal data source
- [ ] Implement 5 real-stat `@beta_tool` functions backed by MLB Stats API with temporal cutoff via existing `data/cache.py` (permanent TTL for historical data):
  - `get_batter_stats` -- real season AVG/OBP/SLG/wOBA from MLB API, cached by (player_id, season, cutoff_date)
  - `get_pitcher_stats` -- real season ERA/FIP/K%/BB% from MLB API
  - `get_matchup_data` -- real BvP history via `mlb_api.get_batter_vs_pitcher()`
  - `get_bullpen_status` -- real bullpen state from game feed + cached season stats
  - `get_pitcher_fatigue_assessment` -- in-game pitch count from game feed + cached season stats
- [ ] Wrap remaining 5 tools as pass-through to existing implementations, logging a warning when called during backtest
- [ ] Include `get_run_expectancy` and `get_win_probability` unchanged (static tables)
- [ ] All tools return responses via `success_response()` / `error_response()` from `tools/response.py`
- [ ] Verify tool response JSON structure is identical to existing tools (schema comparison test, not just "returns valid data")

*Agent integration:*
- [ ] Refactor `_call_agent()` in `game.py`: extract shared `call_agent(client, messages, tools, model, system_prompt, verbose)` with default parameters preserving existing behavior. Both `game.py` and backtest import from it.
- [ ] Implement `build_scenario(decision_point, game_date) -> dict`: populates MatchupState, RosterState, OpponentRosterState from DecisionPoint + real stats. Output structure must match `game_state_to_scenario()` keys and nesting.
- [ ] Implement `run_backtest(game_pk, team, dry_run) -> list[ComparisonEntry]`: orchestrates the full pipeline
- [ ] Implement `compare_decisions(agent_decision, real_manager_action) -> (category_match, action_match, disagreement_type)`
- [ ] Implement `format_report(entries, game_info) -> str`: game summary, agreement rates, disagreement breakdown, top 5 high-leverage disagreements with full agent reasoning, cost summary
- [ ] JSON output saved to `data/game_logs/backtest_{game_pk}_{team}.json`

*CLI:*
- [ ] Implement `backtest.py` CLI with argparse and inline script dependencies (`# /// script` header declaring all transitive deps: `anthropic`, `pydantic`, `httpx`):
  ```
  uv run backtest.py --game-pk 716463 --team "Red Sox"
  uv run backtest.py --game-pk 716463 --team BOS --dry-run
  ```
- [ ] Pre-flight validation: check ANTHROPIC_API_KEY (unless --dry-run), verify game_pk is valid and Final, verify team played in the game, verify season >= 2022
- [ ] `--dry-run` mode: extract decision points and build scenarios without calling Claude API. Print decision point count, breakdown by type (how many real manager actions detected), and estimated token cost.
- [ ] Progress output: show current decision point / total during run

**Tests:**
- [ ] Test extractor against 3 known game feeds with complex substitution patterns (verify lineup, pitch counts, manager actions against boxscore)
- [ ] Test state reconstruction drift detection: introduce a simulated mismatch and verify hard abort
- [ ] Test substitution detection: pitching change mid-inning, pinch hit, pinch runner, double switch
- [ ] Test each of the 5 real-stat tools returns valid data for a known player/date
- [ ] Test temporal filtering: stats for June 15 game exclude July+ data
- [ ] Test file-based caching: second lookup hits cache, not API
- [ ] Test tool response format parity: call both simulation tool and backtest tool, assert identical JSON structure (key names, nesting, types)
- [ ] Test scenario format matches `game_state_to_scenario()` output structure
- [ ] Test comparison logic: agreement on no-action, agreement on pitching change, disagreement cases
- [ ] Test dry-run mode produces decision point summary without API calls
- [ ] Test report generation with mock comparison entries
- [ ] Test CLI argument parsing and pre-flight validation
- [ ] Integration test: `backtest.py --game-pk <known_game> --dry-run` runs end-to-end without errors

**Acceptance criteria:**
- `uv run backtest.py --game-pk <game> --team <team>` runs a complete single-game backtest
- Decision points correctly extracted (verified against boxscore)
- Agent receives scenarios in the same format as live/simulation mode
- 5 core tools return real MLB stats filtered to game date (no future leakage)
- Report shows agreement rate, disagreement breakdown, high-leverage highlights
- JSON log saved for later analysis
- Dry-run mode validates pipeline without API calls

#### Phase 2: Polish

Driven by actual usage after running Phase 1 against real games.

**Potential tasks (prioritize based on what matters after Phase 1):**
- [ ] Add remaining real-stat tools as needed (evaluate_stolen_base with Statcast sprint speed, get_platoon_comparison with real splits, etc.)
- [ ] Handle early-season games (April): supplement current-year stats with prior-year when sample < 50 PA / 20 IP
- [ ] Handle rookies with no prior stats: return available data with a `limited_sample` flag
- [ ] Batch mode: `--start-date` / `--end-date` flags for date ranges (use `get_schedule_by_date()` to resolve game PKs)
- [ ] Cost estimation before batch runs with confirmation prompt
- [ ] Improve report formatting based on what is useful in practice
- [ ] Additional CLI flags as needed (`--model` for cheaper models, `--quiet` for suppressed output)

## Acceptance Criteria

### Functional Requirements

- [ ] Given a game_pk and team, extracts all decision points from the historical game
- [ ] Each decision point presents real game state with real player stats (temporally filtered)
- [ ] Agent receives scenarios in the same format as live mode (no agent prompt changes needed)
- [ ] 5 core tools return real MLB stats filtered to the game date (no future leakage)
- [ ] Agent decisions are compared to real manager decisions at category and action level
- [ ] Report shows agreement rate, disagreement breakdown, and high-leverage highlights
- [ ] Dry-run mode validates pipeline without Claude API calls

### Non-Functional Requirements

- [ ] Temporal stat cache (file-based, permanent TTL) eliminates redundant API calls within and across games
- [ ] Historical game feed cache uses permanent TTL for completed games
- [ ] Single-game backtest completes with reasonable API usage (~50-80 agent calls per game)
- [ ] State reconstruction verified against boxscore with hard abort on drift

### Quality Gates

- [ ] Decision point extraction verified against known game boxscores (3 test fixtures with complex subs)
- [ ] Temporal filtering verified: no stat from after game date appears in tool responses
- [ ] Tool response format matches existing tools (schema comparison test)
- [ ] Integration test: dry-run end-to-end against a real game feed
- [ ] Tests for extractor, tools, runner (scenario + comparison + report)

## Dependencies & Prerequisites

- MLB Stats API access (free, no key required for public data)
- Anthropic API key (for agent calls; not needed for --dry-run)
- pybaseball (already in project, for Statcast data -- used in Phase 2 tool expansions)
- Existing modules: `data/mlb_api.py`, `data/cache.py`, `decision_quality_wpa.py`, `models.py`, `game.py` (call_agent extraction)

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Game state reconstruction errors (wrong lineup after substitution) | Medium | High | Verify against boxscore at each half-inning with hard abort; test against 3 complex game fixtures first |
| Temporal stat filtering incomplete (future data leaks) | Low | High | Unit tests that verify stat dates; schema comparison tests for tool response format |
| `_call_agent()` refactoring breaks existing game.py | Low | High | Default parameters preserve existing behavior; run existing tests after extraction |
| MLB Stats API rate limiting | Medium | Medium | Aggressive file-based caching with permanent TTL; pre-fetch stats before running agent |
| Agent prompt needs modification for real player context | Low | Medium | Test with a single game first; the prompt is player-name-agnostic |
| Incomplete play-by-play data for some games | Low | Low | Skip games with missing/empty allPlays[]; skip non-Final games; log warnings |
| API cost for single-game backtest | Low | Medium | ~50-80 calls per game is manageable; dry-run shows count before committing |

## References & Research

### Internal References

- Brainstorm: `docs/brainstorms/2026-02-06-backtesting-harness-brainstorm.md`
- MLB Stats API integration: `data/mlb_api.py` (get_live_game_feed at line 634, extract_game_situation at line 676)
- Game state ingestion: `game_state_ingestion.py` (ingest_mlb_api_feed at line 319)
- Agent interface: `game.py` (_call_agent at line 310, run_agent_decision at line 431)
- WPA scoring: `decision_quality_wpa.py` (lookup_wp at line 64, lookup_li at line 118)
- Models: `models.py` (MatchupState at line 171, ManagerDecision at line 265)
- Existing tools: `tools/` directory (12 tools, all use sample_rosters.json, all use `@beta_tool` decorator)
- Tool response format: `tools/response.py` (success_response, error_response)
- File-based cache: `data/cache.py` (Cache class with TTL, atomic writes, SHA-256 keys)

### External References

- MLB Stats API documentation: https://statsapi.mlb.com
- pybaseball library: existing integration in `data/statcast.py`

### Review Feedback Incorporated

Changes from plan review (DHH, Kieran, Simplicity reviewers):
- Removed Supabase dependency -- use existing file-based cache with permanent TTL
- Collapsed from 7 modules to 3 (extractor, tools, runner)
- Replaced classes with functions where no meaningful state persists
- MVP tool set: 5 real-stat + 2 static + 5 pass-through (not 12 reimplementations)
- Collapsed from 4 phases to 2 (build it, then polish it)
- Deferred batch mode, extra CLI flags, interrupt/resume
- Added `call_agent()` extraction from `game.py` with configurable tools/model
- Specified `@beta_tool` approach for backtest tools (module-level with configurable data provider)
- Changed dataclasses to Pydantic models for project consistency
- Added hard abort on state reconstruction drift (not soft warning)
- Added tool response format schema comparison tests
- Added integration test for dry-run end-to-end
- Added inline script dependency (`# /// script`) requirement
- Inlined GameContext fields into DecisionPoint (eliminated unnecessary indirection)
