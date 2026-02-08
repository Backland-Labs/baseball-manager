# Backtesting Harness -- Implementation Plan

Source: `docs/plans/2026-02-06-feat-historical-game-backtesting-harness-plan.md`

## Files to create

```
backtest.py                     # CLI entry point (argparse, # /// script header)
backtest/__init__.py            # Package init
backtest/extractor.py           # walk_game_feed() generator + Pydantic models
backtest/tools.py               # 5 real-stat @beta_tool functions + 5 pass-through
backtest/runner.py              # build_scenario(), run_backtest(), compare_decisions(), format_report()
tests/test_backtest_extractor.py
tests/test_backtest_tools.py
tests/test_backtest_runner.py
```

## Files to modify

```
game.py                         # Extract call_agent() with configurable tools/model params
```

## Step 1: Refactor _call_agent() in game.py

Extract `_call_agent()` to accept `tools`, `model`, and `system` as parameters with defaults
preserving current behavior. The backtest will import and call this with different tools.

Current signature:
```python
def _call_agent(client, messages, verbose=True)
```

New signature:
```python
def _call_agent(client, messages, *, tools=None, model=None, system=None, verbose=True)
```

- `tools` defaults to `ALL_TOOLS`
- `model` defaults to `"claude-sonnet-4-5-20250929"`
- `system` defaults to `SYSTEM_PROMPT`
- All existing callers continue working with no changes
- Run existing tests to verify no breakage

## Step 2: backtest/extractor.py -- Models + walk_game_feed()

### Pydantic models

```python
class ActionType(str, Enum):
    PITCHING_CHANGE = "PITCHING_CHANGE"
    PINCH_HIT = "PINCH_HIT"
    PINCH_RUN = "PINCH_RUN"
    STOLEN_BASE = "STOLEN_BASE"
    BUNT = "BUNT"
    IBB = "IBB"
    NO_ACTION = "NO_ACTION"

class RealManagerAction(BaseModel):
    action_type: ActionType
    player_in: str | None = None     # player_id
    player_out: str | None = None    # player_id
    player_in_name: str | None = None
    player_out_name: str | None = None
    details: str = ""

class PlayerState(BaseModel):
    player_id: str
    name: str
    position: str
    bats: str = "R"
    throws: str = "R"

class BullpenEntry(BaseModel):
    player_id: str
    name: str
    throws: str = "R"
    pitch_count: int = 0             # in-game pitch count
    has_pitched: bool = False
    available: bool = True

class DecisionPoint(BaseModel):
    play_index: int
    inning: int
    half: str                        # "TOP" or "BOTTOM"
    outs: int
    runners: dict                    # {"first": {...} | None, "second": ..., "third": ...}
    score_home: int
    score_away: int
    batter_id: str
    batter_name: str
    batter_bats: str
    pitcher_id: str
    pitcher_name: str
    pitcher_throws: str
    pitcher_pitch_count: int
    pitcher_batters_faced: int
    pitcher_innings_pitched: float
    pitcher_runs_allowed: int
    current_lineup: list[PlayerState]
    bench: list[PlayerState]
    bullpen: list[BullpenEntry]
    opp_lineup: list[PlayerState]
    opp_bench: list[PlayerState]
    opp_bullpen: list[BullpenEntry]
    on_deck_batter_id: str | None = None
    on_deck_batter_name: str | None = None
    on_deck_batter_bats: str = "R"
    real_manager_action: RealManagerAction | None = None
    real_outcome: str = ""           # e.g. "Strikeout", "Home Run", "Groundout"
    leverage_index: float = 1.0
    managed_team_side: str = "home"  # "home" or "away"
```

### walk_game_feed()

```python
def walk_game_feed(
    feed: dict,
    managed_team: str,    # team name, abbreviation, or "home"/"away"
) -> Iterator[DecisionPoint]:
```

Logic:
1. Resolve managed_team to "home" or "away" by matching against feed's gameData.teams
2. Extract starting lineups from boxscore.teams.{home,away}.battingOrder + players
3. Extract bullpen from boxscore.teams.{home,away}.bullpen + pitchers
4. Walk `liveData.plays.allPlays[]` chronologically
5. For each play (at-bat):
   - Before yielding: check for substitution events in play's `playEvents` (look for
     `type == "action"` with `event` containing "Pitching Change", "Offensive Substitution",
     "Defensive Sub", etc.). Update lineup/bullpen state accordingly.
   - Detect real_manager_action from these events
   - Determine batter/pitcher from `play.matchup`
   - Determine runners from `play.runners` pre-play state
   - Track pitcher pitch count from `play.pitchIndex` length or `playEvents` with `isPitch`
   - Track outs, score from `play.result`
   - Compute leverage_index using `lookup_li()` from decision_quality_wpa
   - Yield DecisionPoint
   - After yielding: update outs, score, base state from play result
6. At half-inning boundaries: verify score against linescore for drift detection

Key MLB Stats API play structure:
```
allPlays[i] = {
  "result": {"type": "atBat", "event": "Strikeout", "rbi": 0, ...},
  "about": {"atBatIndex": 0, "halfInning": "top", "inning": 1, "isTopInning": true},
  "matchup": {"batter": {"id": 123, "fullName": "..."}, "pitcher": {"id": 456, ...}},
  "runners": [{"movement": {"originBase": null, "start": null, "end": "1B", ...}, ...}],
  "playEvents": [
    {"type": "pitch", "isPitch": true, "details": {"type": {"code": "FF"}, ...}},
    {"type": "action", "details": {"event": "Pitching Change", ...}, "player": {"id": 789}},
  ],
  "count": {"balls": 3, "strikes": 2, "outs": 1}
}
```

Substitution events to detect in playEvents:
- `"Pitching Change"` -> ActionType.PITCHING_CHANGE
- `"Offensive Substitution"` or `"Offensive Sub"` -> ActionType.PINCH_HIT or PINCH_RUN
- `"Defensive Sub"` or `"Defensive Switch"` -> track lineup change (may not be a decision point)
- Intentional walk: `play.result.event == "Intent Walk"` -> ActionType.IBB
- Stolen base: `play.result.event` contains "Stolen Base" -> ActionType.STOLEN_BASE
- Sac bunt: `play.result.event` contains "Sac Bunt" -> ActionType.BUNT

## Step 3: backtest/tools.py -- Real-stat tool implementations

Module-level context set before agent calls:
```python
_backtest_context: dict | None = None  # {"game_date": "2024-06-15", "game_feed": {...}}

def set_backtest_context(game_date: str, game_feed: dict) -> None:
    ...

def clear_backtest_context() -> None:
    ...
```

Cache strategy: use `data/cache.py` Cache class with `ttl=10 * 365 * 86400` (effectively permanent)
for historical stats. Cache key includes `(player_id, season, cutoff_date)`.

### 5 real-stat tools (new implementations, @beta_tool decorated)

Each mirrors the signature and response format of its simulation counterpart:

1. **get_batter_stats(player_id, vs_hand, home_away, recency_window)**
   - Data source: MLB Stats API player season stats
   - Return format: same keys as existing tool (player_ref, bats, splits, traditional, advanced, today)
   - Temporal: use `get_schedule_by_date` + game logs to compute stats through cutoff_date
   - Fallback: if < 50 PA in current season, include prior year stats as "career" section

2. **get_pitcher_stats(player_id, stat_type)**
   - Data source: MLB Stats API pitcher season stats
   - Return format: match existing (player_ref, throws, traditional, advanced, pitch_mix, today)
   - Temporal: same cutoff approach

3. **get_matchup_data(batter_id, pitcher_id)**
   - Data source: `mlb_api.get_batter_vs_pitcher(batter_id, pitcher_id)`
   - Return format: match existing (player_ref x2, career_totals, sample_size, matchup_edge)
   - Temporal: BvP endpoint returns career totals (no date filter available), acceptable

4. **get_bullpen_status(team)**
   - Data source: game feed boxscore + cached season stats for each reliever
   - Return format: match existing (available_arms list with player_ref, role, stats)
   - Uses in-game bullpen state from DecisionPoint

5. **get_pitcher_fatigue_assessment(pitcher_id)**
   - Data source: game feed (pitch count, batters faced) + cached season stats
   - Return format: match existing (player_ref, pitch_count, fatigue_level, recommendation)

### 5 pass-through tools

Wrap existing simulation tools, log a warning when called:
- evaluate_stolen_base
- evaluate_sacrifice_bunt
- get_defensive_positioning
- get_defensive_replacement_value
- get_platoon_comparison

### 2 unchanged tools

Import directly from existing:
- get_run_expectancy (static RE matrix)
- get_win_probability (static WP table)

All 12 tools collected into `BACKTEST_TOOLS` list.

## Step 4: backtest/runner.py -- Orchestration

### build_scenario(decision_point, game_date) -> dict

Convert DecisionPoint to the agent scenario format matching `game_state_to_scenario()` output:
```python
{
    "matchup_state": {
        "inning": ..., "half": ..., "outs": ..., "count": {"balls": 0, "strikes": 0},
        "runners": {"first": {...}|None, "second": ..., "third": ...},
        "score": {"home": ..., "away": ...},
        "batting_team": "HOME"|"AWAY",
        "batter": {"player_id": ..., "name": ..., "bats": ..., "lineup_position": ...},
        "pitcher": {"player_id": ..., "name": ..., "throws": ..., "pitch_count_today": ...,
                    "batters_faced_today": ..., "times_through_order": ...,
                    "innings_pitched_today": ..., "runs_allowed_today": ...,
                    "today_line": {"IP": ..., "H": ..., "R": ..., "ER": ..., "BB": ..., "K": ...}},
        "on_deck_batter": {"player_id": ..., "name": ..., "bats": ...},
    },
    "roster_state": {
        "our_lineup": [...], "our_lineup_position": ...,
        "bench": [...], "bullpen": [...],
        "mound_visits_remaining": 5, "challenge_available": True,
    },
    "opponent_roster_state": {
        "their_lineup": [...], "their_lineup_position": ...,
        "their_bench": [...], "their_bullpen": [...],
    },
    "decision_prompt": "Top of the 6th, 1 out, ..."
}
```

### compare_decisions(agent_decision, real_manager_action) -> dict

```python
{
    "category_match": bool,    # same decision type?
    "action_match": bool,      # same specific action? (same reliever, etc.)
    "agent_type": str,         # agent's decision type
    "real_type": str,          # real manager's decision type
    "disagreement_type": str | None,  # what agent wanted to do differently
}
```

Category matching rules:
- NO_ACTION types: SWING_AWAY, NO_ACTION, NO_CHANGE, CONTINUE, etc. all match NO_ACTION
- PITCHING_CHANGE types: PITCHING_CHANGE, PULL_STARTER, BRING_IN_RELIEVER all match PITCHING_CHANGE
- PINCH_HIT types: PINCH_HIT, PINCH_HITTER
- etc.

### ComparisonEntry (Pydantic model)

```python
class ComparisonEntry(BaseModel):
    play_index: int
    inning: int
    half: str
    outs: int
    score_home: int
    score_away: int
    batter_name: str
    pitcher_name: str
    leverage_index: float
    real_outcome: str
    real_outcome_wpa: float
    agent_decision_type: str
    agent_action_details: str
    agent_reasoning: str
    agent_confidence: float
    real_manager_action_type: str
    real_manager_action_details: str
    category_match: bool
    action_match: bool
    disagreement_type: str | None = None
    tool_calls: list[dict] = []
    token_usage: dict = {}
```

### run_backtest(game_pk, team, dry_run=False) -> list[ComparisonEntry]

1. Fetch game feed via `mlb_api.get_live_game_feed(game_pk)`
2. Validate: game is Final, season >= 2022
3. Resolve team to home/away
4. Set backtest context: `set_backtest_context(game_date, feed)`
5. Walk game feed: `list(walk_game_feed(feed, team))`
6. If dry_run: print summary and return empty list
7. For each DecisionPoint:
   - Build scenario via `build_scenario(dp, game_date)`
   - Format as user message (matching game.py's format)
   - Call `_call_agent(client, [{"role": "user", "content": msg}], tools=BACKTEST_TOOLS)`
   - Compare agent decision to real_manager_action
   - Compute real_outcome_wpa from WP lookup
   - Build ComparisonEntry
   - Print progress
8. Return list of ComparisonEntry

### format_report(entries, game_info) -> str

Human-readable report:
```
=== BACKTEST REPORT ===
Game: NYY vs BOS, 2024-06-15 (Final: 5-3 BOS)
Manager: Alex Cora (BOS)
Decision points: 67 (8 active manager actions)

--- Agreement Summary ---
Category agreement: 62/67 (92.5%)
Action agreement: 5/8 active decisions matched

--- Disagreements ---
#  Inning  Situation              LI    Agent Decision       Real Decision
1  T6      1 out, runner on 1st  2.31  PITCHING_CHANGE      NO_ACTION
   Agent: "Pull [pitcher] (87 pitches, 3rd TTO)..."
   Real: Kept starter in. Result: Single, run scored (WPA -0.08)

--- High-Leverage Highlights ---
[Top 5 highest-LI disagreements with full reasoning]

--- Cost ---
API calls: 67 | Tokens: 412,340 (input: 389,200, output: 23,140)
```

## Step 5: backtest.py -- CLI entry point

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anthropic>=0.78.0",
#     "pydantic>=2.0",
#     "httpx",
# ]
# ///
```

argparse with:
- `--game-pk` (required): integer game PK
- `--team` (required): team name or abbreviation
- `--dry-run`: extract decision points only, no API calls

Main flow:
1. Parse args
2. Pre-flight: validate API key (unless dry-run), fetch + validate game feed
3. Call `run_backtest(game_pk, team, dry_run)`
4. Print `format_report(entries, game_info)`
5. Save JSON to `data/game_logs/backtest_{game_pk}_{team}.json`

## Step 6: Tests

### test_backtest_extractor.py
- Fetch a real game feed (cache it as a fixture) and verify:
  - Correct number of decision points
  - Lineup tracking after substitutions matches boxscore
  - Pitcher pitch counts match boxscore
  - Substitution events correctly classified as ActionType
  - Half-inning boundary score verification works

### test_backtest_tools.py
- Test temporal stat caching with permanent TTL
- Test tool response format parity (same JSON keys as simulation tools)
- Test set_backtest_context / clear_backtest_context

### test_backtest_runner.py
- Test build_scenario output matches game_state_to_scenario structure
- Test compare_decisions with agreement/disagreement cases
- Test format_report with mock entries

## Order of execution

1. Refactor `_call_agent()` in game.py + run existing tests
2. Build `backtest/extractor.py` with models + walk_game_feed (the hard part)
3. Write extractor tests against a real cached game feed
4. Build `backtest/tools.py` with 5 real-stat + pass-through tools
5. Build `backtest/runner.py` with build_scenario, compare, report
6. Build `backtest.py` CLI entry point
7. Write remaining tests
8. End-to-end dry-run test against a real game
