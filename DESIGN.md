# Baseball Manager AI Agent -- Design Document

GOAL: build an AI agent that makes in game decisions as a manager of a baseball team 

## Overview

This document specifies the datasets, tools, game state model, and agent architecture needed to build an AI baseball manager using the Claude Agent SDK. The agent receives a game scenario at a decision point and produces a reasoned managerial decision.

---

## 1. Decision Points the Agent Must Handle

The agent must be able to respond at any of the following decision points during a game:

### Pre-Game
| Decision | Key Inputs |
|----------|-----------|
| Lineup construction (batting order) | Platoon splits vs opposing starter, wOBA/wRC+, defensive requirements |
| Starting pitcher selection | Matchup history, rest days, opponent lineup handedness |
| Defensive positioning plan | Opponent spray charts, batter tendencies |

### Offensive (When Team is Batting)
| Decision | Key Inputs |
|----------|-----------|
| Sacrifice bunt | Run expectancy matrix, batter quality, outs, runners, score differential, inning |
| Stolen base (straight, double, delayed, hit-and-run) | Runner sprint speed, SB success rate, pitcher hold time, catcher pop time, breakeven threshold |
| Pinch hit | Platoon matchup, bench availability, batter vs pitcher data, leverage index |
| Pinch run | Runner speed differential, score, inning, remaining bench |
| Intentional walk | Current vs on-deck batter quality, base-out state, force/DP setup |
| Hit-and-run | Batter contact rate, runner speed, count, ground ball tendency |
| Squeeze play | Runner on 3rd, batter bunt proficiency, outs, score |

### Pitching (When Team is Fielding)
| Decision | Key Inputs |
|----------|-----------|
| Pull the starter | Pitch count, times through order (TTO), velocity trend, spin trend, batted ball quality trend, upcoming hitters |
| Select reliever | Leverage index, platoon matchup, bullpen availability, rest days, matchup vs upcoming 3+ batters (3-batter minimum rule) |
| Intentional walk | Batter quality vs on-deck, base-out state, force/DP setup |
| Mound visit | Pitcher demeanor, velocity drop, control issues, visits remaining (5 per 9 innings) |

### Defensive
| Decision | Key Inputs |
|----------|-----------|
| Infield positioning (standard, DP depth, in, halfway, guard lines) | Batter spray chart, GB%, runners, outs, score, inning |
| Outfield positioning (standard, shallow, deep, shaded, no-doubles) | Batter EV/LA profile, game situation |
| Defensive substitution | Defensive metrics (OAA, DRS) of current player vs replacement, innings remaining, score margin |

### Situational
| Decision | Key Inputs |
|----------|-----------|
| Replay challenge | Video staff confidence level, challenge availability, play importance |
| Counter-strategy (respond to opponent moves) | Opponent's new pitcher handedness, bench remaining, matchup data |

---

## 2. Game State Model

The agent receives three data structures as direct input with every scenario: the **MatchupState**, the **RosterState**, and the **OpponentRosterState**. Analytical context (win probability, leverage index, run expectancy) is NOT passed directly -- the agent must derive it by calling tools.

### What the Agent Receives as Input

The following three structures are passed to the agent as part of the scenario prompt. The agent does not need to call tools to obtain this information.

#### MatchupState (passed as input)

Contains the current game situation and the active batter/pitcher matchup.

```
MatchupState {
    // Core game situation
    inning: int                          // 1-9+
    half: "TOP" | "BOTTOM"
    outs: int                            // 0, 1, 2
    count: {balls: int, strikes: int}    // 0-3 balls, 0-2 strikes

    runners: {
        first:  {player_id, name, sprint_speed, sb_success_rate} | null
        second: {player_id, name, sprint_speed, sb_success_rate} | null
        third:  {player_id, name, sprint_speed, sb_success_rate} | null
    }

    score: {home: int, away: int}
    batting_team: "HOME" | "AWAY"

    // Current matchup
    batter: {
        player_id, name, bats: "L" | "R" | "S",
        lineup_position: int              // 1-9
    }
    pitcher: {
        player_id, name, throws: "L" | "R",
        pitch_count_today: int,
        batters_faced_today: int,
        times_through_order: int,         // 1, 2, 3+
        innings_pitched_today: float,
        runs_allowed_today: int,
        today_line: {IP, H, R, ER, BB, K}
    }
    on_deck_batter: {player_id, name, bats: "L" | "R" | "S"}
}
```

#### RosterState (passed as input)

Contains the agent's own team roster availability.

```
RosterState {
    our_lineup: [{player_id, name, position, bats, in_game: bool}; 9]
    our_lineup_position: int              // 0-8, who is due up next

    bench: [{
        player_id, name, bats, positions: [str],
        available: bool                   // false if already used
    }]

    bullpen: [{
        player_id, name, throws: "L" | "R",
        role: "CLOSER" | "SETUP" | "MIDDLE" | "LONG" | "MOPUP",
        available: bool,
        freshness: "FRESH" | "MODERATE" | "TIRED",
        pitches_last_3_days: [int; 3],
        days_since_last_appearance: int,
        is_warming_up: bool
    }]

    mound_visits_remaining: int           // starts at 5
    challenge_available: bool
}
```

#### OpponentRosterState (passed as input)

Contains the opposing team's roster availability.

```
OpponentRosterState {
    their_lineup: [{player_id, name, position, bats, in_game: bool}; 9]
    their_lineup_position: int
    their_bench: [{player_id, name, bats, available: bool}]
    their_bullpen: [{player_id, name, throws, role, available: bool, freshness}]
}
```

### What the Agent Must Derive via Tools

The following analytical context is NOT provided as input. The agent is expected to call tools to compute or look up these values as part of its decision-making process:

- **Win probability** -- via `get_win_probability` tool
- **Leverage index** -- returned by `get_win_probability` tool
- **Run expectancy** -- via `get_run_expectancy` tool
- **Stolen base breakeven analysis** -- via `evaluate_stolen_base` tool
- **Bunt evaluation** -- via `evaluate_sacrifice_bunt` tool
- **Player statistics and splits** -- via `get_batter_stats`, `get_pitcher_stats` tools
- **Matchup projections** -- via `get_matchup_data` tool
- **Pitcher fatigue assessment** -- via `get_pitcher_fatigue_assessment` tool
- **Defensive positioning recommendations** -- via `get_defensive_positioning` tool
- **Platoon comparisons** -- via `get_platoon_comparison` tool

This design forces the agent to actively gather analytical context before making decisions, rather than having it pre-computed. The agent must decide which tools to call based on the decision at hand.

---

## 3. Tools

Tools are the agent's mechanism for gathering additional context and analytical data beyond what is provided in the input game state. They are strictly informational -- tools allow the agent to look up statistics, compute probabilities, and evaluate scenarios. Tools do NOT execute game actions or decisions. The agent's decision is expressed through its structured response output (the `ManagerDecision` schema), which the simulation engine interprets and applies to the game state.

Each tool is a standalone Python or TypeScript script. The agent passes arguments and receives structured JSON output.

### Player Statistics

| Tool | Description |
|------|-------------|
| `get_batter_stats` | Retrieves batting statistics for a player, including traditional stats (AVG/OBP/SLG), advanced metrics (wOBA, wRC+, barrel rate, xwOBA), plate discipline (K%, BB%, chase rate, whiff rate), batted ball profile (GB%, pull%, EV, LA), and sprint speed. Supports splits by handedness, home/away, and recency windows. |
| `get_pitcher_stats` | Retrieves pitching statistics for a pitcher, including ERA/FIP/xFIP, K% and BB%, ground ball rate, pitch mix with per-pitch velocity/spin/whiff rates, and times-through-order wOBA splits. Supports splits by batter handedness, home/away, and recency windows. |
| `get_matchup_data` | Retrieves head-to-head batter vs pitcher history and similarity-based projections. Returns direct matchup results, a sample-size reliability indicator, similarity-model projected wOBA (for small samples), and pitch-type vulnerability breakdown. |

### Game Situation Analysis

| Tool | Description |
|------|-------------|
| `get_run_expectancy` | Returns the expected runs for a given base-out state, the probability of scoring at least one run, and the run distribution. Backed by the 24-state run expectancy matrix. |
| `get_win_probability` | Returns the win probability, leverage index, and conditional win probabilities (if a run scores, if the inning ends scoreless) given the full game state: inning, half, outs, base state, score differential, and home/away. |
| `evaluate_stolen_base` | Evaluates a stolen base attempt given runner speed, SB success rate, pitcher hold time, catcher pop time, target base, and current base-out state. Returns estimated success probability, breakeven rate, expected run-expectancy change, and a recommendation. |
| `evaluate_sacrifice_bunt` | Evaluates whether a sacrifice bunt is optimal given the batter, base-out state, score differential, and inning. Compares bunt vs swing-away expected runs and probability of scoring at least one run. |

### Pitching and Bullpen

| Tool | Description |
|------|-------------|
| `get_bullpen_status` | Returns the current bullpen state for a team: which relievers are available, their stats, freshness level, days of rest, recent pitch counts, platoon splits, warm-up status, and which relievers are unavailable and why. |
| `get_pitcher_fatigue_assessment` | Assesses the current pitcher's fatigue based on in-game trends: velocity changes from first inning to last, spin rate decline, batted ball quality trend (exit velocity against), pitch count, times through order, and an overall fatigue level rating. |

### Defense

| Tool | Description |
|------|-------------|
| `get_defensive_positioning` | Returns recommended infield and outfield positioning for a given batter-pitcher matchup and game situation. Includes the batter's spray chart summary, infield-in cost/benefit analysis, and shift recommendations within current rule constraints. |
| `get_defensive_replacement_value` | Evaluates the net value of a defensive substitution by comparing the defensive upgrade (OAA difference) against the offensive downgrade, scaled by estimated innings remaining. |

### Platoon and Pinch-Hitting

| Tool | Description |
|------|-------------|
| `get_platoon_comparison` | Compares a potential pinch hitter against the current batter for the active matchup. Returns each player's projected wOBA vs the current pitcher, the platoon advantage delta, the defensive cost of the substitution, and the bench depth impact. |

---

## 4. Analytical Framework Constants

These values should be embedded in the agent's system prompt or available as a reference tool.

### Key Decision Formulas

**Stolen base breakeven:**
```
breakeven_rate = |RE_loss_if_caught| / (|RE_gain_if_success| + |RE_loss_if_caught|)
```

Approximate breakeven rates:
- Runner on 1st, 0 outs: ~71.5%
- Runner on 1st, 1 out: ~74.4%
- Runner on 1st, 2 outs: ~77.5%
- Runner on 2nd (steal 3rd), 0 outs: ~80%+

**Odds ratio for matchup projection:**
```
P(event | batter, pitcher) = (P_batter * P_pitcher) / P_league
```

**Times through order penalty:**
| TTO | Approximate wOBA Increase |
|-----|--------------------------|
| 1st | Baseline |
| 2nd | +0.010 to +0.015 |
| 3rd | +0.020 to +0.035 |
| 4th+ | +0.035 to +0.050+ |

**Platoon advantage:**
- Same-hand matchup: ~-0.015 to -0.020 wOBA for batter
- Opposite-hand matchup: ~+0.015 to +0.020 wOBA for batter
- Total platoon gap: ~0.030-0.040 wOBA

**Leverage index thresholds for bullpen deployment:**
| LI | Action |
|----|--------|
| < 0.5 | Use weakest available reliever |
| 0.5 - 1.5 | Use middle reliever |
| 1.5 - 2.5 | Use setup-quality reliever |
| > 2.5 | Use best available reliever (regardless of "role") |

---

## 5. MLB Rules the Agent Must Know

These rules constrain available decisions and must be in the system prompt:

1. **3-batter minimum**: A reliever must face at least 3 batters or end the half-inning before being removed
2. **Runner on 2nd in extras**: Starting in the 10th inning, each half-inning begins with the last scheduled batter placed on 2nd base
3. **Mound visit limit**: 5 per 9-inning game (1 additional per extra inning); second visit to same pitcher in same inning requires removal
4. **Pitch timer**: 15 seconds with bases empty, 20 seconds with runners on
5. **Pickoff limit**: 2 disengagements (pickoff attempts or step-offs) per plate appearance; third unsuccessful disengagement = balk
6. **Infield positioning**: 2 infielders must be on each side of 2nd base; all 4 must have feet on the infield dirt
7. **Universal DH**: Pitcher does not bat
8. **Replay challenge**: 1 per game; second granted if first is successful; umpire-initiated review available in 7th inning+
9. **Bigger bases**: 18 inches (not 15), increasing stolen base opportunities slightly

---

## 6. Agent Architecture (Claude Agent SDK)

### Technology Stack

- **SDK**: `claude-agent-sdk` (Python)
- **Tools**: Standalone Python/TypeScript scripts the agent can execute
- **Agent pattern**: `ClaudeSDKClient` for multi-turn game management
- **Output**: Structured output via Pydantic model for typed decisions

### System Prompt Design

The system prompt should establish:
1. The agent's identity as a baseball manager
2. The decision-making framework: always compute analytical context first (call `get_win_probability` and `get_run_expectancy` to assess the situation before deciding)
3. Key analytical constants (TTO penalty, platoon splits, breakeven rates)
4. Current MLB rules that constrain decisions
5. Instructions that the agent receives MatchupState, RosterState, and OpponentRosterState as input but must use tools for all statistical lookups and analytical computations
6. That tools are for gathering information only -- the agent expresses its decision through the `ManagerDecision` structured output, not by calling an action tool
7. The output format expected

### Tool Organization

Each tool is a standalone script in a `tools/` directory. The agent executes them and receives JSON output. Scripts can be written in Python or TypeScript.

```
tools/
  get_batter_stats.py
  get_pitcher_stats.py
  get_matchup_data.py
  get_run_expectancy.py
  get_win_probability.py
  evaluate_stolen_base.py
  evaluate_sacrifice_bunt.py
  get_bullpen_status.py
  get_pitcher_fatigue_assessment.py
  get_defensive_positioning.py
  get_defensive_replacement_value.py
  get_platoon_comparison.py
```

### Decision Output Schema

Every agent response should conform to this structure:

```python
class ManagerDecision(BaseModel):
    decision: str                          # e.g., "PULL_STARTER", "STOLEN_BASE", "SWING_AWAY"
    action_details: str                    # Specific action: "Bring in Martinez (RHP) to face Johnson"
    confidence: float                      # 0.0-1.0
    reasoning: str                         # Full statistical justification
    win_probability_before: float          # WP before decision
    win_probability_after_expected: float  # Expected WP after decision
    key_factors: list[str]                 # Top 3-5 factors that drove the decision
    alternatives_considered: list[dict]    # [{decision, expected_wp, reason_rejected}]
    risks: list[str]                       # Potential downsides
```

### Scenario Input Format

Each scenario presented to the agent includes the three state objects and a decision prompt. No pre-computed analytical context is provided.

```json
{
    "scenario_id": "uuid",
    "matchup_state": { ... },
    "roster_state": { ... },
    "opponent_roster_state": { ... },
    "decision_prompt": "Natural language description of the situation and what decision is needed"
}
```

### Agent Flow

1. Agent receives scenario with MatchupState, RosterState, OpponentRosterState, and a decision prompt
2. Agent assesses the situation by calling analytical tools (`get_win_probability`, `get_run_expectancy`) to establish context
3. Agent identifies the decision type and calls relevant data tools (player stats, matchup data, fatigue assessment, etc.)
4. Agent synthesizes the tool results using the analytical framework from the system prompt
5. Agent produces a structured `ManagerDecision` as its response -- this is how the agent communicates its decision, not via a tool call

The simulation engine receives the `ManagerDecision` output, validates it against the current game state and MLB rules, applies it, and advances the game to the next decision point.

---

## 7. Implementation Phases

### Phase 1: Core Framework
- Game state model and scenario input format
- System prompt with analytical framework
- Structured output schema
- 4 core tools: `get_batter_stats`, `get_pitcher_stats`, `get_run_expectancy`, `get_win_probability`

### Phase 2: Decision-Specific Tools
- `evaluate_stolen_base`, `evaluate_sacrifice_bunt`
- `get_matchup_data`, `get_platoon_comparison`
- `get_bullpen_status`, `get_pitcher_fatigue_assessment`

### Phase 3: Defensive and Advanced
- `get_defensive_positioning`, `get_defensive_replacement_value`
- In-game velocity/spin trend tracking
- Multi-game bullpen optimization

### Phase 4: Data Pipeline
- Connect tools to real data sources (pybaseball, MLB Stats API)
- Build pre-computed lookup tables (RE matrix, WP tables)
- Historical scenario generation for testing

### Phase 5: Evaluation
- Generate test scenarios from historical games (Retrosheet play-by-play)
- Compare agent decisions against actual manager decisions
- Compare agent decisions against analytical consensus (The Book recommendations)
- Measure decision quality by expected WPA of agent choices vs alternatives

---

## 8. Data Sources

The tool scripts need access to underlying data. These are the potential sources for populating them.

### Libraries

- **`pybaseball`** (Python) -- pulls Statcast, FanGraphs, and Baseball Reference data via simple API. The primary programmatic interface for historical and current-season stats.
- **MLB Stats API** -- official MLB data feed for live game data, rosters, schedules, and game state. Free and public.

### Databases and Websites

- **Baseball Savant** (baseballsavant.mlb.com) -- Statcast data: pitch-level tracking (velocity, spin, movement, location), batted ball data (exit velocity, launch angle), player tracking (sprint speed, lead distance, jump), expected stats (xBA, xSLG, xwOBA), and catcher framing/pop times.
- **FanGraphs** (fangraphs.com) -- advanced stats (wOBA, wRC+, FIP, xFIP, SIERA), plate discipline metrics, pitch values, defensive metrics (UZR, DRS), splits, leaderboards, and projections (Steamer, ZiPS).
- **Baseball Reference** (baseball-reference.com) -- comprehensive traditional and advanced stats, career splits, batter-vs-pitcher tables, game logs, and historical data.
- **Retrosheet** (retrosheet.org) -- play-by-play event files for every MLB game back to 1921. The standard source for computing run expectancy matrices, win probability tables, and transition probabilities.
- **Brooks Baseball** (brooksbaseball.net) -- detailed pitch-by-pitch breakdowns, release point analysis, and historical pitch data trends.

### Pre-Computed Tables

Some tools will need pre-computed lookup tables rather than live data queries:

- **Run expectancy matrix** -- 24 base-out states, recomputed annually from Retrosheet data or pulled from published sources (FanGraphs, Tom Tango).
- **Win probability tables** -- indexed by inning, half, outs, base state, and score differential. Computed from historical outcomes or available from FanGraphs.
- **Leverage index tables** -- derived from win probability tables, measuring the expected swing in WP for each game state.
- **Stolen base breakeven rates** -- derived from run expectancy matrix, one value per base-out state.

### Reference Texts

- **"The Book: Playing the Percentages in Baseball"** by Tom Tango, Mitchel Lichtman, and Andrew Dolphin -- the definitive analytical reference for managerial decision-making. Covers run expectancy, win probability, platoon splits, sacrifice bunts, stolen bases, intentional walks, pitching changes, and more.
- **Tom Tango's website** (tangotiger.com) -- foundational framework for LI, WPA, and run expectancy analysis, with published tables and tools.
