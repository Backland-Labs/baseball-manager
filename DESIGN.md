# Baseball Manager AI Agent -- Design Document

GOAL: Build an AI agent that watches live MLB games and tweets real-time managerial decisions. The agent receives game state before each at-bat, analyzes the situation using real player data, and outputs a plain-text decision describing what it would do as the team's manager.

---

## 1. Overview

The agent operates as a reactive decision-maker during live MLB games. An external service polls the MLB Stats API for game state updates and pushes the current situation to the agent before each at-bat. The agent:

1. Receives the current game state (matchup, rosters, situation)
2. Calls information-gathering tools to analyze the situation (player stats, matchup data, win probability, etc.)
3. Outputs a plain-text managerial decision suitable for tweeting

The agent manages one specific team per game. Most at-bats require no action (let the current lineup play). When the situation warrants it, the agent makes an active decision: pitching change, pinch hit, stolen base attempt, defensive repositioning, etc.

---

## 2. Architecture

```
MLB Stats API (live game feed)
        |
        v
External Service
  - Polls live feed (~10s interval)
  - Detects new at-bats and state changes
  - Converts MLB API response to agent input models
        |
        v
Agent (Claude + tools)
  - Receives MatchupState + RosterState + OpponentRosterState
  - Calls tools for analytical context (real stats, matchups, probabilities)
  - Outputs plain-text decision
        |
        v
Decision Output
  - Tweets active decisions to Twitter/X
  - Logs all decisions (active and no-action) for analysis
```

The agent is stateless per invocation. Each at-bat is an independent decision. The external service maintains game context (substitution history, bullpen usage, mound visits, etc.) and passes it as part of the game state input.

---

## 3. Decision Points the Agent Must Handle

The agent evaluates the situation before each at-bat and decides whether to take action. The following are the categories of decisions it may make:

### Offensive (When Managed Team is Batting)
| Decision | Key Inputs |
|----------|-----------|
| Sacrifice bunt | Run expectancy matrix, batter quality, outs, runners, score differential, inning |
| Stolen base (straight, double, delayed, hit-and-run) | Runner sprint speed, SB success rate, pitcher hold time, catcher pop time, breakeven threshold |
| Pinch hit | Platoon matchup, bench availability, batter vs pitcher data, leverage index |
| Pinch run | Runner speed differential, score, inning, remaining bench |
| Intentional walk | Current vs on-deck batter quality, base-out state, force/DP setup |
| Hit-and-run | Batter contact rate, runner speed, count, ground ball tendency |
| Squeeze play | Runner on 3rd, batter bunt proficiency, outs, score |

### Pitching (When Managed Team is Fielding)
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

## 4. Game State Model

The agent receives three data structures as input with every at-bat: the **MatchupState**, the **RosterState**, and the **OpponentRosterState**. These are populated by the external service from the MLB Stats API live game feed. Analytical context (win probability, leverage index, run expectancy) is NOT passed directly -- the agent must derive it by calling tools.

### MatchupState (passed as input)

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
    managed_team: "HOME" | "AWAY"

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

### RosterState (passed as input)

Contains the managed team's roster availability.

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

### OpponentRosterState (passed as input)

Contains the opposing team's roster availability.

```
OpponentRosterState {
    their_lineup: [{player_id, name, position, bats, in_game: bool}; 9]
    their_lineup_position: int
    their_bench: [{player_id, name, bats, available: bool}]
    their_bullpen: [{player_id, name, throws, role, available: bool, freshness}]
}
```

---

## 5. Tools

Tools are the agent's mechanism for gathering analytical context beyond what is provided in the input game state. They are strictly informational -- tools query real MLB data sources (Statcast, FanGraphs, MLB Stats API) to provide statistics, projections, and evaluations. The agent's decision is expressed through its plain-text response, not via tool calls.

Each tool is a standalone Python script in the `tools/` directory. The agent passes arguments and receives structured JSON output.

### Player Statistics

| Tool | Description |
|------|-------------|
| `get_batter_stats` | Retrieves real batting statistics from Statcast/FanGraphs: traditional stats (AVG/OBP/SLG), advanced metrics (wOBA, wRC+, barrel rate, xwOBA), plate discipline (K%, BB%, chase rate, whiff rate), batted ball profile (GB%, pull%, EV, LA), and sprint speed. Supports splits by handedness, home/away, and recency. |
| `get_pitcher_stats` | Retrieves real pitching statistics from Statcast/FanGraphs: ERA/FIP/xFIP, K% and BB%, ground ball rate, pitch mix with per-pitch velocity/spin/whiff rates, and times-through-order wOBA splits. Supports splits by batter handedness, home/away, and recency. |
| `get_matchup_data` | Retrieves real batter-vs-pitcher career history from MLB Stats API or Baseball Reference. Returns plate appearances, results, and outcome distribution. For small samples, supplements with similarity-model projections. |

### Game Situation Analysis

| Tool | Description |
|------|-------------|
| `get_run_expectancy` | Returns expected runs for a given base-out state, probability of scoring at least one run, and run distribution. Backed by a pre-computed 24-state run expectancy matrix derived from Retrosheet play-by-play data. |
| `get_win_probability` | Returns win probability, leverage index, and conditional win probabilities (if a run scores, if the inning ends scoreless) given the full game state. Backed by pre-computed win probability tables derived from historical game outcomes. |
| `evaluate_stolen_base` | Evaluates a stolen base attempt given runner speed, SB success rate, pitcher hold time, catcher pop time, target base, and current base-out state. Returns estimated success probability, breakeven rate, expected run-expectancy change, and a recommendation. |
| `evaluate_sacrifice_bunt` | Evaluates whether a sacrifice bunt is optimal given the batter, base-out state, score differential, and inning. Compares bunt vs swing-away expected runs and probability of scoring at least one run. |

### Pitching and Bullpen

| Tool | Description |
|------|-------------|
| `get_bullpen_status` | Returns detailed bullpen status for the managed team: availability, stats, freshness, rest days, recent pitch counts, platoon splits, and warm-up status. Data sourced from the game state and supplemented with season stats from Statcast/FanGraphs. |
| `get_pitcher_fatigue_assessment` | Assesses the current pitcher's fatigue based on in-game trends from the live game feed: velocity changes, spin rate decline, batted ball quality trend, pitch count, times through order, and an overall fatigue rating. |

### Defense

| Tool | Description |
|------|-------------|
| `get_defensive_positioning` | Returns recommended infield and outfield positioning for a given batter-pitcher matchup and game situation. Uses the batter's real spray chart data from Statcast. |
| `get_defensive_replacement_value` | Evaluates the net value of a defensive substitution by comparing the defensive upgrade (OAA difference) against the offensive downgrade, scaled by estimated innings remaining. Uses real defensive metrics from Statcast. |

### Platoon and Pinch-Hitting

| Tool | Description |
|------|-------------|
| `get_platoon_comparison` | Compares a potential pinch hitter against the current batter for the active matchup. Returns each player's projected wOBA vs the current pitcher, the platoon advantage delta, the defensive cost, and the bench depth impact. |

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

---

## 6. Analytical Framework Constants

These values are embedded in the agent's system prompt as reference knowledge for decision-making.

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

## 7. MLB Rules the Agent Must Know

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

## 8. Agent Output

The agent's primary output is a **plain-text decision** suitable for tweeting. The output should read like a baseball manager explaining their move to a broadcast audience.

### When making an active decision:

> Bottom of the 7th, bases loaded, 2 outs. Pulling Smith off the mound -- he's seeing the lineup for the 3rd time and his fastball is down 2.3 mph from the 1st inning. Bringing in Rivera from the pen. His .198 wOBA vs lefties is exactly what we need against Johnson.

### When no action is needed:

The agent responds with a brief "no action" indication. These are not tweeted.

### Output characteristics:

- Conversational but analytically grounded
- References specific stats and matchup data to justify the decision
- Describes the game situation for context
- Names specific players involved
- Fits within tweet length constraints (~280 characters for the core decision)
- The reasoning can be longer for logging purposes, but the tweet-ready text must be concise

---

## 9. Agent Architecture (Claude Agent SDK)

### Technology Stack

- **SDK**: Claude Agent SDK (Python, `anthropic` package)
- **Tools**: Standalone Python scripts querying real MLB data
- **Entry point**: Single-file UV script with PEP 723 inline metadata
- **Output**: Plain-text decision

### System Prompt Design

The system prompt establishes:
1. The agent's identity as the manager of a specific MLB team
2. The decision-making framework: assess the situation via tools before deciding
3. Key analytical constants (TTO penalty, platoon splits, breakeven rates)
4. Current MLB rules that constrain decisions
5. That the agent receives MatchupState, RosterState, and OpponentRosterState as input but must use tools for all statistical lookups
6. That the output is plain-text describing the decision, not a structured object
7. That most at-bats require no action -- the agent should only make active decisions when the situation warrants it

### Agent Flow

1. Agent receives the current game state (MatchupState + RosterState + OpponentRosterState)
2. Agent assesses the situation: Is this a decision point? Is the managed team batting or fielding? What's the leverage?
3. If no action is warranted, agent responds with "no action"
4. If a decision is warranted, agent calls relevant tools to gather context (player stats, matchup data, win probability, fatigue assessment, etc.)
5. Agent synthesizes tool results and outputs a plain-text decision with reasoning

---

## 10. Data Sources

### APIs and Libraries

- **MLB Stats API** (statsapi.mlb.com) -- live game feed, rosters, schedules, player info, batter-vs-pitcher history. Free and public. Primary source for game state and roster data.
- **`pybaseball`** (Python) -- pulls Statcast, FanGraphs, and Baseball Reference data. Primary source for player statistics, pitch-level data, and advanced metrics.

### Databases and Websites

- **Baseball Savant** (baseballsavant.mlb.com) -- Statcast data: pitch-level tracking (velocity, spin, movement, location), batted ball data (exit velocity, launch angle), player tracking (sprint speed, lead distance, jump), expected stats (xBA, xSLG, xwOBA), and catcher framing/pop times.
- **FanGraphs** (fangraphs.com) -- advanced stats (wOBA, wRC+, FIP, xFIP, SIERA), plate discipline metrics, pitch values, defensive metrics (UZR, DRS), splits, leaderboards, and projections (Steamer, ZiPS).
- **Baseball Reference** (baseball-reference.com) -- comprehensive traditional and advanced stats, career splits, batter-vs-pitcher tables, game logs, and historical data.
- **Retrosheet** (retrosheet.org) -- play-by-play event files for every MLB game back to 1921. Source for computing run expectancy matrices, win probability tables, and transition probabilities.

### Pre-Computed Tables

Some tools use pre-computed lookup tables rather than live data queries:

- **Run expectancy matrix** -- 24 base-out states, computed from Retrosheet data or published sources (FanGraphs, Tom Tango).
- **Win probability tables** -- indexed by inning, half, outs, base state, and score differential. Computed from historical outcomes.
- **Leverage index tables** -- derived from win probability tables, measuring the expected swing in WP for each game state.
- **Stolen base breakeven rates** -- derived from run expectancy matrix, one value per base-out state.

### Reference Texts

- **"The Book: Playing the Percentages in Baseball"** by Tom Tango, Mitchel Lichtman, and Andrew Dolphin -- the definitive analytical reference for managerial decision-making.
- **Tom Tango's website** (tangotiger.com) -- foundational framework for LI, WPA, and run expectancy analysis.

---

## 11. Implementation Phases

### Phase 1: Data Foundation
- MLB Stats API client for rosters and player lookups
- pybaseball/Statcast integration for player stats
- Pre-computed run expectancy and win probability tables from Retrosheet
- Data caching layer

### Phase 2: Tools (backed by real data)
- Implement all 12 information-gathering tools using real data sources
- Input validation on all tool parameters
- Consistent JSON response format across tools

### Phase 3: Agent Core
- Game state ingestion (parse pushed game state into MatchupState/RosterState/OpponentRosterState)
- Agent system prompt
- Decision engine (receive state, use tools, output decision)
- Plain-text output formatting

### Phase 4: Integration
- External service that polls MLB Stats API live game feed
- Decision logging with full context
- Tweet output integration
- Error handling for API failures, invalid data, rate limits

### Phase 5: Evaluation
- Test against historical game situations
- Compare agent decisions against actual manager decisions
- Measure decision quality by expected WPA
