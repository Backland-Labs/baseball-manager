# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic>=0.78.0"]
# ///
"""Baseball manager agent tools -- information-gathering tools for the Claude agent."""

from tools.get_batter_stats import get_batter_stats
from tools.get_pitcher_stats import get_pitcher_stats
from tools.get_matchup_data import get_matchup_data
from tools.get_run_expectancy import get_run_expectancy
from tools.get_win_probability import get_win_probability
from tools.evaluate_stolen_base import evaluate_stolen_base
from tools.evaluate_sacrifice_bunt import evaluate_sacrifice_bunt
from tools.get_bullpen_status import get_bullpen_status
from tools.get_pitcher_fatigue_assessment import get_pitcher_fatigue_assessment
from tools.get_defensive_positioning import get_defensive_positioning
from tools.get_defensive_replacement_value import get_defensive_replacement_value
from tools.get_platoon_comparison import get_platoon_comparison

ALL_TOOLS = [
    get_batter_stats,
    get_pitcher_stats,
    get_matchup_data,
    get_run_expectancy,
    get_win_probability,
    evaluate_stolen_base,
    evaluate_sacrifice_bunt,
    get_bullpen_status,
    get_pitcher_fatigue_assessment,
    get_defensive_positioning,
    get_defensive_replacement_value,
    get_platoon_comparison,
]
