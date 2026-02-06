# /// script
# requires-python = ">=3.12"
# dependencies = ["pytest>=7.0"]
# ///
"""Tests for the mlb_stats_api_client feature.

Validates the core MLB Stats API client module (data/mlb_api.py):
  1. Fetch current 26-man active roster for a team by team ID or name
  2. Fetch player metadata: name, position, handedness (bats/throws), jersey number
  3. Fetch live game feed by gamePk: current inning, outs, count, runners, score, batter, pitcher
  4. Fetch game schedule by date to find active gamePks
  5. Fetch batter-vs-pitcher career stats from the vsPlayer endpoint
  6. Map MLB Stats API player IDs to the fields needed by the agent models
  7. Handle API errors gracefully: connection failures, invalid IDs, off-season empty responses
  8. Add request timeout and basic retry logic for transient failures
"""

import json
import sys
import tempfile
import time
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from data.cache import Cache
from data.mlb_api import (
    BASE_URL,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    TEAM_IDS,
    TEAM_NAMES,
    MLBApiError,
    MLBApiNotFoundError,
    MLBApiConnectionError,
    MLBApiTimeoutError,
    _fetch_json,
    _build_url,
    lookup_team_id,
    get_team_name,
    get_team_roster,
    get_player_info,
    get_players_info,
    extract_player_metadata,
    extract_roster_metadata,
    get_live_game_feed,
    extract_game_situation,
    extract_pitcher_game_stats,
    get_schedule_by_date,
    find_active_game_pks,
    get_batter_vs_pitcher,
    map_roster_to_model_fields,
    map_game_situation_to_matchup_state,
    extract_bvp_stats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache(tmp_path):
    """Return a Cache instance backed by a temporary directory."""
    return Cache(root_dir=tmp_path / "cache")


@pytest.fixture
def sample_roster_response():
    """Sample MLB Stats API roster response."""
    return {
        "copyright": "Copyright 2026 MLB...",
        "roster": [
            {
                "person": {
                    "id": 660271,
                    "fullName": "Shohei Ohtani",
                    "link": "/api/v1/people/660271",
                    "firstName": "Shohei",
                    "lastName": "Ohtani",
                    "primaryNumber": "17",
                    "primaryPosition": {
                        "code": "Y",
                        "name": "Two-Way Player",
                        "type": "Two-Way Player",
                        "abbreviation": "TWP",
                    },
                    "batSide": {"code": "L", "description": "Left"},
                    "pitchHand": {"code": "R", "description": "Right"},
                    "active": True,
                },
                "jerseyNumber": "17",
                "position": {
                    "code": "10",
                    "name": "Designated Hitter",
                    "type": "Hitter",
                    "abbreviation": "DH",
                },
                "status": {"code": "A", "description": "Active"},
                "parentTeamId": 119,
            },
            {
                "person": {
                    "id": 605141,
                    "fullName": "Mookie Betts",
                    "link": "/api/v1/people/605141",
                    "firstName": "Markus",
                    "lastName": "Betts",
                    "primaryNumber": "50",
                    "primaryPosition": {
                        "code": "6",
                        "name": "Shortstop",
                        "type": "Infielder",
                        "abbreviation": "SS",
                    },
                    "batSide": {"code": "R", "description": "Right"},
                    "pitchHand": {"code": "R", "description": "Right"},
                    "active": True,
                },
                "jerseyNumber": "50",
                "position": {
                    "code": "6",
                    "name": "Shortstop",
                    "type": "Infielder",
                    "abbreviation": "SS",
                },
                "status": {"code": "A", "description": "Active"},
                "parentTeamId": 119,
            },
        ],
    }


@pytest.fixture
def sample_player_response():
    """Sample MLB Stats API player response."""
    return {
        "copyright": "Copyright 2026 MLB...",
        "people": [
            {
                "id": 545361,
                "fullName": "Mike Trout",
                "firstName": "Michael",
                "lastName": "Trout",
                "primaryNumber": "27",
                "birthDate": "1991-08-07",
                "currentAge": 34,
                "height": "6' 2\"",
                "weight": 235,
                "active": True,
                "primaryPosition": {
                    "code": "8",
                    "name": "Center Field",
                    "type": "Outfielder",
                    "abbreviation": "CF",
                },
                "batSide": {"code": "R", "description": "Right"},
                "pitchHand": {"code": "R", "description": "Right"},
            }
        ],
    }


@pytest.fixture
def sample_schedule_response():
    """Sample MLB Stats API schedule response."""
    return {
        "copyright": "Copyright 2026 MLB...",
        "totalItems": 2,
        "totalGames": 2,
        "dates": [
            {
                "date": "2024-10-01",
                "totalGames": 2,
                "games": [
                    {
                        "gamePk": 775345,
                        "gameDate": "2024-10-01T18:32:00Z",
                        "officialDate": "2024-10-01",
                        "status": {
                            "abstractGameState": "Final",
                            "detailedState": "Final",
                            "statusCode": "F",
                        },
                        "teams": {
                            "away": {
                                "team": {"id": 116, "name": "Detroit Tigers"},
                                "score": 3,
                                "isWinner": True,
                            },
                            "home": {
                                "team": {"id": 117, "name": "Houston Astros"},
                                "score": 1,
                                "isWinner": False,
                            },
                        },
                        "venue": {"id": 2392, "name": "Minute Maid Park"},
                    },
                    {
                        "gamePk": 775346,
                        "gameDate": "2024-10-01T22:08:00Z",
                        "officialDate": "2024-10-01",
                        "status": {
                            "abstractGameState": "Live",
                            "detailedState": "In Progress",
                            "statusCode": "I",
                        },
                        "teams": {
                            "away": {
                                "team": {"id": 111, "name": "Boston Red Sox"},
                                "score": 2,
                            },
                            "home": {
                                "team": {"id": 147, "name": "New York Yankees"},
                                "score": 4,
                            },
                        },
                        "venue": {"id": 3313, "name": "Yankee Stadium"},
                    },
                ],
            }
        ],
    }


@pytest.fixture
def sample_live_feed():
    """Sample MLB Stats API live game feed response."""
    return {
        "copyright": "Copyright 2026 MLB...",
        "gamePk": 775345,
        "gameData": {
            "status": {
                "abstractGameState": "Live",
                "detailedState": "In Progress",
            },
            "teams": {
                "away": {"id": 116, "name": "Detroit Tigers"},
                "home": {"id": 117, "name": "Houston Astros"},
            },
            "players": {
                "ID660271": {
                    "id": 660271,
                    "fullName": "Shohei Ohtani",
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "R"},
                },
                "ID434378": {
                    "id": 434378,
                    "fullName": "Justin Verlander",
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                },
            },
        },
        "liveData": {
            "plays": {
                "allPlays": [],
                "currentPlay": {
                    "matchup": {
                        "batter": {"id": 660271, "fullName": "Shohei Ohtani"},
                        "batSide": {"code": "L"},
                        "pitcher": {"id": 434378, "fullName": "Justin Verlander"},
                        "pitchHand": {"code": "R"},
                    },
                    "count": {"balls": 2, "strikes": 1, "outs": 1},
                },
            },
            "linescore": {
                "currentInning": 5,
                "isTopInning": False,
                "outs": 1,
                "teams": {
                    "home": {"runs": 3, "hits": 7, "errors": 0},
                    "away": {"runs": 2, "hits": 5, "errors": 1},
                },
                "offense": {
                    "first": {"id": 605141, "fullName": "Mookie Betts"},
                    "second": None,
                    "third": None,
                    "ondeck": {"id": 605141, "fullName": "Mookie Betts"},
                },
            },
            "boxscore": {
                "teams": {
                    "home": {
                        "players": {},
                    },
                    "away": {
                        "players": {
                            "ID434378": {
                                "person": {"id": 434378, "fullName": "Justin Verlander"},
                                "stats": {
                                    "pitching": {
                                        "inningsPitched": "5.0",
                                        "hits": 7,
                                        "runs": 3,
                                        "earnedRuns": 3,
                                        "baseOnBalls": 2,
                                        "strikeOuts": 6,
                                        "numberOfPitches": 88,
                                        "battersFaced": 22,
                                        "homeRuns": 1,
                                    }
                                },
                            }
                        },
                    },
                },
            },
        },
    }


@pytest.fixture
def sample_bvp_response():
    """Sample MLB Stats API batter-vs-pitcher response."""
    return {
        "copyright": "Copyright 2026 MLB...",
        "stats": [
            {
                "type": {"displayName": "vsPlayerTotal"},
                "group": {"displayName": "hitting"},
                "totalSplits": 1,
                "splits": [
                    {
                        "stat": {
                            "gamesPlayed": 18,
                            "groundOuts": 5,
                            "airOuts": 22,
                            "doubles": 2,
                            "triples": 0,
                            "homeRuns": 3,
                            "strikeOuts": 13,
                            "baseOnBalls": 9,
                            "hits": 12,
                            "avg": ".267",
                            "atBats": 45,
                            "obp": ".370",
                            "slg": ".489",
                            "ops": ".859",
                            "plateAppearances": 56,
                        },
                        "batter": {"id": 545361, "fullName": "Mike Trout"},
                        "pitcher": {"id": 434378, "fullName": "Justin Verlander"},
                    }
                ],
            },
            {
                "type": {"displayName": "vsPlayer"},
                "group": {"displayName": "hitting"},
                "totalSplits": 3,
                "splits": [
                    {
                        "season": "2012",
                        "stat": {"gamesPlayed": 2, "hits": 1, "homeRuns": 1, "avg": ".333"},
                        "team": {"id": 108, "name": "Los Angeles Angels"},
                        "opponent": {"id": 116, "name": "Detroit Tigers"},
                        "batter": {"id": 545361, "fullName": "Mike Trout"},
                        "pitcher": {"id": 434378, "fullName": "Justin Verlander"},
                    },
                    {
                        "season": "2014",
                        "stat": {"gamesPlayed": 4, "hits": 3, "homeRuns": 0, "avg": ".250"},
                        "team": {"id": 108, "name": "Los Angeles Angels"},
                        "opponent": {"id": 116, "name": "Detroit Tigers"},
                        "batter": {"id": 545361, "fullName": "Mike Trout"},
                        "pitcher": {"id": 434378, "fullName": "Justin Verlander"},
                    },
                    {
                        "season": "2019",
                        "stat": {"gamesPlayed": 5, "hits": 4, "homeRuns": 2, "avg": ".308"},
                        "team": {"id": 108, "name": "Los Angeles Angels"},
                        "opponent": {"id": 117, "name": "Houston Astros"},
                        "batter": {"id": 545361, "fullName": "Mike Trout"},
                        "pitcher": {"id": 434378, "fullName": "Justin Verlander"},
                    },
                ],
            },
        ],
    }


@pytest.fixture
def sample_bvp_empty_response():
    """Sample BvP response when no matchup history exists."""
    return {
        "copyright": "Copyright 2026 MLB...",
        "stats": [
            {
                "type": {"displayName": "vsPlayerTotal"},
                "group": {"displayName": "hitting"},
                "totalSplits": 0,
                "splits": [],
            },
            {
                "type": {"displayName": "vsPlayer"},
                "group": {"displayName": "hitting"},
                "totalSplits": 0,
                "splits": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Helper to mock urllib.request.urlopen
# ---------------------------------------------------------------------------

def _mock_urlopen(response_data: dict, status: int = 200):
    """Create a mock for urllib.request.urlopen that returns JSON data."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ===========================================================================
# Step 1: Fetch current 26-man active roster for a team by team ID or name
# ===========================================================================

class TestStep1RosterFetching:
    """Test roster fetching by team ID or name."""

    def test_team_ids_has_all_30_teams(self):
        """All 30 MLB teams should be represented in TEAM_NAMES."""
        assert len(TEAM_NAMES) == 30

    def test_lookup_team_id_by_name(self):
        assert lookup_team_id("Red Sox") == 111
        assert lookup_team_id("red sox") == 111

    def test_lookup_team_id_by_abbreviation(self):
        assert lookup_team_id("BOS") == 111
        assert lookup_team_id("bos") == 111
        assert lookup_team_id("NYY") == 147
        assert lookup_team_id("LAD") == 119

    def test_lookup_team_id_by_full_name(self):
        assert lookup_team_id("Boston Red Sox") == 111
        assert lookup_team_id("New York Yankees") == 147

    def test_lookup_team_id_by_int(self):
        assert lookup_team_id(111) == 111
        assert lookup_team_id(147) == 147

    def test_lookup_team_id_by_numeric_string(self):
        assert lookup_team_id("111") == 111

    def test_lookup_team_id_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown team"):
            lookup_team_id("Nonexistent Team")

    def test_lookup_team_id_invalid_int_raises(self):
        with pytest.raises(ValueError, match="Unknown team ID"):
            lookup_team_id(999)

    def test_get_team_name(self):
        assert get_team_name(111) == "Boston Red Sox"
        assert get_team_name(147) == "New York Yankees"

    def test_get_team_name_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown team ID"):
            get_team_name(999)

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_by_id(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        roster = get_team_roster(119, cache=tmp_cache)
        assert len(roster) == 2
        assert roster[0]["person"]["fullName"] == "Shohei Ohtani"

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_by_name(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        roster = get_team_roster("Dodgers", cache=tmp_cache)
        assert len(roster) == 2
        mock_fetch.assert_called_once()
        # Verify the URL includes team ID 119
        call_url = mock_fetch.call_args[0][0]
        assert "teams/119/roster" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_by_abbreviation(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        roster = get_team_roster("LAD", cache=tmp_cache)
        assert len(roster) == 2

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_uses_active_type_by_default(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        get_team_roster(119, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "rosterType=active" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_respects_season_param(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        get_team_roster(119, season=2023, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "season=2023" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_hydrates_person_by_default(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        get_team_roster(119, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "hydrate=person" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_caches_result(self, mock_fetch, sample_roster_response, tmp_cache):
        mock_fetch.return_value = sample_roster_response
        # First call fetches from API
        roster1 = get_team_roster(119, cache=tmp_cache)
        assert mock_fetch.call_count == 1
        # Second call should use cache
        roster2 = get_team_roster(119, cache=tmp_cache)
        assert mock_fetch.call_count == 1
        assert roster1 == roster2

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_empty_response(self, mock_fetch, tmp_cache):
        mock_fetch.return_value = {"roster": []}
        roster = get_team_roster(119, cache=tmp_cache)
        assert roster == []


# ===========================================================================
# Step 2: Fetch player metadata
# ===========================================================================

class TestStep2PlayerMetadata:
    """Test player metadata fetching."""

    @patch("data.mlb_api._fetch_json")
    def test_get_player_info(self, mock_fetch, sample_player_response, tmp_cache):
        mock_fetch.return_value = sample_player_response
        player = get_player_info(545361, cache=tmp_cache)
        assert player["fullName"] == "Mike Trout"
        assert player["id"] == 545361
        assert player["primaryPosition"]["abbreviation"] == "CF"
        assert player["batSide"]["code"] == "R"
        assert player["pitchHand"]["code"] == "R"
        assert player["primaryNumber"] == "27"

    @patch("data.mlb_api._fetch_json")
    def test_get_player_info_not_found(self, mock_fetch, tmp_cache):
        mock_fetch.return_value = {"people": []}
        with pytest.raises(MLBApiNotFoundError, match="not found"):
            get_player_info(999999, cache=tmp_cache)

    @patch("data.mlb_api._fetch_json")
    def test_get_player_info_caches_result(self, mock_fetch, sample_player_response, tmp_cache):
        mock_fetch.return_value = sample_player_response
        get_player_info(545361, cache=tmp_cache)
        get_player_info(545361, cache=tmp_cache)
        assert mock_fetch.call_count == 1

    @patch("data.mlb_api._fetch_json")
    def test_get_players_info_batch(self, mock_fetch, tmp_cache):
        mock_fetch.return_value = {
            "people": [
                {"id": 545361, "fullName": "Mike Trout"},
                {"id": 660271, "fullName": "Shohei Ohtani"},
            ]
        }
        players = get_players_info([545361, 660271], cache=tmp_cache)
        assert len(players) == 2
        assert players[0]["fullName"] == "Mike Trout"
        assert players[1]["fullName"] == "Shohei Ohtani"

    def test_get_players_info_empty_list(self, tmp_cache):
        """Empty input should return empty list without API call."""
        result = get_players_info([], cache=tmp_cache)
        assert result == []

    @patch("data.mlb_api._fetch_json")
    def test_get_players_info_uses_cache_for_known_players(self, mock_fetch, sample_player_response, tmp_cache):
        mock_fetch.return_value = sample_player_response
        # Pre-cache one player
        get_player_info(545361, cache=tmp_cache)
        mock_fetch.reset_mock()

        # Now fetch batch -- 545361 should be cached, only 660271 fetched
        mock_fetch.return_value = {"people": [{"id": 660271, "fullName": "Shohei Ohtani"}]}
        players = get_players_info([545361, 660271], cache=tmp_cache)
        assert len(players) == 2
        # Only one API call for the uncached player
        assert mock_fetch.call_count == 1

    def test_extract_player_metadata(self, sample_player_response):
        player = sample_player_response["people"][0]
        meta = extract_player_metadata(player)
        assert meta["player_id"] == 545361
        assert meta["name"] == "Mike Trout"
        assert meta["position"] == "Center Field"
        assert meta["position_abbreviation"] == "CF"
        assert meta["bats"] == "R"
        assert meta["throws"] == "R"
        assert meta["jersey_number"] == "27"
        assert meta["active"] is True

    def test_extract_player_metadata_missing_fields(self):
        """Should handle a player dict with minimal fields."""
        meta = extract_player_metadata({"id": 1, "fullName": "Test"})
        assert meta["player_id"] == 1
        assert meta["name"] == "Test"
        assert meta["bats"] == ""
        assert meta["throws"] == ""
        assert meta["jersey_number"] == ""

    def test_extract_roster_metadata(self, sample_roster_response):
        roster = sample_roster_response["roster"]
        result = extract_roster_metadata(roster)
        assert len(result) == 2

        ohtani = result[0]
        assert ohtani["player_id"] == 660271
        assert ohtani["name"] == "Shohei Ohtani"
        # Roster-level position overrides person-level
        assert ohtani["position_abbreviation"] == "DH"
        assert ohtani["bats"] == "L"
        assert ohtani["throws"] == "R"
        assert ohtani["jersey_number"] == "17"

        betts = result[1]
        assert betts["player_id"] == 605141
        assert betts["name"] == "Mookie Betts"
        assert betts["position_abbreviation"] == "SS"


# ===========================================================================
# Step 3: Fetch live game feed by gamePk
# ===========================================================================

class TestStep3LiveGameFeed:
    """Test live game feed fetching and extraction."""

    @patch("data.mlb_api._fetch_json")
    def test_get_live_game_feed(self, mock_fetch, sample_live_feed, tmp_cache):
        mock_fetch.return_value = sample_live_feed
        feed = get_live_game_feed(775345, cache=tmp_cache)
        assert feed["gamePk"] == 775345
        assert "gameData" in feed
        assert "liveData" in feed

    @patch("data.mlb_api._fetch_json")
    def test_get_live_game_feed_uses_v11(self, mock_fetch, sample_live_feed, tmp_cache):
        mock_fetch.return_value = sample_live_feed
        get_live_game_feed(775345, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "/v1.1/game/" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_live_game_feed_caches_with_short_ttl(self, mock_fetch, sample_live_feed, tmp_cache):
        mock_fetch.return_value = sample_live_feed
        get_live_game_feed(775345, cache=tmp_cache)
        get_live_game_feed(775345, cache=tmp_cache)
        assert mock_fetch.call_count == 1

    @patch("data.mlb_api._fetch_json")
    def test_get_live_game_feed_not_found(self, mock_fetch, tmp_cache):
        mock_fetch.side_effect = MLBApiNotFoundError("Not found", status_code=404)
        with pytest.raises(MLBApiNotFoundError):
            get_live_game_feed(999999, cache=tmp_cache)

    def test_extract_game_situation(self, sample_live_feed):
        situation = extract_game_situation(sample_live_feed)
        assert situation["inning"] == 5
        assert situation["half"] == "BOTTOM"
        assert situation["outs"] == 1
        assert situation["count"] == {"balls": 2, "strikes": 1}
        assert situation["score"] == {"home": 3, "away": 2}
        assert situation["batter"]["fullName"] == "Shohei Ohtani"
        assert situation["pitcher"]["fullName"] == "Justin Verlander"
        assert situation["game_status"] == "Live"

    def test_extract_game_situation_runners(self, sample_live_feed):
        situation = extract_game_situation(sample_live_feed)
        assert situation["runners"]["first"]["fullName"] == "Mookie Betts"
        assert situation["runners"]["second"] is None
        assert situation["runners"]["third"] is None

    def test_extract_game_situation_on_deck(self, sample_live_feed):
        situation = extract_game_situation(sample_live_feed)
        assert situation["on_deck_batter"]["fullName"] == "Mookie Betts"

    def test_extract_game_situation_empty_feed(self):
        """Handle an empty/minimal feed gracefully."""
        feed = {"gameData": {}, "liveData": {}}
        situation = extract_game_situation(feed)
        assert situation["inning"] == 0
        assert situation["half"] == "TOP"
        assert situation["outs"] == 0
        assert situation["game_status"] == "Unknown"

    def test_extract_pitcher_game_stats(self, sample_live_feed):
        stats = extract_pitcher_game_stats(sample_live_feed, 434378)
        assert stats["innings_pitched"] == "5.0"
        assert stats["hits"] == 7
        assert stats["runs"] == 3
        assert stats["earned_runs"] == 3
        assert stats["walks"] == 2
        assert stats["strikeouts"] == 6
        assert stats["pitch_count"] == 88
        assert stats["batters_faced"] == 22
        assert stats["home_runs_allowed"] == 1

    def test_extract_pitcher_game_stats_not_found(self, sample_live_feed):
        stats = extract_pitcher_game_stats(sample_live_feed, 999999)
        assert stats == {}


# ===========================================================================
# Step 4: Fetch game schedule by date
# ===========================================================================

class TestStep4ScheduleFetching:
    """Test schedule fetching by date."""

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_by_date(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        games = get_schedule_by_date("2024-10-01", cache=tmp_cache)
        assert len(games) == 2
        assert games[0]["gamePk"] == 775345
        assert games[1]["gamePk"] == 775346

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_by_date_url_params(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        get_schedule_by_date("2024-10-01", cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "sportId=1" in call_url
        assert "date=2024-10-01" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_by_date_with_team_filter(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        get_schedule_by_date("2024-10-01", team_id=116, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "teamId=116" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_empty_date(self, mock_fetch, tmp_cache):
        mock_fetch.return_value = {"dates": []}
        games = get_schedule_by_date("2024-12-25", cache=tmp_cache)
        assert games == []

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_caches_result(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        get_schedule_by_date("2024-10-01", cache=tmp_cache)
        get_schedule_by_date("2024-10-01", cache=tmp_cache)
        assert mock_fetch.call_count == 1

    @patch("data.mlb_api._fetch_json")
    def test_find_active_game_pks(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        # Monkey-patch to use tmp_cache
        with patch("data.mlb_api.get_default_cache", return_value=tmp_cache):
            pks = find_active_game_pks("2024-10-01")
        assert 775345 in pks
        assert 775346 in pks

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_game_structure(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        games = get_schedule_by_date("2024-10-01", cache=tmp_cache)
        game = games[0]
        assert "gamePk" in game
        assert "gameDate" in game
        assert "status" in game
        assert "teams" in game
        assert "away" in game["teams"]
        assert "home" in game["teams"]
        assert "venue" in game


# ===========================================================================
# Step 5: Fetch batter-vs-pitcher career stats
# ===========================================================================

class TestStep5BatterVsPitcher:
    """Test batter-vs-pitcher career stats fetching."""

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_career_stats(self, mock_fetch, sample_bvp_response, tmp_cache):
        mock_fetch.return_value = sample_bvp_response
        result = get_batter_vs_pitcher(545361, 434378, cache=tmp_cache)
        assert result["plate_appearances"] == 56
        assert result["small_sample"] is False
        assert result["batter"]["fullName"] == "Mike Trout"
        assert result["pitcher"]["fullName"] == "Justin Verlander"
        assert result["career"] is not None
        assert result["career"]["avg"] == ".267"
        assert result["career"]["homeRuns"] == 3

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_season_splits(self, mock_fetch, sample_bvp_response, tmp_cache):
        mock_fetch.return_value = sample_bvp_response
        result = get_batter_vs_pitcher(545361, 434378, cache=tmp_cache)
        assert len(result["seasons"]) == 3
        assert result["seasons"][0]["season"] == "2012"
        assert result["seasons"][2]["season"] == "2019"

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_no_matchup_history(self, mock_fetch, sample_bvp_empty_response, tmp_cache):
        mock_fetch.return_value = sample_bvp_empty_response
        result = get_batter_vs_pitcher(999, 888, cache=tmp_cache)
        assert result["career"] is None
        assert result["plate_appearances"] == 0
        assert result["small_sample"] is True
        assert result["seasons"] == []

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_small_sample(self, mock_fetch, tmp_cache):
        response = {
            "stats": [
                {
                    "type": {"displayName": "vsPlayerTotal"},
                    "group": {"displayName": "hitting"},
                    "totalSplits": 1,
                    "splits": [{
                        "stat": {"plateAppearances": 5, "avg": ".400", "hits": 2},
                        "batter": {"id": 1, "fullName": "A"},
                        "pitcher": {"id": 2, "fullName": "B"},
                    }],
                },
                {"type": {"displayName": "vsPlayer"}, "splits": []},
            ]
        }
        mock_fetch.return_value = response
        result = get_batter_vs_pitcher(1, 2, cache=tmp_cache)
        assert result["small_sample"] is True
        assert result["plate_appearances"] == 5

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_caches_result(self, mock_fetch, sample_bvp_response, tmp_cache):
        mock_fetch.return_value = sample_bvp_response
        get_batter_vs_pitcher(545361, 434378, cache=tmp_cache)
        get_batter_vs_pitcher(545361, 434378, cache=tmp_cache)
        assert mock_fetch.call_count == 1

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_url_params(self, mock_fetch, sample_bvp_response, tmp_cache):
        mock_fetch.return_value = sample_bvp_response
        get_batter_vs_pitcher(545361, 434378, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "people/545361/stats" in call_url
        assert "stats=vsPlayer" in call_url
        assert "group=hitting" in call_url
        assert "opposingPlayerId=434378" in call_url

    def test_extract_bvp_stats(self, sample_bvp_response):
        bvp = {
            "career": sample_bvp_response["stats"][0]["splits"][0]["stat"],
            "seasons": [],
            "plate_appearances": 56,
            "small_sample": False,
            "batter": {"fullName": "Mike Trout"},
            "pitcher": {"fullName": "Justin Verlander"},
        }
        stats = extract_bvp_stats(bvp)
        assert stats["plate_appearances"] == 56
        assert stats["hits"] == 12
        assert stats["home_runs"] == 3
        assert stats["strikeouts"] == 13
        assert stats["walks"] == 9
        assert stats["avg"] == 0.267
        assert stats["obp"] == 0.370
        assert stats["slg"] == 0.489
        assert stats["ops"] == 0.859
        assert stats["small_sample"] is False
        assert stats["batter_name"] == "Mike Trout"
        assert stats["pitcher_name"] == "Justin Verlander"

    def test_extract_bvp_stats_no_history(self):
        bvp = {
            "career": None,
            "seasons": [],
            "plate_appearances": 0,
            "small_sample": True,
            "batter": {"fullName": "Rookie"},
            "pitcher": {"fullName": "Veteran"},
        }
        stats = extract_bvp_stats(bvp)
        assert stats["plate_appearances"] == 0
        assert stats["avg"] is None
        assert stats["small_sample"] is True
        assert stats["batter_name"] == "Rookie"

    def test_extract_bvp_stats_undefined_values(self):
        """Handle -.-- stat values (e.g., atBatsPerHomeRun when 0 HR)."""
        bvp = {
            "career": {
                "plateAppearances": 3,
                "atBats": 3,
                "hits": 0,
                "doubles": 0,
                "triples": 0,
                "homeRuns": 0,
                "baseOnBalls": 0,
                "strikeOuts": 2,
                "avg": ".000",
                "obp": ".000",
                "slg": ".000",
                "ops": ".000",
            },
            "seasons": [],
            "plate_appearances": 3,
            "small_sample": True,
            "batter": {},
            "pitcher": {},
        }
        stats = extract_bvp_stats(bvp)
        assert stats["avg"] == 0.0
        assert stats["ops"] == 0.0


# ===========================================================================
# Step 6: Map MLB Stats API fields to agent models
# ===========================================================================

class TestStep6ModelMapping:
    """Test mapping API fields to the agent model fields."""

    def test_map_roster_to_model_fields(self, sample_roster_response):
        roster = sample_roster_response["roster"]
        mapped = map_roster_to_model_fields(roster)
        assert len(mapped) == 2

        ohtani = mapped[0]
        assert ohtani["player_id"] == "660271"
        assert ohtani["name"] == "Shohei Ohtani"
        assert ohtani["position"] == "DH"
        assert ohtani["bats"] == "L"
        assert ohtani["throws"] == "R"
        assert ohtani["jersey_number"] == "17"
        assert ohtani["is_pitcher"] is False

        betts = mapped[1]
        assert betts["player_id"] == "605141"
        assert betts["name"] == "Mookie Betts"
        assert betts["position"] == "SS"

    def test_map_roster_to_model_fields_pitcher(self):
        roster = [{
            "person": {
                "id": 434378,
                "fullName": "Justin Verlander",
                "batSide": {"code": "R"},
                "pitchHand": {"code": "R"},
            },
            "jerseyNumber": "35",
            "position": {
                "code": "1",
                "name": "Pitcher",
                "type": "Pitcher",
                "abbreviation": "P",
            },
        }]
        mapped = map_roster_to_model_fields(roster)
        assert mapped[0]["is_pitcher"] is True
        assert mapped[0]["position"] == "P"

    def test_map_game_situation_to_matchup_state(self, sample_live_feed):
        result = map_game_situation_to_matchup_state(sample_live_feed, "home")
        assert result["inning"] == 5
        assert result["half"] == "BOTTOM"
        assert result["outs"] == 1
        assert result["count"] == {"balls": 2, "strikes": 1}
        assert result["score"] == {"home": 3, "away": 2}
        assert result["batting_team"] == "HOME"
        assert result["managed_team"] == "HOME"
        assert result["batter"]["name"] == "Shohei Ohtani"
        assert result["batter"]["bats"] == "L"
        assert result["pitcher"]["name"] == "Justin Verlander"
        assert result["pitcher"]["throws"] == "R"
        assert result["pitcher"]["pitch_count_today"] == 88
        assert result["game_status"] == "Live"

    def test_map_game_situation_to_matchup_state_away(self, sample_live_feed):
        result = map_game_situation_to_matchup_state(sample_live_feed, "away")
        assert result["managed_team"] == "AWAY"

    def test_map_game_situation_pitcher_line(self, sample_live_feed):
        result = map_game_situation_to_matchup_state(sample_live_feed, "home")
        line = result["pitcher"]["today_line"]
        assert line["IP"] == 5.0
        assert line["H"] == 7
        assert line["R"] == 3
        assert line["ER"] == 3
        assert line["BB"] == 2
        assert line["K"] == 6

    def test_map_game_situation_runners(self, sample_live_feed):
        result = map_game_situation_to_matchup_state(sample_live_feed, "home")
        assert "first" in result["runners"]
        assert result["runners"]["first"]["name"] == "Mookie Betts"


# ===========================================================================
# Step 7: Handle API errors gracefully
# ===========================================================================

class TestStep7ErrorHandling:
    """Test error handling for connection failures, invalid IDs, etc."""

    def test_mlb_api_error_has_status_code(self):
        err = MLBApiError("test", status_code=500, url="http://example.com")
        assert err.status_code == 500
        assert err.url == "http://example.com"
        assert "test" in str(err)

    def test_mlb_api_not_found_error_is_subclass(self):
        assert issubclass(MLBApiNotFoundError, MLBApiError)

    def test_mlb_api_connection_error_is_subclass(self):
        assert issubclass(MLBApiConnectionError, MLBApiError)

    def test_mlb_api_timeout_error_is_subclass(self):
        assert issubclass(MLBApiTimeoutError, MLBApiError)

    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_404_raises_not_found(self, mock_urlopen):
        error = urllib.error.HTTPError(
            "http://test", 404, "Not Found", {}, None
        )
        mock_urlopen.side_effect = error
        with pytest.raises(MLBApiNotFoundError):
            _fetch_json("http://test")

    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_400_raises_error(self, mock_urlopen):
        error = urllib.error.HTTPError(
            "http://test", 400, "Bad Request", {}, None
        )
        mock_urlopen.side_effect = error
        with pytest.raises(MLBApiError) as exc_info:
            _fetch_json("http://test")
        assert exc_info.value.status_code == 400

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_500_retries(self, mock_urlopen, mock_sleep):
        error = urllib.error.HTTPError(
            "http://test", 500, "Server Error", {}, None
        )
        # First two calls fail, third succeeds
        mock_resp = _mock_urlopen({"success": True})
        mock_urlopen.side_effect = [error, error, mock_resp]
        result = _fetch_json("http://test", max_retries=3)
        assert result == {"success": True}
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_500_all_retries_fail(self, mock_urlopen, mock_sleep):
        error = urllib.error.HTTPError(
            "http://test", 500, "Server Error", {}, None
        )
        mock_urlopen.side_effect = error
        with pytest.raises(MLBApiConnectionError, match="Failed after"):
            _fetch_json("http://test", max_retries=3)
        assert mock_urlopen.call_count == 3

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_timeout_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = TimeoutError("timed out")
        with pytest.raises(MLBApiTimeoutError):
            _fetch_json("http://test", max_retries=2)
        assert mock_urlopen.call_count == 2

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_connection_error_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(MLBApiConnectionError):
            _fetch_json("http://test", max_retries=2)
        assert mock_urlopen.call_count == 2

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_os_error_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = OSError("Network unreachable")
        with pytest.raises(MLBApiConnectionError, match="Network unreachable"):
            _fetch_json("http://test", max_retries=2)
        assert mock_urlopen.call_count == 2

    @patch("data.mlb_api._fetch_json")
    def test_get_team_roster_api_error_propagates(self, mock_fetch, tmp_cache):
        mock_fetch.side_effect = MLBApiConnectionError("Connection failed")
        with pytest.raises(MLBApiConnectionError):
            get_team_roster(119, cache=tmp_cache)

    @patch("data.mlb_api._fetch_json")
    def test_get_player_info_api_not_found_propagates(self, mock_fetch, tmp_cache):
        mock_fetch.side_effect = MLBApiNotFoundError("Not found", status_code=404)
        with pytest.raises(MLBApiNotFoundError):
            get_player_info(999999, cache=tmp_cache)

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_empty_dates_not_error(self, mock_fetch, tmp_cache):
        """Off-season dates return empty list, not an error."""
        mock_fetch.return_value = {"dates": []}
        games = get_schedule_by_date("2024-12-25", cache=tmp_cache)
        assert games == []

    def test_invalid_team_raises_value_error(self):
        with pytest.raises(ValueError):
            lookup_team_id("Nonexistent")


# ===========================================================================
# Step 8: Request timeout and retry logic
# ===========================================================================

class TestStep8TimeoutAndRetry:
    """Test timeout configuration and retry behavior."""

    def test_default_timeout_value(self):
        assert DEFAULT_TIMEOUT == 10

    def test_max_retries_value(self):
        assert MAX_RETRIES == 3

    def test_build_url_basic(self):
        url = _build_url("v1", "teams/111/roster")
        assert url == f"{BASE_URL}/v1/teams/111/roster"

    def test_build_url_with_params(self):
        url = _build_url("v1", "teams/111/roster", {"rosterType": "active", "season": 2024})
        assert f"{BASE_URL}/v1/teams/111/roster?" in url
        assert "rosterType=active" in url
        assert "season=2024" in url

    def test_build_url_filters_none_params(self):
        url = _build_url("v1", "teams/111/roster", {"rosterType": "active", "season": None})
        assert "season" not in url
        assert "rosterType=active" in url

    def test_build_url_v11(self):
        url = _build_url("v1.1", "game/775345/feed/live")
        assert url == f"{BASE_URL}/v1.1/game/775345/feed/live"

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_retry_uses_exponential_backoff(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = OSError("fail")
        with pytest.raises(MLBApiConnectionError):
            _fetch_json("http://test", max_retries=3)
        # Should sleep between retries with increasing delays
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0)  # first retry: backoff(0)
        mock_sleep.assert_any_call(1)  # second retry: backoff(1)

    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_success_no_retries(self, mock_urlopen):
        mock_resp = _mock_urlopen({"data": "test"})
        mock_urlopen.return_value = mock_resp
        result = _fetch_json("http://test")
        assert result == {"data": "test"}
        assert mock_urlopen.call_count == 1

    @patch("data.mlb_api._backoff_sleep")
    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_transient_then_success(self, mock_urlopen, mock_sleep):
        """Transient failure followed by success should return data."""
        error = urllib.error.URLError("Connection refused")
        mock_resp = _mock_urlopen({"recovered": True})
        mock_urlopen.side_effect = [error, mock_resp]
        result = _fetch_json("http://test", max_retries=3)
        assert result == {"recovered": True}
        assert mock_urlopen.call_count == 2


# ===========================================================================
# Additional integration-style tests
# ===========================================================================

class TestMiscellaneous:
    """Additional tests for edge cases and integration."""

    def test_base_url_is_correct(self):
        assert BASE_URL == "https://statsapi.mlb.com/api"

    def test_team_id_lookup_all_abbreviations(self):
        """Verify all common abbreviations are in the lookup table."""
        common_abbrs = [
            "LAA", "ARI", "BAL", "BOS", "CHC", "CIN", "CLE", "COL",
            "DET", "HOU", "KC", "LAD", "WSH", "NYM", "OAK", "PIT",
            "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "MIN",
            "PHI", "ATL", "CWS", "MIA", "NYY", "MIL",
        ]
        for abbr in common_abbrs:
            team_id = lookup_team_id(abbr)
            assert isinstance(team_id, int), f"Failed for {abbr}"
            assert team_id in TEAM_NAMES

    def test_team_name_lookup_covers_all_ids(self):
        """Every team ID should map to a name."""
        all_ids = set(TEAM_NAMES.keys())
        assert len(all_ids) == 30
        for tid in all_ids:
            name = get_team_name(tid)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_extract_game_situation_top_inning(self):
        feed = {
            "gameData": {"status": {"abstractGameState": "Live"}},
            "liveData": {
                "plays": {"currentPlay": {"matchup": {}, "count": {}}},
                "linescore": {
                    "currentInning": 3,
                    "isTopInning": True,
                    "outs": 2,
                    "teams": {
                        "home": {"runs": 1},
                        "away": {"runs": 0},
                    },
                    "offense": {},
                },
            },
        }
        situation = extract_game_situation(feed)
        assert situation["half"] == "TOP"
        assert situation["inning"] == 3
        assert situation["outs"] == 2

    @patch("data.mlb_api._fetch_json")
    def test_get_live_game_feed_with_fields_param(self, mock_fetch, sample_live_feed, tmp_cache):
        mock_fetch.return_value = sample_live_feed
        get_live_game_feed(775345, fields="gameData,liveData.linescore", cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "fields=" in call_url

    @patch("data.mlb_api._fetch_json")
    def test_get_schedule_with_hydrate(self, mock_fetch, sample_schedule_response, tmp_cache):
        mock_fetch.return_value = sample_schedule_response
        get_schedule_by_date("2024-10-01", hydrate="probablePitcher", cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "hydrate=probablePitcher" in call_url

    def test_map_game_situation_empty_pitcher_stats(self):
        """When pitcher is not in boxscore, stats should be zeroed."""
        feed = {
            "gameData": {
                "status": {"abstractGameState": "Live"},
                "players": {},
            },
            "liveData": {
                "plays": {
                    "currentPlay": {
                        "matchup": {
                            "batter": {"id": 1, "fullName": "Batter"},
                            "pitcher": {"id": 2, "fullName": "Pitcher"},
                        },
                        "count": {"balls": 0, "strikes": 0},
                    },
                },
                "linescore": {
                    "currentInning": 1,
                    "isTopInning": True,
                    "outs": 0,
                    "teams": {"home": {"runs": 0}, "away": {"runs": 0}},
                    "offense": {},
                },
                "boxscore": {"teams": {"away": {"players": {}}, "home": {"players": {}}}},
            },
        }
        result = map_game_situation_to_matchup_state(feed, "home")
        assert result["pitcher"]["pitch_count_today"] == 0
        assert result["pitcher"]["batters_faced_today"] == 0

    @patch("data.mlb_api._fetch_json")
    def test_get_bvp_with_season_filter(self, mock_fetch, sample_bvp_response, tmp_cache):
        mock_fetch.return_value = sample_bvp_response
        get_batter_vs_pitcher(545361, 434378, season=2019, cache=tmp_cache)
        call_url = mock_fetch.call_args[0][0]
        assert "season=2019" in call_url

    def test_exception_hierarchy(self):
        """All custom exceptions should inherit from MLBApiError."""
        assert issubclass(MLBApiNotFoundError, MLBApiError)
        assert issubclass(MLBApiConnectionError, MLBApiError)
        assert issubclass(MLBApiTimeoutError, MLBApiError)
        # And MLBApiError from Exception
        assert issubclass(MLBApiError, Exception)

    @patch("data.mlb_api.urllib.request.urlopen")
    def test_fetch_json_sets_accept_header(self, mock_urlopen):
        mock_resp = _mock_urlopen({"ok": True})
        mock_urlopen.return_value = mock_resp
        _fetch_json("http://test")
        # Verify the Request object was created with the right header
        call_args = mock_urlopen.call_args
        request = call_args[0][0]
        assert request.get_header("Accept") == "application/json"
