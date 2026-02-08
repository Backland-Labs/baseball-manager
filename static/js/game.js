/* game.js -- SSE client for live game view */

(function () {
  'use strict';

  var root = document.getElementById('game-root');
  if (!root) return;

  var gameId = root.dataset.gameId;
  var feed = document.getElementById('play-feed');
  var source = new EventSource('/api/game/' + gameId + '/events');

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function updateScoreboard(state) {
    if (!state) return;
    var away = state.away || {};
    var home = state.home || {};
    setText('away-name', away.name || 'Away');
    setText('home-name', home.name || 'Home');
    setText('away-R', state.score_away || 0);
    setText('home-R', state.score_home || 0);

    // Inning runs
    var awayRuns = away.inning_runs || [];
    var homeRuns = home.inning_runs || [];
    for (var i = 0; i < 9; i++) {
      setText('away-' + (i + 1), awayRuns[i] !== undefined ? awayRuns[i] : '-');
      setText('home-' + (i + 1), homeRuns[i] !== undefined ? homeRuns[i] : '-');
    }

    // Count hits from batter stats
    var awayHits = 0, homeHits = 0;
    if (away.batter_stats) {
      Object.values(away.batter_stats).forEach(function (s) { awayHits += (s.H || 0); });
    }
    if (home.batter_stats) {
      Object.values(home.batter_stats).forEach(function (s) { homeHits += (s.H || 0); });
    }
    setText('away-H', awayHits);
    setText('home-H', homeHits);
    setText('away-E', 0);
    setText('home-E', 0);
  }

  function updateDiamond(state) {
    if (!state) return;
    var runners = state.runners || [];
    var bases = { 1: false, 2: false, 3: false };
    runners.forEach(function (r) { bases[r.base] = true; });
    ['1', '2', '3'].forEach(function (b) {
      var el = document.getElementById('base-' + b);
      if (el) {
        el.classList.toggle('occupied', !!bases[parseInt(b)]);
      }
    });
  }

  function updateOuts(state) {
    if (!state) return;
    var outs = state.outs || 0;
    var dots = document.querySelectorAll('#outs-display .out-dot');
    dots.forEach(function (dot, i) {
      dot.classList.toggle('active', i < outs);
    });
  }

  function updateInning(state) {
    if (!state) return;
    var half = state.half === 'BOTTOM' ? 'Bot' : 'Top';
    setText('inning-display', half + ' ' + (state.inning || 1));
  }

  function updateWP(wp) {
    if (wp === undefined || wp === null) return;
    setText('wp-display', Math.round(wp * 100) + '%');
  }

  function updatePitcherBatter(state) {
    if (!state) return;
    // Current pitcher is from fielding team
    var battingTeam, fieldingTeam;
    if (state.half === 'TOP') {
      battingTeam = state.away;
      fieldingTeam = state.home;
    } else {
      battingTeam = state.home;
      fieldingTeam = state.away;
    }

    // Current batter
    if (battingTeam && battingTeam.lineup_index !== undefined) {
      // We don't have lineup names in state dict, so we show index
      setText('batter-display', 'Spot #' + (battingTeam.lineup_index + 1));
    }

    // Current pitcher
    if (fieldingTeam && fieldingTeam.current_pitcher_id) {
      var pid = fieldingTeam.current_pitcher_id;
      var pstats = (fieldingTeam.pitcher_stats || {})[pid] || {};
      var ip = pstats.IP !== undefined ? pstats.IP : '-';
      var k = pstats.K !== undefined ? pstats.K : '-';
      var pitches = pstats.pitches !== undefined ? pstats.pitches : '-';
      setText('pitcher-display', ip + ' IP, ' + k + ' K, ' + pitches + ' P');
    }
  }

  function appendPlay(desc, detail, runsScored, state) {
    var item = document.createElement('div');
    item.className = 'feed-item' + (runsScored > 0 ? ' scoring' : '');

    var descEl = document.createElement('div');
    descEl.className = 'feed-desc';
    descEl.textContent = desc;
    item.appendChild(descEl);

    if (detail) {
      var detailEl = document.createElement('div');
      detailEl.className = 'feed-detail';
      detailEl.textContent = detail;
      item.appendChild(detailEl);
    }

    if (state) {
      var scoreEl = document.createElement('div');
      scoreEl.className = 'feed-score';
      scoreEl.textContent = (state.away ? state.away.name : 'Away') + ' ' +
        (state.score_away || 0) + ' - ' +
        (state.home ? state.home.name : 'Home') + ' ' +
        (state.score_home || 0);
      item.appendChild(scoreEl);
    }

    // Remove "waiting" message if present
    var waiting = feed.querySelector('.muted');
    if (waiting) waiting.remove();

    feed.insertBefore(item, feed.firstChild);
  }

  function renderBoxScore(boxScore) {
    if (!boxScore) return;
    var section = document.getElementById('box-score-section');
    var content = document.getElementById('box-score-content');
    section.style.display = '';

    var html = '';

    ['away', 'home'].forEach(function (side) {
      var team = boxScore[side];
      if (!team) return;
      html += '<h4>' + team.team_name + '</h4>';

      // Batting
      html += '<table class="box-score-table"><thead><tr>' +
        '<th>Name</th><th>AB</th><th>H</th><th>R</th><th>RBI</th><th>BB</th><th>K</th><th>HR</th>' +
        '</tr></thead><tbody>';
      (team.batting || []).forEach(function (b) {
        html += '<tr><td>' + b.name + '</td>' +
          '<td>' + (b.AB || 0) + '</td><td>' + (b.H || 0) + '</td>' +
          '<td>' + (b.R || 0) + '</td><td>' + (b.RBI || 0) + '</td>' +
          '<td>' + (b.BB || 0) + '</td><td>' + (b.K || 0) + '</td>' +
          '<td>' + (b.HR || 0) + '</td></tr>';
      });
      html += '</tbody></table>';

      // Pitching
      html += '<table class="box-score-table"><thead><tr>' +
        '<th>Pitcher</th><th>IP</th><th>H</th><th>R</th><th>ER</th><th>BB</th><th>K</th><th>P</th>' +
        '</tr></thead><tbody>';
      (team.pitching || []).forEach(function (p) {
        html += '<tr><td>' + p.name + '</td>' +
          '<td>' + (p.IP || 0) + '</td><td>' + (p.H || 0) + '</td>' +
          '<td>' + (p.R || 0) + '</td><td>' + (p.ER || 0) + '</td>' +
          '<td>' + (p.BB || 0) + '</td><td>' + (p.K || 0) + '</td>' +
          '<td>' + (p.pitches || 0) + '</td></tr>';
      });
      html += '</tbody></table>';
    });

    content.innerHTML = html;
  }

  // SSE event handlers
  source.addEventListener('game_start', function (e) {
    var data = JSON.parse(e.data);
    updateScoreboard(data.state);
    updateDiamond(data.state);
    updateOuts(data.state);
    updateInning(data.state);
    updateWP(data.wp);
    appendPlay(
      'Game starting: ' + (data.away_team || 'Away') + ' at ' + (data.home_team || 'Home'),
      'Seed: ' + data.seed,
      0,
      data.state
    );
  });

  source.addEventListener('play', function (e) {
    var data = JSON.parse(e.data);
    updateScoreboard(data.state);
    updateDiamond(data.state);
    updateOuts(data.state);
    updateInning(data.state);
    updateWP(data.wp);
    updatePitcherBatter(data.state);
    appendPlay(data.description, data.detail, data.runs_scored || 0, data.state);
  });

  source.addEventListener('game_end', function (e) {
    var data = JSON.parse(e.data);
    updateScoreboard(data.state);
    appendPlay(
      'Game Over! Winner: ' + (data.winning_team || '?'),
      '',
      0,
      data.state
    );
    renderBoxScore(data.box_score);
    source.close();
  });

  source.addEventListener('error', function (e) {
    if (source.readyState === EventSource.CLOSED) {
      appendPlay('Connection closed', '', 0, null);
    }
  });
})();
