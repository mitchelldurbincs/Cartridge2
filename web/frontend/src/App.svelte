<script lang="ts">
  import { onMount } from 'svelte';
  import GenericBoard from './GenericBoard.svelte';
  import Stats from './Stats.svelte';
  import DecisionInspector from './DecisionInspector.svelte';
  import { newGame, makeMove, getHealth, getGameInfo, getGames, getGameState, getHistory, type GameState, type GameInfo } from './lib/api';
  import { boundAnalysis, samePosition, type HistoryResponse, type PositionKey, type ProbabilityMetric } from './lib/analysis';

  let gameState: GameState | null = $state(null);
  let gameInfo: GameInfo | null = $state(null);
  let availableGames: string[] = $state([]);
  let selectedGame = $state('tictactoe');
  let error: string | null = $state(null);
  let loading = $state(false);
  let serverOnline = $state(false);
  let lastBotMove: number | null = $state(null);
  let history: HistoryResponse | null = $state(null);
  let viewedRevision: number | null = $state(null);
  let highlightedAction: number | null = $state(null);
  let metric: ProbabilityMetric = $state('visit_share');
  let historyBusy = $state(false);
  let historyRequest = 0;
  let selectedRecord = $derived.by(() => viewedRevision == null ? null :
    history?.records.find(record => record.state.revision === viewedRevision) ?? null);
  let displayState = $derived(selectedRecord?.state ?? gameState);
  let analysis = $derived(boundAnalysis(selectedRecord));
  let isHistorical = $derived(selectedRecord !== null);
  let namedActions = $derived.by(() => (gameState?.actions ?? []).filter(a => a.target?.kind === 'named' &&
    (a.target.name !== 'Wait' || gameInfo?.board_type !== 'generals')));

  function key(state: GameState): PositionKey { return { session_id: state.session_id, revision: state.revision }; }

  async function refreshHistory(from?: number) {
    if (!gameState) return;
    const expected = key(gameState);
    const request = ++historyRequest;
    historyBusy = true;
    try {
      const response = await getHistory(expected.session_id, from);
      if (request === historyRequest && gameState && samePosition(expected, gameState)) {
        history = response;
        if (viewedRevision != null && !response.records.some(r => r.state.revision === viewedRevision)) viewedRevision = null;
      }
    } finally { if (request === historyRequest) historyBusy = false; }
  }

  async function recoverPosition() {
    try {
      gameState = await getGameState();
      viewedRevision = null;
      lastBotMove = null;
      highlightedAction = null;
      await refreshHistory();
    } catch (recoveryError) {
      error = `${error ?? ''} Current position could not be refreshed: ${String(recoveryError)}`;
    }
  }

  onMount(async () => {
    loading = true;
    try {
      await getHealth();
      serverOnline = true;
      availableGames = await getGames();
      selectedGame = availableGames[0] ?? 'tictactoe';
      gameInfo = await getGameInfo(selectedGame);
      // Opening another tab must not reset a running shared session.
      gameState = await getGameState();
      await refreshHistory();
    } catch (e) { error = String(e); }
    finally { loading = false; }
  });

  async function handleGameChange(event: Event) {
    selectedGame = (event.target as HTMLSelectElement).value;
    try { gameInfo = await getGameInfo(selectedGame); await handleNewGame('player'); }
    catch (e) { error = String(e); }
  }

  async function handleNewGame(first: 'player' | 'bot') {
    if (loading) return;
    loading = true;
    error = null;
    viewedRevision = null;
    highlightedAction = null;
    lastBotMove = null;
    try {
      gameState = await newGame(first, selectedGame, gameState ? key(gameState) : undefined);
      await refreshHistory();
    } catch (e) { error = String(e); await recoverPosition(); }
    finally { loading = false; }
  }

  async function handleCellClick(position: number) {
    if (loading || isHistorical || !gameState || gameState.game_over ||
      gameState.current_player !== gameState.human_player || !gameState.legal_moves.includes(position)) return;
    loading = true;
    error = null;
    try {
      const response = await makeMove(position, key(gameState));
      gameState = response;
      lastBotMove = response.bot_move;
      await refreshHistory();
    } catch (e) {
      error = String(e);
      // The human transition may have committed before a bot-search failure.
      // Read authoritative state; never retry the player's move automatically.
      await recoverPosition();
    } finally { loading = false; }
  }

  function view(revision: number | null) { viewedRevision = revision; highlightedAction = null; }

  async function lastDecision() {
    try {
      await refreshHistory();
      const last = history?.records.slice().reverse().find(r => r.decision && r.decision.source !== 'human');
      if (last) view(last.state.revision);
    } catch (e) { error = String(e); }
  }

  async function navigate(direction: -1 | 1) {
    if (!history || !gameState || historyBusy) return;
    const target = (viewedRevision ?? gameState.revision) + direction;
    if (target < history.first_available_revision || target > gameState.revision) return;
    try {
      if (!history.records.some(r => r.state.revision === target)) {
        await refreshHistory(direction < 0 ? Math.max(history.first_available_revision, target - 63) : target);
      }
      if (history?.records.some(r => r.state.revision === target)) view(target);
    } catch (e) { error = String(e); }
  }
</script>

<main>
  <h1>Cartridge2 {gameInfo?.display_name ?? 'Loading...'}</h1>

  {#if serverOnline && availableGames.length > 1}
    <div class="game-selector">
      <label for="game-select">Select Game:</label>
      <select id="game-select" bind:value={selectedGame} onchange={handleGameChange} disabled={loading}>
        {#each availableGames as game}
          <option value={game}>{game}</option>
        {/each}
      </select>
    </div>
  {/if}

  {#if gameInfo?.description}
    <p class="game-description">{gameInfo.description}</p>
  {/if}

  {#if !serverOnline}
    <div class="error">
      <p>Cannot connect to server.</p>
      <p>Make sure the Rust backend is running:</p>
      <code>cd web && cargo run</code>
    </div>
  {:else}
    <div class="game-container">
      <div class="game-section">
        {#if gameState && gameInfo && displayState}
          <nav class="history-controls" aria-label="Match history">
            <button onclick={() => view(null)} class:active={!isHistorical}>Live</button>
            <button onclick={lastDecision} disabled={loading || historyBusy || gameState.revision === 0}>Last bot decision</button>
            <button aria-label="Previous position" onclick={() => navigate(-1)} disabled={loading || historyBusy || (viewedRevision ?? gameState.revision) <= (history?.first_available_revision ?? 0)}>←</button>
            <select aria-label="History position" value={viewedRevision ?? ''} onchange={(e) => view(e.currentTarget.value === '' ? null : Number(e.currentTarget.value))} disabled={loading || historyBusy}>
              <option value="">Live · {gameState.revision}</option>
              {#each history?.records ?? [] as record}<option value={record.state.revision}>Position {record.state.revision}{record.decision ? ` · ${record.decision.source.replaceAll('_', ' ')}` : ''}</option>{/each}
            </select>
            <button aria-label="Next position" onclick={() => navigate(1)} disabled={loading || historyBusy || viewedRevision === null || viewedRevision >= gameState.revision}>→</button>
          </nav>
          {#if isHistorical}<p class="inspection-label">Position {displayState.revision} · read-only · {metric.replaceAll('_', ' ')}</p>{/if}
          {#key `${displayState.session_id}:${isHistorical ? `history-${displayState.revision}` : 'live'}`}
          <GenericBoard
            cells={displayState.cells}
            legalMoves={displayState.legal_moves}
            gameOver={displayState.game_over}
            lastBotMove={isHistorical ? null : lastBotMove}
            {gameInfo}
            currentPlayer={displayState.current_player}
            humanPlayer={displayState.human_player}
            positionKey={`${displayState.session_id}:${displayState.revision}`}
            actionPresentations={displayState.actions}
            assessments={analysis?.actions ?? []}
            {metric}
            readOnly={isHistorical || loading}
            {highlightedAction}
            onCellClick={handleCellClick}
          />
          {/key}

          <div class="status"
               class:player1-wins={gameState.winner === 1}
               class:player2-wins={gameState.winner === 2}
               class:drop-column={gameInfo?.board_type === 'drop_column'}>
            {isHistorical ? `Inspecting: ${displayState.message}` : gameState.message}
          </div>

          {#if error}
            <div class="error">{error}</div>
          {/if}

          {#if !isHistorical && !gameState.game_over && gameState.current_player === gameState.human_player}
            <div class="pass-section">
              {#each namedActions as action}<button class="pass-button" onclick={() => handleCellClick(action.action)} disabled={loading}>{action.label}</button>{/each}
            </div>
          {/if}

          {#if history && history.first_available_revision > 0}<p class="inspection-label">History retained from position {history.first_available_revision}; older positions were evicted.</p>{/if}

          <div class="controls">
            <button onclick={() => handleNewGame('player')} disabled={loading}>
              New Game (You First)
            </button>
            <button onclick={() => handleNewGame('bot')} disabled={loading}>
              New Game (Bot First)
            </button>
          </div>
        {:else}
          <p>Loading game...</p>
        {/if}
      </div>

      <div class="stats-section">
        <DecisionInspector decision={analysis} position={selectedRecord?.state ?? null}
          presentations={selectedRecord?.state.actions ?? []} bind:metric
          onAction={(action) => highlightedAction = action} />
        <Stats />
      </div>
    </div>
  {/if}
</main>

<style>
  .history-controls { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: 1rem; justify-content: center; }
  .history-controls button,.history-controls select { padding: .45rem; border-radius: 5px; border: 1px solid #50687c; background: #202e40; color: #e6eef7; }
  .history-controls button { cursor: pointer; } .history-controls .active { border-color: #75d9ba; }
  .history-controls button:disabled { opacity: .45; cursor: default; }
  .inspection-label { color: #a9cdbf; font-size: .8rem; }
  main {
    max-width: 1250px;
    margin: 0 auto;
    padding: 2rem;
    text-align: center;
  }

  h1 {
    color: #00d9ff;
    margin-bottom: 1rem;
  }

  .game-selector {
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
  }

  .game-selector label {
    color: #aaa;
  }

  .game-selector select {
    padding: 0.5rem 1rem;
    font-size: 1rem;
    background: #2a2a4a;
    color: #fff;
    border: 1px solid #4a4a6a;
    border-radius: 8px;
    cursor: pointer;
  }

  .game-selector select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .game-description {
    color: #888;
    font-style: italic;
    margin-bottom: 1.5rem;
  }

  .game-container {
    display: flex;
    gap: 3rem;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .game-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }

  .stats-section {
    min-width: 250px;
  }

  .status {
    font-size: 1.2rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    background: #2a2a4a;
  }

  /* Grid games (TicTacToe, Othello): Player 1 = Cyan, Player 2 = Red */
  .status.player1-wins {
    background: #1a3a4a;
    color: #00d9ff;
  }

  .status.player2-wins {
    background: #4a1a1a;
    color: #ff6b6b;
  }

  /* Drop column games (Connect 4): Player 1 = Red, Player 2 = Yellow */
  .status.drop-column.player1-wins {
    background: #4a1a1a;
    color: #ff6b6b;
  }

  .status.drop-column.player2-wins {
    background: #4a3a1a;
    color: #ffe066;
  }

  .error {
    color: #f66;
    background: #4a1a1a;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
  }

  .error code {
    display: block;
    margin-top: 0.5rem;
    background: #333;
    padding: 0.5rem;
    border-radius: 4px;
  }

  .controls {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
  }

  button {
    padding: 0.75rem 1.5rem;
    font-size: 1rem;
    background: #00d9ff;
    color: #1a1a2e;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
  }

  button:hover:not(:disabled) {
    background: #00b8dd;
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Pass button styling */
  .pass-section {
    margin: 0.5rem 0;
  }

  .pass-button {
    background: #ffa500;
    color: #1a1a2e;
    font-weight: bold;
    animation: pulse 2s infinite;
  }

  .pass-button:hover:not(:disabled) {
    background: #ff8c00;
  }

  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.7);
    }
    70% {
      box-shadow: 0 0 0 10px rgba(255, 165, 0, 0);
    }
    100% {
      box-shadow: 0 0 0 0 rgba(255, 165, 0, 0);
    }
  }
</style>
