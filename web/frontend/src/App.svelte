<script lang="ts">
  import { onMount } from 'svelte';
  import GenericBoard from './GenericBoard.svelte';
  import Stats from './Stats.svelte';
  import { newGame, makeMove, getHealth, getGameInfo, getGames, getStats, type GameState, type MoveResponse, type GameInfo } from './lib/api';

  let gameState: GameState | null = $state(null);
  let gameInfo: GameInfo | null = $state(null);
  let availableGames: string[] = $state([]);
  let selectedGame: string = $state('tictactoe');
  let error: string | null = $state(null);
  let loading: boolean = $state(false);
  let serverOnline: boolean = $state(false);
  let lastBotMove: number | null = $state(null);

  onMount(async () => {
    // Check server health
    try {
      await getHealth();
      serverOnline = true;
      // Load available games
      availableGames = await getGames();

      // Check what game is currently being trained and use that as default
      try {
        const stats = await getStats();
        if (stats.env_id && availableGames.includes(stats.env_id)) {
          selectedGame = stats.env_id;
        }
      } catch {
        // Stats not available, keep default
      }

      // Load game metadata
      gameInfo = await getGameInfo(selectedGame);
      // Start a new game with the selected game
      gameState = await newGame('player', selectedGame);
    } catch (e) {
      serverOnline = false;
      error = 'Cannot connect to server. Is the Rust backend running on :8080?';
    }
  });

  async function handleGameChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedGame = target.value;
    loading = true;
    error = null;
    lastBotMove = null;
    try {
      gameInfo = await getGameInfo(selectedGame);
      gameState = await newGame('player', selectedGame);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  async function handleNewGame(first: 'player' | 'bot') {
    loading = true;
    error = null;
    lastBotMove = null;
    try {
      gameState = await newGame(first, selectedGame);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  async function handleCellClick(position: number) {
    if (loading || !gameState || gameState.game_over) return;
    if (gameState.current_player !== gameState.human_player) return; // Not player's turn
    if (!gameState.legal_moves.includes(position)) return; // Illegal move

    loading = true;
    error = null;
    try {
      const response: MoveResponse = await makeMove(position);
      gameState = response;
      lastBotMove = response.bot_move;
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  // Check if pass action is available (for games like Othello)
  function canPass(): boolean {
    if (!gameState || gameState.game_over) return false;
    if (gameState.current_player !== gameState.human_player) return false;
    // Pass action is typically the last action index (e.g., 64 for Othello's 65 actions)
    const passAction = (gameInfo?.num_actions ?? 0) - 1;
    return gameState.legal_moves.includes(passAction);
  }

  // Handle pass action
  async function handlePass() {
    if (loading || !gameState || gameState.game_over) return;
    if (gameState.current_player !== gameState.human_player) return;
    
    const passAction = (gameInfo?.num_actions ?? 0) - 1;
    if (!gameState.legal_moves.includes(passAction)) return;

    loading = true;
    error = null;
    try {
      const response: MoveResponse = await makeMove(passAction);
      gameState = response;
      lastBotMove = response.bot_move;
    } catch (e) {
      error = String(e);
    }
    loading = false;
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
        {#if gameState && gameInfo}
          <GenericBoard
            board={gameState.board}
            legalMoves={gameState.legal_moves}
            gameOver={gameState.game_over}
            {lastBotMove}
            {gameInfo}
            currentPlayer={gameState.current_player}
            onCellClick={handleCellClick}
          />

          <div class="status"
               class:player1-wins={gameState.winner === 1}
               class:player2-wins={gameState.winner === 2}
               class:drop-column={gameInfo?.board_type === 'drop_column'}>
            {gameState.message}
          </div>

          {#if error}
            <div class="error">{error}</div>
          {/if}

          {#if canPass()}
            <div class="pass-section">
              <button class="pass-button" onclick={handlePass} disabled={loading}>
                Pass (No legal moves)
              </button>
            </div>
          {/if}

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
        <Stats />
      </div>
    </div>
  {/if}
</main>

<style>
  main {
    max-width: 900px;
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
