<script lang="ts">
  import type { CellView, GameInfo } from './lib/api';

  // ============================================================================
  // Layout Constants
  // ============================================================================

  /** Maximum container size for grid boards like TicTacToe/Othello (pixels) */
  const GRID_MAX_SIZE = 400;

  /** Divisor for calculating font size from cell size in grid boards */
  const GRID_FONT_DIVISOR = 32;

  /** Maximum container size for drop-column boards like Connect 4 (pixels) */
  const DROP_MAX_SIZE = 420;

  /** Gap between cells in drop-column board (pixels) */
  const DROP_GAP = 4;

  /** Outer frame padding in drop-column board (pixels) */
  const DROP_FRAME_PADDING = 12;

  /** Inner grid padding in drop-column board (pixels) */
  const DROP_GRID_PADDING = 8;

  /** Ratio of hole diameter to cell size in drop-column board */
  const DROP_HOLE_RATIO = 0.8;

  /** Ratio of piece diameter to hole size in drop-column board */
  const DROP_PIECE_RATIO = 0.92;

  /** Height of the column indicator/hover zone (pixels) */
  const DROP_INDICATOR_HEIGHT = 50;

  /** Offset above board where dropping piece animation starts (pixels) */
  const DROP_START_OFFSET = 20;

  /** Duration of player's drop animation before triggering move (ms) */
  const DROP_ANIMATION_DELAY = 400;

  /** Duration of bot move highlight animation (ms) */
  const BOT_HIGHLIGHT_DURATION = 500;

  // ============================================================================
  // Component Props
  // ============================================================================

  interface Props {
    cells: CellView[];
    legalMoves: number[];
    gameOver: boolean;
    lastBotMove: number | null;
    gameInfo: GameInfo;
    currentPlayer: number;
    onCellClick: (position: number) => void;
  }

  let {
    cells,
    legalMoves,
    gameOver,
    lastBotMove,
    gameInfo,
    currentPlayer,
    onCellClick
  }: Props = $props();

  // The grid and drop-column renderers only care about occupancy, and both
  // predate the engine's richer cell projection. Deriving the flat owner array
  // here keeps them untouched rather than threading CellView through both.
  let board = $derived(cells.map((cell) => cell.owner));

  // Extract dimensions from metadata
  let width = $derived(gameInfo.board_width);
  let height = $derived(gameInfo.board_height);
  let boardType = $derived(gameInfo.board_type);
  let playerSymbols = $derived(gameInfo.player_symbols);

  // ============================================================================
  // Grid Board (TicTacToe, Othello style)
  // ============================================================================

  let gridCellSize = $derived(Math.floor(Math.min(GRID_MAX_SIZE / width, GRID_MAX_SIZE / height)));
  let gridStyle = $derived(`grid-template-columns: repeat(${width}, ${gridCellSize}px)`);
  let gridFontSize = $derived(Math.max(1, Math.floor(gridCellSize / GRID_FONT_DIVISOR)));

  function getGridCellSymbol(value: number): string {
    if (value === 0) return '';
    const playerIndex = value - 1;
    return playerSymbols[playerIndex] || String(value);
  }

  function getGridCellClass(index: number, value: number): string {
    let classes = 'grid-cell';
    if (value === 1) classes += ' player1';
    if (value === 2) classes += ' player2';
    if (value === 0 && legalMoves.includes(index) && !gameOver) classes += ' clickable';
    if (index === lastBotMove) classes += ' last-bot-move';
    return classes;
  }

  // ============================================================================
  // Drop Column Board (Connect 4 style)
  // ============================================================================

  let dropCellSize = $derived(Math.floor(Math.min(DROP_MAX_SIZE / width, DROP_MAX_SIZE / height)));
  let dropHoleSize = $derived(Math.floor(dropCellSize * DROP_HOLE_RATIO));
  let dropPieceSize = $derived(Math.floor(dropHoleSize * DROP_PIECE_RATIO));

  // Track dropping pieces for animation (player and bot separately)
  let droppingPiece: { column: number; row: number; player: number } | null = $state(null);
  let botDroppingPiece: { column: number; row: number; player: number } | null = $state(null);
  let animatingCells: Set<number> = $state(new Set());
  // Track the last processed bot move to avoid re-triggering animation
  let processedBotMove: number | null = $state(null);

  // Convert board array to 2D grid (row 0 at bottom for drop_column)
  function getDropCell(col: number, row: number): number {
    const index = row * width + col;
    return board[index] || 0;
  }

  // Find the row where a piece would land in a column
  function findLandingRow(col: number): number {
    for (let row = 0; row < height; row++) {
      if (getDropCell(col, row) === 0) {
        return row;
      }
    }
    return -1; // Column is full
  }

  // Handle column click with animation
  function handleColumnClick(col: number) {
    if (gameOver || !legalMoves.includes(col)) return;

    const landingRow = findLandingRow(col);
    if (landingRow === -1) return;

    // Start drop animation
    droppingPiece = { column: col, row: landingRow, player: currentPlayer };

    // Trigger the actual move after a brief delay to show animation start
    // Note: droppingPiece is NOT cleared here - it will be cleared by the $effect
    // when the board updates with the new piece, preventing the "disappearing piece" gap
    setTimeout(() => {
      onCellClick(col);
    }, DROP_ANIMATION_DELAY);
  }

  // Clear dropping piece when the board updates to include it
  // This prevents the gap where the animation ends but the API hasn't responded yet
  $effect(() => {
    if (droppingPiece) {
      const { column, row } = droppingPiece;
      const cellValue = getDropCell(column, row);
      // If the cell now has a piece, clear the dropping animation
      if (cellValue !== 0) {
        droppingPiece = null;
      }
    }
  });

  // Check if a cell is currently the target of a dropping animation (player or bot)
  // Used to avoid showing both the dropping piece and the board piece simultaneously
  function isCellBeingDropped(col: number, row: number): boolean {
    if (droppingPiece !== null && droppingPiece.column === col && droppingPiece.row === row) {
      return true;
    }
    if (botDroppingPiece !== null && botDroppingPiece.column === col && botDroppingPiece.row === row) {
      return true;
    }
    return false;
  }

  // Find the row where a piece is in a column (top-most piece)
  function findPieceRow(col: number, player: number): number {
    for (let row = height - 1; row >= 0; row--) {
      if (getDropCell(col, row) === player) {
        return row;
      }
    }
    return -1;
  }

  // Reset processedBotMove when a new game starts (lastBotMove becomes null)
  $effect(() => {
    if (lastBotMove === null) {
      processedBotMove = null;
    }
  });

  // Trigger drop animation for bot moves
  $effect(() => {
    if (boardType === 'drop_column' && lastBotMove !== null && lastBotMove >= 0 && lastBotMove < width) {
      // Only trigger if this is a new bot move we haven't processed yet
      if (lastBotMove !== processedBotMove) {
        processedBotMove = lastBotMove;
        const col = lastBotMove;
        const row = findPieceRow(col, 2); // Bot is player 2

        if (row >= 0) {
          // Start bot drop animation
          botDroppingPiece = { column: col, row: row, player: 2 };

          // Clear animation after it completes
          setTimeout(() => {
            botDroppingPiece = null;
            // Also trigger the highlight effect after drop completes
            const index = row * width + col;
            animatingCells.add(index);
            setTimeout(() => {
              animatingCells = new Set([...animatingCells].filter(i => i !== index));
            }, BOT_HIGHLIGHT_DURATION);
          }, DROP_ANIMATION_DELAY);
        }
      }
    }
  });

  function getDropCellClass(col: number, row: number): string {
    const value = getDropCell(col, row);
    let classes = 'drop-cell';
    if (value === 1) classes += ' player1';
    if (value === 2) classes += ' player2';

    const index = row * width + col;
    if (animatingCells.has(index)) {
      classes += ' just-dropped';
    }
    return classes;
  }

  function isColumnClickable(col: number): boolean {
    return !gameOver && legalMoves.includes(col) && !droppingPiece;
  }

  // Calculate the Y position for the dropping animation
  function getDropStartY(): number {
    return -dropCellSize - DROP_START_OFFSET;
  }

  function getDropEndY(row: number): number {
    const visualRow = height - 1 - row;
    const paddingOffset = DROP_FRAME_PADDING + DROP_GRID_PADDING;
    // Position at cell top + offset to center of hole + offset to center piece within hole
    const cellTop = paddingOffset + visualRow * (dropCellSize + DROP_GAP);
    const holeOffset = (dropCellSize - dropHoleSize) / 2;
    const pieceOffset = (dropHoleSize - dropPieceSize) / 2;
    return cellTop + holeOffset + pieceOffset;
  }

  // ============================================================================
  // Generals Board
  // ============================================================================
  //
  // A Generals move is (tile, direction), encoded as `tile * 4 + dir` with
  // dir 0=up, 1=right, 2=down, 3=left (see games-generals/src/action.rs), plus
  // a wait action at the end. So a click cannot be a move on its own: pick a
  // source tile first, then an adjacent target.

  /** Direction offsets in the engine's canonical order: up, right, down, left. */
  const GEN_DIR_DX = [0, 1, 0, -1];
  const GEN_DIR_DY = [-1, 0, 1, 0];

  const GENERALS_MAX_SIZE = 440;

  let generalsCellSize = $derived(
    Math.floor(Math.min(GENERALS_MAX_SIZE / width, GENERALS_MAX_SIZE / height))
  );
  let generalsStyle = $derived(`grid-template-columns: repeat(${width}, ${generalsCellSize}px)`);

  let selectedTile: number | null = $state(null);

  /** The wait action is the last index; every other action is a (tile, dir) move. */
  let waitAction = $derived(gameInfo.num_actions - 1);

  function generalsAction(from: number, dir: number): number {
    return from * 4 + dir;
  }

  /** A tile can be picked when at least one of its four moves is legal. */
  function isSourceTile(index: number): boolean {
    if (gameOver) return false;
    return [0, 1, 2, 3].some((dir) => legalMoves.includes(generalsAction(index, dir)));
  }

  /** Direction from `selectedTile` to `index`, or null if not adjacent. */
  function directionTo(from: number, to: number): number | null {
    const fx = from % width;
    const fy = Math.floor(from / width);
    for (let dir = 0; dir < 4; dir++) {
      const nx = fx + GEN_DIR_DX[dir];
      const ny = fy + GEN_DIR_DY[dir];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      if (ny * width + nx === to) return dir;
    }
    return null;
  }

  function isTargetTile(index: number): boolean {
    if (selectedTile === null) return false;
    const dir = directionTo(selectedTile, index);
    return dir !== null && legalMoves.includes(generalsAction(selectedTile, dir));
  }

  function handleGeneralsClick(index: number) {
    if (gameOver) return;

    if (selectedTile !== null) {
      const dir = directionTo(selectedTile, index);
      if (dir !== null && legalMoves.includes(generalsAction(selectedTile, dir))) {
        const action = generalsAction(selectedTile, dir);
        selectedTile = null;
        onCellClick(action);
        return;
      }
      // Clicking elsewhere re-picks (or clears) the source rather than
      // silently doing nothing.
      selectedTile = index === selectedTile || !isSourceTile(index) ? null : index;
      return;
    }

    if (isSourceTile(index)) selectedTile = index;
  }

  function handleGeneralsWait() {
    if (gameOver || !legalMoves.includes(waitAction)) return;
    selectedTile = null;
    onCellClick(waitAction);
  }

  // A move by either side invalidates the pending selection.
  $effect(() => {
    void cells;
    selectedTile = null;
  });

  function getGeneralsCellClass(index: number, cell: CellView): string {
    let classes = `gen-cell gen-${cell.kind}`;
    if (cell.owner === 1) classes += ' player1';
    if (cell.owner === 2) classes += ' player2';
    if (index === selectedTile) classes += ' selected';
    else if (isTargetTile(index)) classes += ' target';
    else if (selectedTile === null && isSourceTile(index)) classes += ' selectable';
    return classes;
  }

  /** Terrain marker; a general or city keeps it alongside its army count. */
  function terrainGlyph(cell: CellView): string {
    if (cell.kind === 'mountain') return '▲';
    if (cell.kind === 'general') return '★';
    if (cell.kind === 'city') return '◉';
    return '';
  }

  function armyLabel(cell: CellView): string {
    return cell.value > 0 ? String(cell.value) : '';
  }

  function getDropX(col: number): number {
    const paddingOffset = DROP_FRAME_PADDING + DROP_GRID_PADDING;
    // Position at cell left + offset to center of hole + offset to center piece within hole
    const cellLeft = paddingOffset + col * (dropCellSize + DROP_GAP);
    const holeOffset = (dropCellSize - dropHoleSize) / 2;
    const pieceOffset = (dropHoleSize - dropPieceSize) / 2;
    return cellLeft + holeOffset + pieceOffset;
  }
</script>

{#if boardType === 'grid'}
  <!-- Grid-style board (TicTacToe, Othello) -->
  <div class="grid-board" style={gridStyle}>
    {#each board as cell, i}
      <button
        class={getGridCellClass(i, cell)}
        style="width: {gridCellSize}px; height: {gridCellSize}px; font-size: {gridFontSize}rem;"
        onclick={() => onCellClick(i)}
        disabled={cell !== 0 || gameOver || !legalMoves.includes(i)}
      >
        {getGridCellSymbol(cell)}
      </button>
    {/each}
  </div>
{:else if boardType === 'drop_column'}
  <!-- Drop-column style board (Connect 4) -->
  <div class="drop-container">
    <!-- Hover indicators for columns -->
    <div class="column-indicators" style="padding-left: {DROP_FRAME_PADDING + DROP_GRID_PADDING}px; padding-right: {DROP_FRAME_PADDING + DROP_GRID_PADDING}px;">
      {#each Array(width) as _, col}
        <button
          class="column-indicator"
          class:clickable={isColumnClickable(col)}
          style="width: {dropCellSize}px; height: {DROP_INDICATOR_HEIGHT}px;"
          onclick={() => handleColumnClick(col)}
          disabled={!isColumnClickable(col)}
          aria-label={`Drop piece in column ${col + 1}`}
        >
          {#if isColumnClickable(col)}
            <div
              class={`hover-piece player${currentPlayer}-preview`}
              style="width: {dropHoleSize}px; height: {dropHoleSize}px;"
            ></div>
          {/if}
        </button>
      {/each}
    </div>

    <!-- Main board -->
    <div class="board-frame">
      <!-- Dropping piece animation (player) -->
      {#if droppingPiece}
        <div
          class={`dropping-piece player${droppingPiece.player}`}
          style="
            left: {getDropX(droppingPiece.column)}px;
            width: {dropPieceSize}px;
            height: {dropPieceSize}px;
            --drop-start: {getDropStartY()}px;
            --drop-end: {getDropEndY(droppingPiece.row)}px;
          "
        ></div>
      {/if}

      <!-- Dropping piece animation (bot) -->
      {#if botDroppingPiece}
        <div
          class={`dropping-piece player${botDroppingPiece.player}`}
          style="
            left: {getDropX(botDroppingPiece.column)}px;
            width: {dropPieceSize}px;
            height: {dropPieceSize}px;
            --drop-start: {getDropStartY()}px;
            --drop-end: {getDropEndY(botDroppingPiece.row)}px;
          "
        ></div>
      {/if}

      <!-- Board grid (blue frame with holes) -->
      <div class="board-grid" style="gap: {DROP_GAP}px;">
        {#each Array(height) as _, visualRow}
          {@const row = height - 1 - visualRow}
          <div class="board-row" style="gap: {DROP_GAP}px;">
            {#each Array(width) as _, col}
              <div class={getDropCellClass(col, row)} style="width: {dropCellSize}px; height: {dropCellSize}px;">
                <div class="hole" style="width: {dropHoleSize}px; height: {dropHoleSize}px;">
                  {#if getDropCell(col, row) !== 0 && !isCellBeingDropped(col, row)}
                    <div
                      class="piece"
                      class:player1={getDropCell(col, row) === 1}
                      class:player2={getDropCell(col, row) === 2}
                      style="width: {dropPieceSize}px; height: {dropPieceSize}px;"
                    ></div>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        {/each}
      </div>

      <!-- Board stand -->
      <div class="board-stand"></div>
    </div>
  </div>
{:else if boardType === 'generals'}
  <!-- Generals: terrain grid, select a source tile then an adjacent target -->
  <div class="generals-container">
    <div class="generals-board" style={generalsStyle}>
      {#each cells as cell, i}
        <button
          class={getGeneralsCellClass(i, cell)}
          style="width: {generalsCellSize}px; height: {generalsCellSize}px;"
          onclick={() => handleGeneralsClick(i)}
          disabled={gameOver}
          aria-label={`Tile ${i % width},${Math.floor(i / width)}`}
        >
          <span class="gen-terrain">{terrainGlyph(cell)}</span>
          <span class="gen-army">{armyLabel(cell)}</span>
        </button>
      {/each}
    </div>
    <div class="generals-controls">
      <span class="generals-hint">
        {#if gameOver}
          Game over
        {:else if selectedTile === null}
          Pick one of your tiles to move from
        {:else}
          Now pick an adjacent tile — or click again to cancel
        {/if}
      </span>
      <button
        class="generals-wait"
        onclick={handleGeneralsWait}
        disabled={gameOver || !legalMoves.includes(waitAction)}
      >
        Wait
      </button>
    </div>
  </div>
{:else}
  <!-- Fallback for unknown board types -->
  <div class="unknown-board">
    <p>Unknown board type: {boardType}</p>
  </div>
{/if}

<style>
  /* ============================================================================
   * Grid Board Styles (TicTacToe, Othello)
   * ============================================================================ */
  .grid-board {
    display: grid;
    gap: 4px;
    padding: 8px;
    background: #2a2a4a;
    border-radius: 12px;
  }

  .grid-cell {
    font-weight: bold;
    background: #3a3a5a;
    border: 2px solid transparent;
    border-radius: 8px;
    cursor: default;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .grid-cell.player1 {
    color: #00d9ff;
  }

  .grid-cell.player2 {
    color: #ff6b6b;
  }

  .grid-cell.clickable {
    cursor: pointer;
    border-color: #4a4a6a;
  }

  .grid-cell.clickable:hover {
    background: #4a4a6a;
    border-color: #00d9ff;
  }

  .grid-cell.last-bot-move {
    animation: grid-highlight 0.5s ease-out;
  }

  @keyframes grid-highlight {
    0% {
      background: #ff6b6b44;
    }
    100% {
      background: #3a3a5a;
    }
  }

  .grid-cell:disabled {
    cursor: default;
  }

  /* ============================================================================
   * Drop Column Board Styles (Connect 4)
   * ============================================================================ */
  .drop-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    user-select: none;
  }

  .column-indicators {
    display: flex;
    gap: 4px;
    margin-bottom: 8px;
    height: 50px;
  }

  .column-indicator {
    background: transparent;
    border: none;
    cursor: default;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 4px;
  }

  .column-indicator.clickable {
    cursor: pointer;
  }

  .column-indicator.clickable:hover .hover-piece {
    opacity: 1;
    transform: scale(1);
  }

  .hover-piece {
    border-radius: 50%;
    opacity: 0;
    transform: scale(0.8);
    transition: all 0.15s ease;
  }

  .player1-preview {
    background: radial-gradient(circle at 30% 30%, #ff6b6b, #e74c3c);
    box-shadow: 0 2px 8px rgba(231, 76, 60, 0.4);
  }

  .player2-preview {
    background: radial-gradient(circle at 30% 30%, #ffe066, #f1c40f);
    box-shadow: 0 2px 8px rgba(243, 156, 18, 0.4);
  }

  .board-frame {
    position: relative;
    background: linear-gradient(180deg, #1e5799 0%, #2989d8 50%, #1e5799 100%);
    border-radius: 12px;
    padding: 12px;
    box-shadow:
      0 8px 32px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.1),
      inset 0 -2px 4px rgba(0, 0, 0, 0.2);
  }

  .board-grid {
    display: flex;
    flex-direction: column;
    background: linear-gradient(180deg, #2980b9 0%, #3498db 50%, #2980b9 100%);
    padding: 8px;
    border-radius: 8px;
    box-shadow:
      inset 0 2px 8px rgba(0, 0, 0, 0.3),
      inset 0 -1px 2px rgba(255, 255, 255, 0.1);
  }

  .board-row {
    display: flex;
  }

  .drop-cell {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .hole {
    border-radius: 50%;
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    box-shadow:
      inset 0 4px 8px rgba(0, 0, 0, 0.6),
      inset 0 -2px 4px rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  .piece {
    border-radius: 50%;
    transition: transform 0.1s ease;
  }

  .piece.player1 {
    background: radial-gradient(circle at 30% 30%, #ff8a8a, #e74c3c 60%, #c0392b);
    box-shadow:
      0 2px 4px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.3),
      inset 0 -2px 4px rgba(0, 0, 0, 0.2);
  }

  .piece.player2 {
    background: radial-gradient(circle at 30% 30%, #ffe066, #f1c40f 60%, #f39c12);
    box-shadow:
      0 2px 4px rgba(0, 0, 0, 0.3),
      inset 0 2px 4px rgba(255, 255, 255, 0.4),
      inset 0 -2px 4px rgba(0, 0, 0, 0.1);
  }

  .drop-cell.just-dropped .piece {
    animation: pop-in 0.3s ease-out;
  }

  @keyframes pop-in {
    0% {
      transform: scale(0.8);
    }
    50% {
      transform: scale(1.1);
    }
    100% {
      transform: scale(1);
    }
  }

  .dropping-piece {
    position: absolute;
    border-radius: 50%;
    z-index: 10;
    animation: drop-piece 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
  }

  .dropping-piece.player1 {
    background: radial-gradient(circle at 30% 30%, #ff8a8a, #e74c3c 60%, #c0392b);
    box-shadow:
      0 4px 8px rgba(0, 0, 0, 0.4),
      inset 0 2px 4px rgba(255, 255, 255, 0.3),
      inset 0 -2px 4px rgba(0, 0, 0, 0.2);
  }

  .dropping-piece.player2 {
    background: radial-gradient(circle at 30% 30%, #ffe066, #f1c40f 60%, #f39c12);
    box-shadow:
      0 4px 8px rgba(0, 0, 0, 0.4),
      inset 0 2px 4px rgba(255, 255, 255, 0.4),
      inset 0 -2px 4px rgba(0, 0, 0, 0.1);
  }

  @keyframes drop-piece {
    0% {
      top: var(--drop-start);
      opacity: 1;
    }
    80% {
      top: var(--drop-end);
    }
    90% {
      top: calc(var(--drop-end) - 4px);
    }
    100% {
      top: var(--drop-end);
      opacity: 1;
    }
  }

  .board-stand {
    width: 100%;
    height: 20px;
    background: linear-gradient(180deg, #1e5799 0%, #0f3460 100%);
    border-radius: 0 0 8px 8px;
    margin-top: -4px;
    box-shadow:
      0 4px 8px rgba(0, 0, 0, 0.3),
      inset 0 1px 2px rgba(255, 255, 255, 0.1);
  }

  /* ============================================================================
   * Generals Board Styles
   * ============================================================================ */
  .generals-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    user-select: none;
  }

  .generals-board {
    display: grid;
    gap: 2px;
    padding: 8px;
    background: #2a2a4a;
    border-radius: 12px;
  }

  .gen-cell {
    position: relative;
    background: #3a3a5a;
    border: 2px solid transparent;
    border-radius: 4px;
    cursor: default;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #cfcfe6;
    font-size: 0.8rem;
    font-weight: bold;
    padding: 0;
  }

  .gen-terrain {
    position: absolute;
    top: 1px;
    left: 3px;
    font-size: 0.6rem;
    opacity: 0.85;
  }

  .gen-cell.gen-mountain {
    background: #23233a;
    color: #6a6a8a;
  }

  .gen-cell.gen-city {
    background: #45456a;
  }

  .gen-cell.player1 {
    background: #1c5f70;
    color: #d6f7ff;
  }

  .gen-cell.player2 {
    background: #7a2f2f;
    color: #ffe0e0;
  }

  .gen-cell.selectable {
    cursor: pointer;
    border-color: #4a4a6a;
  }

  .gen-cell.selectable:hover {
    border-color: #00d9ff;
  }

  .gen-cell.selected {
    cursor: pointer;
    border-color: #ffd34d;
    box-shadow: 0 0 8px rgba(255, 211, 77, 0.6);
  }

  .gen-cell.target {
    cursor: pointer;
    border-color: #6be36b;
  }

  .gen-cell.target:hover {
    background: #3f6a3f;
  }

  .generals-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .generals-hint {
    color: #9a9ab8;
    font-size: 0.85rem;
  }

  .generals-wait {
    background: #3a3a5a;
    border: 1px solid #4a4a6a;
    border-radius: 6px;
    color: #cfcfe6;
    cursor: pointer;
    padding: 0.35rem 0.9rem;
  }

  .generals-wait:disabled {
    cursor: default;
    opacity: 0.5;
  }

  /* ============================================================================
   * Unknown Board Type
   * ============================================================================ */
  .unknown-board {
    padding: 2rem;
    background: #4a1a1a;
    border-radius: 12px;
    color: #f66;
  }

  /* ============================================================================
   * Responsive adjustments
   * ============================================================================ */
  @media (max-width: 500px) {
    .column-indicator {
      height: 40px;
    }
  }
</style>
