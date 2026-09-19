<script lang="ts">
  import type { CellView } from './lib/api';
  import { percent, targetMass, probabilityHeat as heat, type ActionPresentation, type ActionAssessment, type ProbabilityMetric } from './lib/analysis';

  interface Props {
    cells: CellView[];
    legalMoves: number[];
    gameOver: boolean;
    width: number;
    height: number;
    onCellClick: (action: number) => void | Promise<void>;
    actionPresentations: ActionPresentation[];
    assessments?: ActionAssessment[];
    metric?: ProbabilityMetric;
    readOnly?: boolean;
    highlightedAction?: number | null;
  }

  let {
    cells, legalMoves, gameOver, width, height, onCellClick,
    actionPresentations, assessments = [], metric = 'visit_share',
    readOnly = false, highlightedAction = null,
  }: Props = $props();

  function sourceMass(index: number): number | null {
    return targetMass(actionPresentations, assessments, metric, 'source', index);
  }

  // The engine supplies source/target edges and named actions. A click picks
  // the source first, then a target; the UI never encodes a Generals action.

  const GENERALS_MAX_SIZE = 440;

  let generalsCellSize = $derived(
    Math.floor(Math.min(GENERALS_MAX_SIZE / width, GENERALS_MAX_SIZE / height))
  );
  let generalsStyle = $derived(`grid-template-columns: repeat(${width}, ${generalsCellSize}px)`);

  let selectedTile: number | null = $state(null);

  let waitAction = $derived(actionPresentations.find(a => a.target?.kind === 'named' && a.target.name === 'Wait')?.action ?? null);
  let edges = $derived(actionPresentations.flatMap(a => a.target?.kind === 'edge' ? [{ ...a, from: a.target.from, to: a.target.to }] : []));
  let selectedEdges = $derived(edges.filter(a => a.from === selectedTile));

  /** A tile can be picked when at least one of its four moves is legal. */
  function isSourceTile(index: number): boolean {
    if (gameOver) return false;
    return edges.some(a => a.from === index && legalMoves.includes(a.action));
  }

  function edgeAction(from: number, to: number): number | null {
    return edges.find(a => a.from === from && a.to === to)?.action ?? null;
  }

  function isTargetTile(index: number): boolean {
    if (selectedTile === null) return false;
    const action = edgeAction(selectedTile, index);
    return action !== null && legalMoves.includes(action);
  }

  function handleGeneralsClick(index: number) {
    if (gameOver) return;
    if (readOnly) {
      selectedTile = selectedTile === index ? null : isSourceTile(index) ? index : null;
      return;
    }

    if (selectedTile !== null) {
      const action = edgeAction(selectedTile, index);
      if (action !== null && legalMoves.includes(action)) {
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
    if (readOnly || gameOver || waitAction === null || !legalMoves.includes(waitAction)) return;
    selectedTile = null;
    onCellClick(waitAction);
  }

  // A move by either side invalidates the pending selection.
  $effect(() => {
    void cells;
    selectedTile = null;
  });

  $effect(() => {
    const selected = edges.find(a => a.action === highlightedAction);
    if (readOnly && selected) selectedTile = selected.from;
  });

  function edgeProbability(action: number): number | null {
    return assessments.find(a => a.action.index === action)?.[metric] ?? null;
  }
  function directionGlyph(from: number, to: number): string {
    return to === from - width ? '↑' : to === from + width ? '↓' : to > from ? '→' : '←';
  }

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

</script>

  <!-- Generals: terrain grid, select a source tile then an adjacent target -->
  <div class="generals-container">
    <div class="generals-board" style={generalsStyle}>
      {#each cells as cell, i}
        <button
          class={getGeneralsCellClass(i, cell)}
          style="width: {generalsCellSize}px; height: {generalsCellSize}px; {heat(sourceMass(i))}"
          onclick={() => handleGeneralsClick(i)}
          disabled={gameOver}
          aria-label={`Tile ${i % width},${Math.floor(i / width)}`}
        >
          <span class="gen-terrain">{terrainGlyph(cell)}</span>
          <span class="gen-army">{armyLabel(cell)}</span>
          {#if sourceMass(i) != null}<span class="probability gen-probability">{percent(sourceMass(i))}</span>{/if}
        </button>
      {/each}
    </div>
    <div class="generals-controls">
      <span class="generals-hint">
        {#if readOnly}
          Source totals shown. Select a source to inspect directions; percentages stay global.
        {:else if gameOver}
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
        disabled={readOnly || gameOver || waitAction === null || !legalMoves.includes(waitAction)}
      >
        Wait
      </button>
    </div>
    {#if readOnly && selectedTile !== null}
      <div class="direction-inspector" aria-label="Directional action probabilities">
        {#each selectedEdges as edge}
          <span class:inspected={edge.action === highlightedAction} title={edge.label}>
            <strong>{directionGlyph(edge.from, edge.to)}</strong> {percent(edgeProbability(edge.action))}
            <small>{edge.label}</small>
          </span>
        {/each}
      </div>
    {/if}
  </div>

<style>
  .probability { position: absolute; bottom: 3px; left: 0; width: 100%; color: #fff; font-size: .7rem; font-weight: 600; text-shadow: 0 1px 3px #000; pointer-events: none; }
  .gen-probability { font-size: .56rem; bottom: 1px; }
  .inspected { outline: 2px solid #ffe082; outline-offset: -2px; }
  .direction-inspector { display: flex; gap: .6rem; flex-wrap: wrap; justify-content: center; }
  .direction-inspector span { border: 1px solid #596378; border-radius: 6px; padding: .5rem; color: #dbefe8; }
  .direction-inspector strong { font-size: 1.5rem; } .direction-inspector small { display: block; font-size: .65rem; }
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

</style>
