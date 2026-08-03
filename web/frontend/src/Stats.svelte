<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getStats, getModelInfo, type TrainingStats, type ModelInfo, type EvalStats } from './lib/api';
  import { STATS_POLL_INTERVAL_MS, MS_PER_SECOND } from './lib/constants';
  import LossChart from './LossChart.svelte';

  let stats: TrainingStats | null = $state(null);
  let modelInfo: ModelInfo | null = $state(null);
  let error: string | null = $state(null);
  let pollInterval: number | undefined;

  // Training speed tracking
  let prevStats: { step: number; timestamp: number } | null = $state(null);
  let stepsPerSecond: number | null = $state(null);
  let etaSeconds: number | null = $state(null);

  // Smoothed speed (exponential moving average)
  const SPEED_SMOOTHING = 0.3; // Lower = smoother, higher = more responsive

  async function fetchData() {
    try {
      const [statsResult, modelResult] = await Promise.all([
        getStats(),
        getModelInfo()
      ]);

      // Calculate training speed from delta
      if (prevStats && statsResult.step > prevStats.step && statsResult.timestamp > prevStats.timestamp) {
        const stepDelta = statsResult.step - prevStats.step;
        const timeDelta = statsResult.timestamp - prevStats.timestamp;
        if (timeDelta > 0) {
          const instantSpeed = stepDelta / timeDelta;
          // Apply exponential smoothing
          if (stepsPerSecond === null) {
            stepsPerSecond = instantSpeed;
          } else {
            stepsPerSecond = SPEED_SMOOTHING * instantSpeed + (1 - SPEED_SMOOTHING) * stepsPerSecond;
          }

          // Calculate ETA
          if (stepsPerSecond > 0 && statsResult.total_steps > 0) {
            const remainingSteps = statsResult.total_steps - statsResult.step;
            etaSeconds = remainingSteps / stepsPerSecond;
          }
        }
      }

      // Store current stats for next comparison
      if (statsResult.step > 0) {
        prevStats = { step: statsResult.step, timestamp: statsResult.timestamp };
      }

      stats = statsResult;
      modelInfo = modelResult;
      error = null;
    } catch (e) {
      error = 'Failed to fetch data';
    }
  }

  onMount(() => {
    fetchData();
    pollInterval = setInterval(fetchData, STATS_POLL_INTERVAL_MS);
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
  });

  function formatNumber(n: number | undefined): string {
    if (n === undefined || n === 0) return '-';
    return n.toFixed(4);
  }

  function formatPercent(n: number | undefined | null): string {
    if (n == null) return '-';
    return `${(n * 100).toFixed(1)}%`;
  }

  function formatTimestamp(ts: number | null | undefined): string {
    if (!ts) return '-';
    const date = new Date(ts * MS_PER_SECOND);
    return date.toLocaleTimeString();
  }

  function formatSpeed(speed: number | null): string {
    if (speed === null || speed <= 0) return '-';
    if (speed >= 100) return `${Math.round(speed)} steps/s`;
    if (speed >= 10) return `${speed.toFixed(1)} steps/s`;
    if (speed >= 1) return `${speed.toFixed(2)} steps/s`;
    return `${speed.toFixed(3)} steps/s`;
  }

  function formatEta(seconds: number | null): string {
    if (seconds === null || seconds <= 0) return '-';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      return `${mins}m ${secs}s`;
    }
    if (seconds < 86400) {
      const hours = Math.floor(seconds / 3600);
      const mins = Math.round((seconds % 3600) / 60);
      return `${hours}h ${mins}m`;
    }
    const days = Math.floor(seconds / 86400);
    const hours = Math.round((seconds % 86400) / 3600);
    return `${days}d ${hours}h`;
  }

  function getProgressPercent(step: number, total: number): number {
    if (total <= 0) return 0;
    return Math.min(100, (step / total) * 100);
  }

  function formatDigest(digest: string | null): string {
    if (!digest) return '-';
    return `${digest.slice(0, 12)}…`;
  }

  function getWinRateColor(winRate: number | null | undefined): string {
    if (winRate == null) return '#888';  // Gray - no data
    if (winRate >= 0.7) return '#4f4';  // Green - good
    if (winRate >= 0.5) return '#fa0';  // Orange - okay
    return '#f66';  // Red - poor
  }

  let evaluationHistory = $derived.by(() =>
    [...(stats?.eval_history ?? [])].sort((left, right) => left.step - right.step)
  );

  function chartX(history: EvalStats[], index: number): number {
    if (history.length < 2) return 50;
    const firstStep = history[0].step;
    const lastStep = history[history.length - 1].step;
    if (firstStep === lastStep) return (index / (history.length - 1)) * 100;
    return ((history[index].step - firstStep) / (lastStep - firstStep)) * 100;
  }

  function chartY(winRate: number): number {
    return 60 - Math.max(0, Math.min(1, winRate)) * 60;
  }

  function chartPoints(history: EvalStats[]): string {
    return history
      .map((point, index) => `${chartX(history, index)},${chartY(point.win_rate)}`)
      .join(' ');
  }
</script>

<div class="stats-panel">
  <!-- Model Info Section -->
  <h2>Bot Model</h2>
  {#if modelInfo}
    <div class="model-status" class:loaded={modelInfo.loaded} class:no-model={!modelInfo.loaded}>
      <span class="status-indicator"></span>
      <span class="status-text">{modelInfo.status}</span>
    </div>
    {#if modelInfo.loaded}
      <div class="stat-grid model-grid">
        {#if modelInfo.training_step != null}
          <div class="stat">
            <span class="label">Training Step</span>
            <span class="value">{modelInfo.training_step.toLocaleString()}</span>
          </div>
        {/if}
        <div class="stat">
          <span class="label">Loaded At</span>
          <span class="value">{formatTimestamp(modelInfo.loaded_at)}</span>
        </div>
        {#if modelInfo.checkpoint_id}
          <div class="stat">
            <span class="label">Checkpoint ID</span>
            <span class="value digest" title={modelInfo.checkpoint_id}>{formatDigest(modelInfo.checkpoint_id)}</span>
          </div>
        {/if}
        {#if modelInfo.model_sha256}
          <div class="stat">
            <span class="label">Weight Digest</span>
            <span class="value digest" title={modelInfo.model_sha256}>{formatDigest(modelInfo.model_sha256)}</span>
          </div>
        {/if}
      </div>
    {/if}
  {/if}

  <hr class="divider" />

  <!-- Training Stats Section -->
  <h2>Training Stats</h2>

  {#if error}
    <p class="error">{error}</p>
  {:else if stats && stats.step > 0}
    <!-- Progress Bar -->
    {#if stats.total_steps > 0}
      <div class="progress-container">
        <div class="progress-bar">
          <div
            class="progress-fill"
            style="width: {getProgressPercent(stats.step, stats.total_steps)}%"
          ></div>
        </div>
        <div class="progress-text">
          <span>{stats.step.toLocaleString()} / {stats.total_steps.toLocaleString()} steps</span>
          <span>{getProgressPercent(stats.step, stats.total_steps).toFixed(1)}%</span>
        </div>
      </div>
    {/if}

    <div class="stat-grid">
      <div class="stat">
        <span class="label">Speed</span>
        <span class="value speed-value">{formatSpeed(stepsPerSecond)}</span>
      </div>
      <div class="stat">
        <span class="label">ETA</span>
        <span class="value eta-value">{formatEta(etaSeconds)}</span>
      </div>
      <div class="stat">
        <span class="label">Total Loss</span>
        <span class="value">{formatNumber(stats.total_loss)}</span>
      </div>
      <div class="stat">
        <span class="label">Policy Loss</span>
        <span class="value">{formatNumber(stats.policy_loss)}</span>
      </div>
      <div class="stat">
        <span class="label">Value Loss</span>
        <span class="value">{formatNumber(stats.value_loss)}</span>
      </div>
      <div class="stat">
        <span class="label">Learning Rate</span>
        <span class="value">{formatNumber(stats.learning_rate)}</span>
      </div>
      <div class="stat">
        <span class="label">Replay Records</span>
        <span class="value">{stats.replay_record_count.toLocaleString()}</span>
      </div>
      <div class="stat">
        <span class="label">Last Update</span>
        <span class="value">{formatTimestamp(stats.timestamp)}</span>
      </div>
    </div>

    <!-- Loss Chart -->
    {#if stats.history && stats.history.length > 0}
      <LossChart history={stats.history} />
    {/if}

    <!-- Evaluation Section -->
    {#if stats.last_eval}
      <hr class="divider" />
      <h2>Model Evaluation</h2>

      <div class="eval-card">
        <div class="eval-card-header">
          <span class="opponent-label">vs Random</span>
        </div>
        <div class="win-rate-value" style="color: {getWinRateColor(stats.last_eval.win_rate)}">
          {formatPercent(stats.last_eval.win_rate)}
        </div>
        <div class="eval-details">
          <span>Draw: {formatPercent(stats.last_eval.draw_rate)}</span>
          <span>Loss: {formatPercent(stats.last_eval.loss_rate)}</span>
        </div>
      </div>

      <div class="stat-grid">
        <div class="stat">
          <span class="label">Evaluated Step</span>
          <span class="value">{stats.last_eval.step.toLocaleString()}</span>
        </div>
        <div class="stat">
          <span class="label">Games</span>
          <span class="value">{stats.last_eval.games_played.toLocaleString()}</span>
        </div>
        <div class="stat">
          <span class="label">Average Length</span>
          <span class="value">{stats.last_eval.avg_game_length.toFixed(1)}</span>
        </div>
        <div class="stat">
          <span class="label">Evaluated At</span>
          <span class="value">{formatTimestamp(stats.last_eval.timestamp)}</span>
        </div>
      </div>

      {#if evaluationHistory.length > 1}
        <div class="chart-container">
          <h3>Win Rate by Training Step</h3>
          <svg class="win-rate-chart" viewBox="0 0 100 60" preserveAspectRatio="none" role="img">
            <title>Win rate against random by training step</title>
            <line class="chart-baseline" x1="0" y1="30" x2="100" y2="30"></line>
            <polyline class="chart-line" points={chartPoints(evaluationHistory)}></polyline>
            {#each evaluationHistory as evalPoint, index}
              <circle
                class="chart-point"
                cx={chartX(evaluationHistory, index)}
                cy={chartY(evalPoint.win_rate)}
                r="1.8"
              >
                <title>Step {evalPoint.step}: {formatPercent(evalPoint.win_rate)}</title>
              </circle>
            {/each}
          </svg>
          <div class="chart-labels">
            <span>Step {evaluationHistory[0].step.toLocaleString()}</span>
            <span>50% win rate</span>
            <span>Step {evaluationHistory[evaluationHistory.length - 1].step.toLocaleString()}</span>
          </div>
        </div>
      {/if}
    {:else}
      <hr class="divider" />
      <h2>Model Evaluation</h2>
      <p class="no-data">No evaluation data yet.</p>
      <p class="hint">Evaluation runs automatically during training.</p>
    {/if}
  {:else}
    <p class="no-data">No training data yet.</p>
    <p class="hint">Start the Python trainer to see stats here.</p>
  {/if}

</div>

<style>
  .stats-panel {
    background: #2a2a4a;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: left;
  }

  h2 {
    margin: 0 0 1rem 0;
    color: #00d9ff;
    font-size: 1.2rem;
  }

  .divider {
    border: none;
    border-top: 1px solid #3a3a5a;
    margin: 1.5rem 0;
  }

  /* Progress bar styles */
  .progress-container {
    margin-bottom: 1rem;
  }

  .progress-bar {
    height: 8px;
    background: #3a3a5a;
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d9ff, #00ff88);
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .progress-text {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #888;
    margin-top: 0.25rem;
  }

  .speed-value {
    color: #00d9ff;
  }

  .eta-value {
    color: #00ff88;
  }

  .model-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem;
    border-radius: 8px;
    background: #3a3a5a;
    margin-bottom: 0.75rem;
  }

  .model-status.loaded {
    background: #1a4a2a;
  }

  .model-status.no-model {
    background: #4a3a1a;
  }

  .status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #888;
  }

  .model-status.loaded .status-indicator {
    background: #4f4;
  }

  .model-status.no-model .status-indicator {
    background: #fa0;
  }

  .status-text {
    font-size: 0.9rem;
    color: #fff;
  }

  .model-grid {
    margin-bottom: 0;
  }

  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  .stat {
    background: #3a3a5a;
    padding: 0.75rem;
    border-radius: 8px;
  }

  .label {
    display: block;
    font-size: 0.75rem;
    color: #888;
    margin-bottom: 0.25rem;
  }

  .value {
    font-size: 1.1rem;
    font-weight: bold;
    color: #fff;
  }

  .value.digest {
    display: block;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9rem;
    overflow-wrap: anywhere;
  }

  .error {
    color: #f66;
  }

  .no-data {
    color: #888;
    margin-bottom: 0.5rem;
  }

  .hint {
    font-size: 0.85rem;
    color: #666;
  }

  /* Evaluation styles */
  .eval-card {
    background: #3a3a5a;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 1rem;
  }

  .eval-card-header {
    margin-bottom: 0.5rem;
  }

  .opponent-label {
    font-size: 0.85rem;
    color: #00d9ff;
    font-weight: bold;
  }

  .win-rate-value {
    font-size: 2rem;
    font-weight: bold;
    line-height: 1;
    margin: 0.5rem 0;
  }

  .eval-details {
    display: flex;
    justify-content: center;
    gap: 1rem;
    font-size: 0.75rem;
    color: #888;
  }

  /* Evaluation history chart */
  .chart-container {
    margin-top: 1rem;
  }

  .chart-container h3 {
    font-size: 0.9rem;
    color: #888;
    margin: 0 0 0.5rem 0;
    font-weight: normal;
  }

  .win-rate-chart {
    display: block;
    width: 100%;
    height: 90px;
    background: #3a3a5a;
    border-radius: 8px;
    overflow: visible;
  }

  .chart-baseline {
    stroke: #666;
    stroke-width: 0.5;
    stroke-dasharray: 2 2;
  }

  .chart-line {
    fill: none;
    stroke: #00d9ff;
    stroke-width: 1.5;
    vector-effect: non-scaling-stroke;
  }

  .chart-point {
    fill: #00ff88;
    stroke: #1a1a2e;
    stroke-width: 0.75;
    vector-effect: non-scaling-stroke;
  }

  .chart-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #666;
    margin-top: 0.25rem;
  }

</style>
