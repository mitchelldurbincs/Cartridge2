<script lang="ts">
  import { samePosition, percent, sortedActions, type DecisionAnalysis, type ActionPresentation, type PositionKey, type ProbabilityMetric } from './lib/analysis';
  let { decision, position, presentations, metric = $bindable('visit_share'), onAction = (_action: number) => {} }:
    { decision: DecisionAnalysis | null; position: PositionKey | null; presentations: ActionPresentation[];
      metric?: ProbabilityMetric; onAction?: (action: number) => void } = $props();
  let analysis = $derived(decision?.schema_version === 1 && position && samePosition(decision.position, position) ? decision : null);
  let showAll = $state(false);
  let availableMetrics = $derived((['visit_share', 'network_prior', 'selection_probability'] as const)
    .filter(key => analysis?.actions.some(action => action[key] != null)));
  let effectiveMetric = $derived(availableMetrics.includes(metric) ? metric : availableMetrics[0] ?? metric);
  let rows = $derived(availableMetrics.length ? sortedActions(analysis?.actions ?? [], effectiveMetric) :
    [...(analysis?.actions ?? [])].sort((a, b) => (b.q_value ?? -Infinity) - (a.q_value ?? -Infinity) || a.action.index - b.action.index));
  let labels = $derived(new Map(presentations.map(action => [action.action, action.label])));
  const metricLabels = { visit_share: 'Search visit share', network_prior: 'Network prior', selection_probability: 'Selection probability' };
  $effect(() => { if (availableMetrics.length && !availableMetrics.includes(metric)) metric = availableMetrics[0]; });
  function value(value: number | null | undefined): string { return value == null ? '—' : value.toFixed(3); }
</script>

<section class="inspector" aria-label="Decision inspector">
  <h2>Decision inspector</h2>
  {#if !position}
    <p>Choose “Last bot decision” or a history position to inspect a move. Analysis is never overlaid on a different position.</p>
  {:else if !analysis}
    <p>No recorded decision for position {position.revision}. This may be the live or final position.</p>
  {:else}
    <p class="context">Position {analysis.position.revision} · Agent {analysis.actor} · {analysis.source.replaceAll('_', ' ')}</p>
    <p>Played: <strong>{labels.get(analysis.selected_action.index) ?? `Action ${analysis.selected_action.index}`}</strong></p>
    {#if analysis.source === 'human'}
      <p>Human move. No search or value estimate was recorded.</p>
    {:else if analysis.source === 'random'}
      <p>No model loaded. These are uniform random selection probabilities, not learned preferences.</p>
    {/if}
    {#if analysis.network_value || analysis.search_value}
      <div class="values">
        {#each [['Network', analysis.network_value], ['Search', analysis.search_value]] as [label, estimate]}
          {#if estimate && typeof estimate !== 'string'}
            <div><span>{label} value</span><strong>{value(estimate.value)}</strong>
              <small>Agent {estimate.perspective_agent} · {estimate.quantity.replaceAll('_', ' ')}</small></div>
          {/if}
        {/each}
      </div>
      <p class="note">Value estimates are not win percentages. Action Q uses this decision's agent perspective; “—” means unavailable.</p>
    {/if}
    {#if analysis.search}
      <p class="effort">{analysis.search.completed_simulations} simulations · {(analysis.search.total_time_us / 1000).toFixed(1)} ms</p>
      <details><summary>Search details</summary>
        <p>{analysis.search.root_visits} root visits (includes bootstrap) · {analysis.search.neural_evaluations} evaluated observations · temperature {analysis.search.temperature}</p>
      </details>
    {/if}
    {#if rows.length}
      <div class="toolbar">
        {#if availableMetrics.length}<label>Overlay / rank
          <select bind:value={metric}>{#each availableMetrics as key}<option value={key}>{metricLabels[key]}</option>{/each}</select>
        </label>{/if}
        <span>{rows.length} legal actions</span>
      </div>
      <div class="table-scroll"><table>
        <thead><tr><th>Action</th><th>Prior</th><th>Visits %</th><th>Pick %</th><th>N</th><th>Q</th></tr></thead>
        <tbody>{#each (showAll ? rows : rows.slice(0, 10)) as action (action.action.index)}
          <tr class:played={action.action.index === analysis.selected_action.index}>
            <td><button onclick={() => onAction(action.action.index)} title="Highlight this action, without playing it">
              {action.action.index === analysis.selected_action.index ? '✓ ' : ''}{labels.get(action.action.index) ?? `Action ${action.action.index}`}
            </button>{#if action.expanded === false}<small>not expanded</small>{/if}</td>
            <td>{percent(action.network_prior)}</td><td>{percent(action.visit_share)}</td>
            <td>{percent(action.selection_probability)}</td><td>{action.visits ?? '—'}</td><td>{value(action.q_value)}</td>
          </tr>
        {/each}</tbody>
      </table></div>
      {#if rows.length > 10}<button onclick={() => showAll = !showAll}>{showAll ? 'Top 10' : `All ${rows.length} legal actions`}</button>{/if}
    {/if}
    {#if analysis.checkpoint}
      <p class="checkpoint" title={analysis.checkpoint.checkpoint_id}>Checkpoint {analysis.checkpoint.checkpoint_id.slice(0, 12)} · step {analysis.checkpoint.training_step ?? 'unknown'}</p>
    {/if}
  {/if}
</section>

<style>
  .inspector { text-align: left; background: #151b2b; border: 1px solid #344157; border-radius: 12px; padding: 1rem; color: #e7eef7; }
  h2 { margin: 0 0 .8rem; font-size: 1.2rem; } p { line-height: 1.5; }
  .context,.note,.checkpoint,small { color: #b1bfd2; font-size: .8rem; } small { display: block; }
  .values { display: flex; gap: 1.6rem; } .values strong { display: block; font-size: 1.7rem; }
  .values span { font-size: .85rem; } .effort { color: #83dccd; }
  .toolbar { display: flex; gap: .6rem; flex-wrap: wrap; align-items: center; margin: 1rem 0; font-size: .8rem; }
  select { margin-left: .5rem; padding: .35rem; background: #243148; color: white; border: 1px solid #536883; border-radius: 4px; }
  .table-scroll { overflow: auto; } table { border-collapse: collapse; width: 100%; font-size: .76rem; }
  th,td { padding: .5rem .3rem; text-align: right; border-bottom: 1px solid #2a374a; white-space: nowrap; }
  th:first-child,td:first-child { text-align: left; } .played { background: #1d3b3b; }
  button { color: #c4ece7; background: none; border: 1px solid #41556c; border-radius: 4px; padding: .35rem; cursor: pointer; text-align: left; }
  .checkpoint { overflow-wrap: anywhere; } details { font-size: .8rem; }
</style>
