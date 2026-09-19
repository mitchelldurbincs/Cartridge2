// Board-independent inspector contract. Probabilities are optional: Q learners
// and human decisions do not fabricate policy/search data.
import type { GameState } from './api';

export type ActionTarget =
  | { kind: 'cell'; index: number }
  | { kind: 'column'; index: number }
  | { kind: 'edge'; from: number; to: number }
  | { kind: 'named'; name: string };

export interface ActionPresentation { action: number; label: string; target: ActionTarget | null }
export interface PositionKey { session_id: string; revision: number }
export interface ActionAssessment {
  action: { kind: 'discrete'; index: number };
  network_prior: number | null;
  search_prior: number | null;
  visit_share: number | null;
  selection_probability: number | null;
  visits: number | null;
  q_value: number | null;
  expanded: boolean | null;
}
export interface ValueEstimate {
  value: number;
  perspective_agent: number;
  quantity: 'expected_outcome' | 'discounted_return';
  bounds: [number, number] | null;
}
export interface DecisionAnalysis {
  schema_version: number;
  analysis_id: string;
  position: PositionKey;
  env_id: string;
  env_contract_version: number;
  algorithm_id: string;
  actor: number;
  source: 'alpha_zero_mcts' | 'random' | 'human' | 'dqn_q_values';
  checkpoint: { checkpoint_id: string; training_step: number | null } | null;
  selected_action: { kind: 'discrete'; index: number };
  network_value: ValueEstimate | null;
  search_value: ValueEstimate | null;
  actions: ActionAssessment[];
  search: {
    completed_simulations: number; root_visits: number; neural_evaluations: number;
    total_time_us: number; temperature: number;
  } | null;
}
export interface PositionRecord { state: GameState; decision: DecisionAnalysis | null }
export interface HistoryResponse {
  schema_version: number;
  session_id: string;
  first_available_revision: number;
  current_revision: number;
  records: PositionRecord[];
  next_revision: number | null;
}
export type ProbabilityMetric = 'visit_share' | 'network_prior' | 'selection_probability';

export function samePosition(left: PositionKey, right: PositionKey): boolean {
  return left.session_id === right.session_id && left.revision === right.revision;
}

export function percent(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1)}%`;
}

export function sortedActions(actions: ActionAssessment[], metric: ProbabilityMetric): ActionAssessment[] {
  return [...actions].sort((a, b) =>
    (b[metric] ?? -Infinity) - (a[metric] ?? -Infinity) || a.action.index - b.action.index);
}

/** Aggregate an explicitly declared source/cell/column without renormalizing. */
export function targetMass(presentations: ActionPresentation[], actions: ActionAssessment[],
  metric: ProbabilityMetric, kind: 'cell' | 'column' | 'source', index: number): number | null {
  const ids = new Set(presentations.filter(({ target }) => target && (
    kind === 'source' ? target.kind === 'edge' && target.from === index :
    target.kind === kind && target.index === index
  )).map(a => a.action));
  const values = actions.filter(a => ids.has(a.action.index)).map(a => a[metric]);
  const available = values.filter((v): v is number => v != null && Number.isFinite(v));
  return available.length ? available.reduce((sum, value) => sum + value, 0) : null;
}

export function boundAnalysis(record: PositionRecord | null): DecisionAnalysis | null {
  if (!record?.decision || record.decision.schema_version !== 1 ||
      !samePosition(record.state, record.decision.position)) return null;
  return record.decision;
}
