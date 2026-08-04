// API client for the Cartridge2 backend

/** Terrain of a cell. Mirrors engine_core::board_profile::CellKind. */
export type CellKind = 'normal' | 'general' | 'city' | 'mountain';

/** One board cell, straight from the engine's BoardView. */
export interface CellView {
  /** 0 = empty/neutral, 1/2 = owning player. */
  owner: number;
  kind: CellKind;
  /** Per-cell quantity (Generals' army count); 0 for the flat games. */
  value: number;
}

export interface GameState {
  cells: CellView[];
  current_player: number;
  human_player: number;
  winner: number;
  game_over: boolean;
  legal_moves: number[];
  message: string;
}

export interface GameInfo {
  env_id: string;
  display_name: string;
  board_width: number;
  board_height: number;
  num_actions: number;
  player_count: number;
  player_names: string[];
  player_symbols: string[];
  description: string;
  board_type: 'grid' | 'drop_column' | 'generals';
}

export interface GamesListResponse {
  games: string[];
}

export interface MoveResponse extends GameState {
  bot_move: number | null;
}

export interface EvalStats {
  step: number;
  win_rate: number;
  draw_rate: number;
  loss_rate: number;
  games_played: number;
  avg_game_length: number;
  timestamp: number;
}

export interface HistoryEntry {
  step: number;
  total_loss: number;
  value_loss: number;
  policy_loss: number;
  learning_rate: number;
  grad_norm: number | null;
}

export interface TrainingStats {
  step: number;
  total_steps: number;
  total_loss: number;
  policy_loss: number;
  value_loss: number;
  samples_seen: number;
  replay_record_count: number;
  last_checkpoint: string;
  learning_rate: number;
  timestamp: number;
  env_id: string;
  last_eval: EvalStats | null;
  eval_history: EvalStats[];
  history: HistoryEntry[];
}

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ModelInfo {
  loaded: boolean;
  checkpoint_id: string | null;
  model_sha256: string | null;
  path: string | null;
  loaded_at: number | null;
  training_step: number | null;
  status: string;
}

const API_BASE = '';

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function getGameState(): Promise<GameState> {
  const res = await fetch(`${API_BASE}/game/state`);
  if (!res.ok) throw new Error('Failed to get game state');
  return res.json();
}

export async function newGame(first: 'player' | 'bot' = 'player', game?: string): Promise<GameState> {
  const body: { first: string; game?: string } = { first };
  if (game) {
    body.game = game;
  }
  const res = await fetch(`${API_BASE}/game/new`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Failed to create new game');
  }
  return res.json();
}

export async function makeMove(position: number): Promise<MoveResponse> {
  const res = await fetch(`${API_BASE}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Move failed');
  }
  return res.json();
}

export async function getStats(): Promise<TrainingStats> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to get stats');
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model`);
  if (!res.ok) throw new Error('Failed to get model info');
  return res.json();
}

export async function getGames(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/games`);
  if (!res.ok) throw new Error('Failed to get games list');
  const data: GamesListResponse = await res.json();
  return data.games;
}

export async function getGameInfo(envId: string): Promise<GameInfo> {
  const res = await fetch(`${API_BASE}/game-info/${envId}`);
  if (!res.ok) throw new Error(`Failed to get game info for ${envId}`);
  return res.json();
}
