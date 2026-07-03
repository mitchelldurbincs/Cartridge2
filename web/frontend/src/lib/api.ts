// API client for the Cartridge2 backend

export interface GameState {
  board: number[];
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
  board_type: 'grid' | 'drop_column';
}

export interface GamesListResponse {
  games: string[];
}

export interface MoveResponse extends GameState {
  bot_move: number | null;
}

export interface EvalStats {
  step: number;
  current_iteration: number;
  // Model vs Best (gatekeeper) results
  opponent: 'best' | 'random';
  opponent_iteration: number | null;
  win_rate: number;
  draw_rate: number;
  loss_rate: number;
  became_new_best: boolean;
  // Model vs Random results (optional)
  vs_random_win_rate: number | null;
  vs_random_draw_rate: number | null;
  // Metadata
  games_played: number;
  avg_game_length?: number;
  timestamp: number;
}

export interface HistoryEntry {
  step: number;
  total_loss: number;
  value_loss: number;
  policy_loss: number;
  learning_rate: number;
  grad_norm?: number;  // Optional: only present when gradient clipping is enabled
}

export interface TrainingStats {
  step: number;
  total_steps: number;
  total_loss: number;
  policy_loss: number;
  value_loss: number;
  replay_buffer_size: number;
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
  path: string | null;
  file_modified: number | null;
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

export interface ActorStats {
  env_id: string;
  episodes_completed: number;
  total_steps: number;
  player1_wins: number;
  player2_wins: number;
  draws: number;
  avg_episode_length: number;
  episodes_per_second: number;
  runtime_seconds: number;
  mcts_avg_inference_us: number;
  timestamp: number;
}

export async function getActorStats(): Promise<ActorStats> {
  const res = await fetch(`${API_BASE}/actor-stats`);
  if (!res.ok) throw new Error('Failed to get actor stats');
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
