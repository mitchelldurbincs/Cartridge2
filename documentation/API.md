# Cartridge2 REST API Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Base URL and CORS](#2-base-url-and-cors)
3. [Authentication](#3-authentication)
4. [Common Response Patterns](#4-common-response-patterns)
5. [Endpoints Reference](#5-endpoints-reference)
   - [Health](#51-health)
   - [Games Discovery](#52-games-discovery)
   - [Game Session](#53-game-session)
   - [Training Statistics](#54-training-statistics)
   - [Model Information](#55-model-information)
   - [Actor Statistics](#56-actor-statistics)
   - [Prometheus Metrics](#57-prometheus-metrics)
6. [Type Definitions](#6-type-definitions)
7. [Error Handling](#7-error-handling)
8. [Game-Specific Behavior](#8-game-specific-behavior)
9. [WebSocket Support](#9-websocket-support)
10. [Rate Limiting](#10-rate-limiting)
11. [Examples](#11-examples)

---

## 1. Overview

The Cartridge2 Web API provides a RESTful interface for:

- Playing games against trained AI models
- Monitoring training progress and statistics
- Managing game sessions
- Querying model status

### Technology Stack

| Component | Technology |
|-----------|------------|
| Server | Rust + Axum |
| Serialization | JSON (serde) |
| Default Port | 8080 |

### API Version

Current version: **0.1.0** (returned in `/health` response)

---

## 2. Base URL and CORS

### Base URL

```
http://localhost:8080
```

In Docker:
```
http://web:8080         # From other containers
http://localhost:8080   # From host machine
```

### CORS Configuration

CORS is deny-by-default:

- When `web.allowed_origins` (config.toml) is empty, only localhost origins
  (e.g. `http://localhost:5173`, `http://localhost:8080`) are allowed —
  suitable for local development.
- For production, list explicit origins:

```toml
[web]
allowed_origins = ["https://yourdomain.com"]
```

Allowed methods are `GET`, `POST`, `OPTIONS` with the `Content-Type` and
`Accept` headers.

---

## 3. Authentication

**No authentication required.**

The API is designed for local development and trusted networks. All endpoints are publicly accessible.

---

## 4. Common Response Patterns

### Success Response

All successful responses return JSON with HTTP 200 OK:

```json
{
  "field1": "value1",
  "field2": "value2"
}
```

### Error Response

Errors return an appropriate HTTP status code with a plain text error message:

```
HTTP/1.1 400 Bad Request
Content-Type: text/plain

Illegal move: position/column 9 is not valid
```

### Content Types

| Direction | Content-Type |
|-----------|--------------|
| Request | `application/json` |
| Response (success) | `application/json` |
| Response (error) | `text/plain` |

---

## 5. Endpoints Reference

### 5.1 Health

#### GET /health

Health check endpoint for monitoring and load balancers.

**Request**

```http
GET /health HTTP/1.1
Host: localhost:8080
```

**Response**

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` if server is running |
| `version` | string | Server version from Cargo.toml |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Server is healthy |

**Example**

```bash
curl http://localhost:8080/health
```

---

### 5.2 Games Discovery

#### GET /games

List the playable game environment. Only the currently configured game
(`common.env_id`) is returned — other registered games are hidden so the UI
always matches the game the loaded model was trained for.

**Request**

```http
GET /games HTTP/1.1
Host: localhost:8080
```

**Response**

```json
{
  "games": ["connect4"]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `games` | string[] | Single-element array with the current game ID |

**Example**

```bash
curl http://localhost:8080/games
```

---

#### GET /game-info/:id

Get detailed metadata for a specific game.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Game environment ID (e.g., `tictactoe`, `connect4`) |

**Request**

```http
GET /game-info/tictactoe HTTP/1.1
Host: localhost:8080
```

**Response**

```json
{
  "env_id": "tictactoe",
  "display_name": "Tic-Tac-Toe",
  "board_width": 3,
  "board_height": 3,
  "num_actions": 9,
  "obs_size": 29,
  "legal_mask_offset": 18,
  "player_count": 2,
  "player_names": ["X", "O"],
  "player_symbols": ["X", "O"],
  "description": "Get three in a row to win!",
  "board_type": "grid"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `env_id` | string | Unique game identifier |
| `display_name` | string | Human-readable game name |
| `board_width` | number | Board width in cells |
| `board_height` | number | Board height in cells |
| `num_actions` | number | Total possible actions |
| `obs_size` | number | Observation vector size (for neural network) |
| `legal_mask_offset` | number | Index where legal moves start in observation |
| `player_count` | number | Number of players (always 2) |
| `player_names` | string[] | Player names for display |
| `player_symbols` | string[] | Player symbols for board rendering |
| `description` | string | Game description |
| `board_type` | string | `"grid"` or `"drop_column"` |

**Board Types**

| Type | Games | Description |
|------|-------|-------------|
| `grid` | TicTacToe, Othello | Click any empty cell |
| `drop_column` | Connect4 | Click column, piece drops to bottom |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Success |
| 403 | Game exists but is not the currently configured game |
| 404 | Game not found |

**Error Response (403)**

```
Cannot access game 'tictactoe': only the current game 'connect4' is available
```

**Example**

```bash
curl http://localhost:8080/game-info/connect4
```

---

### 5.3 Game Session

#### GET /game/state

Get the current game board state.

**Request**

```http
GET /game/state HTTP/1.1
Host: localhost:8080
```

**Response**

```json
{
  "board": [0, 0, 0, 0, 1, 0, 0, 0, 2],
  "current_player": 1,
  "human_player": 1,
  "winner": 0,
  "game_over": false,
  "legal_moves": [0, 1, 2, 3, 5, 6, 7],
  "message": "Your turn (X)"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `board` | number[] | Board cells (0=empty, 1=player1, 2=player2) |
| `current_player` | number | Whose turn: 1 or 2 |
| `human_player` | number | Which player is human: 1 or 2 |
| `winner` | number | Game result: 0=ongoing, 1=P1 wins, 2=P2 wins, 3=draw |
| `game_over` | boolean | Is the game finished? |
| `legal_moves` | number[] | Valid positions/columns for next move |
| `message` | string | Human-readable status message |

**Board Layout**

TicTacToe (3x3):
```
Index:  0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8
```

Connect4 (7x6, column indices):
```
Columns:  0   1   2   3   4   5   6
          |   |   |   |   |   |   |
          v   v   v   v   v   v   v
Row 5:  [35][36][37][38][39][40][41]  ← Top
Row 4:  [28][29][30][31][32][33][34]
Row 3:  [21][22][23][24][25][26][27]
Row 2:  [14][15][16][17][18][19][20]
Row 1:  [ 7][ 8][ 9][10][11][12][13]
Row 0:  [ 0][ 1][ 2][ 3][ 4][ 5][ 6]  ← Bottom
```

**Example**

```bash
curl http://localhost:8080/game/state
```

---

#### POST /game/new

Start a new game session.

**Request Body**

```json
{
  "first": "player",
  "game": "tictactoe"
}
```

**Request Fields**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `first` | string | No | `"player"` | Who moves first: `"player"` or `"bot"` |
| `game` | string | No | Current game | Game ID to play |

**Request**

```http
POST /game/new HTTP/1.1
Host: localhost:8080
Content-Type: application/json

{"first": "bot", "game": "connect4"}
```

**Response**

When `first: "player"`:
```json
{
  "board": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "current_player": 1,
  "human_player": 1,
  "winner": 0,
  "game_over": false,
  "legal_moves": [0, 1, 2, 3, 4, 5, 6, 7, 8],
  "message": "Your turn (X)"
}
```

When `first: "bot"`:
```json
{
  "board": [0, 0, 0, 0, 1, 0, 0, 0, 0],
  "current_player": 2,
  "human_player": 2,
  "winner": 0,
  "game_over": false,
  "legal_moves": [0, 1, 2, 3, 5, 6, 7, 8],
  "message": "Your turn (O)"
}
```

**Who Plays What**

| `first` | Human Plays | Bot Plays | Human's Turn |
|---------|-------------|-----------|--------------|
| `"player"` | X (player 1) | O (player 2) | First |
| `"bot"` | O (player 2) | X (player 1) | After bot moves |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Game created successfully |
| 403 | Requested game is not the currently configured game |
| 500 | Failed to create game (invalid game ID) |

**Example**

```bash
# Player goes first (default)
curl -X POST http://localhost:8080/game/new \
  -H "Content-Type: application/json" \
  -d '{}'

# Bot goes first, play Connect4
curl -X POST http://localhost:8080/game/new \
  -H "Content-Type: application/json" \
  -d '{"first": "bot", "game": "connect4"}'
```

---

#### POST /move

Make a move. The server executes the player's move, then the bot responds.

**Request Body**

```json
{
  "position": 4
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `position` | number | Yes | Position index or column number (game-specific) |

**Position Values**

| Game | Valid Positions | Meaning |
|------|-----------------|---------|
| TicTacToe | 0-8 | Cell index (see board layout) |
| Connect4 | 0-6 | Column number (piece drops to bottom) |

**Request**

```http
POST /move HTTP/1.1
Host: localhost:8080
Content-Type: application/json

{"position": 4}
```

**Response**

```json
{
  "board": [0, 0, 0, 0, 1, 2, 0, 0, 0],
  "current_player": 1,
  "human_player": 1,
  "winner": 0,
  "game_over": false,
  "legal_moves": [0, 1, 2, 3, 6, 7, 8],
  "message": "Your turn (X)",
  "bot_move": 5
}
```

**Response Fields**

Includes all fields from `GameStateResponse` plus:

| Field | Type | Description |
|-------|------|-------------|
| `bot_move` | number \| null | Position where bot moved (null if game ended on player's move) |

**Move Sequence**

1. Validate player's move is legal
2. Execute player's move
3. Check if game ended
4. If not ended: Bot selects and executes move
5. Return updated state with `bot_move`

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Move executed successfully |
| 400 | Invalid move (see error messages below) |
| 500 | Internal error (bot move failed) |

**Error Messages (400)**

| Message | Cause |
|---------|-------|
| `Game is already over` | Attempting to move after game ended |
| `Not your turn` | Attempting to move when it's bot's turn |
| `Illegal move: position/column X is not valid` | Invalid or occupied position |

**Examples**

```bash
# TicTacToe: Move to center
curl -X POST http://localhost:8080/move \
  -H "Content-Type: application/json" \
  -d '{"position": 4}'

# Connect4: Drop piece in column 3
curl -X POST http://localhost:8080/move \
  -H "Content-Type: application/json" \
  -d '{"position": 3}'
```

---

### 5.4 Training Statistics

#### GET /stats

Get current training statistics from `stats.json`.

**Request**

```http
GET /stats HTTP/1.1
Host: localhost:8080
```

**Response**

```json
{
  "step": 1500,
  "total_steps": 5000,
  "total_loss": 0.2345,
  "policy_loss": 0.1234,
  "value_loss": 0.1111,
  "replay_buffer_size": 125000,
  "learning_rate": 0.0001,
  "timestamp": 1704067200.0,
  "env_id": "tictactoe",
  "last_eval": {
    "step": 1500,
    "win_rate": 0.72,
    "draw_rate": 0.15,
    "loss_rate": 0.13,
    "games_played": 50,
    "avg_game_length": 7.2,
    "timestamp": 1704067200.0
  },
  "eval_history": [...],
  "history": [
    {
      "step": 100,
      "total_loss": 0.8,
      "policy_loss": 0.4,
      "value_loss": 0.4,
      "learning_rate": 0.001
    }
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `step` | number | Current training step |
| `total_steps` | number | Total planned training steps |
| `total_loss` | number | Combined loss (policy + value) |
| `policy_loss` | number | Policy head loss |
| `value_loss` | number | Value head loss |
| `replay_buffer_size` | number | Number of transitions in buffer |
| `learning_rate` | number | Current learning rate |
| `timestamp` | number | Unix timestamp of last update |
| `env_id` | string | Game being trained |
| `last_eval` | EvalStats \| null | Most recent evaluation results |
| `eval_history` | EvalStats[] | History of all evaluations |
| `history` | HistoryEntry[] | Training loss history (downsampled) |

**EvalStats Fields**

| Field | Type | Description |
|-------|------|-------------|
| `step` | number | Training step when evaluated |
| `win_rate` | number | Win rate (0.0 - 1.0) |
| `draw_rate` | number | Draw rate (0.0 - 1.0) |
| `loss_rate` | number | Loss rate (0.0 - 1.0) |
| `games_played` | number | Games in evaluation |
| `avg_game_length` | number | Average moves per game |
| `timestamp` | number | Unix timestamp |

**HistoryEntry Fields**

| Field | Type | Description |
|-------|------|-------------|
| `step` | number | Training step |
| `total_loss` | number | Total loss at step |
| `policy_loss` | number | Policy loss at step |
| `value_loss` | number | Value loss at step |
| `learning_rate` | number | Learning rate at step |

**Notes**

- Returns empty/default stats if `stats.json` doesn't exist
- File is read from `{data_dir}/stats.json`
- History is downsampled for large training runs

**Example**

```bash
curl http://localhost:8080/stats
```

---

### 5.5 Model Information

#### GET /model

Get information about the currently loaded ONNX model.

**Request**

```http
GET /model HTTP/1.1
Host: localhost:8080
```

**Response (Model Loaded)**

```json
{
  "loaded": true,
  "path": "/app/data/models/latest.onnx",
  "file_modified": 1704067200,
  "loaded_at": 1704067210,
  "training_step": 1500,
  "status": "Model loaded (step 1500)"
}
```

**Response (No Model)**

```json
{
  "loaded": false,
  "path": null,
  "file_modified": null,
  "loaded_at": null,
  "training_step": null,
  "status": "No model loaded - bot plays randomly"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `loaded` | boolean | Is a model currently loaded? |
| `path` | string \| null | Path to model file |
| `file_modified` | number \| null | File modification time (Unix timestamp) |
| `loaded_at` | number \| null | When model was loaded (Unix timestamp) |
| `training_step` | number \| null | Training step (parsed from filename) |
| `status` | string | Human-readable status message |

**Model Loading**

- Server watches `{data_dir}/models/latest.onnx` for changes
- Hot-reloads model automatically when file changes
- Bot plays randomly if no model is loaded

**Example**

```bash
curl http://localhost:8080/model
```

---

### 5.6 Actor Statistics

#### GET /actor-stats

Get self-play statistics written by the actor to `{data_dir}/actor_stats.json`.
Returns zeroed defaults if the file doesn't exist yet.

**Request**

```http
GET /actor-stats HTTP/1.1
Host: localhost:8080
```

**Response**

```json
{
  "env_id": "connect4",
  "episodes_completed": 1250,
  "total_steps": 31875,
  "player1_wins": 640,
  "player2_wins": 545,
  "draws": 65,
  "avg_episode_length": 25.5,
  "episodes_per_second": 3.2,
  "runtime_seconds": 390.6,
  "mcts_avg_inference_us": 850.0,
  "timestamp": 1704067200
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `env_id` | string | Game being self-played |
| `episodes_completed` | number | Episodes finished |
| `total_steps` | number | Total game steps across all episodes |
| `player1_wins` | number | Episodes won by player 1 |
| `player2_wins` | number | Episodes won by player 2 |
| `draws` | number | Episodes ending in a draw |
| `avg_episode_length` | number | Average moves per episode |
| `episodes_per_second` | number | Self-play throughput |
| `runtime_seconds` | number | Total actor runtime |
| `mcts_avg_inference_us` | number | Average MCTS inference time (µs) |
| `timestamp` | number | Unix timestamp of last update |

**Example**

```bash
curl http://localhost:8080/actor-stats
```

---

### 5.7 Prometheus Metrics

#### GET /metrics

Prometheus-format metrics for scraping (request counts, latencies, game
sessions). Returns `text/plain` in the Prometheus exposition format, not JSON.

**Example**

```bash
curl http://localhost:8080/metrics
```

---

## 6. Type Definitions

### TypeScript Definitions

```typescript
// Health check
interface HealthResponse {
  status: string;
  version: string;
}

// Games discovery
interface GamesListResponse {
  games: string[];
}

interface GameInfo {
  env_id: string;
  display_name: string;
  board_width: number;
  board_height: number;
  num_actions: number;
  obs_size: number;
  legal_mask_offset: number;
  player_count: number;
  player_names: string[];
  player_symbols: string[];
  description: string;
  board_type: 'grid' | 'drop_column';
}

// Game session
interface GameState {
  board: number[];
  current_player: number;
  human_player: number;
  winner: number;
  game_over: boolean;
  legal_moves: number[];
  message: string;
}

interface MoveResponse extends GameState {
  bot_move: number | null;
}

interface NewGameRequest {
  first?: 'player' | 'bot';
  game?: string;
}

interface MoveRequest {
  position: number;
}

// Training statistics
interface TrainingStats {
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

interface EvalStats {
  step: number;
  win_rate: number;
  draw_rate: number;
  loss_rate: number;
  games_played: number;
  avg_game_length: number;
  timestamp: number;
}

interface HistoryEntry {
  step: number;
  total_loss: number;
  policy_loss: number;
  value_loss: number;
  learning_rate: number;
}

// Model info
interface ModelInfo {
  loaded: boolean;
  path: string | null;
  file_modified: number | null;
  loaded_at: number | null;
  training_step: number | null;
  status: string;
}

// Actor self-play statistics
interface ActorStats {
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
```

### Rust Definitions

See source files:
- `web/src/types/requests.rs`
- `web/src/types/responses.rs`

---

## 7. Error Handling

### HTTP Status Codes

| Code | Meaning | When Returned |
|------|---------|---------------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid move, game over, not your turn |
| 403 | Forbidden | Game other than the currently configured one |
| 404 | Not Found | Unknown game ID |
| 500 | Internal Server Error | Server-side failure |

### Error Response Format

Errors are returned as plain text:

```http
HTTP/1.1 400 Bad Request
Content-Type: text/plain

Illegal move: position/column 9 is not valid
```

### Common Error Messages

| Endpoint | Error | Cause |
|----------|-------|-------|
| POST /move | `Game is already over` | Move attempted after game ended |
| POST /move | `Not your turn` | Move when it's bot's turn |
| POST /move | `Illegal move: position/column X is not valid` | Invalid or occupied position |
| POST /game/new | `Cannot switch to game 'X': only the current game 'Y' is available` | Non-current game |
| POST /game/new | `Failed to create game 'X': ...` | Unknown game ID |
| GET /game-info/:id | `Cannot access game 'X': only the current game 'Y' is available` | Non-current game |
| GET /game-info/:id | `Game not found: X` | Unknown game ID |

### Client Error Handling

```typescript
async function makeMove(position: number): Promise<MoveResponse> {
  const res = await fetch('/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position }),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || 'Move failed');
  }

  return res.json();
}
```

---

## 8. Game-Specific Behavior

### TicTacToe

| Property | Value |
|----------|-------|
| Board Size | 3x3 (9 cells) |
| Position Range | 0-8 |
| Board Type | `grid` |
| Win Condition | 3 in a row (horizontal, vertical, diagonal) |

**Board Mapping**

```
Position → Row, Col
0 → (0,0)  1 → (0,1)  2 → (0,2)
3 → (1,0)  4 → (1,1)  5 → (1,2)
6 → (2,0)  7 → (2,1)  8 → (2,2)
```

### Connect4

| Property | Value |
|----------|-------|
| Board Size | 7x6 (42 cells) |
| Position Range | 0-6 (columns) |
| Board Type | `drop_column` |
| Win Condition | 4 in a row (horizontal, vertical, diagonal) |

**Column Mapping**

```
Position = Column number (0-6)
Piece drops to lowest empty cell in column
```

**Board Array Layout**

```
Index = row * 7 + col
Row 0 = bottom, Row 5 = top
```

### Othello

| Property | Value |
|----------|-------|
| Board Size | 8x8 (64 cells) |
| Position Range | 0-63 (cells), 64 = pass |
| Board Type | `grid` |
| Win Condition | Most discs when neither player can move |

**Board Mapping**

```
Position = row * 8 + col
Action 64 is the pass move (only legal when no placement is possible)
```

### Winner Values

| Value | Meaning |
|-------|---------|
| 0 | Game ongoing |
| 1 | Player 1 (X/Red) wins |
| 2 | Player 2 (O/Yellow) wins |
| 3 | Draw |

---

## 9. WebSocket Support

**Not currently implemented.**

The API uses polling for real-time updates:

```typescript
// Poll stats every 5 seconds
setInterval(async () => {
  const stats = await getStats();
  updateUI(stats);
}, 5000);
```

Future versions may add WebSocket support for:
- Real-time training progress
- Live game spectating
- Multi-player games

---

## 10. Rate Limiting

**No rate limiting implemented.**

The server handles requests as fast as possible. For production deployments, consider adding rate limiting via:

- Reverse proxy (nginx, Traefik)
- Axum tower middleware
- API gateway

---

## 11. Examples

### Complete Game Flow

```bash
# 1. Check available games
curl http://localhost:8080/games
# Response: {"games":["tictactoe","connect4"]}

# 2. Get game info
curl http://localhost:8080/game-info/tictactoe
# Response: {"env_id":"tictactoe","display_name":"Tic-Tac-Toe",...}

# 3. Start new game (player first)
curl -X POST http://localhost:8080/game/new \
  -H "Content-Type: application/json" \
  -d '{"first": "player", "game": "tictactoe"}'
# Response: {"board":[0,0,0,0,0,0,0,0,0],"current_player":1,...}

# 4. Make moves until game ends
curl -X POST http://localhost:8080/move \
  -H "Content-Type: application/json" \
  -d '{"position": 4}'
# Response: {"board":[0,0,0,0,1,2,0,0,0],"bot_move":5,...}

curl -X POST http://localhost:8080/move \
  -H "Content-Type: application/json" \
  -d '{"position": 0}'
# Response: {"board":[1,0,2,0,1,2,0,0,0],"bot_move":2,...}

# 5. Continue until winner != 0 or game_over == true
```

### Training Monitoring

```bash
# Check if model is loaded
curl http://localhost:8080/model
# Response: {"loaded":true,"training_step":1500,...}

# Get training stats
curl http://localhost:8080/stats
# Response: {"step":1500,"total_loss":0.234,...}

# Monitor training progress
watch -n 5 'curl -s http://localhost:8080/stats | jq .step'
```

### JavaScript Client

```javascript
class CartridgeClient {
  constructor(baseUrl = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
  }

  async getGames() {
    const res = await fetch(`${this.baseUrl}/games`);
    const data = await res.json();
    return data.games;
  }

  async newGame(first = 'player', game = 'tictactoe') {
    const res = await fetch(`${this.baseUrl}/game/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ first, game }),
    });
    return res.json();
  }

  async move(position) {
    const res = await fetch(`${this.baseUrl}/move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position }),
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }
    return res.json();
  }

  async getState() {
    const res = await fetch(`${this.baseUrl}/game/state`);
    return res.json();
  }

  async getStats() {
    const res = await fetch(`${this.baseUrl}/stats`);
    return res.json();
  }

  async getModelInfo() {
    const res = await fetch(`${this.baseUrl}/model`);
    return res.json();
  }
}

// Usage
const client = new CartridgeClient();

async function playGame() {
  await client.newGame('player', 'tictactoe');

  let state = await client.getState();
  while (!state.game_over) {
    // Pick first legal move
    const move = state.legal_moves[0];
    const result = await client.move(move);
    state = result;
    console.log(`Moved to ${move}, bot moved to ${result.bot_move}`);
  }

  console.log(`Game over! Winner: ${state.winner}`);
}
```

### Python Client

```python
import requests

class CartridgeClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

    def get_games(self):
        return requests.get(f"{self.base_url}/games").json()["games"]

    def new_game(self, first="player", game="tictactoe"):
        return requests.post(
            f"{self.base_url}/game/new",
            json={"first": first, "game": game}
        ).json()

    def move(self, position):
        res = requests.post(
            f"{self.base_url}/move",
            json={"position": position}
        )
        res.raise_for_status()
        return res.json()

    def get_state(self):
        return requests.get(f"{self.base_url}/game/state").json()

    def get_stats(self):
        return requests.get(f"{self.base_url}/stats").json()

    def get_model_info(self):
        return requests.get(f"{self.base_url}/model").json()

# Usage
client = CartridgeClient()
client.new_game(first="player", game="connect4")

state = client.get_state()
while not state["game_over"]:
    move = state["legal_moves"][0]  # Pick first legal move
    result = client.move(move)
    print(f"Moved to {move}, bot moved to {result.get('bot_move')}")
    state = result

print(f"Game over! Winner: {state['winner']}")
```

### cURL Cheatsheet

```bash
# Health check
curl http://localhost:8080/health

# List games
curl http://localhost:8080/games

# Get game metadata
curl http://localhost:8080/game-info/tictactoe
curl http://localhost:8080/game-info/connect4

# New game (player first)
curl -X POST http://localhost:8080/game/new \
  -H "Content-Type: application/json" \
  -d '{"first": "player"}'

# New game (bot first, specific game)
curl -X POST http://localhost:8080/game/new \
  -H "Content-Type: application/json" \
  -d '{"first": "bot", "game": "connect4"}'

# Get current state
curl http://localhost:8080/game/state

# Make a move
curl -X POST http://localhost:8080/move \
  -H "Content-Type: application/json" \
  -d '{"position": 4}'

# Get training stats
curl http://localhost:8080/stats

# Get model info
curl http://localhost:8080/model

# Pretty print with jq
curl -s http://localhost:8080/stats | jq .
curl -s http://localhost:8080/game/state | jq '.board | . as $b | [range(0;9)] | map($b[.])'
```

---

## Appendix: Quick Reference

### Endpoint Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/games` | List available games (current game only) |
| GET | `/game-info/:id` | Get game metadata |
| GET | `/game/state` | Get current game state |
| POST | `/game/new` | Start new game |
| POST | `/move` | Make a move |
| GET | `/stats` | Get training statistics |
| GET | `/actor-stats` | Get actor self-play statistics |
| GET | `/model` | Get model info |

### Status Code Quick Reference

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid move, etc.) |
| 404 | Not found (unknown game) |
| 500 | Server error |

### Content Types

| Type | Usage |
|------|-------|
| `application/json` | Request/response bodies |
| `text/plain` | Error messages |
