# Cartridge2 Frontend

Svelte 5 + TypeScript single-page application for playing against trained AlphaZero models and viewing training statistics.

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server (proxies API to localhost:8080)
npm run dev
# Open http://localhost:5173

# Type-check
npm run check

# Production build
npm run build
```

The Rust backend must be running on port 8080 for the app to work. In development, Vite proxies all API requests automatically (configured in `vite.config.ts`).

## Architecture

```
src/
├── main.ts                 # Entry point, hash-based SPA router
├── App.svelte              # Main page: game selector, board, controls
├── GenericBoard.svelte     # Renders any game board (grid or drop-column)
├── Stats.svelte            # Training stats display (polls /stats)
├── LossChart.svelte        # Canvas-based loss visualization chart
├── LossOverTimePage.svelte # Full-page training progress view
└── lib/
    ├── api.ts              # Typed API client (all backend calls)
    ├── chart.ts            # Chart formatting utilities
    └── constants.ts        # Polling intervals, thresholds
```

### Routing

Hash-based SPA routing in `main.ts`:

| Route | Component | Description |
|-------|-----------|-------------|
| `#/` (default) | `App` | Play games and view stats |
| `#/loss-over-time` | `LossOverTimePage` | Training loss charts |

### Key Components

- **App.svelte** - Main game UI. On mount, checks server health, loads available games, auto-selects the game currently being trained. Handles game creation, move submission, and pass actions (for Othello).
- **GenericBoard.svelte** - Renders boards for all game types. Supports two board types: `grid` (TicTacToe, Othello) and `drop_column` (Connect 4). Adapts layout based on `GameInfo` metadata from the backend.
- **Stats.svelte** - Polls `/stats` every 5 seconds, displays training progress (loss, learning rate, eval win rates).
- **LossChart.svelte** - Renders training loss history on a canvas element.

### API Client

`lib/api.ts` provides typed functions for all backend endpoints:

| Function | Endpoint | Description |
|----------|----------|-------------|
| `getHealth()` | `GET /health` | Server health check |
| `getGames()` | `GET /games` | List available games |
| `getGameInfo(id)` | `GET /game-info/:id` | Game metadata (board size, actions) |
| `newGame(first, game)` | `POST /game/new` | Start a new game |
| `getGameState()` | `GET /game/state` | Current board state |
| `makeMove(position)` | `POST /move` | Submit a move |
| `getStats()` | `GET /stats` | Training statistics |
| `getModelInfo()` | `GET /model` | Loaded model info |
| `getActorStats()` | `GET /actor-stats` | Self-play statistics |

### Adding a New Game Visualization

The frontend automatically supports new games if they follow the existing metadata pattern. The backend provides `GameInfo` with `board_width`, `board_height`, `board_type`, `player_symbols`, etc., and `GenericBoard` renders accordingly.

For games that need a new board type beyond `grid` and `drop_column`, add a new rendering branch in `GenericBoard.svelte`.

## Tech Stack

- **Svelte 5** with runes (`$state`, `$derived`)
- **TypeScript** for type safety
- **Vite 6** for dev server and bundling
- Dark-mode UI with responsive layout
