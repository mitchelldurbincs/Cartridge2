# Cartridge2 Frontend

Svelte 5 + TypeScript single-page application for playing against trained AlphaZero models and viewing training statistics.

## Quick Start

```bash
# Install dependencies
npm --prefix web/frontend install

# Start dev server (proxies API to localhost:8080)
npm --prefix web/frontend run dev
# Open http://localhost:5173

# Type-check
npm --prefix web/frontend run check

# Production build
npm --prefix web/frontend run build
```

The Rust backend must be running on port 8080 for the app to work. In development, Vite proxies all API requests automatically (configured in `vite.config.ts`).

## Architecture

```
src/
├── main.ts                 # Entry point, hash-based SPA router
├── App.svelte              # Main page: game selector, board, controls
├── GenericBoard.svelte     # grid, drop-column, and Generals renderers
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
- **GenericBoard.svelte** - Renders the installed presentation types: `grid`
  (TicTacToe, Othello), `drop_column` (Connect 4), and `generals`. It adapts
  layout from `GameInfo` supplied by the serving host.
- **Stats.svelte** - Polls the stats and model endpoints every 5 seconds and
  displays losses, replay-record count, content identities, basic evaluation
  results, and win rate by training step.
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

### Adding a New Game Visualization

Environment registration alone does not make a game web-playable. The selected
algorithm must declare a serving component, the environment manifest's optional
`metadata.board` profile must be present, runtime state must expose a board
presentation, and the frontend must support the profile's renderer. The backend
maps that nested engine profile to the flat `GameInfo` HTTP DTO with dimensions,
symbols, and `board_type`.

For games that need a new board type beyond `grid` and `drop_column`, add a new rendering branch in `GenericBoard.svelte`.

## Tech Stack

- **Svelte 5** with runes (`$state`, `$derived`)
- **TypeScript** for type safety
- **Vite 6** for dev server and bundling
- Dark-mode UI with responsive layout
