- [x] Make it where you can only see the current game on the web UI - you shouldnt be able to see others since you can't play against them properly if another model is trained - right?
      (Done: `/games` now returns only the configured game, and other game IDs return 403 — see `web/src/handlers/game.rs`.)
