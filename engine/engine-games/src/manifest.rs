//! Game-metadata manifest generation.
//!
//! The engine is the single source of truth for game facts (board dimensions,
//! action count, observation layout). The Python trainer needs the same facts,
//! but CI runs Rust and Python as separate jobs with no artifact passing and no
//! Rust toolchain on the Python side, so it cannot read them live.
//!
//! Instead this module renders every registered game's [`GameMetadata`] to a
//! JSON manifest that is checked in at
//! `trainer/src/trainer/game_metadata.json` and shipped inside the Python
//! package. `tests/manifest_golden.rs` fails if the checked-in copy drifts from
//! what the engine would generate, which turns "the trainer and the engine
//! disagree about obs_size" from a silent mistraining bug into a test failure.
//!
//! Regenerate with `make game-manifest`.

use engine_core::{create_game, list_registered_games, GameMetadata};
use serde::Serialize;

use crate::register_all_games;

/// Bumped when the manifest's own shape changes (not when a game changes).
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// The command that regenerates the manifest, embedded so anyone who opens the
/// file knows not to hand-edit it.
pub const REGENERATE_COMMAND: &str = "make game-manifest";

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    generated_by: String,
    games: Vec<GameMetadata>,
}

/// Render the manifest for every registered game.
///
/// Games are sorted by `env_id` — the registry is a `HashMap`, so iteration
/// order is otherwise unstable and the file would churn between runs. Sorted
/// and pretty-printed also keeps a two-branch merge conflict to a single
/// contiguous hunk.
pub fn manifest_json() -> String {
    register_all_games();

    let mut env_ids = list_registered_games();
    env_ids.sort();

    let games: Vec<GameMetadata> = env_ids
        .iter()
        .map(|env_id| {
            create_game(env_id)
                .unwrap_or_else(|| panic!("game '{env_id}' is registered but not constructible"))
                .metadata()
        })
        .collect();

    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generated_by: REGENERATE_COMMAND.to_string(),
        games,
    };

    let mut json = serde_json::to_string_pretty(&manifest).expect("GameMetadata is serializable");
    json.push('\n');
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_deterministic() {
        assert_eq!(manifest_json(), manifest_json());
    }

    #[test]
    fn manifest_lists_games_sorted_by_env_id() {
        let json: serde_json::Value = serde_json::from_str(&manifest_json()).unwrap();
        let ids: Vec<&str> = json["games"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["env_id"].as_str().unwrap())
            .collect();

        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted, "games must be sorted for stable diffs");
        assert!(ids.contains(&"generals_8x8"));
    }

    #[test]
    fn manifest_carries_the_obs_encoding_fields() {
        // These two are the whole reason the manifest exists: they are engine
        // facts the trainer previously hardcoded.
        let json: serde_json::Value = serde_json::from_str(&manifest_json()).unwrap();
        let generals = json["games"]
            .as_array()
            .unwrap()
            .iter()
            .find(|g| g["env_id"] == "generals_8x8")
            .expect("generals_8x8 in manifest");

        assert_eq!(generals["obs_channels"], 9);
        assert_eq!(generals["player_relative_obs"], true);
        assert_eq!(generals["obs_size"], 835);
        assert_eq!(generals["legal_mask_offset"], 576);
    }
}
