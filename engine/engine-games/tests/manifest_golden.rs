//! Golden-file test: the checked-in game manifest must match what the engine
//! generates right now.
//!
//! This is the drift guard for the whole metadata single-source scheme. The
//! Python trainer reads game facts from `trainer/src/trainer/game_metadata.json`
//! instead of hardcoding them, so that file has to stay current with the game
//! crates. A live Rust-to-Python comparison cannot run in CI (the Python job
//! installs no Rust toolchain, and no artifacts pass between jobs), so instead
//! the committed file is compared here — inside the existing `rust-test` job,
//! and locally on every `cargo test`.
//!
//! On failure: run `make game-manifest`, or `UPDATE_GAME_MANIFEST=1 cargo test`.

use engine_games::manifest::{manifest_json, REGENERATE_COMMAND};

/// Compile-time embed, matching how the repo already shares `config.defaults.toml`
/// and `sql/schema.sql` across languages. A missing manifest is a compile error
/// rather than a confusing runtime skip.
const COMMITTED_MANIFEST: &str = include_str!("../../../trainer/src/trainer/game_metadata.json");

fn manifest_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../trainer/src/trainer/game_metadata.json"
    ))
}

#[test]
fn committed_manifest_matches_engine() {
    let generated = manifest_json();

    if generated == COMMITTED_MANIFEST {
        return;
    }

    if std::env::var_os("UPDATE_GAME_MANIFEST").is_some() {
        std::fs::write(manifest_path(), &generated).expect("rewriting manifest");
        panic!("game manifest was stale and has been rewritten; re-run the tests");
    }

    // Point at the first differing line: the whole file is too long to eyeball.
    let diff = generated
        .lines()
        .zip(COMMITTED_MANIFEST.lines())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| {
            format!(
                "first difference at line {}:\n  engine: {a}\n  file:   {b}",
                i + 1
            )
        })
        .unwrap_or_else(|| {
            format!(
                "files share a common prefix but differ in length ({} vs {} lines)",
                generated.lines().count(),
                COMMITTED_MANIFEST.lines().count(),
            )
        });

    panic!(
        "trainer/src/trainer/game_metadata.json is out of date with the game crates.\n\
         The Python trainer reads its game facts from that file, so leaving it stale \
         would silently mistrain.\n\
         Regenerate with `{REGENERATE_COMMAND}` (or UPDATE_GAME_MANIFEST=1 cargo test).\n\n\
         {diff}"
    );
}
