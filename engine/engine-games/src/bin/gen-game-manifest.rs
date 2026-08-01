//! Regenerate the game-metadata manifest consumed by the Python trainer.
//!
//! Run via `make game-manifest`, or directly:
//!
//! ```text
//! cargo run --manifest-path engine/Cargo.toml --bin gen-game-manifest
//! cargo run --manifest-path engine/Cargo.toml --bin gen-game-manifest -- /tmp/out.json
//! ```
//!
//! With no argument it writes the checked-in manifest inside the Python
//! package. Pass a path to write elsewhere, or `-` for stdout.

use std::io::Write;
use std::path::PathBuf;

use engine_games::manifest::manifest_json;

/// The checked-in manifest, relative to this crate.
///
/// It lives inside the Python package (rather than at the repo root) so the
/// trainer can load it with `importlib.resources` — no path arithmetic, and it
/// ships in the wheel via `[tool.setuptools.package-data]`.
fn default_output_path() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../trainer/src/trainer/game_metadata.json"
    ))
}

fn main() {
    let json = manifest_json();

    match std::env::args().nth(1) {
        Some(arg) if arg == "-" => {
            std::io::stdout()
                .write_all(json.as_bytes())
                .expect("write to stdout");
        }
        other => {
            let path = other.map(PathBuf::from).unwrap_or_else(default_output_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|e| panic!("creating {}: {e}", parent.display()));
            }
            std::fs::write(&path, &json)
                .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
            eprintln!("wrote {}", path.display());
        }
    }
}
