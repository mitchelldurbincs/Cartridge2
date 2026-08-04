//! Generate the environment/algorithm manifest consumed by the Python runtime.

use std::io::Write;
use std::path::PathBuf;

use engine_games::manifest::manifest_json;

fn default_output_path() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../trainer/src/trainer/environment_manifest.json"
    ))
}

fn main() {
    let json = manifest_json();
    match std::env::args().nth(1) {
        Some(argument) if argument == "-" => std::io::stdout()
            .write_all(json.as_bytes())
            .expect("write manifest to stdout"),
        output => {
            let path = output
                .map(PathBuf::from)
                .unwrap_or_else(default_output_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .unwrap_or_else(|error| panic!("creating {}: {error}", parent.display()));
            }
            std::fs::write(&path, json)
                .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
            eprintln!("wrote {}", path.display());
        }
    }
}
