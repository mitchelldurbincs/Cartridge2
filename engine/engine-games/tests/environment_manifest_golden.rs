//! Drift guard for the checked-in environment/algorithm contract catalog.

use engine_games::manifest::{manifest_json, REGENERATE_COMMAND};

const COMMITTED_MANIFEST: &str =
    include_str!("../../../trainer/src/trainer/environment_manifest.json");

#[test]
fn committed_manifest_matches_engine() {
    let generated = manifest_json();
    if generated == COMMITTED_MANIFEST {
        return;
    }

    let difference = generated
        .lines()
        .zip(COMMITTED_MANIFEST.lines())
        .enumerate()
        .find(|(_, (generated, committed))| generated != committed)
        .map(|(index, (generated, committed))| {
            format!(
                "first difference at line {}:\n  engine: {generated}\n  file:   {committed}",
                index + 1
            )
        })
        .unwrap_or_else(|| {
            format!(
                "common prefix but different lengths ({} vs {} lines)",
                generated.lines().count(),
                COMMITTED_MANIFEST.lines().count()
            )
        });

    panic!(
        "trainer/src/trainer/environment_manifest.json is stale.\n\
         Regenerate with `{REGENERATE_COMMAND}`.\n\n{difference}"
    );
}
