//! Generate the checked-in environment and algorithm contract catalog.

use std::collections::BTreeMap;

use algorithm_core::{algorithm_descriptors, compatibility_reports, AlgorithmDescriptor};
use engine_core::{list_registered_environments, Capabilities, EngineContext, EnvironmentMetadata};
use serde::Serialize;

use crate::register_all_environments;

/// Bumped only when this document's shape changes.
pub const MANIFEST_SCHEMA_VERSION: u32 = 5;
pub const REGENERATE_COMMAND: &str = "make environment-manifest";

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    generated_by: String,
    algorithms: Vec<&'static AlgorithmDescriptor>,
    environments: Vec<ManifestEnvironment>,
}

#[derive(Serialize)]
struct ManifestEnvironment {
    metadata: EnvironmentMetadata,
    capabilities: Capabilities,
    algorithm_profiles: BTreeMap<&'static str, algorithm_core::CompatibilityReport>,
}

pub fn manifest_json() -> String {
    register_all_environments();
    let mut env_ids = list_registered_environments();
    env_ids.sort();

    let environments = env_ids
        .iter()
        .map(|env_id| {
            let context = EngineContext::new(env_id).unwrap_or_else(|error| {
                panic!("environment '{env_id}' is not constructible: {error}")
            });
            let algorithm_profiles = compatibility_reports(&context)
                .into_iter()
                .map(|report| (report.algorithm_id, report))
                .collect();
            ManifestEnvironment {
                metadata: context.metadata(),
                capabilities: context.capabilities(),
                algorithm_profiles,
            }
        })
        .collect();

    let manifest = Manifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generated_by: REGENERATE_COMMAND.to_string(),
        algorithms: algorithm_descriptors(),
        environments,
    };
    let mut json =
        serde_json::to_string_pretty(&manifest).expect("environment manifest must be serializable");
    json.push('\n');
    json
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_deterministic_and_environment_shaped() {
        let rendered = manifest_json();
        assert_eq!(rendered, manifest_json());
        let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(value["schema_version"], 5);
        assert!(value.get("games").is_none());
        let environments = value["environments"].as_array().unwrap();
        let ids = environments
            .iter()
            .map(|environment| environment["metadata"]["id"].as_str().unwrap())
            .collect::<Vec<_>>();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn board_profile_is_nested_and_algorithm_compatibility_is_explicit() {
        let value: serde_json::Value = serde_json::from_str(&manifest_json()).unwrap();
        let mut saw_non_board = false;
        for environment in value["environments"].as_array().unwrap() {
            let metadata = &environment["metadata"];
            assert!(metadata.get("board_width").is_none());
            assert_eq!(environment["capabilities"]["id"]["env_id"], metadata["id"]);
            let profile = &environment["algorithm_profiles"]["alphazero_board_v1"];
            if profile["compatible"].as_bool().unwrap() {
                let board = &metadata["board"];
                assert!(board["width"].as_u64().unwrap() > 0);
                let observation = &environment["capabilities"]["encoding"]["observation"];
                assert_eq!(observation["kind"], "tensor");
                assert_eq!(observation["spec"]["dtype"], "f32_little_endian");
                assert_eq!(observation["spec"]["dimensions"][0]["name"], "channel");
            } else if metadata["board"].is_null() {
                saw_non_board = true;
            }
        }
        assert!(
            saw_non_board,
            "catalog must exercise a non-board environment"
        );
    }

    #[test]
    fn alphazero_compatible_agent_and_chance_contracts_are_machine_readable() {
        let value: serde_json::Value = serde_json::from_str(&manifest_json()).unwrap();
        for environment in value["environments"].as_array().unwrap() {
            if !environment["algorithm_profiles"]["alphazero_board_v1"]["compatible"]
                .as_bool()
                .unwrap()
            {
                continue;
            }
            let capabilities = &environment["capabilities"];
            assert_eq!(capabilities["agents"]["kind"], "fixed");
            assert_eq!(
                capabilities["agents"]["agents"].as_array().unwrap().len(),
                2
            );
            assert_eq!(capabilities["semantics"]["chance_model"], "none");
            assert_eq!(
                capabilities["semantics"]["reward_model"],
                "terminal_zero_sum"
            );
        }
    }
}
