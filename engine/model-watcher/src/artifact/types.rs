use algorithm_core::ModelArtifactContract;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub(crate) const CHECKPOINT_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub(crate) const RUN_HEAD_SCHEMA_VERSION: u32 = 2;
pub(crate) const RUN_COMMIT_SCHEMA_VERSION: u32 = 1;
pub(crate) const STATS_SNAPSHOT_SCHEMA_VERSION: u32 = 3;
pub(crate) const RUN_HEAD_CHANNEL: &str = "current";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtifactProfile {
    pub algorithm_id: String,
    pub env_id: String,
    pub env_contract_version: u32,
    pub model_artifact_schema_version: u32,
    pub model_contract: String,
}

impl From<&ModelArtifactContract> for ArtifactProfile {
    fn from(identity: &ModelArtifactContract) -> Self {
        Self {
            algorithm_id: identity.algorithm_id.clone(),
            env_id: identity.env_id.clone(),
            env_contract_version: identity.env_contract_version,
            model_artifact_schema_version: identity.schema_version,
            model_contract: identity.model_contract.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BlobReference {
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CheckpointManifestV1 {
    pub schema_version: u32,
    pub profile: ArtifactProfile,
    pub step: u64,
    pub parent_checkpoint_id: Option<String>,
    pub config_sha256: String,
    pub onnx: BlobReference,
    pub learner_state: BlobReference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunHeadV2 {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub run_commit_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChampionReferenceV1 {
    pub checkpoint_id: String,
    pub evaluation_id: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OrchestrationCommitV1 {
    pub iteration: u64,
    pub collection_scope_id: String,
    pub source_checkpoint_id: Option<String>,
    pub episodes_generated: u64,
    pub transitions_generated: u64,
    pub training_steps: u64,
    pub collector_simulations: u32,
    pub collector_seed: Option<u64>,
    pub evaluation_seed: Option<u64>,
    pub actor_time_seconds: f64,
    pub trainer_time_seconds: f64,
    pub eval_time_seconds: f64,
    pub total_time_seconds: f64,
    pub eval_win_rate: Option<f64>,
    pub eval_draw_rate: Option<f64>,
    pub timestamp: String,
    pub evaluation_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunRecipeV1 {
    pub schema_version: u32,
    pub learner_config_sha256: String,
    pub learner_recipe: Box<RawValue>,
    pub total_iterations: u64,
    pub episodes_per_iteration: u32,
    pub training_steps_per_iteration: u64,
    pub num_actors: u32,
    pub collector_episode_timeout_seconds: u64,
    pub collector_eval_batch_size: u32,
    pub collector_onnx_intra_threads: u32,
    pub mcts_start_simulations: u32,
    pub mcts_max_simulations: u32,
    pub mcts_simulation_ramp: u32,
    pub collector_c_puct: f64,
    pub collector_temperature: f64,
    pub collector_late_temperature: f64,
    pub temperature_move_threshold: u32,
    pub collector_dirichlet_alpha: f64,
    pub collector_dirichlet_weight: f64,
    pub collector_seed_strategy: String,
    pub evaluation_interval: u64,
    pub evaluation_games: u32,
    #[serde(rename = "evaluation_simulations")]
    pub _evaluation_simulations: u32,
    pub evaluation_temperature: f64,
    pub evaluation_win_threshold: f64,
    #[serde(rename = "evaluation_vs_random")]
    pub _evaluation_vs_random: bool,
    pub solver_games: u32,
    pub evaluation_seed: u64,
    pub replay_policy: String,
    pub promotion_metric: String,
    pub promotion_margin: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HistoryEntryV1 {
    pub step: u64,
    pub metrics: BTreeMap<String, f64>,
    pub learning_rate: f64,
    pub grad_norm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvaluationStatsV1 {
    pub step: u64,
    pub metrics: BTreeMap<String, f64>,
    pub episodes: u64,
    pub mean_episode_length: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainingStatsV3 {
    pub step: u64,
    pub total_steps: u64,
    pub metrics: BTreeMap<String, f64>,
    pub learning_rate: f64,
    pub samples_seen: u64,
    pub replay_record_count: u64,
    pub last_checkpoint: String,
    pub timestamp: f64,
    pub history: Vec<HistoryEntryV1>,
    pub env_id: String,
    pub last_evaluation: Option<EvaluationStatsV1>,
    pub evaluation_history: Vec<EvaluationStatsV1>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StatsSnapshotV3 {
    pub schema_version: u32,
    pub profile: ArtifactProfile,
    pub config_sha256: String,
    pub checkpoint_id: String,
    pub step: u64,
    pub stats: TrainingStatsV3,
}

#[derive(Debug, Clone)]
pub(crate) struct RunCommitV1 {
    pub profile: ArtifactProfile,
    pub config_sha256: String,
    pub parent_run_commit_id: Option<String>,
    pub checkpoint_id: String,
    pub run_recipe_id: Option<String>,
    pub run_recipe: Option<RunRecipeV1>,
    pub stats_snapshot: StatsSnapshotV3,
    pub champion: Option<ChampionReferenceV1>,
    pub evaluation_head_id: Option<String>,
    pub orchestration: Option<OrchestrationCommitV1>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedRunCommit {
    pub run_commit_id: String,
    pub commit: RunCommitV1,
    pub manifest: CheckpointManifestV1,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedCheckpoint {
    pub checkpoint_id: String,
    pub run_commit_id: String,
    pub manifest: CheckpointManifestV1,
    pub model_path: PathBuf,
}
