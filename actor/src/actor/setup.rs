use algorithm_core::{resolve_algorithm, AlgorithmDescriptor, RuntimeProfile};
use anyhow::{anyhow, Result};
use engine_core::{ActionSpace, AgentId, Capabilities, EngineContext, ObservationEncoding};
use mcts::MctsConfig;
#[cfg(feature = "s3")]
use model_watcher::s3::{S3Config, S3ModelWatcher};
use model_watcher::{ModelInfo, ModelLoadSpec, ModelSelection, ModelWatcher};
use std::sync::Arc;
use tracing::{debug, info};

use crate::algorithms::AlphaZeroCollectorConfig;
use crate::config::Config;
use crate::mcts_policy::MctsPolicy;
use crate::storage::{
    create_replay_store, ReplayProfile, ReplaySelection, ReplayStore, StorageConfig,
};

pub(super) struct CollectorDependencies {
    pub engine: EngineContext,
    pub mcts_policy: MctsPolicy,
    pub replay: Arc<dyn ReplayStore>,
    pub replay_selection: ReplaySelection,
    pub obs_size: usize,
    pub num_actions: usize,
}

struct EnvironmentSetup {
    engine: EngineContext,
    capabilities: Capabilities,
    algorithm: &'static AlgorithmDescriptor,
    runtime_profile: RuntimeProfile,
    max_horizon: u32,
    obs_size: usize,
    num_actions: usize,
}

enum RuntimeModelWatcher {
    Filesystem(ModelWatcher),
    #[cfg(feature = "s3")]
    S3(S3ModelWatcher),
}

impl RuntimeModelWatcher {
    async fn try_load_existing(&self) -> Result<bool> {
        match self {
            Self::Filesystem(watcher) => watcher.try_load_existing(),
            #[cfg(feature = "s3")]
            Self::S3(watcher) => watcher.try_load_existing().await,
        }
    }

    fn model_info(&self) -> Arc<std::sync::RwLock<ModelInfo>> {
        match self {
            Self::Filesystem(watcher) => watcher.model_info(),
            #[cfg(feature = "s3")]
            Self::S3(watcher) => watcher.model_info(),
        }
    }
}

pub(super) async fn build(
    config: &Config,
    cartridge_config: &AlphaZeroCollectorConfig,
) -> Result<CollectorDependencies> {
    config.validate()?;
    cartridge_config.validate()?;
    let environment = prepare_environment(config, cartridge_config)?;
    let mcts_policy = build_mcts_policy(config, cartridge_config, &environment)?;
    pin_source_model(config, cartridge_config, &environment, &mcts_policy).await?;
    let (replay, replay_selection) = open_replay(config, &environment).await?;
    Ok(CollectorDependencies {
        engine: environment.engine,
        mcts_policy,
        replay,
        replay_selection,
        obs_size: environment.obs_size,
        num_actions: environment.num_actions,
    })
}

fn prepare_environment(
    config: &Config,
    cartridge_config: &AlphaZeroCollectorConfig,
) -> Result<EnvironmentSetup> {
    engine_games::register_all_environments();
    let engine = EngineContext::new(&config.env_id)
        .map_err(|error| anyhow!("Environment '{}' is unavailable: {error}", config.env_id))?;
    let algorithm = resolve_algorithm(&config.algorithm_id)?;
    let compatibility = algorithm.compatibility(&engine);
    compatibility.require_compatible()?;
    let algorithm = algorithm.descriptor();
    info!(
        algorithm = algorithm.id,
        env_id = %config.env_id,
        model_contract = algorithm.components.model_contract,
        experience_schema = algorithm.components.experience_schema,
        "Algorithm compatibility validated"
    );
    debug!(
        algorithm = algorithm.id,
        assumptions = ?compatibility.unverified_assumptions,
        "Environment semantics not yet machine-verifiable"
    );
    let capabilities = engine.capabilities();
    let max_horizon = require_max_horizon(&capabilities)?;
    require_reachable_temperature_threshold(cartridge_config.temp_threshold, max_horizon)?;
    engine.metadata().require_board()?;
    let num_actions = discrete_action_count(&capabilities)?;
    let obs_size = observation_size(&capabilities)?;
    let runtime_profile = RuntimeProfile::new(
        algorithm.id,
        config.env_id.clone(),
        capabilities.contract_version,
    )?;
    info!(
        actor_id = %config.actor_id,
        env_id = %capabilities.id.env_id,
        max_horizon,
        preferred_batch = capabilities.preferred_batch,
        "Actor environment initialized"
    );
    Ok(EnvironmentSetup {
        engine,
        capabilities,
        algorithm,
        runtime_profile,
        max_horizon,
        obs_size,
        num_actions,
    })
}

fn build_mcts_policy(
    config: &Config,
    cartridge_config: &AlphaZeroCollectorConfig,
    environment: &EnvironmentSetup,
) -> Result<MctsPolicy> {
    let eval_batch_size = usize::try_from(cartridge_config.eval_batch_size)
        .map_err(|_| anyhow!("eval_batch_size does not fit this platform's usize"))?;
    let mut mcts_config = MctsConfig::for_training()
        .with_simulations(cartridge_config.num_simulations)
        .with_eval_batch_size(eval_batch_size)
        .with_c_puct(cartridge_config.c_puct)
        .with_temperature(cartridge_config.temperature);
    mcts_config.dirichlet_alpha = cartridge_config.dirichlet_alpha;
    mcts_config.dirichlet_epsilon = cartridge_config.dirichlet_weight;
    mcts_config
        .validate()
        .map_err(|error| anyhow!("invalid authenticated MCTS configuration: {error}"))?;
    info!(
        num_simulations = cartridge_config.num_simulations,
        c_puct = cartridge_config.c_puct,
        temperature = cartridge_config.temperature,
        late_temperature = cartridge_config.late_temperature,
        temp_threshold = cartridge_config.temp_threshold,
        dirichlet_alpha = cartridge_config.dirichlet_alpha,
        dirichlet_weight = cartridge_config.dirichlet_weight,
        eval_batch_size = cartridge_config.eval_batch_size,
        virtual_loss = mcts_config.virtual_loss,
        "Authenticated collector MCTS configuration"
    );
    Ok(MctsPolicy::new(
        config.env_id.clone(),
        environment.num_actions,
        environment.obs_size,
    )
    .with_config(mcts_config)
    .with_temp_schedule(
        cartridge_config.temp_threshold,
        cartridge_config.late_temperature,
    ))
}

async fn pin_source_model(
    config: &Config,
    cartridge_config: &AlphaZeroCollectorConfig,
    environment: &EnvironmentSetup,
    mcts_policy: &MctsPolicy,
) -> Result<()> {
    let onnx_intra_threads = usize::try_from(cartridge_config.onnx_intra_threads)
        .map_err(|_| anyhow!("onnx_intra_threads does not fit this platform's usize"))?;
    let model_contract = environment.algorithm.model_artifact_contract(
        config.env_id.clone(),
        environment.capabilities.contract_version,
    );
    let model_spec = ModelLoadSpec::new(
        environment.obs_size,
        environment.num_actions,
        onnx_intra_threads,
        environment.max_horizon,
        model_contract,
    )?;
    let watcher = build_model_watcher(config, environment, model_spec, mcts_policy).await?;
    let loaded = watcher.try_load_existing().await?;
    let model_info = watcher
        .model_info()
        .read()
        .map_err(|error| anyhow!("failed to read loaded model identity: {error}"))?
        .clone();
    require_source_checkpoint(config.source_checkpoint_id.as_deref(), loaded, &model_info)?;
    if loaded {
        info!(
            checkpoint_id = model_info.checkpoint_id.as_deref().unwrap_or("unknown"),
            "Pinned required source model for this one-shot collection"
        );
    } else {
        info!("Confirmed root collection has no RunHead; using uniform evaluator");
    }
    Ok(())
}

async fn build_model_watcher(
    config: &Config,
    environment: &EnvironmentSetup,
    model_spec: ModelLoadSpec,
    mcts_policy: &MctsPolicy,
) -> Result<RuntimeModelWatcher> {
    match crate::config::central_config()
        .storage
        .model_backend
        .as_str()
    {
        "filesystem" => {
            let model_dir = environment.runtime_profile.model_dir(&config.data_dir);
            std::fs::create_dir_all(&model_dir)?;
            Ok(RuntimeModelWatcher::Filesystem(ModelWatcher::new(
                model_dir,
                model_spec,
                ModelSelection::Latest,
                mcts_policy.evaluator_ref(),
            )))
        }
        "s3" => build_s3_watcher(config, environment, model_spec, mcts_policy).await,
        backend => Err(anyhow!("unsupported model storage backend '{backend}'")),
    }
}

#[cfg(feature = "s3")]
async fn build_s3_watcher(
    config: &Config,
    environment: &EnvironmentSetup,
    model_spec: ModelLoadSpec,
    mcts_policy: &MctsPolicy,
) -> Result<RuntimeModelWatcher> {
    let storage = &crate::config::central_config().storage;
    let bucket = storage
        .s3_bucket
        .clone()
        .ok_or_else(|| anyhow!("storage.s3_bucket is required for S3 model watching"))?;
    let profile_dir = environment.runtime_profile.data_dir(&config.data_dir);
    Ok(RuntimeModelWatcher::S3(
        S3ModelWatcher::new(
            S3Config {
                bucket,
                prefix: environment.runtime_profile.model_prefix(),
                endpoint_url: storage.s3_endpoint.clone(),
                region: None,
                cache_dir: profile_dir.join("model-cache"),
            },
            model_spec,
            ModelSelection::Latest,
            mcts_policy.evaluator_ref(),
        )
        .await?,
    ))
}

#[cfg(not(feature = "s3"))]
async fn build_s3_watcher(
    _config: &Config,
    _environment: &EnvironmentSetup,
    _model_spec: ModelLoadSpec,
    _mcts_policy: &MctsPolicy,
) -> Result<RuntimeModelWatcher> {
    Err(anyhow!(
        "storage.model_backend is 's3' but the actor binary was built without the s3 feature"
    ))
}

async fn open_replay(
    config: &Config,
    environment: &EnvironmentSetup,
) -> Result<(Arc<dyn ReplayStore>, ReplaySelection)> {
    let replay_selection = ReplaySelection {
        profile: ReplayProfile {
            env_id: config.env_id.clone(),
            env_contract_version: environment.capabilities.contract_version,
            algorithm_id: environment.algorithm.id.to_string(),
            experience_schema: environment
                .algorithm
                .components
                .experience_schema
                .to_string(),
        },
        collection_scope_id: config.collection_scope_id.clone(),
        source_checkpoint_id: config.source_checkpoint_id.clone(),
    };
    let replay = create_replay_store(&StorageConfig {
        postgres_url: config.postgres_url.clone(),
        pool_config: config.pool_config(),
        selection: replay_selection.clone(),
    })
    .await?;
    info!(selection = ?replay_selection, "Opaque replay record store initialized (PostgreSQL)");
    Ok((Arc::from(replay), replay_selection))
}

fn discrete_action_count(capabilities: &Capabilities) -> Result<usize> {
    match capabilities.action_space(AgentId(1)) {
        Some(ActionSpace::Discrete { size }) => Ok(*size as usize),
        other => Err(anyhow!(
            "AlphaZero requires discrete actions, got {other:?}"
        )),
    }
}

fn observation_size(capabilities: &Capabilities) -> Result<usize> {
    match &capabilities.encoding.observation {
        ObservationEncoding::Tensor { spec } => spec
            .fixed_elements()
            .ok_or_else(|| anyhow!("AlphaZero requires a fixed observation tensor")),
        other => Err(anyhow!(
            "AlphaZero requires tensor observations, got {other:?}"
        )),
    }
}

fn require_max_horizon(capabilities: &Capabilities) -> Result<u32> {
    capabilities
        .max_horizon
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            anyhow!(
                "AlphaZero environment '{}' must declare a finite non-zero max_horizon",
                capabilities.id.env_id
            )
        })
}

pub(super) fn require_reachable_temperature_threshold(
    temp_threshold: u32,
    max_horizon: u32,
) -> Result<()> {
    if temp_threshold > 0 && temp_threshold >= max_horizon {
        return Err(anyhow!(
            "temp_threshold {temp_threshold} must be zero or less than environment max_horizon {max_horizon}; otherwise late_temperature is unreachable"
        ));
    }
    Ok(())
}

pub(super) fn require_source_checkpoint(
    expected_source_checkpoint_id: Option<&str>,
    loaded: bool,
    model_info: &ModelInfo,
) -> Result<()> {
    match expected_source_checkpoint_id {
        None if loaded || model_info.loaded || model_info.checkpoint_id.is_some() => Err(anyhow!(
            "root collection requires an absent RunHead, but checkpoint '{}' was loaded",
            model_info.checkpoint_id.as_deref().unwrap_or("unknown")
        )),
        None => Ok(()),
        Some(expected) if !loaded || !model_info.loaded => Err(anyhow!(
            "collection requires source checkpoint '{expected}', but no RunHead model was loaded"
        )),
        Some(expected) if model_info.checkpoint_id.as_deref() != Some(expected) => Err(anyhow!(
            "loaded RunHead checkpoint '{}' does not match required source checkpoint '{expected}'",
            model_info.checkpoint_id.as_deref().unwrap_or("unknown")
        )),
        Some(_) => Ok(()),
    }
}
