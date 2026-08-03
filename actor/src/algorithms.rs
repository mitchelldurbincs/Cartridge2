//! Runtime dispatch from an algorithm ID to its self-play collector.

mod alphazero_board_v1;
mod dqn_v1;

#[cfg(test)]
pub(crate) use alphazero_board_v1::COLLECTOR_CONFIG_SCHEMA_VERSION;
pub(crate) use alphazero_board_v1::{encode_experience, AlphaZeroCollectorConfig};
pub(crate) use dqn_v1::{
    encode_transition as encode_dqn_transition, DqnCollectorConfig, DqnTransition,
};

use crate::actor::AlphaZeroCollector;
use crate::config::Config;
use crate::dqn_actor::DqnCollector;
use algorithm_core::{resolve_algorithm, BuiltinAlgorithm};
use anyhow::Result;
use async_trait::async_trait;

/// Collector surface owned by an algorithm cartridge.
#[async_trait]
pub trait CollectorAlgorithm: Send + Sync {
    async fn run(&self) -> Result<()>;
    fn shutdown(&self);
}

#[async_trait]
impl CollectorAlgorithm for AlphaZeroCollector {
    async fn run(&self) -> Result<()> {
        AlphaZeroCollector::run(self).await
    }

    fn shutdown(&self) {
        AlphaZeroCollector::shutdown(self);
    }
}

#[async_trait]
impl CollectorAlgorithm for DqnCollector {
    async fn run(&self) -> Result<()> {
        DqnCollector::run(self).await
    }

    fn shutdown(&self) {
        DqnCollector::shutdown(self);
    }
}

/// Build the collector selected by configuration.
pub async fn build_collector(config: Config) -> Result<Box<dyn CollectorAlgorithm>> {
    match resolve_algorithm(&config.algorithm_id)? {
        BuiltinAlgorithm::AlphaZeroBoardV1 => {
            let cartridge_config = AlphaZeroCollectorConfig::parse(&config.collector_config)?;
            Ok(Box::new(
                AlphaZeroCollector::new(config, cartridge_config).await?,
            ))
        }
        BuiltinAlgorithm::DqnV1 => {
            let cartridge_config = DqnCollectorConfig::parse(&config.collector_config)?;
            Ok(Box::new(DqnCollector::new(config, cartridge_config).await?))
        }
    }
}
