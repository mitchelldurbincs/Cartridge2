//! Runtime dispatch from an algorithm ID to its self-play collector.

mod alphazero_board_v1;

pub(crate) use alphazero_board_v1::encode_experience;

use crate::actor::AlphaZeroCollector;
use crate::config::Config;
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

/// Build the collector selected by configuration.
pub async fn build_collector(config: Config) -> Result<Box<dyn CollectorAlgorithm>> {
    match resolve_algorithm(&config.algorithm_id)? {
        BuiltinAlgorithm::AlphaZeroBoardV1 => Ok(Box::new(AlphaZeroCollector::new(config).await?)),
    }
}
