//! Process-local registry of validated typed environments.

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::adapter::EnvironmentAdapter;
use crate::board_game::{BoardGame, BoardGameEnvironment};
use crate::contract;
use crate::erased::{ErasedEnvironment, ErasedEnvironmentError};
use crate::metadata::EnvironmentMetadata;
use crate::typed::{Capabilities, EngineId, Environment};

type EnvironmentFactory = fn() -> Result<Box<dyn ErasedEnvironment>, ErasedEnvironmentError>;

#[derive(Clone)]
struct RegisteredEnvironment {
    factory: EnvironmentFactory,
    id: EngineId,
    capabilities: Capabilities,
    metadata: EnvironmentMetadata,
}

static REGISTRY: Lazy<Mutex<HashMap<String, RegisteredEnvironment>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RegistryError {
    #[error("environment '{env_id}' is already registered")]
    AlreadyRegistered { env_id: String },
    #[error("environment '{env_id}' is not registered")]
    NotRegistered { env_id: String },
    #[error("invalid environment factory: {message}")]
    InvalidEnvironment { message: String },
    #[error(
        "environment factory for '{env_id}' produced descriptors that differ from registration"
    )]
    FactoryDescriptorsChanged { env_id: String },
}

fn typed_factory<E>() -> Result<Box<dyn ErasedEnvironment>, ErasedEnvironmentError>
where
    E: Environment + Default,
{
    Ok(Box::new(EnvironmentAdapter::try_new(E::default())?))
}

fn board_game_factory<G>() -> Result<Box<dyn ErasedEnvironment>, ErasedEnvironmentError>
where
    G: BoardGame + Default,
{
    let environment = BoardGameEnvironment::new(G::default());
    if environment.metadata().board.is_none() {
        return Err(ErasedEnvironmentError::ContractViolation(
            "BoardGame implementation must publish board metadata".to_string(),
        ));
    }
    Ok(Box::new(EnvironmentAdapter::try_new(environment)?))
}

fn validate_factory(
    factory: EnvironmentFactory,
) -> Result<(RegisteredEnvironment, Box<dyn ErasedEnvironment>), RegistryError> {
    let environment = factory().map_err(|error| RegistryError::InvalidEnvironment {
        message: error.to_string(),
    })?;
    let id = environment.engine_id();
    let capabilities = environment.capabilities();
    let metadata = environment.metadata();
    contract::validate_descriptors(&id, &capabilities, &metadata).map_err(|error| {
        RegistryError::InvalidEnvironment {
            message: error.to_string(),
        }
    })?;
    Ok((
        RegisteredEnvironment {
            factory,
            id,
            capabilities,
            metadata,
        },
        environment,
    ))
}

fn register_factory(factory: EnvironmentFactory) -> Result<(), RegistryError> {
    let (registered, _) = validate_factory(factory)?;
    let env_id = registered.id.env_id.clone();
    let mut registry = REGISTRY.lock().unwrap();
    if registry.contains_key(&env_id) {
        return Err(RegistryError::AlreadyRegistered { env_id });
    }
    registry.insert(env_id, registered);
    Ok(())
}

/// Register a typed environment. Its own validated descriptor is its registry key.
pub fn register_environment<E>() -> Result<(), RegistryError>
where
    E: Environment + Default,
{
    register_factory(typed_factory::<E>)
}

/// Register an implementation of the explicitly narrow board-game profile.
pub fn register_board_game<G>() -> Result<(), RegistryError>
where
    G: BoardGame + Default,
{
    register_factory(board_game_factory::<G>)
}

pub(crate) fn create_environment(
    env_id: &str,
) -> Result<Box<dyn ErasedEnvironment>, RegistryError> {
    let registered = REGISTRY
        .lock()
        .unwrap()
        .get(env_id)
        .cloned()
        .ok_or_else(|| RegistryError::NotRegistered {
            env_id: env_id.to_string(),
        })?;
    let (actual, environment) = validate_factory(registered.factory)?;
    if actual.id != registered.id
        || actual.capabilities != registered.capabilities
        || actual.metadata != registered.metadata
    {
        return Err(RegistryError::FactoryDescriptorsChanged {
            env_id: env_id.to_string(),
        });
    }
    Ok(environment)
}

pub fn list_registered_environments() -> Vec<String> {
    REGISTRY.lock().unwrap().keys().cloned().collect()
}

pub fn is_registered(env_id: &str) -> bool {
    REGISTRY.lock().unwrap().contains_key(env_id)
}

#[cfg(test)]
pub(crate) fn clear_registry() {
    REGISTRY.lock().unwrap().clear();
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
