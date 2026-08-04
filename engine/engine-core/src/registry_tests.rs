use crate::board_view::Presentation;
use crate::erased::{ErasedEnvironment, ErasedEnvironmentError, ErasedTimestep};
use crate::metadata::EnvironmentMetadata;
use crate::registry::*;
use crate::test_utils::REGISTRY_TEST_MUTEX;
use crate::typed::{
    ActionSpace, AgentId, AgentModel, Capabilities, Encoding, EngineId, EnvironmentSemantics,
    InformationModel,
};
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug)]
struct RegistryEnvironment {
    build_id: &'static str,
    preferred_batch: u32,
    descriptor_variant: u32,
}

impl ErasedEnvironment for RegistryEnvironment {
    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "registered".into(),
            build_id: if self.descriptor_variant == 1 {
                "changed-build".into()
            } else {
                self.build_id.into()
            },
        }
    }

    fn capabilities(&self) -> Capabilities {
        let mut capabilities = Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::custom("state:v1", "action:v1", "obs:v1"),
            semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
            max_horizon: None,
            agents: AgentModel::fixed_homogeneous([AgentId(1)], ActionSpace::discrete(1)),
            preferred_batch: self.preferred_batch,
        };
        match self.descriptor_variant {
            2 => capabilities.contract_version = 2,
            3 => {
                capabilities.agents =
                    AgentModel::fixed_homogeneous([AgentId(1)], ActionSpace::discrete(2));
            }
            4 => capabilities.encoding.state = "state:v2".into(),
            5 => {
                capabilities.semantics.information_model = InformationModel::PartiallyObserved;
            }
            _ => {}
        }
        capabilities
    }

    fn metadata(&self) -> EnvironmentMetadata {
        let metadata = EnvironmentMetadata::new("registered", "Registered");
        if self.descriptor_variant == 6 {
            metadata.with_description("changed metadata")
        } else {
            metadata
        }
    }

    fn reset(
        &mut self,
        _seed: u64,
        _hint: &[u8],
        _out_state: &mut Vec<u8>,
        _out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        Ok(())
    }

    fn step(
        &mut self,
        _state: &[u8],
        _action: &[u8],
        _out_state: &mut Vec<u8>,
        _out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        Ok(())
    }

    fn presentation(&self, _state: &[u8]) -> Result<Option<Presentation>, ErasedEnvironmentError> {
        Ok(None)
    }
}

fn factory_a() -> Result<Box<dyn ErasedEnvironment>, ErasedEnvironmentError> {
    Ok(Box::new(RegistryEnvironment {
        build_id: "a",
        preferred_batch: 1,
        descriptor_variant: 0,
    }))
}

fn factory_b() -> Result<Box<dyn ErasedEnvironment>, ErasedEnvironmentError> {
    Ok(Box::new(RegistryEnvironment {
        build_id: "b",
        preferred_batch: 1,
        descriptor_variant: 0,
    }))
}

static DESCRIPTOR_VARIANT: AtomicU32 = AtomicU32::new(0);

fn descriptor_drifting_factory() -> Result<Box<dyn ErasedEnvironment>, ErasedEnvironmentError> {
    Ok(Box::new(RegistryEnvironment {
        build_id: "stable-build",
        preferred_batch: if DESCRIPTOR_VARIANT.load(Ordering::SeqCst) == 7 {
            2
        } else {
            1
        },
        descriptor_variant: DESCRIPTOR_VARIANT.load(Ordering::SeqCst),
    }))
}

#[test]
fn registry_constructs_and_lists_environments() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    clear_registry();
    register_factory(factory_a).unwrap();

    assert!(is_registered("registered"));
    assert_eq!(list_registered_environments(), vec!["registered"]);
    assert_eq!(
        create_environment("registered")
            .unwrap()
            .engine_id()
            .build_id,
        "a"
    );
    assert_eq!(
        create_environment("missing").unwrap_err(),
        RegistryError::NotRegistered {
            env_id: "missing".into()
        }
    );
}

#[test]
fn duplicate_registration_is_rejected_without_replacement() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    clear_registry();
    register_factory(factory_a).unwrap();
    assert_eq!(
        register_factory(factory_b),
        Err(RegistryError::AlreadyRegistered {
            env_id: "registered".into()
        })
    );
    assert_eq!(
        create_environment("registered")
            .unwrap()
            .engine_id()
            .build_id,
        "a"
    );
}

#[test]
fn every_factory_descriptor_field_is_pinned_at_registration() {
    let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
    for (descriptor_name, descriptor_variant) in [
        ("engine ID", 1),
        ("contract version", 2),
        ("action space", 3),
        ("encoding", 4),
        ("semantics", 5),
        ("metadata", 6),
        ("preferred batch", 7),
    ] {
        clear_registry();
        DESCRIPTOR_VARIANT.store(0, Ordering::SeqCst);
        register_factory(descriptor_drifting_factory).unwrap();
        DESCRIPTOR_VARIANT.store(descriptor_variant, Ordering::SeqCst);

        assert_eq!(
            create_environment("registered").unwrap_err(),
            RegistryError::FactoryDescriptorsChanged {
                env_id: "registered".into(),
            },
            "{descriptor_name} drift was not rejected"
        );
    }
}
