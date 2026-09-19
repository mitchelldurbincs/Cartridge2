use super::*;
use crate::artifact::{sha256_hex, ArtifactProfile, BlobReference};
use crate::tests::{identity, CONTRACT_TEST_MODEL};
use crate::ModelLoadSpec;
use tempfile::tempdir;

fn evaluator() -> SharedOnnxEvaluator {
    let root = tempdir().unwrap();
    let path = root.path().join("model.onnx");
    std::fs::write(&path, CONTRACT_TEST_MODEL).unwrap();
    ModelLoadSpec::new(3, 3, 1, 9, identity("contract_test"))
        .unwrap()
        .load(&path)
        .unwrap()
}

fn manifest() -> CheckpointManifestV1 {
    CheckpointManifestV1 {
        schema_version: 1,
        profile: ArtifactProfile::from(&identity("contract_test")),
        step: 42,
        parent_checkpoint_id: None,
        config_sha256: "a".repeat(64),
        onnx: BlobReference {
            sha256: sha256_hex(CONTRACT_TEST_MODEL),
            size_bytes: CONTRACT_TEST_MODEL.len() as u64,
        },
        learner_state: BlobReference {
            sha256: "b".repeat(64),
            size_bytes: 1,
        },
    }
}

fn head(checkpoint: char, commit: char) -> AcceptedHead {
    AcceptedHead {
        model_checkpoint_id: checkpoint.to_string().repeat(64),
        run_commit_id: commit.to_string().repeat(64),
    }
}

fn loaded_state() -> ReloadState {
    let state = ReloadState::new(Arc::new(RwLock::new(None)));
    assert_eq!(
        state
            .commit_candidate(
                head('a', 'b'),
                &manifest(),
                "original.onnx".into(),
                Some(evaluator()),
                None,
            )
            .unwrap(),
        LoadOutcome::Loaded
    );
    state
}

struct StateSnapshot {
    accepted_head: Option<AcceptedHead>,
    evaluator_id: Option<usize>,
    model_info: ModelInfo,
}

impl StateSnapshot {
    fn read(state: &ReloadState) -> Self {
        // Test-only poison recovery lets assertions inspect state after a failed
        // lock acquisition; production always propagates poisoned-lock errors.
        Self {
            accepted_head: state
                .accepted_head
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
            evaluator_id: state
                .evaluator
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_ref()
                .map(|value| value.instance_ptr() as usize),
            model_info: state
                .model_info
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        }
    }

    #[track_caller]
    fn assert_matches(&self, state: &ReloadState) {
        let actual = Self::read(state);
        assert_eq!(actual.accepted_head, self.accepted_head);
        assert_eq!(actual.evaluator_id, self.evaluator_id);
        assert_eq!(actual.model_info.loaded, self.model_info.loaded);
        assert_eq!(
            actual.model_info.checkpoint_id,
            self.model_info.checkpoint_id
        );
        assert_eq!(actual.model_info.model_sha256, self.model_info.model_sha256);
        assert_eq!(actual.model_info.path, self.model_info.path);
        assert_eq!(actual.model_info.loaded_at, self.model_info.loaded_at);
        assert_eq!(
            actual.model_info.training_step,
            self.model_info.training_step
        );
    }
}

#[test]
fn loaded_candidate_publishes_evaluator_and_transport_location() {
    for location in [
        "/tmp/fixture/model.onnx",
        "s3://fixture/profile/blobs/model.onnx",
    ] {
        let state = loaded_state();
        let candidate = head('c', 'd');
        let new_evaluator = evaluator();
        let evaluator_id = new_evaluator.instance_ptr() as usize;
        let manifest = manifest();
        assert_eq!(
            state
                .commit_candidate(
                    candidate.clone(),
                    &manifest,
                    location.into(),
                    Some(new_evaluator),
                    state.accepted_head().unwrap(),
                )
                .unwrap(),
            LoadOutcome::Loaded
        );
        let actual = StateSnapshot::read(&state);
        assert_eq!(actual.accepted_head, Some(candidate.clone()));
        assert_eq!(actual.evaluator_id, Some(evaluator_id));
        assert!(actual.model_info.loaded);
        assert_eq!(
            actual.model_info.checkpoint_id,
            Some(candidate.model_checkpoint_id)
        );
        assert_eq!(actual.model_info.model_sha256, Some(manifest.onnx.sha256));
        assert_eq!(actual.model_info.training_step, Some(manifest.step));
        assert_eq!(actual.model_info.path.as_deref(), Some(location));
        assert!(actual.model_info.loaded_at.is_some());
    }
}

#[test]
fn retained_checkpoint_advances_generation_without_changing_model_info_or_evaluator() {
    let state = loaded_state();
    let mut expected = StateSnapshot::read(&state);
    let candidate = head('a', 'c');
    assert_eq!(
        state
            .commit_candidate(
                candidate.clone(),
                &manifest(),
                "unused-location.onnx".into(),
                None,
                expected.accepted_head.clone(),
            )
            .unwrap(),
        LoadOutcome::Advanced
    );
    expected.accepted_head = Some(candidate);
    expected.assert_matches(&state);
}

#[test]
fn unsafe_evaluator_reuse_preserves_all_state() {
    for case in [
        "missing accepted head",
        "different checkpoint",
        "missing evaluator",
    ] {
        let state = loaded_state();
        let mut candidate = head('a', 'c');
        match case {
            "missing accepted head" => *state.accepted_head.write().unwrap() = None,
            "different checkpoint" => candidate.model_checkpoint_id = "d".repeat(64),
            "missing evaluator" => *state.evaluator.write().unwrap() = None,
            _ => unreachable!(),
        }
        let before = StateSnapshot::read(&state);
        let error = state
            .commit_candidate(
                candidate,
                &manifest(),
                "rejected.onnx".into(),
                None,
                before.accepted_head.clone(),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("cannot reuse an evaluator"),
            "{case}: {error}"
        );
        before.assert_matches(&state);
    }
}

#[test]
fn duplicate_and_raced_candidates_preserve_all_state() {
    for (case, candidate, expected_head) in [
        (
            "already accepted generation",
            head('a', 'b'),
            Some(head('a', 'b')),
        ),
        ("stale accepted snapshot", head('c', 'd'), None),
    ] {
        let state = loaded_state();
        let before = StateSnapshot::read(&state);
        assert_eq!(
            state
                .commit_candidate(
                    candidate,
                    &manifest(),
                    "discarded.onnx".into(),
                    Some(evaluator()),
                    expected_head,
                )
                .unwrap(),
            LoadOutcome::Unchanged,
            "{case}"
        );
        before.assert_matches(&state);
    }
}

#[test]
fn poisoned_locks_fail_before_any_state_is_published() {
    for lock in ["accepted head", "evaluator", "model info"] {
        let state = loaded_state();
        let before = StateSnapshot::read(&state);
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match lock {
            "accepted head" => {
                let _guard = state.accepted_head.write().unwrap();
                panic!("poison accepted head");
            }
            "evaluator" => {
                let _guard = state.evaluator.write().unwrap();
                panic!("poison evaluator");
            }
            "model info" => {
                let _guard = state.model_info.write().unwrap();
                panic!("poison model info");
            }
            _ => unreachable!(),
        }));
        assert!(poisoned.is_err());
        let error = state
            .commit_candidate(
                head('c', 'd'),
                &manifest(),
                "rejected.onnx".into(),
                Some(evaluator()),
                before.accepted_head.clone(),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("failed to lock"),
            "{lock}: {error}"
        );
        before.assert_matches(&state);
    }
}
