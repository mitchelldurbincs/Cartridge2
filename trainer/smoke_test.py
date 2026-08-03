#!/usr/bin/env python3
"""Standalone smoke test script for the installed AlphaZero cartridge.

This script runs a minimal test suite without requiring pytest.
It verifies algorithm resolution, the network, storage wiring, and learner
construction.

Note: Integration tests requiring PostgreSQL are skipped unless
CARTRIDGE_STORAGE_POSTGRES_URL is set.

Usage:
    python smoke_test.py
"""

import os
import secrets
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trainer.algorithms.alphazero_board_v1 import ALGORITHM_ID
from trainer.algorithms.alphazero_config import AlphaZeroLearnerConfig
from trainer.algorithms.registry import get_algorithm
from trainer.environment_catalog import get_environment
from trainer.network import PolicyValueNetwork, create_network
from trainer.storage import ReplayProfile, ReplaySelection, create_replay_store
from trainer.trainer import AlphaZeroLearner


def get_alphazero_profile(env_id: str = "tictactoe") -> ReplayProfile:
    """Resolve the exact replay profile declared by the installed cartridge."""
    algorithm = get_algorithm(ALGORITHM_ID)
    return ReplayProfile(
        env_id=env_id,
        env_contract_version=get_environment(env_id).contract_version,
        algorithm_id=algorithm.descriptor.id,
        experience_schema=algorithm.descriptor.components.experience_schema,
    )


def new_root_replay_selection(env_id: str = "tictactoe") -> ReplaySelection:
    """Allocate one exact, fresh root collection fence for a smoke run."""
    return ReplaySelection(
        profile=get_alphazero_profile(env_id),
        collection_scope_id=secrets.token_hex(32),
        source_checkpoint_id=None,
    )


def test_algorithm_registry():
    """Test that the installed algorithm resolves and accepts TicTacToe."""
    print("Testing algorithm registry...")
    algorithm = get_algorithm(ALGORITHM_ID)
    environment = get_environment("tictactoe")

    algorithm.compatibility(environment).require_compatible()

    assert algorithm.descriptor.id == ALGORITHM_ID
    print("Algorithm registry tests passed!")


def test_network():
    """Test that the network produces valid outputs."""
    import torch

    print("Testing network...")
    net = create_network("tictactoe")
    assert isinstance(net, PolicyValueNetwork)

    # Test forward pass
    batch = torch.randn(4, 18)
    policy_logits, value = net(batch)

    assert policy_logits.shape == (4, 9), f"Expected (4, 9), got {policy_logits.shape}"
    assert value.shape == (4, 1), f"Expected (4, 1), got {value.shape}"
    assert (value >= -1).all() and (value <= 1).all(), "Value should be in [-1, 1]"

    print("  Network forward pass: OK")

    # Test with legal mask
    legal_mask = torch.ones(4, 9)
    legal_mask[:, :3] = 0  # Mark first 3 positions as illegal

    policy_probs, _ = net.predict(batch, legal_mask)
    assert (policy_probs[:, :3] == 0).all(), "Illegal moves should have 0 probability"

    print("  Network with legal mask: OK")
    print("Network tests passed!")


def test_storage_factory():
    """Test that storage factory works correctly."""
    print("\nTesting storage factory...")

    # Test that it requires PostgreSQL URL
    with patch.dict(os.environ, {}, clear=True):
        try:
            create_replay_store(new_root_replay_selection())
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "PostgreSQL connection string required" in str(e)
            print("  Factory requires PostgreSQL URL: OK")

    print("Storage factory tests passed!")


def test_learner_creation():
    """Test that the installed algorithm can construct its learner."""
    print("\nTesting AlphaZero learner creation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        stats_path = Path(tmpdir) / "stats.json"
        selection = new_root_replay_selection()

        config = AlphaZeroLearnerConfig(
            model_dir=str(model_dir),
            stats_path=str(stats_path),
            total_steps=10,
            replay_selection=selection,
        )

        algorithm = get_algorithm(ALGORITHM_ID)
        learner = algorithm.build_learner(config)

        assert isinstance(learner, AlphaZeroLearner)
        assert learner.network is not None
        assert learner.replay_profile == get_alphazero_profile()
        assert learner.replay_selection == selection
        assert model_dir.exists()
        print("  AlphaZero learner creation: OK")

    print("AlphaZero learner creation tests passed!")


def test_training_with_postgres():
    """Run a minimal training loop with PostgreSQL (if configured)."""
    postgres_url = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")
    if not postgres_url:
        print("\nSkipping PostgreSQL integration test (CARTRIDGE_STORAGE_POSTGRES_URL not set)")
        return

    print("\nTesting training loop with PostgreSQL...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        stats_path = Path(tmpdir) / "stats.json"
        selection = new_root_replay_selection()

        config = AlphaZeroLearnerConfig(
            model_dir=str(model_dir),
            stats_path=str(stats_path),
            total_steps=10,
            batch_size=32,
            checkpoint_interval=5,
            max_wait=10.0,
            replay_selection=selection,
        )

        learner = get_algorithm(ALGORITHM_ID).build_learner(config)

        # Check if there's data in the database
        replay = create_replay_store(selection)
        count = replay.count()
        replay.close()

        if count < 32:
            print(f"  Not enough data in database ({count} transitions), skipping training")
            return

        stats = learner.train()
        print(f"  Training completed: {stats.step} steps, loss={stats.total_loss:.4f}")

    print("PostgreSQL integration tests passed!")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Cartridge2 Trainer Smoke Test")
    print("=" * 60)

    try:
        test_algorithm_registry()
        test_network()
        test_storage_factory()
        test_learner_creation()
        test_training_with_postgres()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
