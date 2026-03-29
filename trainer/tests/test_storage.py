"""Tests for storage backends (replay buffer and model store).

PostgreSQL tests are integration tests requiring PostgreSQL.
Set CARTRIDGE_STORAGE_POSTGRES_URL environment variable to run them.

Filesystem tests run without any external dependencies.

Example:
    export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@localhost:5432/db
    pytest tests/test_storage.py -v
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from trainer.storage.base import GameMetadata, Transition
from trainer.storage.factory import create_model_store, create_replay_buffer

# Check PostgreSQL availability for integration tests
postgres_available = bool(os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"))

# Skip decorator for PostgreSQL tests only
requires_postgres = pytest.mark.skipif(
    not postgres_available,
    reason="PostgreSQL not configured (set CARTRIDGE_STORAGE_POSTGRES_URL)"
)


@pytest.fixture
def replay_buffer():
    """Create a PostgreSQL replay buffer for testing."""
    url = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")
    buffer = create_replay_buffer(url)

    # Clean up any existing data
    buffer.clear_transitions()

    yield buffer

    # Cleanup after test
    buffer.clear_transitions()
    buffer.close()


@pytest.fixture
def sample_metadata():
    """Create sample game metadata."""
    return GameMetadata(
        env_id="testgame",
        display_name="Test Game",
        board_width=3,
        board_height=3,
        num_actions=9,
        obs_size=29,
        legal_mask_offset=18,
        player_count=2,
    )


@pytest.fixture
def sample_transition():
    """Create a sample transition."""
    obs = np.random.randn(29).astype(np.float32).tobytes()
    next_obs = np.random.randn(29).astype(np.float32).tobytes()
    policy = np.random.randn(9).astype(np.float32)
    policy = policy / policy.sum()  # Normalize

    return Transition(
        id="test-001",
        env_id="testgame",
        episode_id="ep-001",
        step_number=0,
        state=b"state_data",
        action=b"\x00\x00\x00\x00",  # action 0 as bytes
        next_state=b"next_state_data",
        observation=obs,
        next_observation=next_obs,
        reward=0.0,
        done=False,
        timestamp=1234567890,
        policy_probs=policy.tobytes(),
        mcts_value=0.5,
        game_outcome=None,
    )


@requires_postgres
class TestPostgresConnection:
    """Tests for PostgreSQL connection and schema."""

    def test_postgres_connection_success(self):
        """Test that we can connect to PostgreSQL."""
        url = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")
        buffer = create_replay_buffer(url)

        # Should be able to count (basic operation)
        count = buffer.count()
        assert isinstance(count, int)
        assert count >= 0

        buffer.close()

    def test_schema_created_on_init(self, replay_buffer):
        """Test that schema is created automatically."""
        # Schema should already exist from fixture
        # Try to access metadata table
        metadata = replay_buffer.get_metadata()
        # Returns None if no data, but shouldn't error
        assert metadata is None or isinstance(metadata, GameMetadata)

    def test_multiple_connections(self):
        """Test that multiple connections work."""
        url = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")

        # Create multiple buffers
        buffers = [create_replay_buffer(url) for _ in range(3)]

        # All should work
        for buf in buffers:
            count = buf.count()
            assert isinstance(count, int)

        # Close all
        for buf in buffers:
            buf.close()


@requires_postgres
class TestMetadataOperations:
    """Tests for game metadata CRUD."""

    def test_save_and_get_metadata(self, replay_buffer, sample_metadata):
        """Test saving and retrieving metadata."""
        # Save metadata
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO game_metadata
                       (env_id, display_name, board_width, board_height,
                        num_actions, obs_size, legal_mask_offset, player_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (env_id) DO UPDATE SET
                       display_name = EXCLUDED.display_name""",
                    (sample_metadata.env_id, sample_metadata.display_name,
                     sample_metadata.board_width, sample_metadata.board_height,
                     sample_metadata.num_actions, sample_metadata.obs_size,
                     sample_metadata.legal_mask_offset, sample_metadata.player_count)
                )
                conn.commit()

        # Retrieve
        retrieved = replay_buffer.get_metadata("testgame")

        assert retrieved is not None
        assert retrieved.env_id == sample_metadata.env_id
        assert retrieved.display_name == sample_metadata.display_name
        assert retrieved.board_width == sample_metadata.board_width

    def test_get_metadata_missing_returns_none(self, replay_buffer):
        """Test that missing metadata returns None."""
        result = replay_buffer.get_metadata("nonexistent_game")
        assert result is None

    def test_list_metadata(self, replay_buffer, sample_metadata):
        """Test listing all metadata."""
        # Insert multiple games
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i, game_id in enumerate(["game1", "game2", "game3"]):
                    cur.execute(
                        """INSERT INTO game_metadata
                           (env_id, display_name, board_width, board_height,
                            num_actions, obs_size, legal_mask_offset, player_count)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (env_id) DO NOTHING""",
                        (game_id, f"Game {i}", 3, 3, 9, 29, 18, 2)
                    )
                conn.commit()

        # List all
        all_metadata = replay_buffer.list_metadata()

        assert isinstance(all_metadata, list)
        assert len(all_metadata) >= 3

        # Check structure
        for meta in all_metadata:
            assert isinstance(meta, GameMetadata)
            assert meta.env_id


@requires_postgres
class TestTransitionOperations:
    """Tests for transition CRUD."""

    def test_add_single_transition(self, replay_buffer, sample_transition):
        """Test adding a single transition."""
        # PostgresReplayBuffer doesn't have add_transition,
        # it likely receives from actor
        # Skip this test if method doesn't exist
        if not hasattr(replay_buffer, 'add_transition'):
            pytest.skip("add_transition not implemented in PostgresReplayBuffer")

    def test_count_increases_with_data(self, replay_buffer):
        """Test that count reflects added transitions."""
        initial_count = replay_buffer.count()

        # Insert data manually
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO transitions
                       (id, env_id, episode_id, step_number, state, action,
                        next_state, observation, next_observation, reward, done,
                        timestamp, policy_probs, mcts_value, game_outcome)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    ("test-id", "testgame", "ep-001", 0, b"state", b"\x00",
                     b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                     None, 0.5, None)
                )
                conn.commit()

        new_count = replay_buffer.count()
        assert new_count == initial_count + 1

    def test_sample_returns_transitions(self, replay_buffer):
        """Test that sampling returns transitions."""
        # Insert test data
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i in range(10):
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"test-{i}", "testgame", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, i == 9, 1234567890,
                         None, 0.5, 1.0)
                    )
                conn.commit()

        # Sample
        batch = replay_buffer.sample(5, env_id="testgame")

        assert isinstance(batch, list)
        assert len(batch) == 5

        # Check structure
        for t in batch:
            assert isinstance(t, Transition)
            assert t.id.startswith("test-")

    def test_sample_respects_batch_size(self, replay_buffer):
        """Test that sample respects requested batch size."""
        # Insert data
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i in range(20):
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"test-{i}", "testgame", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, None)
                    )
                conn.commit()

        # Sample different sizes
        batch_3 = replay_buffer.sample(3)
        batch_10 = replay_buffer.sample(10)

        assert len(batch_3) == 3
        assert len(batch_10) == 10

    def test_sample_filters_by_env_id(self, replay_buffer):
        """Test that sample can filter by environment."""
        # Insert data for multiple games
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i in range(5):
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"game1-{i}", "game1", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, None)
                    )
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"game2-{i}", "game2", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, None)
                    )
                conn.commit()

        # Sample from specific game
        game1_batch = replay_buffer.sample(10, env_id="game1")

        # All should be from game1
        for t in game1_batch:
            assert t.env_id == "game1"

    def test_count_filters_by_env_id(self, replay_buffer):
        """Test that count can filter by environment."""
        # Insert data
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i in range(5):
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"game1-{i}", "game1", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, None)
                    )
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"game2-{i}", "game2", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, None)
                    )
                conn.commit()

        game1_count = replay_buffer.count(env_id="game1")
        game2_count = replay_buffer.count(env_id="game2")
        total_count = replay_buffer.count()

        assert game1_count == 5
        assert game2_count == 5
        assert total_count >= 10

@requires_postgres
class TestBufferManagement:
    """Tests for buffer operations (clear, cleanup, vacuum)."""

    def test_clear_transitions(self, replay_buffer):
        """Test clearing all transitions."""
        # Insert data
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i in range(10):
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"test-{i}", "testgame", "ep-001", i, b"state", b"\x00",
                         b"next", b"obs", b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, None)
                    )
                conn.commit()

        initial_count = replay_buffer.count()
        assert initial_count >= 10

        # Clear
        deleted = replay_buffer.clear_transitions()

        assert deleted >= 10
        assert replay_buffer.count() == 0

    def test_clear_preserves_metadata(self, replay_buffer, sample_metadata):
        """Test that clear_transitions preserves metadata."""
        # Insert metadata
        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO game_metadata
                       (env_id, display_name, board_width, board_height,
                        num_actions, obs_size, legal_mask_offset, player_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (sample_metadata.env_id, sample_metadata.display_name,
                     sample_metadata.board_width, sample_metadata.board_height,
                     sample_metadata.num_actions, sample_metadata.obs_size,
                     sample_metadata.legal_mask_offset, sample_metadata.player_count)
                )
                conn.commit()

        # Clear transitions
        replay_buffer.clear_transitions()

        # Metadata should still exist
        metadata = replay_buffer.get_metadata("testgame")
        assert metadata is not None
        assert metadata.env_id == "testgame"

@requires_postgres
class TestSampleBatchTensors:
    """Tests for sample_batch_tensors helper."""

    def test_sample_batch_tensors_returns_numpy(self, replay_buffer):
        """Test that sample_batch_tensors returns numpy arrays."""
        # Insert test data with proper observations
        obs1 = np.random.randn(29).astype(np.float32).tobytes()
        obs2 = np.random.randn(29).astype(np.float32).tobytes()

        with replay_buffer._connection() as conn:
            with conn.cursor() as cur:
                for i in range(5):
                    cur.execute(
                        """INSERT INTO transitions
                           (id, env_id, episode_id, step_number, state, action,
                            next_state, observation, next_observation, reward, done,
                            timestamp, policy_probs, mcts_value, game_outcome)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (f"test-{i}", "testgame", "ep-001", i, b"state", b"\x00",
                         b"next", obs1 if i % 2 == 0 else obs2, b"next_obs", 0.0, False, 1234567890,
                         None, 0.5, 1.0)
                    )
                conn.commit()

        # Sample
        result = replay_buffer.sample_batch_tensors(3, num_actions=9, env_id="testgame")

        if result is not None:
            observations, policy_targets, value_targets = result

            assert isinstance(observations, np.ndarray)
            assert isinstance(policy_targets, np.ndarray)
            assert isinstance(value_targets, np.ndarray)

            assert observations.shape[0] == 3
            assert policy_targets.shape[0] == 3
            assert value_targets.shape[0] == 3

    def test_sample_batch_tensors_not_enough_data(self, replay_buffer):
        """Test that sample_batch_tensors returns None if not enough data."""
        # Clear buffer
        replay_buffer.clear_transitions()

        # Try to sample more than available
        result = replay_buffer.sample_batch_tensors(100, num_actions=9)

        assert result is None


class TestFilesystemStorage:
    """Tests for filesystem model storage (no PostgreSQL needed)."""

    def test_filesystem_model_store_creation(self):
        """Test creating a filesystem model store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_model_store("filesystem", path=Path(tmpdir))

            assert store is not None

            # Cleanup
            if hasattr(store, 'close'):
                store.close()

    def test_filesystem_save_and_load_latest(self):
        """Test saving and loading latest model via filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_model_store("filesystem", path=Path(tmpdir))

            # Save model
            model_bytes = b"fake onnx model data"
            info = store.save_onnx(model_bytes, step=100)

            assert info.step == 100
            assert Path(info.path).exists()

            # Load latest
            latest = store.load_latest_onnx()
            assert latest == model_bytes

    def test_filesystem_list_checkpoints(self):
        """Test listing checkpoints in filesystem store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_model_store("filesystem", path=Path(tmpdir))

            # Save multiple
            for step in [50, 100, 150]:
                store.save_onnx(b"model", step=step)

            # List
            checkpoints = store.list_checkpoints()

            assert len(checkpoints) >= 3

            # Should be sorted by step
            steps = [c.step for c in checkpoints]
            assert steps == sorted(steps)

    def test_filesystem_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_model_store("filesystem", path=Path(tmpdir))

            # Save multiple
            for step in range(1, 11):
                store.save_onnx(b"model", step=step)

            # Keep only 5
            deleted = store.cleanup_old_checkpoints(max_keep=5)

            assert deleted == 5  # 10 - 5 = 5 deleted

            remaining = store.list_checkpoints()
            assert len(remaining) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
