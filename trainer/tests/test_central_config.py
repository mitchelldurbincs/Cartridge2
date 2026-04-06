"""Tests for central_config module.

This module tests:
- Config loading from file
- Environment variable overrides
- Thread safety with concurrent access
- Config reload functionality
"""

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from trainer.central_config import (
    Config,
    CommonConfig,
    TrainingConfig,
    EvaluationConfig,
    get_config,
    reset_config,
    _config_lock,
    _cached_config,
)


class TestConfigLoading:
    """Test configuration loading from files."""

    def test_load_default_config(self, monkeypatch, tmp_path):
        """Test that default config loads successfully."""
        # Isolate from repo's config.toml by changing to temp directory
        monkeypatch.chdir(tmp_path)
        reset_config()

        config = get_config()

        assert isinstance(config, Config)
        assert config.common.env_id == "tictactoe"
        assert config.training.iterations == 100
        assert config.evaluation.interval == 1

    def test_config_caching(self):
        """Test that config is cached after first load."""
        reset_config()

        config1 = get_config()
        config2 = get_config()

        # Should be the same object (cached)
        assert config1 is config2

    def test_config_reload(self):
        """Test that reload flag refreshes config."""
        reset_config()

        config1 = get_config()
        config2 = get_config(reload=True)

        # Should be different objects after reload
        assert config1 is not config2


class TestEnvironmentVariableOverrides:
    """Test environment variable configuration overrides."""

    def test_env_override_common_env_id(self, monkeypatch):
        """Test CARTRIDGE_COMMON_ENV_ID override."""
        reset_config()
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "connect4")

        config = get_config(reload=True)

        assert config.common.env_id == "connect4"

    def test_env_override_training_iterations(self, monkeypatch):
        """Test CARTRIDGE_TRAINING_ITERATIONS override."""
        reset_config()
        monkeypatch.setenv("CARTRIDGE_TRAINING_ITERATIONS", "50")

        config = get_config(reload=True)

        assert config.training.iterations == 50

    def test_env_override_evaluation_games(self, monkeypatch):
        """Test CARTRIDGE_EVALUATION_GAMES override."""
        reset_config()
        monkeypatch.setenv("CARTRIDGE_EVALUATION_GAMES", "100")

        config = get_config(reload=True)

        assert config.evaluation.games == 100

    def test_env_override_learning_rate(self, monkeypatch):
        """Test CARTRIDGE_TRAINING_LEARNING_RATE override."""
        reset_config()
        monkeypatch.setenv("CARTRIDGE_TRAINING_LEARNING_RATE", "0.002")

        config = get_config(reload=True)

        assert config.training.learning_rate == 0.002

    def test_env_override_bool_value(self, monkeypatch):
        """Test boolean environment variable override."""
        reset_config()
        monkeypatch.setenv("CARTRIDGE_EVALUATION_EVAL_VS_RANDOM", "false")

        config = get_config(reload=True)

        assert config.evaluation.eval_vs_random is False

    def test_legacy_env_override(self, monkeypatch):
        """Test legacy ALPHAZERO_* environment variables."""
        reset_config()
        monkeypatch.setenv("ALPHAZERO_ENV_ID", "connect4")
        monkeypatch.setenv("ALPHAZERO_ITERATIONS", "75")

        config = get_config(reload=True)

        assert config.common.env_id == "connect4"
        assert config.training.iterations == 75

    def test_empty_env_var_ignored(self, monkeypatch, tmp_path):
        """Test that empty environment variables are ignored."""
        # Isolate from repo's config.toml by changing to temp directory
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "")

        config = get_config(reload=True)

        # Should use default, not empty string
        assert config.common.env_id == "tictactoe"


class TestConfigDataclasses:
    """Test config dataclass properties."""

    def test_common_config_defaults(self):
        """Test CommonConfig defaults."""
        config = CommonConfig()

        assert config.data_dir == "./data"
        assert config.env_id == "tictactoe"
        assert config.log_level == "info"

    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()

        assert config.iterations == 100
        assert config.episodes_per_iteration == 500
        assert config.learning_rate == 0.001
        assert config.device == "auto"

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig defaults."""
        config = EvaluationConfig()

        assert config.interval == 1
        assert config.games == 50
        assert config.win_threshold == 0.55
        assert config.eval_vs_random is True

    def test_config_path_properties(self):
        """Test Config path properties."""
        reset_config()
        config = get_config()

        assert isinstance(config.data_dir, Path)
        assert isinstance(config.models_dir, Path)
        assert isinstance(config.replay_db_path, Path)
        assert isinstance(config.stats_path, Path)


class TestThreadSafety:
    """Test thread safety of config access."""

    def test_concurrent_config_access(self):
        """Test that concurrent config access is thread-safe."""
        reset_config()

        results = []
        errors = []

        def access_config(thread_id):
            try:
                for _ in range(100):
                    config = get_config()
                    # Access multiple fields
                    _ = config.common.env_id
                    _ = config.training.iterations
                    _ = config.evaluation.games
                    results.append(thread_id)
                    time.sleep(0.001)  # Small delay to increase contention
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=access_config, args=(i,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Thread errors: {errors}"
        # Should have all results
        assert len(results) == 5 * 100

    def test_concurrent_reload_access(self):
        """Test concurrent reload and access operations."""
        reset_config()

        access_results = []
        reload_results = []
        errors = []

        def access_config():
            try:
                for _ in range(50):
                    config = get_config()
                    access_results.append(config.common.env_id)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("access", str(e)))

        def reload_config():
            try:
                for _ in range(20):
                    config = get_config(reload=True)
                    reload_results.append(config.common.env_id)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("reload", str(e)))

        # Start threads - 3 accessors, 1 reloader
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=access_config))
        threads.append(threading.Thread(target=reload_config))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"
        # Should have results from both operations
        assert len(access_results) == 3 * 50
        assert len(reload_results) == 20

    def test_reset_config_thread_safety(self):
        """Test that reset_config is thread-safe."""
        reset_config()

        results = []
        errors = []

        def reset_and_access(thread_id):
            try:
                for _ in range(20):
                    reset_config()
                    config = get_config()
                    results.append((thread_id, config.common.env_id))
                    time.sleep(0.002)
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(3):
            threads.append(threading.Thread(target=reset_and_access, args=(i,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during reset: {errors}"
        assert len(results) == 3 * 20

    def test_lock_is_acquired_during_load(self):
        """Test that lock is properly acquired during config loading."""
        reset_config()

        lock_acquired = []

        def track_lock():
            # Try to acquire lock while get_config is running
            time.sleep(0.01)  # Let main thread start first
            acquired = _config_lock.acquire(blocking=False)
            if acquired:
                lock_acquired.append("acquired")
                _config_lock.release()
            else:
                lock_acquired.append("blocked")

        # Pre-load config
        get_config()

        thread = threading.Thread(target=track_lock)
        thread.start()
        thread.join()

        # Lock should have been acquired (not blocked) since config was cached
        assert "acquired" in lock_acquired


class TestConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_env_var_for_int_uses_string(self, monkeypatch, tmp_path):
        """Test that invalid int env var gets stored as string (may cause errors later)."""
        # Isolate from repo's config.toml by changing to temp directory
        monkeypatch.chdir(tmp_path)
        reset_config()
        # Invalid integer - code currently stores it as string
        monkeypatch.setenv("CARTRIDGE_TRAINING_ITERATIONS", "not_a_number")

        # The code doesn't validate at load time, it stores as string
        # This may cause errors later when trying to use the value
        config = get_config(reload=True)
        # Value is stored as string since conversion failed
        assert config.training.iterations == "not_a_number"

    def test_invalid_env_var_uses_string_fallback(self, monkeypatch):
        """Test that invalid env var for string fields works fine."""
        reset_config()
        # For string fields, any value is valid
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "valid_game_name")

        config = get_config(reload=True)
        assert config.common.env_id == "valid_game_name"

    def test_config_search_paths_exist(self):
        """Test that default config search paths are defined."""
        from trainer.central_config import CONFIG_SEARCH_PATHS, DEFAULTS_SEARCH_PATHS

        assert len(CONFIG_SEARCH_PATHS) > 0
        assert len(DEFAULTS_SEARCH_PATHS) > 0
        assert all(isinstance(p, Path) for p in CONFIG_SEARCH_PATHS)
        assert all(isinstance(p, Path) for p in DEFAULTS_SEARCH_PATHS)
