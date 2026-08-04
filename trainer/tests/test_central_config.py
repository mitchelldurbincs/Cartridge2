"""Tests for central_config module.

This module tests:
- Config loading from file
- Environment variable overrides
- Thread safety with concurrent access
- Config reload functionality
"""

import threading
import time
from pathlib import Path

import pytest

from trainer.central_config import (
    ActorConfig,
    AlgorithmConfig,
    CommonConfig,
    Config,
    EvaluationConfig,
    LoggingConfig,
    MctsConfig,
    TrainingConfig,
    _config_lock,
    get_config,
    reset_config,
)


class TestConfigLoading:
    """Test configuration loading from files."""

    def test_load_default_config(self, monkeypatch, tmp_path):
        """Test that default config loads successfully."""
        # Isolate from repo's config.toml by changing to temp directory
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "trainer.central_config.CONFIG_SEARCH_PATHS", [tmp_path / "config.toml"]
        )
        reset_config()

        config = get_config()

        assert isinstance(config, Config)
        assert config.common.env_id == "tictactoe"
        assert config.training.iterations == 100
        assert config.evaluation.interval == 1
        assert config.algorithm.id == "alphazero_board_v1"

    def test_shared_defaults_resolve_to_the_cartridge2_root(self):
        from trainer import central_config

        defaults = central_config._find_defaults_file()
        assert defaults == central_config._PROJECT_ROOT / "config.defaults.toml"
        assert defaults.is_file()

    def test_packaged_defaults_are_byte_identical_to_the_root(self):
        from trainer import central_config

        root = central_config._PROJECT_ROOT / "config.defaults.toml"
        packaged = Path(central_config.__file__).with_name("config.defaults.toml")
        assert packaged.is_file()
        assert packaged.read_bytes() == root.read_bytes()

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

    def test_unknown_algorithm_override_is_rejected(self, monkeypatch):
        reset_config()
        monkeypatch.setenv("CARTRIDGE_ALGORITHM_ID", "test_algorithm")

        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_config(reload=True)

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

    def test_scheduled_evaluation_requires_first_candidate_evidence(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_EVALUATION_EVAL_VS_RANDOM", "false")

        with pytest.raises(ValueError, match="first-candidate evidence"):
            get_config(reload=True)

    def test_empty_identity_env_var_is_rejected(self, monkeypatch, tmp_path):
        """An explicitly empty profile identity must not fall back."""
        # Isolate from repo's config.toml by changing to temp directory
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "")

        with pytest.raises(ValueError, match="common.env_id"):
            get_config(reload=True)


class TestConfigDataclasses:
    """Test config dataclass properties."""

    def test_no_argument_dataclasses_use_canonical_toml_defaults(self):
        from dataclasses import fields

        from trainer.config_sections import CANONICAL_DEFAULTS_DATA

        config = Config()
        for section_name, values in CANONICAL_DEFAULTS_DATA.items():
            section = getattr(config, section_name)
            actual = {item.name: getattr(section, item.name) for item in fields(section)}
            for key, expected in values.items():
                if isinstance(expected, float):
                    assert actual[key] == pytest.approx(expected)
                else:
                    assert actual[key] == expected

    def test_common_config_defaults(self):
        """Test CommonConfig defaults."""
        config = CommonConfig()

        assert config.data_dir == "./data"
        assert config.env_id == "tictactoe"
        assert config.log_level == "info"

    def test_algorithm_config_defaults(self):
        assert AlgorithmConfig().id == "alphazero_board_v1"

    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()

        assert config.iterations == 100
        assert config.episodes_per_iteration == 500
        assert config.learning_rate == 0.001
        assert config.device == "cpu"

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig defaults."""
        config = EvaluationConfig()

        assert config.interval == 1
        assert config.games == 50
        assert config.win_threshold == 0.55
        assert config.eval_vs_random is True
        assert config.simulations == 0
        assert config.temperature == pytest.approx(0.2)
        assert config.solver_games == 0

    def test_mcts_config_defaults(self):
        config = MctsConfig()

        assert config.c_puct == pytest.approx(1.4)
        assert config.temperature == 1.0
        assert config.late_temperature == 1.0
        assert config.dirichlet_alpha == pytest.approx(0.3)
        assert config.dirichlet_weight == 0.25
        assert config.eval_batch_size == 32

    def test_config_path_properties(self):
        """Test Config path properties."""
        reset_config()
        config = get_config()

        assert isinstance(config.data_root, Path)
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.models_dir, Path)
        assert isinstance(config.stats_path, Path)
        assert config.data_dir == (
            config.data_root / "profiles" / config.algorithm.id / config.common.env_id / "v2"
        )


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

    def test_invalid_env_var_for_int_fails_during_config_load(self, monkeypatch, tmp_path):
        """Typed configuration rejects invalid integer overrides immediately."""
        # Isolate from repo's config.toml by changing to temp directory
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_TRAINING_ITERATIONS", "not_a_number")

        with pytest.raises(ValueError, match="not_a_number"):
            get_config(reload=True)

    def test_unknown_environment_override_is_rejected(self, monkeypatch, tmp_path):
        """Selected environment IDs must exist in the generated catalog."""
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "valid_game_name")

        with pytest.raises(ValueError, match="Unknown environment"):
            get_config(reload=True)

    def test_invalid_boolean_override_is_rejected(self, monkeypatch):
        reset_config()
        monkeypatch.setenv("CARTRIDGE_EVALUATION_EVAL_VS_RANDOM", "sometimes")

        with pytest.raises(ValueError, match="Invalid boolean.*true/false"):
            get_config(reload=True)

    def test_unknown_key_is_rejected(self, monkeypatch, tmp_path):
        (tmp_path / "config.toml").write_text('[algorithm]\nidd = "alphazero_board_v1"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "trainer.central_config.CONFIG_SEARCH_PATHS", [tmp_path / "config.toml"]
        )
        reset_config()

        with pytest.raises(ValueError, match=r"\[algorithm\].*idd"):
            get_config(reload=True)

    def test_missing_explicit_config_is_rejected(self, monkeypatch, tmp_path):
        missing = tmp_path / "missing.toml"
        monkeypatch.setenv("CARTRIDGE_CONFIG", str(missing))
        reset_config()

        with pytest.raises(FileNotFoundError, match="CARTRIDGE_CONFIG"):
            get_config(reload=True)

    def test_missing_canonical_defaults_are_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "trainer.central_config.DEFAULTS_SEARCH_PATHS",
            [tmp_path / "missing-defaults.toml"],
        )
        reset_config()

        with pytest.raises(FileNotFoundError, match="canonical config.defaults.toml"):
            get_config(reload=True)

    def test_divergent_canonical_default_copies_are_rejected(self, monkeypatch, tmp_path):
        first = tmp_path / "first.toml"
        second = tmp_path / "second.toml"
        first.write_text("[common]\nenv_id = 'one'\n")
        second.write_text("[common]\nenv_id = 'two'\n")
        monkeypatch.setattr("trainer.central_config.DEFAULTS_SEARCH_PATHS", [first, second])
        reset_config()

        with pytest.raises(RuntimeError, match="copies diverge"):
            get_config(reload=True)

    def test_incomplete_canonical_defaults_are_rejected(self, monkeypatch, tmp_path):
        incomplete = tmp_path / "config.defaults.toml"
        incomplete.write_text(
            '[common]\ndata_dir = "./data"\nenv_id = "tictactoe"\nlog_level = "info"\n'
        )
        monkeypatch.setattr("trainer.central_config.DEFAULTS_SEARCH_PATHS", [incomplete])
        monkeypatch.setattr(
            "trainer.central_config.CONFIG_SEARCH_PATHS", [tmp_path / "missing.toml"]
        )
        reset_config()

        with pytest.raises(ValueError, match="missing sections"):
            get_config(reload=True)

    def test_missing_required_canonical_default_key_is_rejected(self, monkeypatch, tmp_path):
        from trainer import central_config

        incomplete = tmp_path / "config.defaults.toml"
        canonical = (central_config._PROJECT_ROOT / "config.defaults.toml").read_text()
        incomplete.write_text(canonical.replace('device = "cpu"\n', ""))
        monkeypatch.setattr("trainer.central_config.DEFAULTS_SEARCH_PATHS", [incomplete])
        monkeypatch.setattr(
            "trainer.central_config.CONFIG_SEARCH_PATHS", [tmp_path / "missing.toml"]
        )
        reset_config()

        with pytest.raises(ValueError, match=r"\[training\].*device"):
            get_config(reload=True)

    def test_removed_mcts_num_simulations_config_is_rejected(self, monkeypatch, tmp_path):
        (tmp_path / "config.toml").write_text("[mcts]\nnum_simulations = 800\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "trainer.central_config.CONFIG_SEARCH_PATHS", [tmp_path / "config.toml"]
        )
        reset_config()

        with pytest.raises(ValueError, match=r"\[mcts\].*num_simulations"):
            get_config(reload=True)

    def test_removed_mcts_num_simulations_env_is_rejected(self, monkeypatch):
        monkeypatch.setenv("CARTRIDGE_MCTS_NUM_SIMULATIONS", "800")
        reset_config()

        with pytest.raises(ValueError, match="CARTRIDGE_MCTS_NUM_SIMULATIONS"):
            get_config(reload=True)

    def test_config_search_paths_exist(self):
        """Test that default config search paths are defined."""
        from trainer.central_config import CONFIG_SEARCH_PATHS, DEFAULTS_SEARCH_PATHS

        assert len(CONFIG_SEARCH_PATHS) > 0
        assert len(DEFAULTS_SEARCH_PATHS) > 0
        assert all(isinstance(p, Path) for p in CONFIG_SEARCH_PATHS)
        assert all(isinstance(p, Path) for p in DEFAULTS_SEARCH_PATHS)


class TestLoggingConfigSection:
    """Regression tests for the [logging] section.

    _dict_to_config() used to omit the logging section, so [logging] values
    from config.toml (and CARTRIDGE_LOGGING_* overrides) never reached
    Config.logging.
    """

    def test_logging_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        reset_config()

        config = get_config(reload=True)

        assert isinstance(config.logging, LoggingConfig)
        assert config.logging.format == "text"
        assert config.logging.include_timestamps is True
        assert config.logging.include_target is True

    def test_toml_parses_logging_section(self, monkeypatch, tmp_path):
        (tmp_path / "config.toml").write_text("""
[logging]
format = "json"
include_timestamps = false
include_target = false
""")
        monkeypatch.chdir(tmp_path)
        reset_config()

        config = get_config(reload=True)

        assert config.logging.format == "json"
        assert config.logging.include_timestamps is False
        assert config.logging.include_target is False

    def test_env_overrides_logging_section(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_LOGGING_FORMAT", "json")
        monkeypatch.setenv("CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS", "false")

        config = get_config(reload=True)

        assert config.logging.format == "json"
        assert config.logging.include_timestamps is False


class TestWandbAndSolverConfig:
    """Test the [wandb] section and the new [evaluation] solver/promotion keys."""

    def test_wandb_defaults(self, monkeypatch, tmp_path):
        from trainer.central_config import WandbConfig

        monkeypatch.chdir(tmp_path)
        reset_config()
        config = get_config(reload=True)

        assert isinstance(config.wandb, WandbConfig)
        assert config.wandb.enabled is False
        assert config.wandb.required is False
        assert config.wandb.project == "cartridge2"
        assert config.wandb.entity == ""
        assert config.wandb.tags == []
        assert config.wandb.init_timeout_seconds == 30.0

    def test_evaluation_solver_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        reset_config()
        config = get_config(reload=True)

        assert config.evaluation.solver_games == 0
        assert config.evaluation.evaluation_seed == 42
        assert config.evaluation.promotion_metric == "win_rate"
        assert config.evaluation.promotion_margin == 0.0

    def test_toml_parses_wandb_and_solver_keys(self, monkeypatch, tmp_path):
        (tmp_path / "config.toml").write_text("""
[common]
env_id = "connect4"

[evaluation]
solver_games = 25
evaluation_seed = 7
promotion_metric = "solver_optimal"
promotion_margin = 0.02
win_threshold = 0.0

[wandb]
enabled = true
project = "my-project"
entity = "my-entity"
group = "exp-group"
tags = ["a", "b"]
""")
        monkeypatch.chdir(tmp_path)
        reset_config()
        config = get_config(reload=True)

        assert config.evaluation.solver_games == 25
        assert config.evaluation.evaluation_seed == 7
        assert config.evaluation.promotion_metric == "solver_optimal"
        assert config.evaluation.promotion_margin == 0.02
        assert config.wandb.enabled is True
        assert config.wandb.project == "my-project"
        assert config.wandb.entity == "my-entity"
        assert config.wandb.group == "exp-group"
        assert config.wandb.tags == ["a", "b"]

    def test_env_overrides_wandb_and_promotion(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        reset_config()
        monkeypatch.setenv("CARTRIDGE_WANDB_ENABLED", "true")
        monkeypatch.setenv("CARTRIDGE_WANDB_INIT_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "connect4")
        monkeypatch.setenv("CARTRIDGE_EVALUATION_PROMOTION_METRIC", "solver_optimal")
        monkeypatch.setenv("CARTRIDGE_EVALUATION_SOLVER_GAMES", "12")
        monkeypatch.setenv("CARTRIDGE_EVALUATION_WIN_THRESHOLD", "0")

        config = get_config(reload=True)

        assert config.wandb.enabled is True
        assert config.wandb.init_timeout_seconds == 60.0
        assert config.evaluation.promotion_metric == "solver_optimal"
        assert config.evaluation.solver_games == 12

    @pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
    def test_nonfinite_wandb_timeout_is_rejected(self, monkeypatch, tmp_path, timeout):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CARTRIDGE_WANDB_INIT_TIMEOUT_SECONDS", timeout)
        reset_config()

        with pytest.raises(ValueError, match="wandb.init_timeout_seconds"):
            get_config(reload=True)

    def test_evaluation_temperature_env_override(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CARTRIDGE_EVALUATION_TEMPERATURE", "0.35")
        reset_config()

        assert get_config(reload=True).evaluation.temperature == pytest.approx(0.35)

    @pytest.mark.parametrize(
        ("env_id", "solver_games", "promotion_metric", "message"),
        [
            ("tictactoe", "1", "win_rate", "solver_games"),
            ("connect4", "0", "solver_optimal", "solver_optimal"),
            ("tictactoe", "1", "solver_optimal", "solver_games"),
        ],
    )
    def test_ineffective_solver_configuration_is_rejected(
        self,
        monkeypatch,
        tmp_path,
        env_id,
        solver_games,
        promotion_metric,
        message,
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", env_id)
        monkeypatch.setenv("CARTRIDGE_EVALUATION_SOLVER_GAMES", solver_games)
        monkeypatch.setenv("CARTRIDGE_EVALUATION_PROMOTION_METRIC", promotion_metric)
        if promotion_metric == "solver_optimal":
            monkeypatch.setenv("CARTRIDGE_EVALUATION_WIN_THRESHOLD", "0")
        reset_config()

        with pytest.raises(ValueError, match=message):
            get_config(reload=True)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: EvaluationConfig(temperature=float("nan")), "temperature"),
        (lambda: EvaluationConfig(temperature=float("inf")), "temperature"),
        (lambda: EvaluationConfig(promotion_margin=0.1), "promotion_margin"),
        (
            lambda: EvaluationConfig(promotion_metric="solver_optimal", solver_games=1),
            "win_threshold",
        ),
        (lambda: MctsConfig(c_puct=float("nan")), "c_puct"),
        (lambda: MctsConfig(late_temperature=float("inf")), "late_temperature"),
        (lambda: MctsConfig(dirichlet_weight=1.01), "dirichlet_weight"),
        (lambda: MctsConfig(start_sims=0), "start_sims"),
        (lambda: MctsConfig(start_sims=5, max_sims=4), "cannot exceed"),
    ],
)
def test_search_numeric_contract_rejects_invalid_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("alpha", "weight"),
    [(0.0, 0.25), (0.3, 0.0)],
)
def test_dirichlet_noise_cannot_be_half_disabled(alpha, weight):
    with pytest.raises(ValueError, match="must both be zero"):
        MctsConfig(dirichlet_alpha=alpha, dirichlet_weight=weight)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "CARTRIDGE_MCTS_MAX_SIMS": "50",
        },
        {
            "CARTRIDGE_MCTS_MAX_SIMS": "60",
            "CARTRIDGE_MCTS_SIM_RAMP_RATE": "11",
        },
        {
            "CARTRIDGE_TRAINING_ITERATIONS": "2",
            "CARTRIDGE_MCTS_MAX_SIMS": "60",
            "CARTRIDGE_MCTS_SIM_RAMP_RATE": "5",
        },
    ],
)
def test_noncanonical_mcts_schedules_are_rejected(monkeypatch, tmp_path, overrides):
    monkeypatch.chdir(tmp_path)
    reset_config()
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="MCTS|mcts"):
        get_config(reload=True)


def test_unreachable_temperature_threshold_is_rejected(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reset_config()
    monkeypatch.setenv("CARTRIDGE_COMMON_ENV_ID", "tictactoe")
    monkeypatch.setenv("CARTRIDGE_MCTS_TEMP_THRESHOLD", "9")
    monkeypatch.setenv("CARTRIDGE_MCTS_LATE_TEMPERATURE", "0.5")

    with pytest.raises(ValueError, match="max_horizon"):
        get_config(reload=True)


def test_shared_integer_widths_accept_values_above_i32():
    value = 1 << 31

    TrainingConfig(
        iterations=value,
        episodes_per_iteration=value,
        steps_per_iteration=value,
        batch_size=value,
        checkpoint_interval=value,
        num_actors=value,
    )
    EvaluationConfig(interval=value, games=value, solver_games=value)
    ActorConfig(episode_timeout_secs=value, log_interval=value)
    MctsConfig(
        late_temperature=0.5,
        temp_threshold=value,
        start_sims=value,
        max_sims=value,
        sim_ramp_rate=0,
        eval_batch_size=value,
        onnx_intra_threads=value,
    )


@pytest.mark.parametrize(
    ("factory", "width"),
    [
        (lambda: TrainingConfig(iterations=1 << 64), "u64"),
        (lambda: TrainingConfig(episodes_per_iteration=1 << 32), "u32"),
        (lambda: EvaluationConfig(interval=1 << 64), "u64"),
        (lambda: EvaluationConfig(games=1 << 32), "u32"),
        (lambda: ActorConfig(episode_timeout_secs=1 << 64), "u64"),
        (lambda: ActorConfig(log_interval=1 << 32), "u32"),
        (lambda: MctsConfig(eval_batch_size=1 << 32), "u32"),
    ],
)
def test_shared_integer_widths_reject_values_past_wire_bounds(factory, width):
    with pytest.raises(ValueError, match=width):
        factory()
