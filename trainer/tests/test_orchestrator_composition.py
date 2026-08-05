"""Composition tests: Cartridge2's orchestrator wiring into crucible.

The orchestrator loop and its eval/config/actor modules live in
crucible with injected seams; trainer.orchestrator is this repo's
composition root. Each test here constructs through THIS repo's production
wiring and pins behavior that crucible deliberately does not provide.
They would stay green against crucible alone only if the composition
root silently lost its wiring -- which is exactly the regression they exist
to catch.
"""

import json
import tempfile

import pytest
import torch
from crucible.orchestrator.orchestrator import TrainSpec
from torch.optim import Adam

from trainer import trainer as trainer_mod
from trainer.algorithms import get_algorithm
from trainer.algorithms.alphazero_board_v1 import (
    ALGORITHM_ID,
    DESCRIPTOR,
    get_game_config,
    policy_value_artifact_contract,
)
from trainer.algorithms.alphazero_config import AlphaZeroLearnerConfig
from trainer.central_config import WandbConfig
from trainer.checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    learner_config_sha256,
    write_learner_state_artifact,
)
from trainer.environment_catalog import get_environment
from trainer.network import create_network
from trainer.orchestrator import eval_runner as eval_runner_module
from trainer.orchestrator import orchestrator as orchestrator_module
from trainer.orchestrator.actor_runner import _PROJECT_ROOT
from trainer.orchestrator.actor_runner import ActorRunner as ShimActorRunner
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.orchestrator import Orchestrator
from trainer.solver_eval import SolverEvalResults, judge_move
from trainer.stats import TrainerStats, prepare_stats_snapshot
from trainer.storage import ReplayProfile, ReplaySelection
from trainer.storage.publisher import (
    ArtifactValidationError,
    FilesystemCheckpointPublisher,
)
from trainer.structured_logging import get_trace_context


class StubPlayer:
    """Stand-in for the ModelPlayer built by the policy loader."""

    def __init__(self, model_path, temperature=0.0, simulations=0):
        self.model_path = str(model_path)
        self.temperature = temperature
        self.simulations = simulations

    @property
    def name(self):
        return f"Stub({self.model_path})"

    def cli_args(self, slot):
        return [f"--{slot}", self.model_path]


class FakeReplayStore:
    """Replay-store stand-in for the composition root's factory seam."""

    def __init__(
        self,
        selection: ReplaySelection,
        records: int = 0,
        episodes: int = 0,
    ):
        self.selection = selection
        self.records = records
        self.episodes = episodes
        self.close_calls = 0

    def clear(self) -> int:
        self.clear_calls += 1
        return 0

    def vacuum(self) -> None:
        self.vacuum_calls += 1

    def count(self) -> int:
        return self.records

    def count_episodes(self) -> int:
        return self.episodes

    def close(self) -> None:
        self.close_calls += 1


def replay_profile(env_id: str) -> ReplayProfile:
    return ReplayProfile(
        env_id=env_id,
        env_contract_version=2,
        algorithm_id=ALGORITHM_ID,
        experience_schema=DESCRIPTOR.components.experience_schema,
    )


def make_stub_learner(monkeypatch):
    """Replace ``AlphaZeroLearner`` at the algorithm factory's outer seam.

    The algorithm implementation resolves the learner at call time, so
    patching the source module stubs every learner the loop builds.
    Returns the list the stubs are recorded into.
    """
    built = []

    class StubLearner:
        def __init__(self, config):
            self.config = config
            self.last_checkpoint_ref = None
            self.last_prepared_stats = None
            built.append(self)

        def train(self):
            game = get_game_config(self.config.env_id)
            environment = get_environment(self.config.env_id)
            contract = policy_value_artifact_contract(
                algorithm_id=ALGORITHM_ID,
                env_id=self.config.env_id,
                env_contract_version=environment.contract_version,
                model_artifact_schema_version=DESCRIPTOR.model_artifact_schema_version,
                model_contract=DESCRIPTOR.components.model_contract,
                obs_size=game.obs_size,
                num_actions=game.num_actions,
            )
            network = create_network(self.config.env_id, config=game)
            optimizer = Adam(network.parameters(), lr=0.001)
            publisher = FilesystemCheckpointPublisher(
                model_root=self.config.model_dir, contract=contract
            )
            parent = publisher.resolve_head()
            step = self.config.start_step + self.config.total_steps
            config_sha256 = learner_config_sha256(self.config)
            with tempfile.TemporaryDirectory() as staging:
                onnx_path = export_onnx_artifact(
                    network,
                    f"{staging}/model.onnx",
                    torch.device("cpu"),
                    contract,
                )
                learner_path = write_learner_state_artifact(
                    network,
                    optimizer,
                    step,
                    f"{staging}/learner.pt",
                    contract,
                    config_sha256,
                )
                self.last_checkpoint_ref = publisher.stage_checkpoint(
                    onnx_path,
                    learner_path,
                    step=step,
                    parent_checkpoint_id=(parent.checkpoint_id if parent is not None else None),
                    config_sha256=config_sha256,
                    learner_state_contract=LearnerStateContract(network, optimizer),
                )
            stats = TrainerStats(
                step=step,
                total_steps=step,
                samples_seen=step * self.config.batch_size,
                metrics={"loss/total": 0.25},
                last_checkpoint=self.last_checkpoint_ref.checkpoint_id,
                env_id=self.config.env_id,
            )
            self.last_prepared_stats = prepare_stats_snapshot(stats, self.last_checkpoint_ref)
            return stats

    monkeypatch.setattr(trainer_mod, "AlphaZeroLearner", StubLearner)
    return built


def make_solver_results(value_optimal_rate: float, positions: int = 10) -> SolverEvalResults:
    """A real SolverEvalResults with a chosen overall value-optimal rate."""
    results = SolverEvalResults(
        env_id="connect4",
        model_name="Stub(immutable-checkpoint)",
        model_path=f"blobs/sha256/{'0' * 64}.onnx",
        checkpoint_id=None,
        step=None,
        opponent_name="Random",
        games=6,
        seed=42,
        temperature=0.0,
    )
    optimal = round(positions * value_optimal_rate)
    for i in range(positions):
        if i < optimal:
            judgment = judge_move({3: 5, 0: -1}, chosen=3)  # optimal
        else:
            judgment = judge_move({3: 5, 0: -1}, chosen=0)  # blunder
        results.overall.add(judgment)
        results.by_ply["ply_1_8"].add(judgment)
        results.by_seat["first" if i % 2 == 0 else "second"].add(judgment)
    results.model_wins = 4
    results.model_losses = 2
    results.solver_queries = positions
    results.timestamp = "2026-07-03T00:00:00"
    return results


def publish_current_checkpoint(
    config: LoopConfig, *, step: int = 3, parent_checkpoint_id: str | None = None
):
    game = get_game_config(config.env_id)
    environment = get_environment(config.env_id)
    contract = policy_value_artifact_contract(
        algorithm_id=ALGORITHM_ID,
        env_id=config.env_id,
        env_contract_version=environment.contract_version,
        model_artifact_schema_version=DESCRIPTOR.model_artifact_schema_version,
        model_contract=DESCRIPTOR.components.model_contract,
        obs_size=game.obs_size,
        num_actions=game.num_actions,
    )
    network = create_network(config.env_id, config=game)
    optimizer = Adam(network.parameters(), lr=0.001)
    with tempfile.TemporaryDirectory() as staging:
        onnx_path = export_onnx_artifact(
            network,
            f"{staging}/model.onnx",
            torch.device("cpu"),
            contract,
        )
        learner_path = write_learner_state_artifact(
            network,
            optimizer,
            step,
            f"{staging}/learner.pt",
            contract,
            "a" * 64,
        )
        return FilesystemCheckpointPublisher(
            model_root=config.models_dir, contract=contract
        ).stage_checkpoint(
            onnx_path,
            learner_path,
            step=step,
            parent_checkpoint_id=parent_checkpoint_id,
            config_sha256="a" * 64,
            learner_state_contract=LearnerStateContract(network, optimizer),
        )


class TestEvalRunnerComposition:
    def test_clean_eval_api_is_wired(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            orchestrator_module,
            "create_replay_store",
            lambda selection: FakeReplayStore(selection),
        )
        runner = Orchestrator(LoopConfig(data_dir=tmp_path, env_id="tictactoe")).eval_runner

        assert type(runner) is eval_runner_module.EvalRunner
        assert callable(runner.prepare)
        assert callable(runner.commit)
        assert not hasattr(runner, "run")


class TestAlgorithmLearnerFactory:
    def test_build_loop_learner_maps_spec_field_for_field(self, monkeypatch):
        """The selected algorithm converts ``TrainSpec`` without losing fields."""
        built = make_stub_learner(monkeypatch)

        def hook(payload, step):
            return None

        def check():
            return False

        spec = TrainSpec(
            model_dir="m",
            stats_path="s",
            env_id="connect4",
            total_steps=7,
            start_step=3,
            batch_size=16,
            learning_rate=0.01,
            checkpoint_interval=9,
            device="cpu",
            max_wait=12.5,
            eval_interval=0,
            lr_total_steps=70,
            shutdown_check=check,
            metrics_hook=hook,
        )

        loop_config = LoopConfig(
            weight_decay=0.02,
            grad_clip_norm=3.5,
        )
        learner = get_algorithm(ALGORITHM_ID).build_loop_learner(spec, loop_config)

        assert built == [learner]
        tc = learner.config
        assert isinstance(tc, AlphaZeroLearnerConfig)
        assert (tc.model_dir, tc.stats_path, tc.env_id) == ("m", "s", "connect4")
        assert (tc.total_steps, tc.start_step) == (7, 3)
        assert (tc.batch_size, tc.learning_rate) == (16, 0.01)
        assert (tc.weight_decay, tc.grad_clip_norm) == (0.02, 3.5)
        assert (tc.checkpoint_interval, tc.device) == (9, "cpu")
        assert (tc.max_wait, tc.lr_total_steps) == (12.5, 70)
        assert tc.shutdown_check is check
        assert tc.metrics_hook is hook
        assert tc.defer_run_commit is True


class TestTraceComposition:
    def test_trace_starter_sets_context_and_returns_id(self):
        """The trace seam creates a fresh 32-hex trace
        id is returned AND installed (with a 16-hex span) in this repo's
        structured-logging context."""
        trace_id = orchestrator_module._start_iteration_trace()

        assert len(trace_id) == 32
        ctx = get_trace_context()
        assert ctx["trace_id"] == trace_id
        assert len(ctx["span_id"]) == 16


class TestOrchestratorComposition:
    def test_stale_projection_without_head_is_discarded(self, tmp_path, monkeypatch):
        config = LoopConfig(data_dir=tmp_path / "data", env_id="tictactoe")
        config.loop_stats_path.parent.mkdir(parents=True, exist_ok=True)
        config.loop_stats_path.write_text("{corrupt")
        monkeypatch.setattr(
            orchestrator_module,
            "create_replay_store",
            lambda selection: FakeReplayStore(selection),
        )

        Orchestrator(config)

        assert not config.loop_stats_path.exists()

    def test_unknown_algorithm_fails_before_replay_is_opened(self, tmp_path, monkeypatch):
        def unexpected_replay_open(profile):
            raise AssertionError("replay storage must not open before algorithm preflight")

        monkeypatch.setattr(orchestrator_module, "create_replay_store", unexpected_replay_open)

        with pytest.raises(ValueError, match="Unknown algorithm"):
            Orchestrator(
                LoopConfig(
                    data_dir=tmp_path,
                    env_id="connect4",
                    algorithm_id="missing_algorithm",
                )
            )

    def test_wired_orchestrator_runs_one_iteration(self, tmp_path, monkeypatch):
        """The full production object graph assembles and one iteration runs.

        All factories are the composition root's real ones; stubs sit only
        at the outermost seams: the replay buffer (module global), the
        algorithm's AlphaZero learner, a touched-but-never-executed actor
        binary with zero episodes, and W&B on its enabled=False null
        logger. No actor binaries run, no PostgreSQL, no network.
        """
        buffers: list[FakeReplayStore] = []

        def fake_replay_factory(selection):
            assert selection.profile == replay_profile("tictactoe")
            buffer = FakeReplayStore(selection)
            buffers.append(buffer)
            return buffer

        monkeypatch.setattr(orchestrator_module, "create_replay_store", fake_replay_factory)
        reaped: list[tuple] = []
        monkeypatch.setattr(
            orchestrator_module,
            "reap_profile_scopes",
            lambda *, profile, retained_scopes: reaped.append((profile, retained_scopes)) or 0,
        )
        built = make_stub_learner(monkeypatch)

        fake_binary = tmp_path / "actor-stub.exe"
        fake_binary.touch()
        config = LoopConfig(
            iterations=1,
            episodes_per_iteration=1,
            steps_per_iteration=2,
            data_dir=tmp_path / "data",
            env_id="tictactoe",
            actor_binary=fake_binary,  # must exist; never executed
            mcts_max_sims=50,
            mcts_sim_ramp_rate=0,
            eval_interval=0,
        )

        def run_actor(self, *args, **kwargs):
            buffers[-1].records = 5
            buffers[-1].episodes = 1
            return True, 0.0

        monkeypatch.setattr(ShimActorRunner, "run", run_actor)
        orchestrator = Orchestrator(config)

        # Production graph: Cartridge2's bindings, not fakes or core classes.
        assert orchestrator.algorithm.descriptor.id == ALGORITHM_ID
        assert type(orchestrator.actor_runner) is ShimActorRunner
        assert type(orchestrator.eval_runner) is eval_runner_module.EvalRunner

        orchestrator.run()

        # One iteration completed end to end and was persisted.
        assert [s.iteration for s in orchestrator.iteration_history] == [1]
        stats = orchestrator.iteration_history[0]
        assert stats.transitions_generated == 5
        assert stats.eval_win_rate is None  # eval ran without a model
        assert len(buffers) == 2
        assert all(buffer.close_calls == 1 for buffer in buffers)
        with open(config.loop_stats_path) as f:
            saved = json.load(f)
        assert [it["iteration"] for it in saved["iterations"]] == [1]

        # The algorithm-owned TrainSpec conversion carried the loop's values.
        (learner,) = built
        tc = learner.config
        assert isinstance(tc, AlphaZeroLearnerConfig)
        assert tc.model_dir == str(config.models_dir)
        assert tc.stats_path == str(config.stats_path)
        assert tc.env_id == "tictactoe"
        assert tc.total_steps == 2
        assert tc.start_step == 0
        assert tc.max_wait == 60.0
        assert tc.lr_total_steps == 2  # iterations * steps_per_iteration
        assert tc.weight_decay == config.weight_decay
        assert tc.grad_clip_norm == config.grad_clip_norm
        assert tc.shutdown_check() is False
        assert tc.metrics_hook is not None
        assert tc.replay_selection == buffers[-1].selection
        assert buffers[0].selection.collection_scope_id != buffers[1].selection.collection_scope_id
        head = orchestrator.eval_runner.checkpoints.resolve_run_head()
        assert head is not None
        committed = orchestrator.run_commits.resolve(head.run_commit_id).commit
        assert committed.orchestration is not None
        assert committed.orchestration.collection_scope_id == (
            buffers[-1].selection.collection_scope_id
        )
        assert committed.orchestration.source_checkpoint_id is None
        # The reaper ran exactly once, after the commit, over this profile.
        assert reaped == [(replay_profile("tictactoe"), 2)]

    def test_exact_episode_seal_aborts_before_learning_or_commit(self, tmp_path, monkeypatch):
        buffers: list[FakeReplayStore] = []

        def fake_replay_factory(selection):
            buffer = FakeReplayStore(selection)
            buffers.append(buffer)
            return buffer

        monkeypatch.setattr(orchestrator_module, "create_replay_store", fake_replay_factory)
        built = make_stub_learner(monkeypatch)
        monkeypatch.setattr(
            ShimActorRunner,
            "run",
            lambda self, *args, **kwargs: (True, 0.0),
        )
        config = LoopConfig(
            iterations=1,
            episodes_per_iteration=2,
            data_dir=tmp_path,
            env_id="tictactoe",
            mcts_max_sims=50,
            mcts_sim_ramp_rate=0,
            eval_interval=0,
        )
        orchestrator = Orchestrator(config)

        with pytest.raises(ArtifactValidationError, match="exact configured episode"):
            orchestrator.run_iteration(1)

        assert built == []
        assert orchestrator.eval_runner.checkpoints.resolve_run_head() is None
        assert not (config.models_dir / "run-preparations").exists()
        assert not (config.models_dir / "run-commits").exists()
        orchestrator._replay_buffer.close()


class TestLoopConfigComposition:
    def test_algorithm_is_explicit_in_the_loop_request(self):
        assert LoopConfig().algorithm_id == ALGORITHM_ID

    def test_wandb_default_is_real_wandb_config(self):
        """The host config supplies this repo's WandbConfig default.

        crucible's LoopConfig defaults ``wandb`` to None; every
        Cartridge2 caller relies on getting a ready-to-use WandbConfig.
        """
        assert isinstance(LoopConfig().wandb, WandbConfig)

    def test_authenticated_fields_are_immutable_after_construction(self):
        config = LoopConfig()

        with pytest.raises(AttributeError, match="cannot mutate authenticated"):
            config.c_puct = 2.0

        config._set_start_iteration(2)
        assert config.start_iteration == 2

    @pytest.mark.parametrize("metrics_port", [0, 65536, True])
    def test_metrics_port_must_be_a_real_tcp_port(self, metrics_port):
        with pytest.raises(ValueError, match="metrics_port"):
            LoopConfig(metrics_port=metrics_port)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"data_dir": "./data"}, "data_dir"),
            ({"actor_binary": "actor"}, "actor_binary"),
            ({"actor_log_interval": -1}, "actor_log_interval"),
            ({"actor_episode_timeout_seconds": 0}, "actor_episode_timeout_seconds"),
            ({"actor_eval_batch_size": 0}, "actor_eval_batch_size"),
            ({"actor_onnx_intra_threads": 0}, "actor_onnx_intra_threads"),
            ({"iterations": 0}, "iterations"),
            ({"episodes_per_iteration": 0}, "episodes_per_iteration"),
            ({"steps_per_iteration": 0}, "steps_per_iteration"),
            ({"num_actors": 2, "episodes_per_iteration": 1}, "num_actors"),
            ({"batch_size": 0}, "batch_size"),
            ({"learning_rate": float("nan")}, "learning_rate"),
            ({"weight_decay": -0.1}, "weight_decay"),
            ({"grad_clip_norm": float("inf")}, "grad_clip_norm"),
            ({"eval_interval": -1}, "eval_interval"),
            ({"eval_games": 0}, "eval_games"),
            ({"eval_vs_random": 1}, "eval_vs_random"),
            ({"eval_vs_random": False}, "first-candidate evidence"),
            ({"evaluation_seed": 1 << 64}, "evaluation_seed"),
            ({"device": "tpu"}, "device"),
            ({"log_level": "TRACE"}, "log_level"),
            ({"wandb": None}, "wandb"),
            ({"c_puct": float("nan")}, "c_puct"),
            ({"dirichlet_weight": 1.1}, "dirichlet_weight"),
            ({"dirichlet_alpha": 0.0}, "both be zero"),
            ({"eval_temperature": float("inf")}, "eval_temperature"),
            ({"temp_threshold": 1}, "must differ"),
            (
                {"mcts_max_sims": 50, "mcts_sim_ramp_rate": 1},
                "must be zero",
            ),
            (
                {"mcts_start_sims": 50, "mcts_max_sims": 60, "mcts_sim_ramp_rate": 11},
                "ramped MCTS",
            ),
            (
                {
                    "iterations": 2,
                    "mcts_start_sims": 50,
                    "mcts_max_sims": 60,
                    "mcts_sim_ramp_rate": 5,
                },
                "must reach",
            ),
            (
                {
                    "env_id": "tictactoe",
                    "temp_threshold": 9,
                    "late_temperature": 0.5,
                },
                "max_horizon",
            ),
            ({"promotion_margin": 0.1}, "promotion_margin"),
            (
                {
                    "env_id": "connect4",
                    "solver_games": 1,
                    "promotion_metric": "solver_optimal",
                },
                "eval_win_threshold",
            ),
        ],
    )
    def test_operational_and_search_values_fail_at_config_construction(self, overrides, message):
        with pytest.raises((TypeError, ValueError), match=message):
            LoopConfig(**overrides)

    def test_evidence_source_is_optional_only_when_evaluation_is_disabled(self):
        config = LoopConfig(eval_interval=0, eval_vs_random=False, solver_games=0)

        assert config.eval_interval == 0
        assert config.eval_vs_random is False
        assert config.solver_games == 0

    def test_connect4_solver_is_a_valid_first_candidate_evidence_source(self):
        config = LoopConfig(
            env_id="connect4",
            eval_vs_random=False,
            solver_games=1,
        )

        assert config.eval_interval > 0
        assert config.eval_vs_random is False
        assert config.solver_games == 1

    def test_run_recipe_authenticates_every_collector_search_knob(self):
        config = LoopConfig(
            c_puct=1.75,
            temperature=0.9,
            late_temperature=0.15,
            temp_threshold=5,
            dirichlet_alpha=0.45,
            dirichlet_weight=0.3,
        )

        recipe = orchestrator_module._run_recipe(config, get_algorithm(config.algorithm_id))

        assert recipe.collector_c_puct == pytest.approx(1.75)
        assert recipe.collector_temperature == pytest.approx(0.9)
        assert recipe.collector_late_temperature == pytest.approx(0.15)
        assert recipe.collector_dirichlet_alpha == pytest.approx(0.45)
        assert recipe.collector_dirichlet_weight == pytest.approx(0.3)


class TestActorRunnerComposition:
    def test_project_root_resolves_to_repo_root(self):
        """Shape-pin the runner's Path(__file__).parents[4] arithmetic.

        The injected cargo-target candidates are computed relative to
        _PROJECT_ROOT; if the runner module ever moves in the tree, this
        catches the stale parents[] math before binary discovery breaks.
        """
        assert (_PROJECT_ROOT / "trainer").is_dir()

    def test_collector_command_carries_the_selected_algorithm(self, tmp_path):
        binary = tmp_path / "actor"
        binary.touch()
        config = LoopConfig(
            data_dir=tmp_path / "data",
            env_id="connect4",
            algorithm_id=ALGORITHM_ID,
            actor_binary=binary,
            c_puct=1.75,
            temperature=0.9,
            late_temperature=0.15,
            temp_threshold=15,
            dirichlet_alpha=0.45,
            dirichlet_weight=0.3,
        )
        algorithm = get_algorithm(ALGORITHM_ID)
        runner = ShimActorRunner(config, algorithm.collector_config)
        selection = ReplaySelection(
            replay_profile("connect4"),
            "a" * 64,
            "b" * 64,
        )
        runner.select_replay(selection)

        command = runner._build_command(
            binary,
            actor_id="actor-1",
            num_episodes=3,
            num_simulations=25,
        )

        assert command[1:3] == ["--algorithm", ALGORITHM_ID]
        assert "--no-watch" not in command
        encoded = command[command.index("--collector-config") + 1]
        assert json.loads(encoded) == algorithm.collector_config(config, 25)
        assert "--num-simulations" not in command
        assert "--c-puct" not in command
        assert command[command.index("--log-level") + 1] == config.log_level.lower()
        assert command[-4:] == [
            "--collection-scope-id",
            selection.collection_scope_id,
            "--source-checkpoint-id",
            selection.source_checkpoint_id,
        ]

    def test_root_collector_command_omits_source_checkpoint_flag(self, tmp_path):
        binary = tmp_path / "actor"
        binary.touch()
        config = LoopConfig(data_dir=tmp_path, env_id="tictactoe")
        runner = ShimActorRunner(
            config,
            get_algorithm(ALGORITHM_ID).collector_config,
        )
        selection = ReplaySelection(replay_profile("tictactoe"), "c" * 64, None)
        runner.select_replay(selection)

        command = runner._build_command(binary, "actor-1", 1, 5)

        assert "--collection-scope-id" in command
        assert "--source-checkpoint-id" not in command
