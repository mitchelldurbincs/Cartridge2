"""Vertical contract tests for the DQN counter cartridge."""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from trainer import __main__ as cli
from trainer.algorithms import get_algorithm
from trainer.algorithms.dqn_config import DqnLearnerConfig
from trainer.algorithms.dqn_learner import DqnLearner, DqnQNetwork
from trainer.algorithms.dqn_loop import DqnLoop, DqnLoopConfig
from trainer.algorithms.dqn_requests import DqnCollectRequest
from trainer.algorithms.dqn_v1 import (
    ALGORITHM_ID,
    DqnEvaluationResults,
    decode_replay_batch,
)
from trainer.checkpoint import export_onnx_artifact
from trainer.environment_catalog import get_environment
from trainer.storage import ReplayProfile, ReplaySelection


def selection() -> ReplaySelection:
    environment = get_environment("counter")
    descriptor = get_algorithm(ALGORITHM_ID).descriptor
    return ReplaySelection(
        profile=ReplayProfile(
            env_id=environment.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=ALGORITHM_ID,
            experience_schema=descriptor.components.experience_schema,
        ),
        collection_scope_id="a" * 64,
        source_checkpoint_id=None,
    )


def payload(
    observation=(0.0, 1.0),
    action=1,
    reward=-0.01,
    next_observation=(1 / 3, 7 / 8),
    terminated=False,
    truncated=False,
    availability=(1, 1),
) -> bytes:
    return b"".join(
        [
            struct.pack("<2f", *observation),
            struct.pack("<I", action),
            struct.pack("<f", reward),
            struct.pack("<2f", *next_observation),
            bytes((terminated, truncated)),
            bytes(availability),
        ]
    )


def test_counter_is_the_first_dqn_compatible_environment():
    environment = get_environment("counter")
    assert environment.compatibility(ALGORITHM_ID).compatible
    assert not get_environment("tictactoe").compatibility(ALGORITHM_ID).compatible
    contract = get_algorithm(ALGORITHM_ID).artifact_contract(environment)
    assert contract.input("observation").shape == ("batch_size", 2)
    assert contract.output("q_values").shape == ("batch_size", 2)


def test_dqn_replay_codec_preserves_immediate_transition_semantics():
    selected = selection()
    record = selected.record(
        id="transition-0",
        episode_id="episode-0",
        step_number=0,
        payload=payload(),
    )
    batch = decode_replay_batch([record], selection=selected, obs_size=2, num_actions=2)
    np.testing.assert_allclose(batch.observations, [[0.0, 1.0]])
    np.testing.assert_allclose(batch.next_observations, [[1 / 3, 7 / 8]])
    assert batch.actions.tolist() == [1]
    assert batch.rewards[0] == np.float32(-0.01)
    assert batch.next_availability.tolist() == [[True, True]]
    assert not batch.terminated[0]
    assert not batch.truncated[0]


def test_q_network_exports_and_executes_its_declared_single_output(tmp_path):
    environment = get_environment("counter")
    contract = get_algorithm(ALGORITHM_ID).artifact_contract(environment)
    network = DqnQNetwork(2, 2, 16)
    path = export_onnx_artifact(
        network,
        tmp_path / "q-values.onnx",
        torch.device("cpu"),
        contract,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    observation = np.asarray([[0.0, 1.0]], dtype=np.float32)
    [actual] = session.run(["q_values"], {"observation": observation})
    with torch.no_grad():
        expected = network(torch.from_numpy(observation)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_dqn_train_step_updates_online_q_parameters(tmp_path):
    selected = selection()
    config = DqnLearnerConfig(
        env_id="counter",
        model_dir=str(tmp_path / "models"),
        stats_path=str(tmp_path / "stats.json"),
        total_steps=1,
        batch_size=1,
        hidden_size=16,
        replay_selection=selected,
    )
    learner = DqnLearner(config)
    record = selected.record(
        id="transition-0",
        episode_id="episode-0",
        step_number=0,
        payload=payload(terminated=True, availability=(0, 0)),
    )
    batch = decode_replay_batch([record], selection=selected, obs_size=2, num_actions=2)
    before = [parameter.detach().clone() for parameter in learner.network.online.parameters()]
    loss, _ = learner.train_step(batch)
    assert np.isfinite(loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, learner.network.online.parameters(), strict=True)
    )


def test_dqn_learner_publishes_a_q_value_run_head(tmp_path, monkeypatch):
    selected = selection()
    records = [
        selected.record(
            id="transition-running",
            episode_id="episode-0",
            step_number=0,
            payload=payload(),
        ),
        selected.record(
            id="transition-terminal",
            episode_id="episode-0",
            step_number=1,
            payload=payload(
                observation=(2 / 3, 6 / 8),
                reward=1.0,
                next_observation=(1.0, 5 / 8),
                terminated=True,
                availability=(0, 0),
            ),
        ),
    ]

    class MemoryReplay:
        def count(self):
            return len(records)

        def sample(self, batch_size):
            return [records[index % len(records)] for index in range(batch_size)]

        def close(self):
            pass

    monkeypatch.setattr(
        "trainer.algorithms.dqn_learner.create_replay_store",
        lambda replay_selection: MemoryReplay(),
    )
    config = DqnLearnerConfig(
        env_id="counter",
        model_dir=str(tmp_path / "models"),
        stats_path=str(tmp_path / "stats.json"),
        total_steps=2,
        batch_size=2,
        target_sync_interval=1,
        hidden_size=16,
        replay_selection=selected,
    )
    learner = DqnLearner(config)
    stats = learner.train()

    head = learner.checkpoints.resolve_run_head()
    assert head is not None
    checkpoint = learner.checkpoints.resolve_checkpoint(head.checkpoint_id)
    assert checkpoint.manifest.profile.model_contract == "onnx_q_values_v1"
    assert stats.samples_seen == 4
    assert stats.last_checkpoint == head.checkpoint_id
    assert (tmp_path / "stats.json").is_file()

    first_checkpoint_id = head.checkpoint_id
    selected = ReplaySelection(
        profile=selected.profile,
        collection_scope_id="c" * 64,
        source_checkpoint_id=first_checkpoint_id,
    )
    records = [
        selected.record(
            id="descendant-terminal",
            episode_id="episode-1",
            step_number=0,
            payload=payload(terminated=True, reward=1.0, availability=(0, 0)),
        )
    ]
    resumed = DqnLearner(
        DqnLearnerConfig(
            env_id="counter",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=2,
            batch_size=2,
            target_sync_interval=1,
            hidden_size=16,
            replay_selection=selected,
        )
    )
    resumed_stats = resumed.train()
    resumed_head = resumed.checkpoints.resolve_run_head()
    assert resumed_head is not None
    assert resumed_head.checkpoint_id != first_checkpoint_id
    resumed_checkpoint = resumed.checkpoints.resolve_checkpoint(resumed_head.checkpoint_id)
    assert resumed_checkpoint.manifest.parent_checkpoint_id == first_checkpoint_id
    assert resumed_checkpoint.manifest.step == 4
    assert resumed_stats.samples_seen == 8


def test_dqn_owns_a_clean_collect_and_train_cli_surface():
    algorithm = get_algorithm(ALGORITHM_ID)
    assert [command.name for command in algorithm.commands()] == [
        "collect",
        "train",
        "evaluate",
        "loop",
    ]
    parser = cli.build_parser(algorithm)
    args = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "train",
            "--env-id",
            "counter",
            "--collection-scope-id",
            "b" * 64,
            "--source-root",
            "--steps",
            "7",
        ]
    )
    assert args.steps == 7

    resumed = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "train",
            "--env-id",
            "counter",
            "--collection-scope-id",
            "c" * 64,
            "--source-checkpoint-id",
            "d" * 64,
        ]
    )
    assert resumed.source_checkpoint_id == "d" * 64

    loop = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "loop",
            "--iterations",
            "3",
            "--episodes-per-iteration",
            "20",
            "--steps-per-iteration",
            "50",
        ]
    )
    assert loop.iterations == 3
    assert loop.episodes_per_iteration == 20
    assert loop.steps_per_iteration == 50
    assert not hasattr(loop, "model_dir")
    assert not hasattr(loop, "stats_path")


def test_dqn_cli_defaults_are_stable():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))
    collect = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "collect",
            "--episodes",
            "1",
            "--collection-scope-id",
            "a" * 64,
            "--source-root",
        ]
    )
    assert collect.env_id == "counter"
    assert collect.epsilon == 1.0
    assert collect.seed == 0
    assert collect.onnx_intra_threads == 1
    assert collect.actor_id == "dqn-collector"
    assert collect.episode_timeout_secs == 30
    assert collect.log_level == "INFO"

    evaluate = parser.parse_args(["--algorithm", ALGORITHM_ID, "evaluate", "--random"])
    assert evaluate.env_id == "counter"
    assert evaluate.episodes == 100
    assert evaluate.seed == 42
    assert evaluate.onnx_intra_threads == 1

    loop = parser.parse_args(["--algorithm", ALGORITHM_ID, "loop", "--iterations", "1"])
    assert loop.env_id == "counter"
    assert loop.episodes_per_iteration == 100
    assert loop.evaluation_episodes == 100
    assert loop.episode_timeout_secs == 30
    assert loop.log_level == "INFO"


def test_typed_collect_request_preserves_subprocess_arguments_and_exit_code(tmp_path, monkeypatch):
    actor_binary = tmp_path / "actor"
    actor_binary.touch()
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("trainer.algorithms.dqn_application.subprocess.run", run)
    request = DqnCollectRequest(
        env_id="counter",
        episodes=3,
        collection_scope_id="a" * 64,
        source_checkpoint_id="b" * 64,
        epsilon=0.25,
        seed=9,
        onnx_intra_threads=2,
        actor_id="collector-9",
        episode_timeout_secs=17,
        actor_binary=actor_binary,
        data_root=tmp_path / "data",
        log_level="WARNING",
    )

    assert get_algorithm(ALGORITHM_ID).collect(request) == 7
    assert calls == [
        (
            [
                str(actor_binary),
                "--algorithm",
                ALGORITHM_ID,
                "--env-id",
                "counter",
                "--max-episodes",
                "3",
                "--collection-scope-id",
                "a" * 64,
                "--collector-config",
                '{"epsilon":0.25,"onnx_intra_threads":2,"schema_version":1,"seed":9}',
                "--actor-id",
                "collector-9",
                "--episode-timeout-secs",
                "17",
                "--data-dir",
                str(tmp_path / "data"),
                "--log-level",
                "warning",
                "--source-checkpoint-id",
                "b" * 64,
            ],
            {"check": False},
        )
    ]


def test_dqn_loop_calls_typed_application_methods_and_honors_runtime_options(
    tmp_path, monkeypatch, capsys
):
    cartridge = get_algorithm(ALGORITHM_ID)
    requests = {"collect": [], "train": [], "evaluate": []}

    class Checkpoints:
        calls = 0

        def resolve_run_head(self):
            self.calls += 1
            if self.calls == 1:
                return None
            return SimpleNamespace(checkpoint_id="c" * 64)

    monkeypatch.setattr(
        "trainer.algorithms.dqn_loop.create_checkpoint_publisher",
        lambda contract, model_dir: Checkpoints(),
    )
    reaped: list[tuple] = []
    monkeypatch.setattr(
        "trainer.algorithms.dqn_loop.reap_profile_scopes",
        lambda *, profile, retained_scopes: reaped.append((profile, retained_scopes)) or 0,
    )
    monkeypatch.setattr(
        cartridge,
        "collect",
        lambda request: requests["collect"].append(request) or 0,
    )
    monkeypatch.setattr(
        cartridge,
        "train",
        lambda request: requests["train"].append(request),
    )

    def evaluate(request):
        requests["evaluate"].append(request)
        return DqnEvaluationResults(
            env_id="counter",
            player_name="ONNX(model.onnx)",
            episodes_played=2,
            terminated_episodes=2,
            truncated_episodes=0,
            mean_return=1.0,
            min_return=1.0,
            max_return=1.0,
            avg_episode_length=3.0,
        )

    monkeypatch.setattr(cartridge, "evaluate", evaluate)
    config = DqnLoopConfig(
        env_id="counter",
        iterations=1,
        episodes_per_iteration=2,
        steps_per_iteration=3,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        gamma=0.99,
        target_sync_interval=5,
        hidden_size=16,
        grad_clip_norm=10.0,
        device="cpu",
        epsilon_start=0.25,
        epsilon_end=0.01,
        epsilon_decay=0.95,
        seed=11,
        onnx_intra_threads=2,
        evaluation_episodes=2,
        episode_timeout_secs=19,
        actor_binary=None,
        eval_binary=None,
        data_root=Path(tmp_path),
        log_level="WARNING",
    )

    assert DqnLoop(cartridge, config).run() == 0
    assert len(requests["collect"]) == len(requests["train"]) == len(requests["evaluate"]) == 1
    assert requests["collect"][0].episode_timeout_secs == 19
    assert requests["collect"][0].log_level == "WARNING"
    assert requests["train"][0].log_level == "WARNING"
    assert (
        requests["train"][0].replay_selection.collection_scope_id
        == requests["collect"][0].collection_scope_id
    )
    assert requests["train"][0].replay_selection.source_checkpoint_id is None
    assert requests["evaluate"][0].checkpoint_id == "c" * 64
    assert capsys.readouterr().out == (
        '{"avg_episode_length":3.0,"env_id":"counter","episodes_played":2,'
        '"max_return":1.0,"mean_return":1.0,"min_return":1.0,'
        '"player_name":"ONNX(model.onnx)","terminated_episodes":2,'
        '"truncated_episodes":0}\n'
    )


def test_dqn_evaluation_result_is_a_single_agent_return_contract():
    result = DqnEvaluationResults.from_json(
        {
            "env_id": "counter",
            "player_name": "ONNX(model.onnx)",
            "episodes_played": 10,
            "terminated_episodes": 7,
            "truncated_episodes": 3,
            "mean_return": 0.4,
            "min_return": -0.08,
            "max_return": 0.98,
            "avg_episode_length": 4.2,
        }
    )
    assert result.mean_return == 0.4
