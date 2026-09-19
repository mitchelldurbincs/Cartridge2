"""Time complete AlphaZero replay decoding using synthetic, in-memory records.

From the repository root, with trainer dependencies installed:
    PYTHONPATH=trainer/src python trainer/benchmarks/bench_replay_decode.py

Use --baseline-source with another checkout's alphazero_board_v1.py to alternate
baseline/current samples in one process. Fixture construction, imports, and
bitwise output comparisons are outside timing. No storage clients or training
processes are created. JSON includes CPU and wall-clock samples in ms/batch.
"""

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

from trainer.algorithms.alphazero_board_v1 import (
    ALGORITHM_ID,
    DESCRIPTOR,
    decode_replay_batch,
    get_game_config,
)
from trainer.environment_catalog import get_environment
from trainer.storage.base import ReplayProfile, ReplaySelection


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--samples", type=positive_int, default=21)
    parser.add_argument("--batches-per-sample", type=positive_int, default=50)
    parser.add_argument("--baseline-source", help="path to baseline alphazero_board_v1.py")
    args = parser.parse_args()
    decoders = {"current": decode_replay_batch}
    if args.baseline_source:
        name = "trainer.algorithms._benchmark_baseline"
        spec = importlib.util.spec_from_file_location(name, args.baseline_source)
        if spec is None or spec.loader is None:
            parser.error("cannot load baseline Python source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        decoders = {"baseline": module.decode_replay_batch, **decoders}
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "seed": 42,
                **vars(args),
            }
        )
    )
    for env_id in ("tictactoe", "connect4", "othello", "generals_8x8"):
        config = get_game_config(env_id)
        selection = ReplaySelection(
            ReplayProfile(
                env_id,
                get_environment(env_id).capabilities.contract_version,
                ALGORITHM_ID,
                DESCRIPTOR.components.experience_schema,
            ),
            "a" * 64,
            None,
        )
        rng = np.random.default_rng(42)
        records = []
        for index in range(args.batch_size):
            observation = rng.uniform(-1, 1, config.obs_size).astype("<f4")
            policy = rng.uniform(0, 1, config.num_actions).astype("<f4")
            policy /= policy.sum(dtype=np.float32)
            payload = np.concatenate((observation, policy, [index % 3 - 1])).astype("<f4")
            records.append(
                selection.record(
                    id=str(index),
                    episode_id=str(index // 16),
                    step_number=index % 16,
                    payload=payload.tobytes(),
                )
            )
        kwargs = {
            "selection": selection,
            "obs_size": config.obs_size,
            "num_actions": config.num_actions,
        }
        expected = decode_replay_batch(records, **kwargs)
        for decode in decoders.values():
            for actual, wanted in zip(decode(records, **kwargs), expected, strict=True):
                assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
                assert actual.flags.c_contiguous and actual.flags.writeable
                assert actual.tobytes() == wanted.tobytes()
            for _ in range(args.warmup):
                decode(records, **kwargs)
        samples = {name: {"wall_ms": [], "cpu_ms": []} for name in decoders}
        for index in range(args.samples):
            order = list(decoders.items())
            if index % 2:
                order.reverse()
            for name, decode in order:
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                for _ in range(args.batches_per_sample):
                    decode(records, **kwargs)
                cpu_elapsed = time.process_time_ns() - cpu_start
                wall_elapsed = time.perf_counter_ns() - wall_start
                samples[name]["wall_ms"].append(wall_elapsed / args.batches_per_sample / 1e6)
                samples[name]["cpu_ms"].append(cpu_elapsed / args.batches_per_sample / 1e6)
        print(
            json.dumps(
                {
                    "environment": env_id,
                    "obs_size": config.obs_size,
                    "num_actions": config.num_actions,
                    "medians_ms": {
                        name: {clock: statistics.median(values) for clock, values in times.items()}
                        for name, times in samples.items()
                    },
                    "samples_ms": samples,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
