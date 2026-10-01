"""Small portable Connect4 plumbing smoke; run through trainer.supervise."""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

from .device import resolve_device


def smoke_arguments(device: str, data_dir: Path) -> list[str]:
    return [
        "--algorithm",
        "alphazero_board_v1",
        "loop",
        "--env-id",
        "connect4",
        "--iterations",
        "2",
        "--episodes",
        "8",
        "--steps",
        "4",
        "--num-actors",
        "2",
        "--batch-size",
        "8",
        "--mcts-start-sims",
        "8",
        "--mcts-max-sims",
        "8",
        "--mcts-sim-ramp-rate",
        "0",
        "--actor-eval-batch-size",
        "1",
        "--actor-onnx-intra-threads",
        "1",
        "--actor-episode-timeout-seconds",
        "60",
        "--eval-interval",
        "1",
        "--eval-games",
        "4",
        "--eval-simulations",
        "8",
        "--solver-games",
        "0",
        "--device",
        device,
        "--wandb-enabled",
        "false",
        "--data-dir",
        str(data_dir),
    ]


def preflight(device: str) -> None:
    """Exercise device kernels, backward, ONNX export and the Rust consumer."""
    import torch

    from .algorithms import get_algorithm
    from .algorithms.alphazero_board_v1 import get_game_config
    from .checkpoint import export_onnx_artifact
    from .environment_catalog import get_environment
    from .evaluator import evaluate
    from .network import create_network
    from .players import ModelPlayer, RandomPlayer

    device = resolve_device(device)
    print(
        f"Pilot preflight: device={device} torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )
    game = get_game_config("connect4")
    network = create_network("connect4", config=game).to(device)
    inputs = torch.zeros((2, game.obs_size), device=device)
    policy, value = network(inputs)
    loss = policy.square().mean() + value.square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not all(
        torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None
    ):
        raise RuntimeError("Device preflight produced non-finite loss or gradients")
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    contract = get_algorithm("alphazero_board_v1").artifact_contract(get_environment("connect4"))
    with tempfile.TemporaryDirectory(prefix="cartridge-pilot-") as staging:
        path = export_onnx_artifact(
            network, Path(staging) / "model.onnx", torch.device(device), contract
        )
        result = evaluate(
            ModelPlayer(str(path)),
            RandomPlayer(),
            algorithm_id="alphazero_board_v1",
            env_id="connect4",
            num_games=2,
        )
        if result.games_played != 2:
            raise RuntimeError("Rust ONNX preflight did not complete both fixture games")
    print("Pilot preflight passed", flush=True)


def configure_postgres() -> None:
    """Adapt a mounted Compose secret only at the deployment boundary."""
    password_file = os.environ.get("CARTRIDGE_PILOT_POSTGRES_PASSWORD_FILE")
    if password_file:
        password = Path(password_file).read_text().rstrip("\r\n")
        if not password:
            raise RuntimeError("Pilot PostgreSQL password file is empty")
        os.environ["CARTRIDGE_STORAGE_POSTGRES_URL"] = (
            f"postgresql://cartridge:{quote(password, safe='')}@postgres:5432/cartridge"
        )
    if not os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"):
        raise RuntimeError(
            "Set CARTRIDGE_STORAGE_POSTGRES_URL or mount the pilot PostgreSQL secret"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Dedicated smoke experiment root"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    resolve_device(args.device)  # fail before any data/DB activity
    args.data_dir.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / ".pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pilot already owns this data root") from exc
        preflight(args.device)
        if args.preflight_only:
            return 0
        configure_postgres()
        from .__main__ import main as trainer_main

        return trainer_main(smoke_arguments(args.device, args.data_dir))


if __name__ == "__main__":
    sys.exit(main())
