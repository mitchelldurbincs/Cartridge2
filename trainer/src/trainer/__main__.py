"""Canonical command-line entrypoint for installed algorithm cartridges.

Usage is always::

    trainer [--algorithm ALGORITHM_ID] COMMAND [COMMAND_OPTIONS]
    python -m trainer [--algorithm ALGORITHM_ID] COMMAND [COMMAND_OPTIONS]

``--algorithm`` is a global option and therefore precedes the command.  The
selected algorithm owns the command set, parser configuration, and command
runners installed below.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .algorithms import get_algorithm, list_algorithms
from .algorithms.base import Algorithm, AlgorithmCommand
from .central_config import get_config
from .structured_logging import setup_logging


def _select_algorithm(argv: Sequence[str]) -> Algorithm:
    """Resolve the cartridge before building its command-specific parser."""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--algorithm",
        choices=list_algorithms(),
        default=get_config().algorithm.id,
    )
    selected, _ = selector.parse_known_args(argv)
    return get_algorithm(selected.algorithm)


def _install_command(subparsers: argparse._SubParsersAction, command: AlgorithmCommand) -> None:
    parser = subparsers.add_parser(
        command.name,
        help=command.help,
        description=command.description,
        formatter_class=command.formatter_class,
    )
    command.configure_parser(parser)
    parser.set_defaults(_command_runner=command.run)


def build_parser(algorithm: Algorithm) -> argparse.ArgumentParser:
    """Build the canonical parser from one selected cartridge's bindings."""
    parser = argparse.ArgumentParser(
        description="Cartridge2 reinforcement-learning trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=list_algorithms(),
        default=algorithm.descriptor.id,
        help="Installed algorithm cartridge to run",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        description=f"Commands provided by {algorithm.descriptor.id}",
        dest="command",
        required=True,
    )

    commands = algorithm.commands()
    names = [command.name for command in commands]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Algorithm '{algorithm.descriptor.id}' exports duplicate CLI commands")
    for command in commands:
        _install_command(subparsers, command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Select an algorithm, parse its command, and hand over execution."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    algorithm = _select_algorithm(arguments)
    parser = build_parser(algorithm)
    args = parser.parse_args(arguments)

    log_level = getattr(args, "log_level", "INFO")
    setup_logging(level=log_level, component=args.command)
    return args._command_runner(args)


if __name__ == "__main__":
    sys.exit(main())
