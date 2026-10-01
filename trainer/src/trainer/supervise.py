"""POSIX wall-clock boundary outside the learner, actors and evaluator.

Children inherit a dedicated process group. Do not detach workers from it.
Docker's init process reaps orphaned descendants; we reap the direct child.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return seconds


def supervise(command: list[str], *, timeout: float, grace: float, output=None) -> int:
    """Return child status, 124 on timeout, or 128+signal on cancellation.

    Always clean the entire group, including when the parent exits before its
    workers. TERM grace is shared; a stubborn descendant gets KILL even if the
    group leader already exited. The maximum work window is timeout + grace.
    """
    if os.name != "posix":
        raise RuntimeError("The pilot supervisor requires a POSIX host (Linux or macOS)")
    positive_seconds(str(timeout))
    positive_seconds(str(grace))
    received = 0

    def on_signal(signum, _frame):
        nonlocal received
        received = received or signum

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    process = None

    def send(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    try:
        deadline = time.monotonic() + timeout
        process = subprocess.Popen(
            command, start_new_session=True, stdout=output, stderr=subprocess.STDOUT
        )
        while True:
            if received:
                return 128 + received
            status = process.poll()
            if status is not None:
                return status if status >= 0 else 128 - status
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(f"Pilot exceeded {timeout:g}s; terminating process group", file=sys.stderr)
                return 124
            time.sleep(min(0.05, remaining))
    finally:
        try:
            if process is not None:
                send(signal.SIGTERM)
                stop_at = time.monotonic() + grace
                while time.monotonic() < stop_at:
                    process.poll()  # reap the leader without losing track of its group
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(min(0.05, max(0, stop_at - time.monotonic())))
                send(signal.SIGKILL)
                process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=positive_seconds, required=True)
    parser.add_argument("--grace-seconds", type=positive_seconds, default=10.0)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    if args.log_file:
        # Parent directory must already be on the intended persistent mount.
        with args.log_file.open("ab", buffering=0) as output:
            return supervise(
                command, timeout=args.timeout_seconds, grace=args.grace_seconds, output=output
            )
    return supervise(command, timeout=args.timeout_seconds, grace=args.grace_seconds)


if __name__ == "__main__":
    sys.exit(main())
