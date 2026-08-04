# CI/CD Pipeline

GitHub Actions workflow for Cartridge2. Defined in `ci.yml`.

## Triggers

- **Push** to `main`/`master`: runs all checks
- **Pull requests** to `main`/`master`: runs all checks
- **Manual dispatch**: `workflow_dispatch`

Concurrent runs on the same branch are cancelled automatically.

## Jobs

### Rust

| Job | What it does |
|-----|-------------|
| **rust-fmt** | Checks `cargo fmt` on engine, actor, and web |
| **rust-clippy** | Runs `cargo clippy` with `-D warnings`, including all-feature model-watcher/actor/web builds and web without ONNX |
| **rust-test** | Tests engine, all-feature model-watcher and actor, web, and web without ONNX |
| **rust-build** | Release build of all Rust components |
| **rust-security-audit** | `cargo audit` on engine, actor, web (non-blocking) |

Note: `rust-test` also runs the environment-manifest golden test, which fails
when `trainer/src/trainer/environment_manifest.json` drifts from the Rust
environment and algorithm contracts. Regenerate with
`make environment-manifest`.

### Python (Trainer)

| Job | What it does |
|-----|-------------|
| **python-lint** | Checks Ruff linting and formatting on trainer source, tests, and smoke test |
| **python-test** | Installs the trainer with dev deps (which pulls the pinned `crucible` orchestration core), then runs `pytest` |
| **python-security-audit** | `pip-audit` (non-blocking) |

`crucible` is a hard, import-time dependency of `trainer.orchestrator`. It is
declared in `trainer/pyproject.toml`, pinned to a commit — the single place that
pin lives. A local `pytest` on a fresh clone gets it automatically from
`pip install -e "trainer/.[dev]"`; install a sibling checkout editable first if
you are developing crucible alongside.

### Frontend

| Job | What it does |
|-----|-------------|
| **frontend** | Runs `npm audit`, `svelte-check` (TypeScript), and `npm run build` |

### Other

| Job | What it does |
|-----|-------------|
| **docker-build** | Buildx validation of `Dockerfile.alphazero` and `web/Dockerfile` with the deployment-required `s3` feature, plus `web/frontend/Dockerfile` (built, not pushed or run) |
| **secrets-scan** | GitLeaks secret scanning (non-blocking) |

## Check-Only Behavior

CI never rewrites a branch. Formatting and lint jobs use check-only commands,
and the workflow has read-only repository permissions. Apply fixes locally,
then push them explicitly.

## Caching

- **Rust**: Uses `Swatinem/rust-cache@v2` with separate workspaces for engine, actor, and web
- **Python**: pip cache keyed on `trainer/pyproject.toml`
- **Node**: npm cache keyed on `web/frontend/package-lock.json`

## Running Checks Locally

```bash
make lint && make test    # covers Rust + Python

# Or piecewise:

# Rust
cargo fmt --all --manifest-path engine/Cargo.toml -- --check
cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path engine/Cargo.toml
# Repeat for actor/ and web/

# Python (needs crucible - see the python-test note above)
cd trainer
ruff check src/ tests/ smoke_test.py
ruff format --check src/ tests/ smoke_test.py
python -m pytest tests/

# Frontend
cd web/frontend
npm run check
npm run build
```

These are the same check-only forms CI runs.
