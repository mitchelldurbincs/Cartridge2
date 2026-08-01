# CI/CD Pipeline

GitHub Actions workflow for Cartridge2. Defined in `ci.yml`.

## Triggers

- **Push** to `main`/`master`: runs all checks
- **Pull requests** to `main`/`master`: runs all checks + auto-fixes formatting
- **Manual dispatch**: `workflow_dispatch`

Concurrent runs on the same branch are cancelled automatically.

## Jobs

### Rust

| Job | What it does | Auto-fixes on PR? |
|-----|-------------|-------------------|
| **rust-fmt** | Runs `cargo fmt` on engine, actor, and web | Yes - commits formatted code |
| **rust-clippy** | Runs `cargo clippy` with `-D warnings` on all crates | No |
| **rust-test** | Runs `cargo test` on engine, actor, and web | No |
| **rust-build** | Release build of all Rust components | No |
| **rust-security-audit** | `cargo audit` on engine, actor, web (non-blocking) | No |

Note: `rust-test` also runs the game-metadata golden test, which fails when
`trainer/src/trainer/game_metadata.json` drifts from the Rust game crates.
Regenerate with `make game-manifest`.

### Python (Trainer)

| Job | What it does | Auto-fixes on PR? |
|-----|-------------|-------------------|
| **python-lint** | Runs `ruff check --fix` and `black` on `trainer/src/` | Yes - commits fixed code |
| **python-test** | Installs the pinned `crucible` orchestration core from GitHub, then the trainer with dev deps, then runs `pytest` | No |
| **python-security-audit** | `pip-audit` (non-blocking) | No |

`crucible` is a hard, import-time dependency of `trainer.orchestrator`. It is
declared in `trainer/pyproject.toml` pinned to a commit; CI installs the same
pin explicitly. A local `pytest` on a fresh clone needs it too — `pip install -e
"trainer/.[dev]"` pulls it, or install a sibling checkout editable first.

### Frontend

| Job | What it does | Auto-fixes on PR? |
|-----|-------------|-------------------|
| **frontend** | Runs `npm audit`, `svelte-check` (TypeScript) and `npm run build` | No |

### Other

| Job | What it does | Auto-fixes on PR? |
|-----|-------------|-------------------|
| **docker-build** | Buildx validation of `Dockerfile.alphazero`, `web/Dockerfile`, `web/frontend/Dockerfile` (built, not pushed or run) | No |
| **secrets-scan** | GitLeaks secret scanning (non-blocking) | No |

## Auto-Fix Behavior

On pull requests, the `rust-fmt` and `python-lint` jobs automatically fix formatting issues and commit the changes back to the PR branch using `stefanzweifel/git-auto-commit-action`. This means:

1. You push code with formatting issues
2. CI reformats and commits a fix
3. Your PR is updated automatically

On pushes to `main`/`master`, these jobs only check formatting (no auto-commit).

## Caching

- **Rust**: Uses `Swatinem/rust-cache@v2` with separate workspaces for engine, actor, and web
- **Python**: pip cache keyed on `trainer/pyproject.toml`
- **Node**: npm cache keyed on `web/frontend/package-lock.json`

## Running Checks Locally

```bash
make lint && make test    # covers Rust + Python

# Or piecewise:

# Rust
cargo fmt --check --manifest-path engine/Cargo.toml
cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path engine/Cargo.toml
# Repeat for actor/ and web/

# Python (needs crucible - see the python-test note above)
cd trainer
ruff check src/          # CI runs `ruff check --fix` (mutating)
black --check src/       # CI runs `black` (mutating)
python -m pytest tests/

# Frontend
cd web/frontend
npm run check
npm run build
```

Note the asymmetry: CI's lint jobs *mutate and commit*, while the recipe above
only checks. Run the `--fix`/mutating forms locally if you want to match what CI
will do to your branch.
