# CI/CD Pipeline

GitHub Actions workflow for Cartridge2. Defined in `workflows/ci.yml`.

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

### Python (Trainer)

| Job | What it does | Auto-fixes on PR? |
|-----|-------------|-------------------|
| **python-lint** | Runs `ruff check --fix` and `black` on `trainer/src/` | Yes - commits fixed code |
| **python-test** | Installs trainer with dev deps, runs `pytest` | No |

### Frontend

| Job | What it does | Auto-fixes on PR? |
|-----|-------------|-------------------|
| **frontend** | Runs `svelte-check` (TypeScript) and `npm run build` | No |

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
# Rust
cargo fmt --check --manifest-path engine/Cargo.toml
cargo clippy --manifest-path engine/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path engine/Cargo.toml
# Repeat for actor/ and web/

# Python
cd trainer
ruff check src/
black --check src/
pytest

# Frontend
cd web/frontend
npm run check
npm run build
```
