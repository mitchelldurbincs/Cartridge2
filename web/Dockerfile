# Web Backend Service Dockerfile
# Multi-stage build for Rust backend with ONNX support
#
# Build args:
#   CARGO_FEATURES - Comma-separated cargo features (e.g., "s3")

# ============================================
# Stage 0: Rust toolchain (shared base)
# ============================================
# Using Ubuntu 24.04 for glibc >= 2.38 (required by onnxruntime prebuilt binaries)
FROM ubuntu:24.04 AS rust-base

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# ============================================
# Stage 1: Manifests only
# ============================================
# Reduce the source tree to just the files that determine the dependency graph.
# This stage re-runs on every build (it copies everything), but what leaves it
# is only ~14 small files, so the COPY --from in the builder below is cached
# unless a Cargo.toml or Cargo.lock actually changed.
FROM rust-base AS manifests

COPY engine/ ./engine/
COPY web/ ./web/

RUN find . -type f ! -name 'Cargo.toml' ! -name 'Cargo.lock' -delete \
 && find . -type d -empty -delete

# ============================================
# Stage 2: Build Backend (Rust)
# ============================================
FROM rust-base AS builder

# Optional features: s3 (S3 model storage)
ARG CARGO_FEATURES=""

# Only manifests cross this boundary, so everything below stays cached across
# source-only changes. Previously the dependency build sat behind
# `COPY engine/ web/`, so editing any .rs file -- even a test or a comment --
# forced a full ~30 minute rebuild of all ~340 third-party crates.
COPY --from=manifests /app /app

# Give every package a placeholder source file so cargo can build the
# dependency graph without any of our real code. Derived from the manifests
# rather than hardcoded, so adding a crate (e.g. a new game) needs no change
# here. Packages declaring [[bin]] get the src/main.rs their manifest names.
RUN set -eu; \
    : > /tmp/placeholders.txt; \
    for manifest in $(find . -name Cargo.toml); do \
        dir=$(dirname "$manifest"); \
        grep -q '^\[package\]' "$manifest" || continue; \
        mkdir -p "$dir/src"; \
        : > "$dir/src/lib.rs"; \
        echo "$dir/src/lib.rs" >> /tmp/placeholders.txt; \
        if grep -q '^\[\[bin\]\]' "$manifest"; then \
            echo 'fn main() {}' > "$dir/src/main.rs"; \
            echo "$dir/src/main.rs" >> /tmp/placeholders.txt; \
        fi; \
    done

# Compile dependencies against the placeholders. This is the expensive layer.
WORKDIR /app/web
RUN if [ -z "$CARGO_FEATURES" ]; then \
        cargo build --release --locked; \
    else \
        cargo build --release --locked --features "$CARGO_FEATURES"; \
    fi

# Remove exactly the placeholders that were created -- by recorded path, never
# by glob. A `find -name '*.rs' -delete` here would also delete the .rs files
# that build scripts generate under target/, silently destroying the dependency
# artifacts this whole stage exists to produce.
RUN cd /app && xargs rm -f < /tmp/placeholders.txt

COPY config.defaults.toml /app/
COPY engine/ /app/engine/
COPY web/ /app/web/

# touch: COPY carries the build context's mtimes, which in CI predate this
# build (they are set at checkout). Cargo's fingerprints are mtime-based, so
# without this it can mistake the real sources for the already-compiled
# placeholders and ship a binary built from empty files. target/ is excluded so
# generated sources keep their fingerprints.
#
# --locked: build the dependency graph recorded in the committed Cargo.lock,
# matching what CI tested.
RUN find /app/engine /app/web -path '*/target' -prune -o -name '*.rs' -exec touch {} + \
 && if [ -z "$CARGO_FEATURES" ]; then \
        cargo build --release --locked; \
    else \
        cargo build --release --locked --features "$CARGO_FEATURES"; \
    fi

# ============================================
# Stage 2: Runtime
# ============================================
# Using Ubuntu 24.04 for glibc >= 2.38 (required by onnxruntime prebuilt binaries)
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Keep the serving image on the same explicit non-root identity as the trainer;
# both processes read the shared profile projection volume in Kubernetes.
# Ubuntu 24.04 already has a UID-1000 user, so reuse that account when present.
RUN set -eux; \
    existing_user="$(getent passwd 1000 | cut -d: -f1)"; \
    if [ -n "$existing_user" ]; then \
        existing_group="$(id -gn "$existing_user")"; \
        usermod --login cartridge --home /home/cartridge --move-home "$existing_user"; \
        if [ "$existing_group" != "cartridge" ]; then \
            groupmod --new-name cartridge "$existing_group"; \
        fi; \
    else \
        useradd --create-home --uid 1000 --user-group cartridge; \
    fi

WORKDIR /app

# Copy the compiled binary from builder
COPY --from=builder /app/web/target/release/web /app/web

# Create the root used for profile-scoped stats:
# profiles/{algorithm_id}/{env_id}/v{env_contract_version}.
RUN mkdir -p /app/data && chown -R cartridge:cartridge /app

ENV CARTRIDGE_COMMON_DATA_DIR="/app/data"
ENV RUST_LOG="info"

# Expose the web server port
EXPOSE 8080

# Run as non-root user
USER cartridge

# Healthcheck using the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the web server
ENTRYPOINT ["/app/web"]
