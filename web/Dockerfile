# Web Backend Service Dockerfile
# Multi-stage build for Rust backend with ONNX support
#
# Build args:
#   CARGO_FEATURES - Comma-separated cargo features (e.g., "s3")

# ============================================
# Stage 1: Build Backend (Rust)
# ============================================
# Using Ubuntu 24.04 for glibc >= 2.38 (required by onnxruntime prebuilt binaries)
FROM ubuntu:24.04 AS builder

# Optional features: s3 (S3 model storage)
ARG CARGO_FEATURES=""

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

# Copy the workspace (engine + web)
COPY config.defaults.toml ./
COPY engine/ ./engine/
COPY web/ ./web/

# Build the web binary in release mode with optional features
WORKDIR /app/web
RUN if [ -z "$CARGO_FEATURES" ]; then \
        cargo build --release; \
    else \
        cargo build --release --features "$CARGO_FEATURES"; \
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
RUN useradd --create-home --uid 1000 --user-group cartridge

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
