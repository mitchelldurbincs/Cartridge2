//! Health check HTTP server for Kubernetes probes.
//!
//! Provides liveness and readiness endpoints for container orchestration.

use axum::{routing::get, Router};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};

/// Shared health state between main actor loop and health server.
#[derive(Debug, Clone)]
pub struct HealthState {
    /// Set to true once the actor has completed initialization.
    ready: Arc<AtomicBool>,
    /// Set to false if the actor encounters a fatal error.
    healthy: Arc<AtomicBool>,
    /// Timestamp of last successful episode completion (Unix seconds).
    last_episode_time: Arc<AtomicU64>,
}

impl HealthState {
    pub fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            healthy: Arc::new(AtomicBool::new(true)),
            last_episode_time: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Mark the actor as ready to receive traffic.
    pub fn set_ready(&self) {
        self.ready.store(true, Ordering::SeqCst);
        info!("Actor marked as ready");
    }

    /// Mark the actor as unhealthy (will trigger restart).
    /// Called when the actor encounters a fatal error that requires restart.
    #[allow(dead_code)]
    pub fn set_unhealthy(&self) {
        self.healthy.store(false, Ordering::SeqCst);
        error!("Actor marked as unhealthy");
    }

    /// Update the last episode completion time.
    pub fn record_episode_complete(&self) {
        self.last_episode_time.store(unix_now(), Ordering::SeqCst);
    }

    /// Check if the actor is ready.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Check if the actor is healthy.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }

    /// Check if the actor has completed an episode recently (within timeout).
    pub fn is_making_progress(&self, timeout_secs: u64) -> bool {
        let last = self.last_episode_time.load(Ordering::SeqCst);
        if last == 0 {
            // No episodes completed yet, but that's ok during startup
            return true;
        }
        unix_now().saturating_sub(last) < timeout_secs
    }
}

/// Current Unix time in seconds (0 if the system clock is before the epoch).
fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Default for HealthState {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum number of consecutive ports to try when binding the health server.
const MAX_PORT_ATTEMPTS: u16 = 16;

/// Bind the health listener, scanning forward from `preferred_port` if it is
/// already in use.
///
/// Multiple actors on one host all default to the same health port; without
/// this fallback every actor after the first fails with "Address already in
/// use". Only `AddrInUse` triggers the scan — any other bind error (e.g.
/// permission denied) is returned immediately.
async fn bind_health_listener(preferred_port: u16) -> std::io::Result<TcpListener> {
    let mut last_err = None;

    for offset in 0..MAX_PORT_ATTEMPTS {
        let Some(port) = preferred_port.checked_add(offset) else {
            break;
        };
        match TcpListener::bind(("0.0.0.0", port)).await {
            Ok(listener) => {
                if port != preferred_port {
                    info!(
                        preferred_port,
                        actual_port = port,
                        "Preferred health port in use (another actor?), bound fallback port"
                    );
                }
                return Ok(listener);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }

    Err(last_err.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::AddrInUse,
            format!("no free health port at or above {}", preferred_port),
        )
    }))
}

/// Start the health check HTTP server on the given port.
///
/// If the port is already taken (e.g. by another actor on the same host), the
/// server falls back to the next free port; see [`bind_health_listener`].
pub async fn start_health_server(
    port: u16,
    state: HealthState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = Router::new()
        .route(
            "/health",
            get({
                let state = state.clone();
                move || health_handler(state.clone())
            }),
        )
        .route(
            "/ready",
            get({
                let state = state.clone();
                move || ready_handler(state.clone())
            }),
        )
        .route("/metrics", get(metrics_handler));

    let listener = bind_health_listener(port).await?;
    info!("Health server listening on {}", listener.local_addr()?);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler(state: HealthState) -> axum::http::StatusCode {
    // Liveness: is the process fundamentally healthy?
    // Check both health flag and progress (no stuck episodes)
    // 5 minute timeout for progress
    if state.is_healthy() && state.is_making_progress(300) {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn ready_handler(state: HealthState) -> axum::http::StatusCode {
    // Readiness: is the actor ready to generate episodes?
    if state.is_ready() && state.is_healthy() {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    }
}

// Sibling: web/src/handlers/health.rs::metrics_handler (web serves the same
// format but has no memory gauge to refresh).
async fn metrics_handler() -> (
    axum::http::StatusCode,
    [(axum::http::header::HeaderName, &'static str); 1],
    String,
) {
    // Update memory metrics before responding
    crate::metrics::update_memory_metrics();

    (
        axum::http::StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            metrics_common::PROMETHEUS_TEXT_CONTENT_TYPE,
        )],
        crate::metrics::encode_metrics(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_state_initial() {
        let state = HealthState::new();
        assert!(!state.is_ready());
        assert!(state.is_healthy());
    }

    #[test]
    fn test_health_state_ready() {
        let state = HealthState::new();
        state.set_ready();
        assert!(state.is_ready());
    }

    #[test]
    fn test_health_state_unhealthy() {
        let state = HealthState::new();
        state.set_unhealthy();
        assert!(!state.is_healthy());
    }

    #[test]
    fn test_progress_tracking() {
        let state = HealthState::new();
        // Initially, no episodes completed but should report as making progress
        assert!(state.is_making_progress(300));

        // Record an episode completion
        state.record_episode_complete();
        assert!(state.is_making_progress(300));
    }

    #[test]
    fn test_progress_timeout() {
        let state = HealthState::new();
        // Manually set an old timestamp (1 second in the past)
        state
            .last_episode_time
            .store(unix_now() - 1, Ordering::SeqCst);

        // With a 2 second timeout, should still be making progress
        assert!(state.is_making_progress(2));

        // With a 0 second timeout, should NOT be making progress
        assert!(!state.is_making_progress(0));
    }

    #[test]
    fn test_progress_tolerates_future_timestamp() {
        let state = HealthState::new();
        // A timestamp ahead of the clock (skew) must not underflow/panic
        state
            .last_episode_time
            .store(unix_now() + 100, Ordering::SeqCst);
        assert!(state.is_making_progress(1));
    }

    /// The bind tests occupy and release OS-assigned ports, which the OS hands
    /// out roughly sequentially — run concurrently, one test's fallback scan can
    /// steal the port another test just released. Serialize them.
    static PORT_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Bind an OS-assigned port, keeping the listener alive so the port stays taken.
    /// Retries until the port is low enough that the fallback scan cannot overflow u16.
    async fn occupy_free_port() -> (TcpListener, u16) {
        loop {
            let listener = TcpListener::bind(("0.0.0.0", 0)).await.unwrap();
            let port = listener.local_addr().unwrap().port();
            if port < u16::MAX - MAX_PORT_ATTEMPTS {
                return (listener, port);
            }
        }
    }

    #[tokio::test]
    async fn test_bind_health_listener_uses_preferred_port() {
        let _guard = PORT_TEST_LOCK.lock().await;
        // Find a free port, release it, then ask for it explicitly
        let (listener, port) = occupy_free_port().await;
        drop(listener);

        let bound = bind_health_listener(port).await.unwrap();
        assert_eq!(bound.local_addr().unwrap().port(), port);
    }

    #[tokio::test]
    async fn test_bind_health_listener_falls_back_when_port_taken() {
        let _guard = PORT_TEST_LOCK.lock().await;
        // Hold the preferred port so the fallback scan must kick in
        let (_held, port) = occupy_free_port().await;

        let bound = bind_health_listener(port).await.unwrap();
        let actual = bound.local_addr().unwrap().port();
        assert_ne!(actual, port, "should not bind the occupied port");
        assert!(
            actual > port && actual < port + MAX_PORT_ATTEMPTS,
            "fallback port {} should be in ({}, {})",
            actual,
            port,
            port + MAX_PORT_ATTEMPTS
        );
    }

    #[tokio::test]
    async fn test_bind_health_listener_two_actors_get_distinct_ports() {
        let _guard = PORT_TEST_LOCK.lock().await;
        // Simulates two actors launched with the same --health-port
        let (first, port) = occupy_free_port().await;
        let second = bind_health_listener(port).await.unwrap();
        assert_ne!(
            first.local_addr().unwrap().port(),
            second.local_addr().unwrap().port()
        );
    }
}
