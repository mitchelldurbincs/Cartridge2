//! Health check and metrics endpoints.

use axum::{
    http::{header, StatusCode},
    Json,
};

use crate::metrics;
use crate::types::HealthResponse;

/// Health check handler.
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Prometheus metrics handler.
pub async fn metrics_handler() -> (StatusCode, [(header::HeaderName, &'static str); 1], String) {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics::encode_metrics(),
    )
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_returns_ok_status() {
        let response = health().await;
        assert_eq!(response.0.status, "ok");
    }

    #[tokio::test]
    async fn test_health_returns_version() {
        let response = health().await;
        assert!(!response.0.version.is_empty());
        // Version should match cargo package version format (x.y.z)
        assert!(response.0.version.contains('.'));
    }

    #[tokio::test]
    async fn test_health_response_is_json() {
        // The Json wrapper ensures proper JSON serialization
        let json_response = health().await;
        // Should be able to serialize to string
        let json_str = serde_json::to_string(&json_response.0);
        assert!(json_str.is_ok());
        
        let json_str = json_str.unwrap();
        assert!(json_str.contains("status"));
        assert!(json_str.contains("version"));
        assert!(json_str.contains("ok"));
    }

    #[tokio::test]
    async fn test_metrics_handler_returns_success() {
        // Initialize metrics first
        metrics::init_metrics();
        
        let (status, _headers, body) = metrics_handler().await;
        
        assert_eq!(status, StatusCode::OK);
        assert!(!body.is_empty());
    }

    #[tokio::test]
    async fn test_metrics_handler_returns_correct_content_type() {
        metrics::init_metrics();
        
        let (_status, headers, _body) = metrics_handler().await;
        
        assert_eq!(headers[0].0, header::CONTENT_TYPE);
        assert!(headers[0].1.contains("text/plain"));
        assert!(headers[0].1.contains("version=0.0.4"));
    }

    #[tokio::test]
    async fn test_metrics_handler_includes_expected_metrics() {
        metrics::init_metrics();
        
        // Increment some metrics to ensure they appear
        metrics::GAMES_CREATED.inc();
        metrics::MOVES_PLAYED.inc_by(5);
        
        let (_status, _headers, body) = metrics_handler().await;
        
        // Should contain standard metric names
        assert!(body.contains("web_games_created_total"));
        assert!(body.contains("web_moves_played_total"));
    }

    #[test]
    fn test_status_code_is_ok() {
        // Verify StatusCode::OK is what we expect
        assert_eq!(StatusCode::OK.as_u16(), 200);
    }

    #[test]
    fn test_content_type_header_name() {
        // Verify the header name constant
        assert_eq!(header::CONTENT_TYPE.as_str(), "content-type");
    }
}
