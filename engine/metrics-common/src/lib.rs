//! Shared Prometheus metrics plumbing for the actor and web binaries.
//!
//! Each binary keeps its own `Registry` and metric definitions (they monitor
//! different things); this crate holds the plumbing that is identical across
//! them: collector registration, text-format encoding, and the exposition
//! content type served by their `/metrics` endpoints.

use prometheus::core::Collector;
use prometheus::{Encoder, Registry, TextEncoder};

/// Content-Type for the Prometheus text exposition format (v0.0.4).
pub const PROMETHEUS_TEXT_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Register every collector with the registry.
///
/// # Panics
///
/// Panics if a collector is already registered — duplicate registration is a
/// programming error, matching the previous per-binary `unwrap` behavior.
pub fn register_all(registry: &Registry, collectors: Vec<Box<dyn Collector>>) {
    for collector in collectors {
        registry.register(collector).unwrap();
    }
}

/// Encode all metrics from the registry into Prometheus text format.
pub fn encode_metrics(registry: &Registry) -> String {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prometheus::{IntCounter, Opts};

    #[test]
    fn test_register_all_and_encode() {
        let registry = Registry::new();
        let counter =
            IntCounter::with_opts(Opts::new("test_counter_total", "A test counter")).unwrap();
        register_all(&registry, vec![Box::new(counter.clone())]);

        counter.inc();
        let output = encode_metrics(&registry);
        assert!(output.contains("test_counter_total"));
    }

    #[test]
    #[should_panic]
    fn test_register_all_panics_on_duplicate() {
        let registry = Registry::new();
        let counter =
            IntCounter::with_opts(Opts::new("dup_counter_total", "A test counter")).unwrap();
        register_all(&registry, vec![Box::new(counter.clone())]);
        register_all(&registry, vec![Box::new(counter)]);
    }

    #[test]
    fn test_encode_empty_registry() {
        let registry = Registry::new();
        assert!(encode_metrics(&registry).is_empty());
    }
}
