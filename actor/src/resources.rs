//! Lightweight process-resource diagnostics for structured actor logs.

fn parse_rss_kb(contents: &str) -> Option<u64> {
    let line = contents.lines().find(|line| line.starts_with("VmRSS:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Current resident set size in MB, when Linux `/proc` is available.
pub fn rss_mb() -> Option<f64> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_rss_kb(&contents).map(|kb| kb as f64 / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_rss_status_line() {
        assert_eq!(parse_rss_kb("Name:\tactor\nVmRSS:\t2048 kB\n"), Some(2048));
        assert_eq!(parse_rss_kb("Name:\tactor\n"), None);
    }
}
