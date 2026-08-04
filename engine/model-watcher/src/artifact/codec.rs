use anyhow::{bail, Context, Result};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};

use super::types::{RunHeadV2, RUN_HEAD_SCHEMA_VERSION};

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn validate_digest(label: &str, digest: &str) -> Result<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be a lowercase 64-character SHA-256 digest, got {digest:?}");
    }
    Ok(())
}

pub(crate) fn parse_canonical_json<T: DeserializeOwned>(label: &str, bytes: &[u8]) -> Result<T> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .with_context(|| format!("failed to parse {label} as JSON"))?;
    let canonical =
        serde_json::to_vec(&value).with_context(|| format!("failed to canonicalize {label}"))?;
    if canonical != bytes {
        bail!("{label} is not canonical JSON");
    }
    serde_json::from_value(value).with_context(|| format!("invalid {label} contract"))
}

pub(crate) fn validate_run_head(head: &RunHeadV2) -> Result<()> {
    if head.schema_version != RUN_HEAD_SCHEMA_VERSION {
        bail!(
            "unsupported run-head schema {}, expected {}",
            head.schema_version,
            RUN_HEAD_SCHEMA_VERSION
        );
    }
    validate_digest("run-head checkpoint_id", &head.checkpoint_id)?;
    validate_digest("run-head run_commit_id", &head.run_commit_id)
}
