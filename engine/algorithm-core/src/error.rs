#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AlgorithmError {
    #[error("unknown algorithm '{requested}'; available algorithms: {available}")]
    UnknownAlgorithm {
        requested: String,
        available: String,
    },
    #[error("algorithm '{algorithm_id}' is incompatible with environment '{env_id}': {reasons}")]
    Incompatible {
        algorithm_id: String,
        env_id: String,
        reasons: String,
    },
    #[error(
        "invalid runtime profile {field} {value:?}; IDs must contain only lowercase ASCII letters, digits, '_' or '-', and contract versions must be positive"
    )]
    InvalidRuntimeProfile { field: &'static str, value: String },
}
