//! Optional display metadata for discrete actions. Never a legality oracle.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionPresentation {
    pub action: u32,
    pub label: String,
    pub target: Option<ActionTarget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionTarget {
    Cell { index: usize },
    Column { index: usize },
    Edge { from: usize, to: usize },
    Named { name: String },
}

impl ActionPresentation {
    pub fn new(action: u32, label: impl Into<String>, target: ActionTarget) -> Self {
        Self { action, label: label.into(), target: Some(target) }
    }

    pub fn cell(action: u32, width: usize) -> Self {
        let index = action as usize;
        Self::new(action, format!("Row {}, column {}", index / width + 1, index % width + 1),
            ActionTarget::Cell { index })
    }

    pub fn named(action: u32, name: &str) -> Self {
        Self::new(action, name, ActionTarget::Named { name: name.into() })
    }
}
