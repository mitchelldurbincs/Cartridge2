//! Versioned decision data, separate from board rendering and training replay.
use serde::{Deserialize, Serialize};
use super::GameStateResponse;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionKey {
    pub session_id: String,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointIdentity {
    pub checkpoint_id: String,
    pub training_step: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActionReference {
    Discrete { index: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionAssessment {
    pub action: ActionReference,
    pub network_prior: Option<f32>,
    pub search_prior: Option<f32>,
    pub visit_share: Option<f32>,
    pub selection_probability: Option<f32>,
    pub visits: Option<u32>,
    pub q_value: Option<f32>,
    pub expanded: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueQuantity { ExpectedOutcome, DiscountedReturn }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueEstimate {
    pub value: f32,
    pub perspective_agent: u32,
    pub quantity: ValueQuantity,
    pub bounds: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEffort {
    pub completed_simulations: u32,
    pub root_visits: u32,
    pub neural_evaluations: u32,
    pub total_time_us: u64,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSource { AlphaZeroMcts, Random, Human, DqnQValues }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAnalysis {
    pub schema_version: u32,
    pub analysis_id: String,
    pub position: PositionKey,
    pub env_id: String,
    pub env_contract_version: u32,
    pub algorithm_id: String,
    pub actor: u32,
    pub source: DecisionSource,
    pub checkpoint: Option<CheckpointIdentity>,
    pub selected_action: ActionReference,
    pub network_value: Option<ValueEstimate>,
    pub search_value: Option<ValueEstimate>,
    pub actions: Vec<ActionAssessment>,
    pub search: Option<SearchEffort>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionRecord {
    pub state: GameStateResponse,
    /// Decision made FROM this snapshot; the next record is its result.
    pub decision: Option<DecisionAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryResponse {
    pub schema_version: u32,
    pub session_id: String,
    pub first_available_revision: u64,
    pub current_revision: u64,
    pub records: Vec<PositionRecord>,
    pub next_revision: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_board_agent_zero_q_values_need_no_board_or_search() {
        let value = ValueEstimate { value: 2.5, perspective_agent: 0,
            quantity: ValueQuantity::DiscountedReturn, bounds: None };
        let data = serde_json::to_value(value).unwrap();
        assert_eq!(data["perspective_agent"], 0);
        assert_eq!(data["quantity"], "discounted_return");
        assert_eq!(data["value"], 2.5);
        assert!(data["bounds"].is_null());
    }
}
