"""Engine descriptor and component identities implemented by AlphaZero."""

from ..environment_catalog import get_algorithm_descriptor

ALGORITHM_ID = "alphazero_board_v1"
DESCRIPTOR = get_algorithm_descriptor(ALGORITHM_ID)

EXPECTED_COMPONENTS = {
    "collector": "alphazero_mcts_self_play_v1",
    "learner": "alphazero_policy_value_v1",
    "orchestration": "synchronized_alphazero_v1",
    "experience_schema": "alphazero_transition_v1",
    "model_contract": "onnx_policy_value_v1",
    "evaluation_suite": "two_player_seat_balanced_v1",
    "serving": "alphazero_mcts_web_v1",
}
MODEL_ARTIFACT_SCHEMA_VERSION = 1
