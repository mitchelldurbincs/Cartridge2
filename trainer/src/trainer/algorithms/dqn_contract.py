"""Engine descriptor and component identities implemented by DQN."""

from ..environment_catalog import get_algorithm_descriptor

ALGORITHM_ID = "dqn_v1"
DESCRIPTOR = get_algorithm_descriptor(ALGORITHM_ID)

EXPECTED_COMPONENTS = {
    "collector": "dqn_epsilon_greedy_v1",
    "learner": "dqn_q_learning_v1",
    "orchestration": "off_policy_dqn_v1",
    "experience_schema": "dqn_transition_v1",
    "model_contract": "onnx_q_values_v1",
    "evaluation_suite": "single_agent_return_v1",
    "serving": "dqn_greedy_v1",
}
