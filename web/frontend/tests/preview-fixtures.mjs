// Explicit UI-only fixtures. This does not simulate games or evaluate a model.
export const definitions = {
  tictactoe: ['TicTacToe', 3, 3, 9, 'grid'],
  connect4: ['Connect 4', 7, 6, 7, 'drop_column'],
  othello: ['Othello forced-pass fixture', 8, 8, 65, 'grid'],
  generals_8x8: ['Generals', 8, 8, 257, 'generals'],
};
export function fixture(id) {
  const [name, width, height, count, board_type] = definitions[id];
  const info = { env_id: id, display_name: name, board_width: width, board_height: height,
    num_actions: count, player_count: 2, player_names: ['Player 1', 'Player 2'],
    player_symbols: ['X', 'O'], board_type, description: 'UI TEST FIXTURE — not a trained model or engine simulation' };
  const cells = Array.from({ length: width * height }, () => ({ owner: 0, kind: 'normal', value: 0 }));
  cells[0].owner = 1;
  let legal = id === 'tictactoe' ? [1, 2, 3, 4, 5, 6, 7, 8] : id === 'connect4' ? [0, 1, 2, 3, 4, 5, 6] : id === 'othello' ? [64] : [36, 37, 38, 39, 160, 161, 162, 163, 256];
  if (id === 'generals_8x8') {
    cells[9] = { owner: 2, kind: 'general', value: 12 };
    cells[40] = { owner: 2, kind: 'normal', value: 8 };
    cells[45] = { owner: 0, kind: 'mountain', value: 0 };
    cells[27] = { owner: 0, kind: 'city', value: 20 };
  }
  function describe(action) {
    if (id === 'connect4') return { action, label: `Column ${action + 1}`, target: { kind: 'column', index: action } };
    if (id === 'othello' && action === 64) return { action, label: 'Pass', target: { kind: 'named', name: 'Pass' } };
    if (id === 'generals_8x8') {
      if (action === 256) return { action, label: 'Wait', target: { kind: 'named', name: 'Wait' } };
      const from = Math.floor(action / 4), to = from + [-8, 1, 8, -1][action % 4];
      return { action, label: `(${Math.floor(from / 8) + 1}, ${from % 8 + 1}) → (${Math.floor(to / 8) + 1}, ${to % 8 + 1})`, target: { kind: 'edge', from, to } };
    }
    return { action, label: `Row ${Math.floor(action / width) + 1}, column ${action % width + 1}`, target: { kind: 'cell', index: action } };
  }
  const state = { session_id: `fixture-${id}`, revision: 1, cells, current_player: 2, human_player: 1,
    winner: 0, game_over: false, legal_moves: legal, actions: legal.map(describe), message: "Bot's turn" };
  const weights = legal.map((_, i) => i + 1), total = weights.reduce((a, b) => a + b, 0);
  const pickTotal = weights.reduce((a, b) => a + b * b, 0);
  const selected = legal[Math.max(0, legal.length - 2)];
  const decision = { schema_version: 1, analysis_id: `${state.session_id}:1`,
    position: { session_id: state.session_id, revision: 1 }, env_id: id, env_contract_version: 2,
    algorithm_id: 'alphazero_board_v1', actor: 2, source: 'alpha_zero_mcts',
    checkpoint: { checkpoint_id: 'a'.repeat(64), training_step: 1500 },
    selected_action: { kind: 'discrete', index: selected },
    network_value: { value: 0.12, perspective_agent: 2, quantity: 'expected_outcome', bounds: [-1, 1] },
    search_value: { value: 0.38, perspective_agent: 2, quantity: 'expected_outcome', bounds: [-1, 1] },
    actions: legal.map((index, i) => ({ action: { kind: 'discrete', index }, network_prior: 1 / legal.length,
      search_prior: 1 / legal.length, visit_share: weights[i] / total, selection_probability: weights[i] ** 2 / pickTotal,
      visits: weights[i], q_value: 0.38, expanded: true })),
    search: { completed_simulations: total, root_visits: total + 1, neural_evaluations: 12, total_time_us: 183400, temperature: 0.5 } };
  const current = structuredClone(state);
  current.revision = 2; current.current_player = 1; current.message = 'Your turn (fixture)';
  return { info, state: current, history: { schema_version: 1, session_id: state.session_id,
    first_available_revision: 1, current_revision: 2, records: [{ state, decision }, { state: current, decision: null }], next_revision: null } };
}
