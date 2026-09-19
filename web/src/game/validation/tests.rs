use super::*;
use crate::game::GameSession;

fn initial_position(env_id: &str) -> (EngineContext, engine_core::ResetResult, BoardGameMetadata) {
    engine_games::register_all_environments();
    let mut ctx = EngineContext::new(env_id).unwrap();
    let board = ctx.metadata().require_board().unwrap().clone();
    let reset = ctx.reset(42, &[]).unwrap();
    (ctx, reset, board)
}

#[test]
fn bundled_games_validate_reset_and_alternating_step() {
    for env_id in ["tictactoe", "connect4", "othello", "generals_8x8"] {
        let (mut ctx, reset, board) = initial_position(env_id);
        let view = validate_position(
            &ctx,
            &reset.state,
            &reset.timestep,
            &board,
            ExpectedTransition::Reset,
        )
        .unwrap();
        assert_eq!(view.winner, 0, "{env_id}");
        assert_eq!(view.cells.len(), board.board_size().unwrap(), "{env_id}");

        let (actor, _, legal) = active_observation(&reset.timestep).unwrap();
        let action = legal.iter_ones().next().unwrap() as u32;
        let step = ctx.step(&reset.state, &action.to_le_bytes()).unwrap();
        let next = validate_position(
            &ctx,
            &step.state,
            &step.timestep,
            &board,
            ExpectedTransition::Agent(actor),
        )
        .unwrap();
        assert_ne!(next.current_player, view.current_player, "{env_id}");
        assert_eq!(next.winner, 0, "{env_id}");
    }
}

#[test]
fn validates_wins_for_both_seats_and_draws() {
    let cases: &[(&[u32], u8)] = &[
        (&[0, 3, 1, 4, 2], 1),
        (&[0, 3, 1, 4, 8, 5], 2),
        (&[0, 1, 2, 4, 3, 5, 7, 6, 8], 3),
    ];
    for (actions, winner) in cases {
        let (mut ctx, reset, board) = initial_position("tictactoe");
        let mut state = reset.state;
        let mut timestep = reset.timestep;
        let mut view = None;
        for action in *actions {
            let actor = active_observation(&timestep).unwrap().0;
            let step = ctx.step(&state, &action.to_le_bytes()).unwrap();
            view = Some(
                validate_position(
                    &ctx,
                    &step.state,
                    &step.timestep,
                    &board,
                    ExpectedTransition::Agent(actor),
                )
                .unwrap(),
            );
            state = step.state;
            timestep = step.timestep;
        }
        assert_eq!(timestep.episode, EpisodeStatus::Terminated);
        assert_eq!(view.unwrap().winner, *winner);
    }
}

#[test]
fn validates_truncation_without_fabricating_a_winner() {
    let (ctx, reset, board) = initial_position("tictactoe");
    let mut timestep = reset.timestep;
    timestep.episode = EpisodeStatus::Truncated;
    timestep.decision = Decision::None;
    timestep.source = TransitionSource::Agents {
        agent_ids: vec![AgentId(2)],
    };
    for outcome in &mut timestep.outcomes {
        outcome.truncated = true;
    }
    let view = validate_position(
        &ctx,
        &reset.state,
        &timestep,
        &board,
        ExpectedTransition::Agent(AgentId(2)),
    )
    .unwrap();
    assert_eq!(view.winner, 0);
    assert!(timestep.episode.is_done());

    timestep.outcomes[0].reward = 1.0;
    assert_eq!(
        validate_timestep(&timestep, ExpectedTransition::Agent(AgentId(2)))
            .unwrap_err()
            .to_string(),
        "AlphaZero terminal-only reward contract emitted (1, 0) for Truncated"
    );
}

type TimestepCase = (fn(&mut ErasedTimestep), &'static str);

#[test]
fn rejects_malformed_running_timesteps_with_existing_errors() {
    let (_, reset, _) = initial_position("tictactoe");
    let cases: &[TimestepCase] = &[
        (
            |t| t.source = TransitionSource::Chance,
            "reset timestep has invalid transition source Chance",
        ),
        (
            |t| {
                t.outcomes.pop();
            },
            "AlphaZero board timestep requires exactly two per-agent outcomes, got 1",
        ),
        (
            |t| t.outcomes[1].agent_id = AgentId(1),
            "AlphaZero board timestep requires exactly one outcome for agent 1, got 2",
        ),
        (
            |t| t.outcomes[0].reward = f32::NAN,
            "AlphaZero board timestep has a non-finite reward for agent 1",
        ),
        (
            |t| t.outcomes[0].truncated = true,
            "outcome flags for agent 1 disagree with episode status Running",
        ),
        (
            |t| t.outcomes[0].reward = 1.0,
            "AlphaZero terminal-only reward contract emitted (1, 0) for Running",
        ),
        (
            |t| t.decision = Decision::single(AgentId(3), ActionAvailability::All),
            "AlphaZero board serving only supports seats 1 and 2, got 3",
        ),
    ];
    for (mutate, expected) in cases {
        let mut timestep = reset.timestep.clone();
        mutate(&mut timestep);
        assert_eq!(
            validate_timestep(&timestep, ExpectedTransition::Reset)
                .unwrap_err()
                .to_string(),
            *expected
        );
    }
}

#[test]
fn rejects_wrong_step_source_and_repeated_actor() {
    let (_, reset, _) = initial_position("tictactoe");
    let mut timestep = reset.timestep;
    assert_eq!(
        validate_timestep(&timestep, ExpectedTransition::Agent(AgentId(1)))
            .unwrap_err()
            .to_string(),
        "step by agent 1 has invalid transition source Reset"
    );
    timestep.source = TransitionSource::Agents {
        agent_ids: vec![AgentId(1)],
    };
    assert_eq!(
        validate_timestep(&timestep, ExpectedTransition::Agent(AgentId(1)))
            .unwrap_err()
            .to_string(),
        "alternating-turn board transition kept agent 1 active"
    );
    for agent_ids in [vec![AgentId(2)], vec![AgentId(1), AgentId(2)]] {
        timestep.source = TransitionSource::Agents { agent_ids };
        assert_eq!(
            validate_timestep(&timestep, ExpectedTransition::Agent(AgentId(1)))
                .unwrap_err()
                .to_string(),
            format!(
                "step by agent 1 has invalid transition source {:?}",
                timestep.source
            )
        );
    }
}

#[test]
fn rejects_inconsistent_completed_timesteps() {
    let (_, reset, _) = initial_position("tictactoe");
    let mut terminal = reset.timestep;
    terminal.episode = EpisodeStatus::Terminated;
    terminal.decision = Decision::None;
    for outcome in &mut terminal.outcomes {
        outcome.terminated = true;
    }
    let cases: &[TimestepCase] = &[
        (
            |t| t.outcomes[0].reward = 1.0,
            "AlphaZero terminal rewards must be zero-sum, got (1, 0)",
        ),
        (
            |t| t.decision = Decision::Chance,
            "completed board timestep must have no next decision, got Chance",
        ),
        (
            |t| t.observations[0].agent_id = AgentId(3),
            "completed board observation belongs to unsupported agent 3",
        ),
    ];
    for (mutate, expected) in cases {
        let mut timestep = terminal.clone();
        mutate(&mut timestep);
        assert_eq!(
            validate_timestep(&timestep, ExpectedTransition::Reset)
                .unwrap_err()
                .to_string(),
            *expected
        );
    }
    assert_eq!(
        active_observation(&terminal).unwrap_err().to_string(),
        "AlphaZero board action selection requires a running episode, got Terminated"
    );
}

type ViewCase = (fn(&mut BoardView), &'static str);

#[test]
fn rejects_inconsistent_board_projections() {
    let (_, reset, board) = initial_position("tictactoe");
    let cases: &[ViewCase] = &[
        (
            |v| {
                v.cells.pop();
            },
            "board presentation has 8 cells, metadata declares 3x3 (9 cells)",
        ),
        (
            |v| v.current_player = 0,
            "board presentation current player must be seat 1 or 2, got 0",
        ),
        (
            |v| v.cells[0].owner = 3,
            "board presentation contains an owner outside seats 1 and 2",
        ),
        (
            |v| v.current_player = 2,
            "board presentation current player 2 disagrees with sole observation agent 1",
        ),
        (
            |v| v.winner = 1,
            "running episode has terminal board winner 1",
        ),
    ];
    for (mutate, expected) in cases {
        let mut view = BoardView::from_owners(&[0; 9], 1, 0);
        mutate(&mut view);
        assert_eq!(
            validate_board_view(view, &reset.timestep, &board)
                .unwrap_err()
                .to_string(),
            *expected
        );
    }
}

#[test]
fn board_winner_must_match_episode_and_rewards() {
    let (_, reset, board) = initial_position("tictactoe");
    let mut timestep = reset.timestep;
    timestep.episode = EpisodeStatus::Terminated;
    timestep.outcomes[0].reward = 1.0;
    timestep.outcomes[1].reward = -1.0;
    for (winner, expected) in [
        (
            0,
            "terminated episode requires winner 1, 2, or draw marker 3, got 0",
        ),
        (2, "board winner 2 disagrees with per-agent rewards (1, -1)"),
        (3, "board winner 3 disagrees with per-agent rewards (1, -1)"),
    ] {
        let view = BoardView::from_owners(&[0; 9], 1, winner);
        assert_eq!(
            validate_board_view(view, &timestep, &board)
                .unwrap_err()
                .to_string(),
            expected
        );
    }
    timestep.episode = EpisodeStatus::Truncated;
    let view = BoardView::from_owners(&[0; 9], 1, 1);
    assert_eq!(
        validate_board_view(view, &timestep, &board)
            .unwrap_err()
            .to_string(),
        "truncated episode must not fabricate a winner, got 1"
    );
}

#[test]
fn transition_errors_precede_presentation_lookup() {
    let (ctx, reset, board) = initial_position("tictactoe");
    let mut timestep = reset.timestep;
    timestep.source = TransitionSource::Chance;
    timestep.outcomes.clear();
    // Empty state cannot be presented, but source and outcomes must fail first.
    assert_eq!(
        validate_position(&ctx, &[], &timestep, &board, ExpectedTransition::Reset)
            .unwrap_err()
            .to_string(),
        "reset timestep has invalid transition source Chance"
    );
    timestep.source = TransitionSource::Reset;
    assert_eq!(
        validate_position(&ctx, &[], &timestep, &board, ExpectedTransition::Reset)
            .unwrap_err()
            .to_string(),
        "AlphaZero board timestep requires exactly two per-agent outcomes, got 0"
    );
}

#[test]
fn legal_moves_rejects_a_missing_discrete_mask() {
    engine_games::register_all_environments();

    let mut session = GameSession::new("tictactoe").unwrap();
    session.timestep.decision = Decision::single(AgentId(1), ActionAvailability::All);

    assert!(session.legal_moves().is_err());
    assert!(session.is_legal_move(0).is_err());
}

#[test]
fn active_position_rejects_chance_and_simultaneous_decisions() {
    engine_games::register_all_environments();
    let mut session = GameSession::new("tictactoe").unwrap();

    session.timestep.decision = Decision::Chance;
    assert!(session
        .legal_moves()
        .unwrap_err()
        .to_string()
        .contains("Chance"));

    session.timestep.decision = Decision::agents([AgentId(1), AgentId(2)]);
    assert!(session
        .legal_moves()
        .unwrap_err()
        .to_string()
        .contains("exactly one active decision agent"));
}

#[test]
fn active_position_requires_one_matching_observation() {
    engine_games::register_all_environments();
    let mut session = GameSession::new("tictactoe").unwrap();

    session.timestep.observations[0].agent_id = AgentId(2);
    assert!(session
        .legal_moves()
        .unwrap_err()
        .to_string()
        .contains("sole observation belongs to agent 2"));

    session.timestep.observations.clear();
    assert!(session
        .legal_moves()
        .unwrap_err()
        .to_string()
        .contains("exactly one observation"));
}

#[test]
fn timestep_validation_requires_per_agent_outcomes_matching_episode_status() {
    engine_games::register_all_environments();
    let session = GameSession::new("tictactoe").unwrap();
    let mut timestep = session.timestep.clone();

    timestep.outcomes.pop();
    assert!(validate_timestep(&timestep, ExpectedTransition::Reset)
        .unwrap_err()
        .to_string()
        .contains("exactly two per-agent outcomes"));

    let mut timestep = session.timestep.clone();
    timestep.outcomes[0].terminated = true;
    assert!(validate_timestep(&timestep, ExpectedTransition::Reset)
        .unwrap_err()
        .to_string()
        .contains("disagree with episode status"));
}

#[test]
fn board_serving_rejects_custom_or_missing_presentations() {
    let custom = Presentation::Custom {
        contract: "counter_text_v1".to_string(),
        payload: Vec::new(),
    };
    assert!(require_board_presentation(Some(custom))
        .unwrap_err()
        .to_string()
        .contains("custom contract 'counter_text_v1'"));
    assert!(require_board_presentation(None)
        .unwrap_err()
        .to_string()
        .contains("requires the environment to expose a board presentation"));
}
