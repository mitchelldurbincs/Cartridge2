//! `cartridge-eval` — play evaluation games and write the result as JSON.
//!
//! The trainer's orchestrator runs this the same way it runs the actor: as a
//! subprocess over shared files. That keeps the Python side free of any game
//! rules, and lets evaluation use MCTS.
//!
//! ```text
//! cartridge-eval --env-id connect4 --games 50 \
//!     --p1 ./data/models/latest.onnx --p1-temperature 0.2 --p1-sims 100 \
//!     --p2 random \
//!     --output eval.json
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use evaluator::{run_evaluation, Player, PositionRecord};

/// The `--p1`/`--p2` value meaning "the uniform random baseline"; anything
/// else is treated as a path to an ONNX model.
const RANDOM_SPEC: &str = "random";

#[derive(Parser, Debug)]
#[command(
    name = "cartridge-eval",
    about = "Play evaluation games through the Cartridge engine"
)]
struct Args {
    /// Game to play (e.g. tictactoe, connect4, othello, generals_8x8)
    #[arg(long)]
    env_id: String,

    /// Player 1: "random", or a path to an ONNX model
    #[arg(long, default_value = RANDOM_SPEC)]
    p1: String,

    /// Player 2: "random", or a path to an ONNX model
    #[arg(long, default_value = RANDOM_SPEC)]
    p2: String,

    /// Player 1 sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    p1_temperature: f32,

    /// Player 2 sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    p2_temperature: f32,

    /// Player 1 MCTS simulations per move (0 = play the policy head directly)
    #[arg(long, default_value_t = 0)]
    p1_sims: u32,

    /// Player 2 MCTS simulations per move (0 = play the policy head directly)
    #[arg(long, default_value_t = 0)]
    p2_sims: u32,

    /// Number of games; the first half is played with player 1 in seat 1
    #[arg(long, default_value_t = 100)]
    games: u32,

    /// Base RNG seed; game N uses seed + N
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// ONNX intra-op threads (0 = auto)
    #[arg(long, default_value_t = 1)]
    onnx_intra_threads: usize,

    /// Write the summary here as JSON instead of stdout
    #[arg(long)]
    output: Option<PathBuf>,

    /// Also write every move played, as JSONL, for external move scoring
    #[arg(long)]
    dump_positions: Option<PathBuf>,
}

fn build_player(
    env_id: &str,
    spec: &str,
    temperature: f32,
    simulations: u32,
    intra_threads: usize,
) -> Result<Player> {
    if spec == RANDOM_SPEC {
        return Ok(Player::Random);
    }
    Player::model(env_id, spec, temperature, simulations, intra_threads)
}

fn write_positions(path: &PathBuf, positions: &[PositionRecord]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("Failed to create position dump at {}", path.display()))?;
    let mut out = BufWriter::new(file);
    for record in positions {
        serde_json::to_writer(&mut out, record)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    engine_games::register_all_games();

    let mut player1 = build_player(
        &args.env_id,
        &args.p1,
        args.p1_temperature,
        args.p1_sims,
        args.onnx_intra_threads,
    )?;
    let mut player2 = build_player(
        &args.env_id,
        &args.p2,
        args.p2_temperature,
        args.p2_sims,
        args.onnx_intra_threads,
    )?;

    let mut positions = args.dump_positions.as_ref().map(|_| Vec::new());
    let summary = run_evaluation(
        &args.env_id,
        &mut player1,
        &mut player2,
        args.games,
        args.seed,
        positions.as_mut(),
    )?;

    if let (Some(path), Some(positions)) = (&args.dump_positions, &positions) {
        write_positions(path, positions)?;
    }

    let json = serde_json::to_string_pretty(&summary)?;
    match &args.output {
        Some(path) => std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("Failed to write summary to {}", path.display()))?,
        None => println!("{json}"),
    }

    Ok(())
}
