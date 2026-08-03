//! `cartridge-eval` — play evaluation games and write the result as JSON.
//!
//! The trainer's orchestrator runs this the same way it runs the actor: as a
//! subprocess over shared files. That keeps the Python side free of any game
//! rules, and lets evaluation use MCTS.
//!
//! ```text
//! cartridge-eval --algorithm alphazero_board_v1 --env-id connect4 --games 50 \
//!     --p1 ./data/models/blobs/sha256/<digest>.onnx --p1-temperature 0.2 --p1-sims 100 \
//!     --p2 random \
//!     --output eval.json
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use algorithm_core::ModelArtifactContract;
use anyhow::{Context, Result};
use clap::Parser;
use evaluator::{
    canonical_evaluation_temperature, preflight_evaluation, run_evaluation,
    validate_evaluation_schedule, validate_onnx_intra_threads, Player, PositionRecord,
};

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

    /// Algorithm cartridge whose policy adapter and evaluation suite to use
    #[arg(long)]
    algorithm: String,

    /// Player 1: "random", or a path to an ONNX model
    #[arg(long, default_value = RANDOM_SPEC)]
    p1: String,

    /// Player 2: "random", or a path to an ONNX model
    #[arg(long, default_value = RANDOM_SPEC)]
    p2: String,

    /// Player 1 sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    p1_temperature: f32,

    /// Player 2 sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    p2_temperature: f32,

    /// Player 1 MCTS simulations per move (0 = play the policy head directly)
    #[arg(long, default_value_t = 0)]
    p1_sims: u32,

    /// Player 2 MCTS simulations per move (0 = play the policy head directly)
    #[arg(long, default_value_t = 0)]
    p2_sims: u32,

    /// Number of games; player 1 takes seat 1 on even game indices
    #[arg(long, default_value_t = 100)]
    games: u32,

    /// Base RNG seed; game N uses seed + N
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Positive ONNX intra-op thread count
    #[arg(long, default_value_t = 1)]
    onnx_intra_threads: usize,

    /// Write the summary here as JSON instead of stdout
    #[arg(long)]
    output: Option<PathBuf>,

    /// Also write every move played, as JSONL, for external move scoring
    #[arg(long)]
    dump_positions: Option<PathBuf>,
}

impl Args {
    /// Validate the complete CLI request before loading models or opening output files.
    fn validated(mut self) -> Result<Self> {
        validate_evaluation_schedule(self.games, self.seed)?;
        validate_onnx_intra_threads(self.onnx_intra_threads)?;
        self.p1_temperature = canonical_evaluation_temperature(self.p1_temperature)?;
        self.p2_temperature = canonical_evaluation_temperature(self.p2_temperature)?;
        Ok(self)
    }
}

fn build_player(
    model_contract: &ModelArtifactContract,
    spec: &str,
    temperature: f32,
    simulations: u32,
    intra_threads: usize,
) -> Result<Player> {
    if spec == RANDOM_SPEC {
        return Ok(Player::Random);
    }
    Player::model(
        model_contract,
        spec,
        temperature,
        simulations,
        intra_threads,
    )
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
    let args = Args::parse().validated()?;

    // Validate dispatch before opening an ONNX model or any output files.
    let algorithm = preflight_evaluation(&args.algorithm, &args.env_id)?;
    let env_contract_version = engine_core::EngineContext::new(&args.env_id)
        .map_err(|error| anyhow::anyhow!("Environment '{}' is unavailable: {error}", args.env_id))?
        .capabilities()
        .contract_version;
    let model_contract = algorithm
        .descriptor()
        .model_artifact_contract(args.env_id.clone(), env_contract_version);

    let mut player1 = build_player(
        &model_contract,
        &args.p1,
        args.p1_temperature,
        args.p1_sims,
        args.onnx_intra_threads,
    )?;
    let mut player2 = build_player(
        &model_contract,
        &args.p2,
        args.p2_temperature,
        args.p2_sims,
        args.onnx_intra_threads,
    )?;

    let mut positions = args.dump_positions.as_ref().map(|_| Vec::new());
    let summary = run_evaluation(
        &args.algorithm,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Args> {
        let mut argv = vec![
            "cartridge-eval",
            "--algorithm",
            algorithm_core::ALPHAZERO_BOARD_V1_ID,
            "--env-id",
            "tictactoe",
        ];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv)?.validated()
    }

    #[test]
    fn cli_accepts_an_explicit_valid_evaluation_request() {
        let args = parse(&[
            "--games",
            "2",
            "--seed",
            "7",
            "--p1-temperature",
            "0.25",
            "--onnx-intra-threads",
            "1",
        ])
        .unwrap();
        assert_eq!(args.games, 2);
        assert_eq!(args.seed, 7);
        assert_eq!(args.p1_temperature, 0.25);
    }

    #[test]
    fn cli_rejects_empty_games_invalid_temperatures_and_auto_threads() {
        assert!(parse(&["--games", "0"]).is_err());
        assert!(parse(&["--p1-temperature", "NaN"]).is_err());
        assert!(parse(&["--p2-temperature", "-0.1"]).is_err());
        assert!(parse(&["--onnx-intra-threads", "0"]).is_err());
    }

    #[test]
    fn cli_canonicalizes_signed_zero_temperature() {
        let args = parse(&["--p1-temperature", "-0.0"]).unwrap();
        assert_eq!(args.p1_temperature.to_bits(), 0);
    }
}
