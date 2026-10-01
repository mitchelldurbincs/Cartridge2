//! Strict parsing and RunCommit binding for immutable EvaluationArtifactV2 evidence.
use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use super::codec::{sha256_hex, validate_digest};
use super::types::{ArtifactProfile, ChampionReferenceV1, EvaluationStatsV1, RunCommitV1};

#[derive(Debug, Clone)]
pub(crate) struct RandomEvaluationV1 {
    games_played: u64,
    candidate_win_rate: f64,
    draw_rate: f64,
    average_game_length: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct EvaluationEvidenceV2 {
    pub iteration: u64,
    pub candidate_checkpoint_id: String,
    pub previous_evaluation_id: Option<String>,
    pub champion_before: Option<ChampionReferenceV1>,
    pub seed: u64,
    pub promoted: bool,
    pub candidate_win_rate: Option<f64>,
    pub draw_rate: Option<f64>,
    pub completed_at: String,
    pub started_at: String,
    pub simulations: u64,
    pub temperature: f64,
    pub promotion_metric: String,
    pub promotion_margin: f64,
    pub win_threshold: f64,
    pub requested_games: [u64; 4],
    pub vs_random: Option<RandomEvaluationV1>,
}

fn exact(value: &Value, label: &str, fields: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("{label} must be an object"))?;
    let missing: Vec<_> = fields
        .iter()
        .filter(|key| !object.contains_key(**key))
        .copied()
        .collect();
    let extra: Vec<_> = object
        .keys()
        .filter(|key| !fields.contains(&key.as_str()))
        .collect();
    if !missing.is_empty() || !extra.is_empty() || object.len() != fields.len() {
        bail!("{label} fields must be exact (missing={missing:?}, extra={extra:?})");
    }
    Ok(())
}
fn uint(value: &Value, label: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| anyhow!("{label} must be a nonnegative integer"))
}
fn checked_sum(values: &[u64], label: &str) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |sum, value| sum.checked_add(*value))
        .ok_or_else(|| anyhow!("{label} count overflows u64"))
}
fn canonical_python_json(value: &Value) -> Result<Vec<u8>> {
    fn write(value: &Value, out: &mut Vec<u8>) -> Result<()> {
        match value {
            Value::Null => out.extend_from_slice(b"null"),
            Value::Bool(v) => out.extend_from_slice(if *v { b"true" } else { b"false" }),
            Value::Number(n) if n.is_f64() => {
                let f = n
                    .as_f64()
                    .ok_or_else(|| anyhow!("invalid floating JSON number"))?;
                if !f.is_finite() {
                    bail!("non-finite JSON number");
                }
                if f == 0.0 {
                    out.extend_from_slice(b"0.0");
                    return Ok(());
                }
                let repr = format!("{f:?}");
                if let Some((mantissa, exponent)) = repr.split_once('e') {
                    let exponent: i32 = exponent.parse()?;
                    let sign = if exponent >= 0 { b'+' } else { b'-' };
                    out.extend_from_slice(mantissa.as_bytes());
                    out.push(b'e');
                    out.push(sign);
                    let magnitude = exponent.unsigned_abs();
                    if magnitude < 10 {
                        out.push(b'0');
                    }
                    out.extend_from_slice(magnitude.to_string().as_bytes());
                } else {
                    out.extend_from_slice(repr.as_bytes());
                }
            }
            Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
            Value::String(s) => out.extend_from_slice(serde_json::to_string(s)?.as_bytes()),
            Value::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write(item, out)?;
                }
                out.push(b']');
            }
            Value::Object(fields) => {
                out.push(b'{');
                for (i, (key, item)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                    out.push(b':');
                    write(item, out)?;
                }
                out.push(b'}');
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    write(value, &mut out)?;
    Ok(out)
}
fn number(value: &Value, label: &str) -> Result<f64> {
    let numeric = value
        .as_number()
        .ok_or_else(|| anyhow!("{label} must be a JSON float"))?;
    if !numeric.is_f64() {
        bail!("{label} must be a JSON float");
    }
    let result = numeric
        .as_f64()
        .ok_or_else(|| anyhow!("{label} must be a finite number"))?;
    if !result.is_finite() {
        bail!("{label} must be finite");
    }
    Ok(if result == 0.0 { 0.0 } else { result })
}
fn digest(value: &Value, label: &str) -> Result<String> {
    let value = value
        .as_str()
        .ok_or_else(|| anyhow!("{label} must be a string"))?;
    validate_digest(label, value)?;
    Ok(value.to_owned())
}
fn optional_digest(value: &Value, label: &str) -> Result<Option<String>> {
    if value.is_null() {
        Ok(None)
    } else {
        digest(value, label).map(Some)
    }
}
fn rate(value: &Value, label: &str) -> Result<f64> {
    let result = number(value, label)?;
    if !(0.0..=1.0).contains(&result) {
        bail!("{label} must be in [0, 1]");
    }
    Ok(result)
}
fn validate_timestamp(value: &Value, label: &str) -> Result<String> {
    let text = value
        .as_str()
        .ok_or_else(|| anyhow!("{label} must be a timestamp string"))?;
    let b = text.as_bytes();
    if b.len() != 27
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'.'
        || b[26] != b'Z'
        || !b
            .iter()
            .enumerate()
            .all(|(i, c)| [4, 7, 10, 13, 16, 19, 26].contains(&i) || c.is_ascii_digit())
    {
        bail!("{label} must use UTC YYYY-MM-DDTHH:MM:SS.ffffffZ format");
    }
    let parse = |a: usize, z: usize| text[a..z].parse::<u32>().unwrap_or(0);
    let (year, month, day, hour, minute, second) = (
        parse(0, 4),
        parse(5, 7),
        parse(8, 10),
        parse(11, 13),
        parse(14, 16),
        parse(17, 19),
    );
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > month_days || hour > 23 || minute > 59 || second > 59 {
        bail!("{label} is not a valid UTC date/time");
    }
    Ok(text.to_owned())
}
fn timestamp_epoch(value: &str) -> f64 {
    let p = |a: usize, b: usize| value[a..b].parse::<i64>().unwrap();
    let (mut y, m, d) = (p(0, 4), p(5, 7), p(8, 10));
    y -= i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let days = era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
    let seconds = days * 86400 + p(11, 13) * 3600 + p(14, 16) * 60 + p(17, 19);
    seconds as f64 + p(20, 26) as f64 / 1_000_000.0
}

pub(crate) fn parse_evaluation(
    label: &str,
    evaluation_id: &str,
    bytes: &[u8],
    expected_profile: &ArtifactProfile,
) -> Result<EvaluationEvidenceV2> {
    validate_digest("evaluation_id", evaluation_id)?;
    let actual = sha256_hex(bytes);
    if actual != evaluation_id {
        bail!("{label} SHA-256 is {actual}, expected {evaluation_id}");
    }
    let value: Value = serde_json::from_slice(bytes)?;
    if canonical_python_json(&value)? != bytes {
        bail!("{label} is not canonical JSON");
    }
    exact(
        &value,
        label,
        &[
            "schema_version",
            "profile",
            "iteration",
            "candidate_checkpoint_id",
            "previous_evaluation_id",
            "champion_before",
            "recipe",
            "results",
            "decision",
            "started_at",
            "completed_at",
        ],
    )?;
    if value["schema_version"].as_u64() != Some(2) {
        bail!("unsupported evaluation schema_version");
    }
    exact(
        &value["profile"],
        "evaluation profile",
        &[
            "algorithm_id",
            "env_id",
            "env_contract_version",
            "model_artifact_schema_version",
            "model_contract",
        ],
    )?;
    let profile: ArtifactProfile = serde_json::from_value(value["profile"].clone())?;
    if &profile != expected_profile {
        bail!("evaluation profile does not match runtime profile");
    }
    let iteration = uint(&value["iteration"], "evaluation.iteration")?;
    if iteration == 0 {
        bail!("evaluation.iteration must be positive");
    }
    let candidate_checkpoint_id = digest(
        &value["candidate_checkpoint_id"],
        "evaluation.candidate_checkpoint_id",
    )?;
    let previous_evaluation_id = optional_digest(
        &value["previous_evaluation_id"],
        "evaluation.previous_evaluation_id",
    )?;
    let champion_before = if value["champion_before"].is_null() {
        None
    } else {
        exact(
            &value["champion_before"],
            "champion_before",
            &["checkpoint_id", "evaluation_id"],
        )?;
        Some(serde_json::from_value::<ChampionReferenceV1>(
            value["champion_before"].clone(),
        )?)
    };
    let recipe = &value["recipe"];
    exact(
        recipe,
        "evaluation recipe",
        &[
            "simulations",
            "temperature",
            "promotion_metric",
            "promotion_margin",
            "win_threshold",
            "seed",
            "seat_schedule",
            "requested_games",
        ],
    )?;
    let seed = uint(&recipe["seed"], "recipe.seed")?;
    let simulations = uint(&recipe["simulations"], "recipe.simulations")?;
    if simulations > u32::MAX as u64 {
        bail!("recipe.simulations exceeds u32");
    }
    let temperature = number(&recipe["temperature"], "recipe.temperature")?;
    if temperature < 0.0 || f64::from(temperature as f32) != temperature {
        bail!("recipe.temperature must be a canonical nonnegative f32");
    }
    let metric = recipe["promotion_metric"]
        .as_str()
        .ok_or_else(|| anyhow!("recipe.promotion_metric must be a string"))?;
    if metric != "win_rate" && metric != "solver_optimal" {
        bail!("unsupported evaluation promotion_metric");
    }
    let margin = rate(&recipe["promotion_margin"], "recipe.promotion_margin")?;
    let threshold = rate(&recipe["win_threshold"], "recipe.win_threshold")?;
    if (metric == "win_rate" && margin != 0.0) || (metric == "solver_optimal" && threshold != 0.0) {
        bail!("inactive promotion parameter must be zero");
    }
    let temperature = number(&recipe["temperature"], "recipe.temperature")?;
    if temperature < 0.0 || f64::from(temperature as f32) != temperature {
        bail!("recipe.temperature must be a canonical finite f32");
    }
    let schedule = &recipe["seat_schedule"];
    exact(
        schedule,
        "seat_schedule",
        &["kind", "candidate_first", "seed_rule"],
    )?;
    if schedule
        != &serde_json::json!({"kind":"alternating_v1","candidate_first":"even_game_indices","seed_rule":"base_plus_game_index"})
    {
        bail!("unsupported evaluation seat schedule");
    }
    let requested = &recipe["requested_games"];
    exact(
        requested,
        "requested_games",
        &[
            "vs_champion",
            "vs_random",
            "candidate_solver",
            "champion_solver",
        ],
    )?;
    let mut req = [0u64; 4];
    for (i, key) in [
        "vs_champion",
        "vs_random",
        "candidate_solver",
        "champion_solver",
    ]
    .iter()
    .enumerate()
    {
        req[i] = uint(&requested[*key], key)?;
        if req[i] > u32::MAX as u64 {
            bail!("requested games exceeds u32");
        }
    }
    if req[0] + req[1] + req[2] == 0 {
        bail!("evaluation recipe requests no candidate evidence");
    }
    if metric == "win_rate" && req[3] != 0 {
        bail!("win_rate promotion cannot request champion solver evidence");
    }
    let results = &value["results"];
    exact(
        results,
        "evaluation results",
        &[
            "vs_champion",
            "vs_random",
            "candidate_solver",
            "champion_solver",
        ],
    )?;
    let keys = [
        "vs_champion",
        "vs_random",
        "candidate_solver",
        "champion_solver",
    ];
    let mut h2h_rate = None;
    let mut h2h_draw = None;
    let mut vs_random = None;
    for i in 0..4 {
        let item = &results[keys[i]];
        if req[i] == 0 {
            if !item.is_null() {
                bail!("{} result exists with zero request", keys[i]);
            }
            continue;
        }
        if item.is_null() {
            bail!("{} result missing", keys[i]);
        }
        if i < 2 {
            exact(
                item,
                "head-to-head result",
                &[
                    "games_played",
                    "candidate_wins",
                    "opponent_wins",
                    "draws",
                    "candidate_wins_as_first",
                    "candidate_wins_as_second",
                    "opponent_wins_while_candidate_first",
                    "opponent_wins_while_candidate_second",
                    "average_game_length",
                ],
            )?;
            let games = uint(&item["games_played"], "games_played")?;
            let wins = uint(&item["candidate_wins"], "candidate_wins")?;
            let losses = uint(&item["opponent_wins"], "opponent_wins")?;
            let draws = uint(&item["draws"], "draws")?;
            if games != req[i]
                || checked_sum(&[wins, losses, draws], "head-to-head outcome")? != games
            {
                bail!("head-to-head results do not partition requested games");
            }
            let avg = number(&item["average_game_length"], "average_game_length")?;
            if avg < 0.0 || (games == 0) != (avg == 0.0) {
                bail!("head-to-head average game length is invalid");
            }
            let candidate_first =
                uint(&item["candidate_wins_as_first"], "candidate_wins_as_first")?;
            let candidate_second = uint(
                &item["candidate_wins_as_second"],
                "candidate_wins_as_second",
            )?;
            let opponent_first = uint(
                &item["opponent_wins_while_candidate_first"],
                "opponent_wins_while_candidate_first",
            )?;
            let opponent_second = uint(
                &item["opponent_wins_while_candidate_second"],
                "opponent_wins_while_candidate_second",
            )?;
            if checked_sum(&[candidate_first, candidate_second], "candidate seat wins")? != wins
                || checked_sum(&[opponent_first, opponent_second], "opponent seat wins")? != losses
                || checked_sum(&[candidate_first, opponent_first], "first seat outcomes")?
                    > games.div_ceil(2)
                || checked_sum(&[candidate_second, opponent_second], "second seat outcomes")?
                    > games / 2
            {
                bail!("head-to-head seat outcomes are inconsistent");
            }
            for (a, b) in [
                ("candidate_wins_as_first", "candidate_wins_as_second"),
                (
                    "opponent_wins_while_candidate_first",
                    "opponent_wins_while_candidate_second",
                ),
            ] {
                let x = uint(&item[a], a)?;
                let y = uint(&item[b], b)?;
                if checked_sum(&[x, y], "seat outcomes")?
                    != if a.starts_with("candidate") {
                        wins
                    } else {
                        losses
                    }
                {
                    bail!("seat outcomes do not sum to results");
                }
            }
            if i == 0 {
                h2h_rate = Some(wins as f64 / games as f64);
                h2h_draw = Some(draws as f64 / games as f64);
            } else {
                vs_random = Some(RandomEvaluationV1 {
                    games_played: games,
                    candidate_win_rate: wins as f64 / games as f64,
                    draw_rate: draws as f64 / games as f64,
                    average_game_length: avg,
                });
            }
        } else {
            // Solver results have a fixed schema and must report the requested game count.
            exact(
                item,
                "solver result",
                &[
                    "games_played",
                    "candidate_wins",
                    "opponent_wins",
                    "draws",
                    "average_game_length",
                    "overall",
                    "by_ply",
                    "by_seat",
                    "solver_queries",
                    "solver_cache_hits",
                    "solver_time_seconds",
                    "wall_time_seconds",
                    "solver_version",
                ],
            )?;
            if uint(&item["games_played"], "solver.games_played")? != req[i] {
                bail!("solver games_played does not match request");
            }
            if item["solver_version"].as_str().is_none_or(str::is_empty) {
                bail!("solver_version must be nonempty");
            }
            let games = uint(&item["games_played"], "solver.games_played")?;
            let wins = uint(&item["candidate_wins"], "solver.candidate_wins")?;
            let losses = uint(&item["opponent_wins"], "solver.opponent_wins")?;
            let draws = uint(&item["draws"], "solver.draws")?;
            if checked_sum(&[wins, losses, draws], "solver outcomes")? != games {
                bail!("solver outcomes do not partition games");
            }
            let avg = number(&item["average_game_length"], "solver.average_game_length")?;
            if avg < 0.0 || (games == 0) != (avg == 0.0) {
                bail!("solver average game length is invalid");
            }
            let by_ply = &item["by_ply"];
            exact(
                by_ply,
                "solver.by_ply",
                &["ply_1_8", "ply_9_20", "ply_21_plus"],
            )?;
            let by_seat = &item["by_seat"];
            exact(by_seat, "solver.by_seat", &["first", "second"])?;
            let mut buckets = vec![&item["overall"]];
            for key in ["ply_1_8", "ply_9_20", "ply_21_plus"] {
                buckets.push(&by_ply[key]);
            }
            for key in ["first", "second"] {
                buckets.push(&by_seat[key]);
            }
            for bucket in &buckets {
                exact(
                    bucket,
                    "solver bucket",
                    &[
                        "positions",
                        "value_optimal",
                        "exact_best",
                        "blunders_win_to_draw",
                        "blunders_win_to_loss",
                        "blunders_draw_to_loss",
                        "forced",
                    ],
                )?;
                let positions = uint(&bucket["positions"], "solver.positions")?;
                let optimal = uint(&bucket["value_optimal"], "solver.value_optimal")?;
                let exact_best = uint(&bucket["exact_best"], "solver.exact_best")?;
                let forced = uint(&bucket["forced"], "solver.forced")?;
                let b1 = uint(
                    &bucket["blunders_win_to_draw"],
                    "solver.blunders_win_to_draw",
                )?;
                let b2 = uint(
                    &bucket["blunders_win_to_loss"],
                    "solver.blunders_win_to_loss",
                )?;
                let b3 = uint(
                    &bucket["blunders_draw_to_loss"],
                    "solver.blunders_draw_to_loss",
                )?;
                if optimal > positions
                    || exact_best > optimal
                    || forced > exact_best
                    || checked_sum(&[b1, b2, b3], "solver blunders")? != positions - optimal
                {
                    bail!("solver bucket counts are inconsistent");
                }
            }
            let overall = &item["overall"];
            for field in [
                "positions",
                "value_optimal",
                "exact_best",
                "blunders_win_to_draw",
                "blunders_win_to_loss",
                "blunders_draw_to_loss",
                "forced",
            ] {
                for slices in [by_ply, by_seat] {
                    let mut sum = 0u64;
                    for bucket in slices.as_object().unwrap().values() {
                        sum = sum
                            .checked_add(uint(&bucket[field], "solver slice")?)
                            .ok_or_else(|| anyhow!("solver slice count overflow"))?;
                    }
                    if sum != uint(&overall[field], "solver overall")? {
                        bail!("solver slices do not partition overall counts");
                    }
                }
            }
            let queries = uint(&item["solver_queries"], "solver_queries")?;
            let cache = uint(&item["solver_cache_hits"], "solver_cache_hits")?;
            if cache > queries || queries != uint(&overall["positions"], "solver positions")? {
                bail!("solver query/cache counts are inconsistent");
            }
            if number(&item["solver_time_seconds"], "solver_time_seconds")? < 0.0
                || number(&item["wall_time_seconds"], "wall_time_seconds")? < 0.0
            {
                bail!("solver times must be nonnegative");
            }
        }
    }
    if champion_before.is_none() && (req[0] != 0 || req[3] != 0) {
        bail!("champion evidence requires a champion");
    }
    if champion_before.is_some() && (previous_evaluation_id.is_none() || req[0] == 0) {
        bail!("evaluation with champion must link and compare against prior champion");
    }
    if champion_before.is_none() && previous_evaluation_id.is_some() {
        bail!("first evaluation cannot refer to a prior evaluation");
    }
    if champion_before
        .as_ref()
        .is_some_and(|c| c.checkpoint_id == candidate_checkpoint_id)
    {
        bail!("candidate checkpoint cannot be its own champion");
    }
    let decision = &value["decision"];
    exact(decision, "promotion decision", &["promoted", "reason"])?;
    let promoted = decision["promoted"]
        .as_bool()
        .ok_or_else(|| anyhow!("decision.promoted must be boolean"))?;
    if decision["reason"].as_str().is_none_or(str::is_empty) {
        bail!("decision.reason must be a nonempty string");
    }
    if champion_before.is_none() && !promoted {
        bail!("the first valid candidate must establish champion state");
    }
    if metric == "win_rate" && champion_before.is_some() {
        let observed =
            h2h_rate.ok_or_else(|| anyhow!("win-rate decision requires head-to-head evidence"))?;
        if promoted != (observed > threshold) {
            bail!("promotion decision disagrees with win-rate evidence");
        }
    }
    if metric == "solver_optimal" && champion_before.is_some() {
        if req[2] == 0 || req[3] == 0 || req[2] != req[3] {
            bail!("solver-optimal promotion requires equal candidate/champion solver evidence");
        }
        let c = &results["candidate_solver"];
        let h = &results["champion_solver"];
        if c["solver_version"] != h["solver_version"] {
            bail!("solver versions differ");
        }
        for side in [c, h] {
            exact(
                &side["overall"],
                "solver overall",
                &[
                    "positions",
                    "value_optimal",
                    "exact_best",
                    "blunders_win_to_draw",
                    "blunders_win_to_loss",
                    "blunders_draw_to_loss",
                    "forced",
                ],
            )?;
        }
        let cpositions = uint(&c["overall"]["positions"], "positions")?;
        let hpositions = uint(&h["overall"]["positions"], "positions")?;
        let c_rate = if cpositions == 0 {
            0.0
        } else {
            uint(&c["overall"]["value_optimal"], "value_optimal")? as f64 / cpositions as f64
        };
        let h_rate = if hpositions == 0 {
            0.0
        } else {
            uint(&h["overall"]["value_optimal"], "value_optimal")? as f64 / hpositions as f64
        };
        if promoted != (c_rate > h_rate + margin) {
            bail!("promotion decision disagrees with solver evidence");
        }
    }
    let started = validate_timestamp(&value["started_at"], "evaluation.started_at")?;
    let completed = validate_timestamp(&value["completed_at"], "evaluation.completed_at")?;
    if completed <= started {
        bail!("evaluation timestamps are invalid or non-increasing");
    }
    Ok(EvaluationEvidenceV2 {
        iteration,
        candidate_checkpoint_id,
        previous_evaluation_id,
        champion_before,
        seed,
        promoted,
        candidate_win_rate: h2h_rate,
        draw_rate: h2h_draw,
        completed_at: completed.to_owned(),
        started_at: started,
        simulations,
        temperature,
        promotion_metric: metric.to_owned(),
        promotion_margin: margin,
        win_threshold: threshold,
        requested_games: req,
        vs_random,
    })
}

pub(crate) fn bind_evaluation(
    commit: &RunCommitV1,
    parent: Option<&RunCommitV1>,
    evidence: &EvaluationEvidenceV2,
    previous_evidence: Option<&EvaluationEvidenceV2>,
    evaluation_id: &str,
) -> Result<()> {
    let orchestration = commit
        .orchestration
        .as_ref()
        .ok_or_else(|| anyhow!("evaluation evidence requires orchestration"))?;
    let recipe = commit
        .run_recipe
        .as_ref()
        .ok_or_else(|| anyhow!("evaluation requires an immutable RunRecipe"))?;
    let inherited_champion = parent.and_then(|p| p.champion.as_ref());
    let inherited_evaluation = parent.and_then(|p| p.evaluation_head_id.as_deref());
    let has_champion = inherited_champion.is_some();
    let solver_enabled = commit.profile.env_id == "connect4" && recipe.solver_games > 0;
    let expected_requested = [
        if has_champion {
            u64::from(recipe.evaluation_games)
        } else {
            0
        },
        if recipe._evaluation_vs_random {
            u64::from(recipe.evaluation_games)
        } else {
            0
        },
        if solver_enabled {
            u64::from(recipe.solver_games)
        } else {
            0
        },
        if solver_enabled && has_champion {
            u64::from(recipe.solver_games)
        } else {
            0
        },
    ];
    if evidence.simulations != u64::from(recipe._evaluation_simulations)
        || evidence.temperature != recipe.evaluation_temperature
        || evidence.promotion_metric != recipe.promotion_metric
        || evidence.promotion_margin != recipe.promotion_margin
        || evidence.win_threshold != recipe.evaluation_win_threshold
        || evidence.seed != recipe.evaluation_seed
        || evidence.requested_games != expected_requested
    {
        bail!("Evaluation evidence disagrees with immutable RunRecipe");
    }
    let mut expected_history = parent.map_or_else(Vec::new, |p| {
        p.stats_snapshot.stats.evaluation_history.clone()
    });
    if let Some(random) = &evidence.vs_random {
        let mut metrics = std::collections::BTreeMap::new();
        metrics.insert("outcome/win_rate".to_owned(), random.candidate_win_rate);
        metrics.insert("outcome/draw_rate".to_owned(), random.draw_rate);
        metrics.insert(
            "outcome/loss_rate".to_owned(),
            1.0 - random.candidate_win_rate - random.draw_rate,
        );
        expected_history.push(EvaluationStatsV1 {
            step: commit.stats_snapshot.step,
            metrics,
            episodes: random.games_played,
            mean_episode_length: random.average_game_length,
            timestamp: timestamp_epoch(&evidence.completed_at),
        });
        if expected_history.len() > 50 {
            expected_history = expected_history.split_off(expected_history.len() - 50);
        }
    }
    if commit.stats_snapshot.stats.evaluation_history != expected_history {
        bail!("RunCommit stats evaluation history disagrees with immutable evaluation evidence");
    }
    if evidence.iteration != orchestration.iteration
        || evidence.candidate_checkpoint_id != commit.checkpoint_id
        || evidence.previous_evaluation_id.as_deref() != inherited_evaluation
        || evidence.champion_before.as_ref() != inherited_champion
        || evidence.seed != orchestration.evaluation_seed.unwrap_or(u64::MAX)
        || commit.evaluation_head_id.as_deref() != Some(evaluation_id)
    {
        bail!("RunCommit evaluation does not match its authoritative transition");
    }
    if evidence.completed_at > orchestration.timestamp {
        bail!("RunCommit timestamp precedes evaluation completion");
    }
    if let Some(previous) = previous_evidence {
        if evidence.iteration <= previous.iteration || evidence.started_at <= previous.completed_at
        {
            bail!("evaluation chronology must strictly follow its prior evidence");
        }
    } else if evidence.previous_evaluation_id.is_some() {
        bail!("previous evaluation evidence is absent from the RunCommit lineage");
    }
    if orchestration.eval_win_rate != evidence.candidate_win_rate
        || orchestration.eval_draw_rate != evidence.draw_rate
    {
        bail!("RunCommit evaluation rates disagree with immutable evidence");
    }
    let expected_champion = if evidence.promoted {
        Some(ChampionReferenceV1 {
            checkpoint_id: commit.checkpoint_id.clone(),
            evaluation_id: evaluation_id.to_owned(),
        })
    } else {
        inherited_champion.cloned()
    };
    if commit.champion != expected_champion {
        bail!("RunCommit champion disagrees with evaluation decision");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::json!({
            "schema_version":2,
            "profile":{"algorithm_id":"algo","env_id":"connect4","env_contract_version":1,"model_artifact_schema_version":1,"model_contract":"contract"},
            "iteration":2,"candidate_checkpoint_id":"a".repeat(64),"previous_evaluation_id":"b".repeat(64),
            "champion_before":{"checkpoint_id":"c".repeat(64),"evaluation_id":"b".repeat(64)},
            "recipe":{"simulations":0,"temperature":0.0,"promotion_metric":"win_rate","promotion_margin":0.0,"win_threshold":0.5,"seed":5,
                "seat_schedule":{"kind":"alternating_v1","candidate_first":"even_game_indices","seed_rule":"base_plus_game_index"},
                "requested_games":{"vs_champion":2,"vs_random":0,"candidate_solver":0,"champion_solver":0}},
            "results":{"vs_champion":{"games_played":2,"candidate_wins":2,"opponent_wins":0,"draws":0,"candidate_wins_as_first":1,"candidate_wins_as_second":1,"opponent_wins_while_candidate_first":0,"opponent_wins_while_candidate_second":0,"average_game_length":1.0},"vs_random":null,"candidate_solver":null,"champion_solver":null},
            "decision":{"promoted":true,"reason":"candidate exceeded threshold"},"started_at":"2026-01-01T00:00:00.000000Z","completed_at":"2026-01-01T00:00:01.000000Z"
        })
    }
    fn reject(value: Value) {
        let bytes = serde_json::to_vec(&value).unwrap();
        let id = sha256_hex(&bytes);
        let profile: ArtifactProfile = serde_json::from_value(value["profile"].clone()).unwrap();
        assert!(parse_evaluation("test evaluation", &id, &bytes, &profile).is_err());
    }
    #[test]
    fn rejects_decision_and_seat_capacity_that_disagree_with_evidence() {
        let mut bad = fixture();
        bad["decision"]["promoted"] = Value::Bool(false);
        reject(bad);
        let mut bad = fixture();
        bad["results"]["vs_champion"]["candidate_wins_as_first"] = Value::from(2);
        bad["results"]["vs_champion"]["candidate_wins_as_second"] = Value::from(0);
        reject(bad);
    }

    #[test]
    fn accepts_python_padded_exponent_and_rejects_int_normalized_float() {
        let mut value = fixture();
        value["recipe"]["win_threshold"] = serde_json::json!(0.000001);
        let initial = serde_json::to_vec(&value).unwrap();
        let python_bytes = String::from_utf8(initial)
            .unwrap()
            .replace("1e-6", "1e-06")
            .into_bytes();
        let id = sha256_hex(&python_bytes);
        let profile: ArtifactProfile = serde_json::from_value(value["profile"].clone()).unwrap();
        assert!(parse_evaluation("test evaluation", &id, &python_bytes, &profile).is_ok());
        value["recipe"]["win_threshold"] = Value::from(1);
        let bytes = serde_json::to_vec(&value).unwrap();
        let id = sha256_hex(&bytes);
        assert!(parse_evaluation("test evaluation", &id, &bytes, &profile).is_err());
    }

    #[test]
    fn python_float_canonicalization_normalizes_negative_zero() {
        let mut value = fixture();
        value["recipe"]["win_threshold"] = Value::from(-0.0f64);
        let bytes = serde_json::to_vec(&value).unwrap();
        let id = sha256_hex(&bytes);
        let profile: ArtifactProfile = serde_json::from_value(value["profile"].clone()).unwrap();
        assert!(parse_evaluation("test evaluation", &id, &bytes, &profile).is_err());
    }
}
