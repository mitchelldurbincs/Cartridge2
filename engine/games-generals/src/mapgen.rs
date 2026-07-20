//! Procedural map generation, ported from the Go engine's
//! `mapgen/generator.go` and driven by the `reset()` ChaCha20 RNG so the
//! same seed always produces the same map.

use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::board::{idx, manhattan, new_board, xy, Tile, TileKind};
use crate::params::{
    BOARD_SIZE, CITY_RATIO, CITY_START_ARMY, GENERAL_START_ARMY, HEIGHT, MAX_VEIN_LENGTH,
    MIN_GENERAL_SPACING, MIN_VEIN_LENGTH, NUM_MOUNTAIN_VEINS, WIDTH,
};

/// A generated map: tiles plus each player's general tile index
/// (index 0 = player 1, index 1 = player 2).
pub struct GeneratedMap {
    pub tiles: Vec<Tile>,
    pub generals: [usize; 2],
}

/// Generate a map: mountains first, then cities, then generals.
///
/// General placement is guaranteed to succeed: after random attempts with
/// the configured spacing, it falls back to scanning for any spaced normal
/// tile, and finally relaxes spacing entirely (an 8x8 board with 1 vein and
/// 3 cities always has free tiles).
pub fn generate_map(rng: &mut ChaCha20Rng) -> GeneratedMap {
    let mut tiles = new_board();
    place_mountains(&mut tiles, rng);
    place_cities(&mut tiles, rng);
    let generals = place_generals(&mut tiles, rng);
    GeneratedMap { tiles, generals }
}

fn place_mountains(tiles: &mut [Tile], rng: &mut ChaCha20Rng) {
    for _ in 0..NUM_MOUNTAIN_VEINS {
        // Find a seed on a normal, neutral tile
        let mut seed = None;
        for _ in 0..100 {
            let i = rng.gen_range(0..BOARD_SIZE);
            if tiles[i].kind == TileKind::Normal && tiles[i].is_neutral() {
                seed = Some(i);
                break;
            }
        }
        let Some(start) = seed else { continue };

        let mut current = start;
        tiles[current].kind = TileKind::Mountain;
        tiles[current].army = 0;

        let vein_length = if MAX_VEIN_LENGTH > MIN_VEIN_LENGTH {
            MIN_VEIN_LENGTH + rng.gen_range(0..=(MAX_VEIN_LENGTH - MIN_VEIN_LENGTH))
        } else {
            MIN_VEIN_LENGTH
        };

        for _ in 1..vein_length {
            let (x, y) = xy(current);
            let mut candidates = [0usize; 4];
            let mut n = 0;
            for (dx, dy) in [(0isize, -1isize), (1, 0), (0, 1), (-1, 0)] {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && (nx as usize) < WIDTH && ny >= 0 && (ny as usize) < HEIGHT {
                    let ni = idx(nx as usize, ny as usize);
                    if tiles[ni].kind == TileKind::Normal && tiles[ni].is_neutral() {
                        candidates[n] = ni;
                        n += 1;
                    }
                }
            }
            if n == 0 {
                break;
            }
            current = candidates[rng.gen_range(0..n)];
            tiles[current].kind = TileKind::Mountain;
            tiles[current].army = 0;
        }
    }
}

fn place_cities(tiles: &mut [Tile], rng: &mut ChaCha20Rng) {
    let want = BOARD_SIZE / CITY_RATIO;
    let mut placed = 0;
    let mut attempts = 0;
    let max_attempts = want * 20;

    while placed < want && attempts < max_attempts {
        let i = rng.gen_range(0..BOARD_SIZE);
        if tiles[i].is_neutral() && tiles[i].kind == TileKind::Normal {
            tiles[i].kind = TileKind::City;
            tiles[i].army = CITY_START_ARMY;
            placed += 1;
        }
        attempts += 1;
    }
}

fn place_generals(tiles: &mut [Tile], rng: &mut ChaCha20Rng) -> [usize; 2] {
    // Clamp spacing so placement is feasible (Go DefaultMapConfig)
    let spacing = MIN_GENERAL_SPACING.min(WIDTH / 2 + HEIGHT / 2);
    let mut generals = [0usize; 2];

    for player in 0..2usize {
        let placed_so_far = &generals[..player];
        let spot = find_general_spot(tiles, placed_so_far, spacing, rng)
            // Last resort: relax spacing entirely rather than fail reset()
            .or_else(|| find_general_spot(tiles, placed_so_far, 0, rng))
            .expect("8x8 board must have a free tile for a general");

        tiles[spot] = Tile {
            owner: player as u8 + 1,
            army: GENERAL_START_ARMY,
            kind: TileKind::General,
        };
        generals[player] = spot;
    }

    generals
}

fn find_general_spot(
    tiles: &[Tile],
    existing: &[usize],
    spacing: usize,
    rng: &mut ChaCha20Rng,
) -> Option<usize> {
    let spaced_ok = |i: usize| existing.iter().all(|&g| manhattan(i, g) >= spacing);

    // Random attempts first (Go: maxAttempts = W*H)
    for _ in 0..BOARD_SIZE {
        let i = rng.gen_range(0..BOARD_SIZE);
        if tiles[i].is_neutral() && tiles[i].kind == TileKind::Normal && spaced_ok(i) {
            return Some(i);
        }
    }
    // Deterministic fallback scan
    (0..BOARD_SIZE)
        .find(|&i| tiles[i].is_neutral() && tiles[i].kind == TileKind::Normal && spaced_ok(i))
}
