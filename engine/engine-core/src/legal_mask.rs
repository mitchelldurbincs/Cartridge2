//! Dynamic-width discrete action-availability mask.
//!
//! A `LegalMask` holds one bit per action for any action-space size and is
//! carried directly by the decision envelope. It is not embedded in an
//! algorithm-specific observation tensor.

use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

/// Bit mask of legal actions with no fixed width limit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegalMask {
    num_actions: usize,
    words: Box<[u64]>,
}

impl<'de> Deserialize<'de> for LegalMask {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            num_actions: usize,
            words: Box<[u64]>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let expected_words = wire.num_actions.div_ceil(64);
        if wire.words.len() != expected_words {
            return Err(D::Error::custom(format!(
                "legal mask for {} actions requires {expected_words} words, got {}",
                wire.num_actions,
                wire.words.len()
            )));
        }
        if let Some(last) = wire.words.last() {
            let used_bits = wire.num_actions % 64;
            if used_bits != 0 && (*last >> used_bits) != 0 {
                return Err(D::Error::custom(
                    "legal mask has set padding bits above num_actions",
                ));
            }
        }
        Ok(Self {
            num_actions: wire.num_actions,
            words: wire.words,
        })
    }
}

impl LegalMask {
    /// Create a mask with all actions illegal.
    pub fn new(num_actions: usize) -> Self {
        Self {
            num_actions,
            words: vec![0u64; num_actions.div_ceil(64)].into_boxed_slice(),
        }
    }

    /// Create a mask with all `num_actions` actions legal.
    pub fn all_legal(num_actions: usize) -> Self {
        let mut mask = Self::new(num_actions);
        for word_idx in 0..mask.words.len() {
            let bits_in_word = (num_actions - word_idx * 64).min(64);
            mask.words[word_idx] = if bits_in_word == 64 {
                u64::MAX
            } else {
                (1u64 << bits_in_word) - 1
            };
        }
        mask
    }

    /// Create from a compact `u64` bitmask. Only the low `num_actions` bits
    /// are used; `num_actions` must be at most 64.
    pub fn from_u64(bits: u64, num_actions: usize) -> Self {
        assert!(num_actions <= 64, "from_u64 requires num_actions <= 64");
        let mut mask = Self::new(num_actions);
        if num_actions > 0 {
            let keep = if num_actions == 64 {
                u64::MAX
            } else {
                (1u64 << num_actions) - 1
            };
            mask.words[0] = bits & keep;
        }
        mask
    }

    /// Mark an action as legal.
    #[inline]
    pub fn set(&mut self, action: usize) {
        debug_assert!(action < self.num_actions);
        self.words[action / 64] |= 1u64 << (action % 64);
    }

    /// Whether an action is legal. Out-of-range actions are illegal.
    #[inline]
    pub fn is_legal(&self, action: usize) -> bool {
        if action >= self.num_actions {
            return false;
        }
        (self.words[action / 64] >> (action % 64)) & 1 == 1
    }

    /// Number of legal actions.
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// True if no action is legal.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Size of the action space this mask covers.
    #[inline]
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Iterate over the indices of legal actions, ascending.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            let base = word_idx * 64;
            std::iter::successors((word != 0).then_some(word), |w| {
                let next = w & (w - 1); // clear lowest set bit
                (next != 0).then_some(next)
            })
            .map(move |w| base + w.trailing_zeros() as usize)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let mask = LegalMask::new(257);
        assert!(mask.is_empty());
        assert_eq!(mask.count_ones(), 0);
        assert_eq!(mask.num_actions(), 257);
        assert!(!mask.is_legal(0));
        assert!(!mask.is_legal(256));
    }

    #[test]
    fn test_all_legal_small() {
        let mask = LegalMask::all_legal(9);
        assert_eq!(mask.count_ones(), 9);
        assert!(mask.is_legal(0));
        assert!(mask.is_legal(8));
        assert!(!mask.is_legal(9));
    }

    #[test]
    fn test_all_legal_word_boundaries() {
        for n in [63, 64, 65, 128, 257] {
            let mask = LegalMask::all_legal(n);
            assert_eq!(mask.count_ones(), n as u32, "n={}", n);
            assert!(mask.is_legal(n - 1));
            assert!(!mask.is_legal(n));
        }
    }

    #[test]
    fn test_set_and_get_across_words() {
        let mut mask = LegalMask::new(257);
        for action in [0, 63, 64, 100, 255, 256] {
            mask.set(action);
        }
        assert_eq!(mask.count_ones(), 6);
        assert!(mask.is_legal(64));
        assert!(mask.is_legal(256));
        assert!(!mask.is_legal(65));
    }

    #[test]
    fn test_from_u64() {
        let mask = LegalMask::from_u64(0b101010001, 9);
        assert_eq!(mask.count_ones(), 4);
        assert!(mask.is_legal(0));
        assert!(mask.is_legal(4));
        assert!(mask.is_legal(6));
        assert!(mask.is_legal(8));
        assert!(!mask.is_legal(1));
    }

    #[test]
    fn test_from_u64_truncates_high_bits() {
        let mask = LegalMask::from_u64(u64::MAX, 9);
        assert_eq!(mask.count_ones(), 9);
    }

    #[test]
    fn test_iter_ones() {
        let mut mask = LegalMask::new(257);
        let expected = [0usize, 3, 63, 64, 200, 256];
        for &a in &expected {
            mask.set(a);
        }
        let collected: Vec<usize> = mask.iter_ones().collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_iter_ones_empty() {
        let mask = LegalMask::new(100);
        assert_eq!(mask.iter_ones().count(), 0);
    }

    #[test]
    fn test_large_action_space_matches_othello_and_generals() {
        // Othello: 65 actions, Generals 8x8: 257 actions — both past the u64 cliff
        for n in [65usize, 257] {
            let mut mask = LegalMask::new(n);
            mask.set(n - 1);
            assert!(mask.is_legal(n - 1));
            assert_eq!(mask.iter_ones().collect::<Vec<_>>(), vec![n - 1]);
        }
    }

    #[test]
    fn deserialization_rejects_malformed_word_counts_and_padding() {
        assert!(serde_json::from_str::<LegalMask>(r#"{"num_actions":65,"words":[1]}"#).is_err());
        assert!(serde_json::from_str::<LegalMask>(r#"{"num_actions":65,"words":[1,2]}"#).is_err());
        let valid: LegalMask = serde_json::from_str(r#"{"num_actions":65,"words":[1,1]}"#).unwrap();
        assert!(valid.is_legal(64));
    }
}
