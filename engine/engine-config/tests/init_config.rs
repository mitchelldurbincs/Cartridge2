//! `init_config` installs a process-wide value, so it gets its own test
//! binary.
//!
//! Cargo runs unit tests from one crate in a single process, in parallel. A
//! test that populates the process-global config would therefore change what
//! `load_config()` returns for every other test in that binary -- notably the
//! env-override tests, which depend on `load_config()` re-reading. An
//! integration test is a separate process, so the global starts clean and
//! affects nothing else.

use engine_config::{init_config, load_config};

#[test]
fn init_config_installs_the_value_that_later_loads_return() {
    let installed = init_config().expect("config should load in the repo checkout");

    // The whole point: this does NOT re-read the file, it returns what was
    // validated above. Without the seeding, a config replaced between the two
    // reads would be validated and then not used.
    let later = load_config();

    assert_eq!(installed.common.env_id, later.common.env_id);
    assert_eq!(installed.common.data_dir, later.common.data_dir);
    assert_eq!(installed.web.port, later.web.port);
    assert_eq!(installed.mcts.num_simulations, later.mcts.num_simulations);
}

#[test]
fn init_config_is_idempotent() {
    let first = init_config().expect("first init");
    let second = init_config().expect("second init");

    assert_eq!(first.common.env_id, second.common.env_id);
    assert!(
        std::ptr::eq(first, second),
        "init_config must hand back the same installed value, not a fresh read"
    );
}
