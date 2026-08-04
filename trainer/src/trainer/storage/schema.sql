-- Cartridge2 replay schema v3.
--
-- Storage owns only the immutable replay envelope. The payload is opaque and
-- is decoded by the algorithm cartridge named by algorithm_id and
-- experience_schema. Every operation is fenced to one exact collection scope
-- and source checkpoint. Changing an algorithm payload never changes this table.

CREATE TABLE IF NOT EXISTS cartridge_schema_versions (
    component TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
);

INSERT INTO cartridge_schema_versions (component, schema_version)
VALUES ('replay', 3)
ON CONFLICT (component) DO NOTHING;

CREATE TABLE IF NOT EXISTS replay_records (
    id TEXT NOT NULL,
    env_id TEXT NOT NULL,
    env_contract_version BIGINT NOT NULL
        CHECK (env_contract_version BETWEEN 1 AND 4294967295),
    algorithm_id TEXT NOT NULL,
    experience_schema TEXT NOT NULL,
    collection_scope_id TEXT NOT NULL
        CHECK (collection_scope_id ~ '^[0-9a-f]{64}$'),
    source_checkpoint_id TEXT
        CHECK (
            source_checkpoint_id IS NULL
            OR source_checkpoint_id ~ '^[0-9a-f]{64}$'
        ),
    episode_id TEXT NOT NULL,
    step_number BIGINT NOT NULL
        CHECK (step_number BETWEEN 0 AND 4294967295),
    payload BYTEA NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (
        env_id, env_contract_version, algorithm_id, experience_schema,
        collection_scope_id, id
    )
);

CREATE INDEX IF NOT EXISTS idx_replay_records_selection_created
    ON replay_records(
        env_id, env_contract_version, algorithm_id, experience_schema,
        collection_scope_id, source_checkpoint_id, created_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_replay_records_selection_episode
    ON replay_records(
        env_id, env_contract_version, algorithm_id, experience_schema,
        collection_scope_id, source_checkpoint_id, episode_id, step_number
    );
