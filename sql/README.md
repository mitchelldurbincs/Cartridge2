# Replay database schema

Cartridge2 uses PostgreSQL as an algorithm-neutral replay store. Storage owns
only an immutable profile envelope and an opaque payload. The cartridge named
by `algorithm_id` and `experience_schema` owns the payload codec; adding a new
algorithm does not add algorithm-specific columns or tables.

## Schema files

| File | Purpose | Used by |
|------|---------|---------|
| `schema.sql` | Canonical replay-v4 DDL | Rust actor and local setup |
| `../trainer/src/trainer/storage/schema.sql` | Byte-identical packaged mirror | Python trainer |
| `../scripts/init-postgres.sql` | Byte-identical container mirror | Docker Compose |
| `../k8s/base/postgres/init-configmap.yaml` | Canonical DDL embedded for Kubernetes | In-cluster PostgreSQL |

Every form creates exactly `cartridge_schema_versions`, `collection_scopes`,
and `replay_records`. Actor and trainer create the
schema only when the database has no tables; otherwise startup requires the
exact table set, columns, PostgreSQL types, nullability, primary-key order, and
schema marker. Partial, extra, unversioned, or wrong-version schemas fail fast.

## Replay protocol v4

`cartridge_schema_versions` must contain exactly:

```text
('replay', 4)
```

`replay_records` has this storage-owned envelope:

| Column | PostgreSQL type | Contract |
|--------|-----------------|----------|
| `id` | `TEXT` | Non-empty record ID, unique within the scoped profile |
| `env_id` | `TEXT` | Environment ID |
| `env_contract_version` | `BIGINT` | Immutable environment contract version in `[1, 2^32-1]` |
| `algorithm_id` | `TEXT` | Algorithm cartridge that owns the payload |
| `experience_schema` | `TEXT` | Algorithm-owned payload codec ID |
| `collection_scope_id` | `TEXT` | Required lowercase 64-hex identity unique to one collection attempt |
| `source_checkpoint_id` | `TEXT NULL` | Lowercase 64-hex model generation, null only for root collection |
| `episode_id` | `TEXT` | Algorithm grouping key for one episode |
| `step_number` | `BIGINT` | Record order within the episode in `[0, 2^32-1]` |
| `payload` | `BYTEA` | Opaque algorithm-owned bytes |
| `created_at` | `TIMESTAMP` | Server-side insertion time |

The primary key is:

```text
(env_id, env_contract_version, algorithm_id, experience_schema,
 collection_scope_id, id)
```

The selection/created index supports newest-first retention; sampling fetches
rows by ID from an in-memory snapshot of the selection. Every count, distinct-episode count, sample, clear, cleanup, and write
operation is bound to the full `ReplaySelection`: the four-field profile,
`collection_scope_id`, and exact nullable `source_checkpoint_id`. SQL compares
the nullable source with `IS NOT DISTINCT FROM`, so null means root and never
means “all sources.” A record from another profile, attempt, or model generation
is rejected before storage.

The table intentionally contains no state, action, observation, reward,
terminal, policy, value, board, or presentation columns. Those meanings belong
to a cartridge codec, not PostgreSQL.

### Installed AlphaZero payload

`alphazero_transition_v1` encodes one payload as the concatenation below, with
every element represented as little-endian IEEE-754 `f32`:

```text
observation[obs_size] || policy[num_actions] || terminal_value[1]
```

The AlphaZero decoder requires the exact byte length, finite values, policy
entries in `[0, 1]` summing to one, and a terminal value in `[-1, 1]`. Storage
does not know or validate those fields; the installed AlphaZero cartridge does.

## Clean cutover

Replay v4 has no migration, compatibility view, legacy table reader, default
profile inference, or implicit collection scope. Older replay versions are
deliberately rejected. Preserve any data you need, then recreate
the replay database from the current schema.

## Setup

```bash
# Docker Compose initializes a fresh volume automatically.
docker compose up postgres

# Or initialize a fresh local database explicitly.
createdb cartridge
psql cartridge -f sql/schema.sql
psql cartridge -c "CREATE USER cartridge WITH PASSWORD 'cartridge'; GRANT ALL ON DATABASE cartridge TO cartridge;"
```

Default local DSN:

```text
postgresql://cartridge:cartridge@localhost:5432/cartridge
```

Override it for actor and trainer with:

```bash
export CARTRIDGE_STORAGE_POSTGRES_URL=postgresql://user:pass@host:5432/dbname
```
