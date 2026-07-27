-- Live identifier output (adsb-predict). Both tables are derived data,
-- rebuildable by re-running adsb-predict over the messages index — never
-- a source of truth.

-- One row per scored message: the nearest enrolled signature at score
-- time, its cosine similarity, the margin over the runner-up signature,
-- and the similarity-weighted type vote of the nearest signatures.
create table if not exists predictions (
    message_id     bigint      primary key references messages (id) on delete cascade,
    model_run      text        not null,
    predicted_icao char(6),
    similarity     real,
    margin         real,
    predicted_type text,
    created_at     timestamptz not null default now()
);

create index if not exists predictions_created_at_idx on predictions (created_at);

-- Snapshot of the enrolled signature roster as of the loop's last tick —
-- lets the web UI distinguish "never enrolled" from "enrolled but not
-- recognized". weight is the recency-decayed message weight behind the
-- signature; messages is the raw count.
create table if not exists signatures (
    icao       char(6)     primary key,
    model_run  text        not null,
    weight     real        not null,
    messages   integer     not null,
    updated_at timestamptz not null
);
