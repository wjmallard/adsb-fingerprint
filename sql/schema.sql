-- One row per detected Mode S message. Rebuildable from the raw
-- captures by re-running `adsb-index` — the IQ itself stays in files.
create table if not exists messages (
    id            bigserial   primary key,
    capture_file  text        not null,
    sample_offset bigint      not null,
    n_samples     integer     not null,
    captured_at   timestamptz not null,
    session       text        not null,
    df            smallint,
    icao          char(6),
    type_code     smallint,
    crc_ok        boolean     not null,
    rssi_db       real,
    unique (capture_file, sample_offset)
);

create index if not exists messages_icao_idx on messages (icao);

create index if not exists messages_session_idx on messages (session);

-- Per-message fields decoded with pyModeS (nullable — each message type
-- carries only some of them). Added via ALTER so existing DBs migrate.
alter table messages add column if not exists hex           text;
alter table messages add column if not exists altitude_ft   integer;
alter table messages add column if not exists latitude      double precision;
alter table messages add column if not exists longitude     double precision;
alter table messages add column if not exists callsign      text;
alter table messages add column if not exists ground_speed  real;
alter table messages add column if not exists track         real;
alter table messages add column if not exists vertical_rate integer;
