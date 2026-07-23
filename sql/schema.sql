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
