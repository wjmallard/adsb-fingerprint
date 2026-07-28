-- The collector's depth-1 state buffer: one row per airframe ever heard,
-- holding the newest decoded value of each field family, updated by EVERY
-- valid message — including the ones the per-aircraft cap discards before
-- storage — so the map can show real current positions while the messages
-- index stays capped. This is a cache of the live stream, not an index of
-- stored snippets: capped messages exist nowhere else, so the table is not
-- rebuildable from captures and simply repopulates as new traffic arrives.
-- msg_heard counts everything decoded (capped included); field-family
-- timestamps record when each group was last refreshed.

create table if not exists live_state (
    icao          char(6)     primary key,
    last_seen     timestamptz not null,
    msg_heard     bigint      not null default 0,
    rssi_db       real,
    callsign      text,
    callsign_at   timestamptz,
    latitude      double precision,
    longitude     double precision,
    altitude_ft   integer,
    position_at   timestamptz,
    ground_speed  real,
    track         real,
    vertical_rate integer,
    velocity_at   timestamptz
);
