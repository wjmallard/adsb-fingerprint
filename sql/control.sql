-- The collector's control channel with the web UI: one row the collector
-- heartbeats (mode, applied gain, the tuner's supported gain steps) and
-- the web app writes gain requests into. Requests take effect only in
-- listen (--no-logging) mode: a logging session's gain is provenance,
-- recorded once in session.yaml, and must not drift mid-session.

create table if not exists collector_control (
    id             boolean     primary key default true check (id),
    heartbeat_at   timestamptz not null,
    listen_mode    boolean     not null,
    applied_gain   text,
    applied_at     timestamptz,
    valid_gains_db real[],
    requested_gain text,
    requested_at   timestamptz
);
