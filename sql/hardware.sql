-- Receive-chain era log: one row per hardware change (antenna, filter,
-- LNA, dongle, gain staging), stamped so any message or session can be
-- assigned to an era by comparing timestamps. Insert rows ad hoc via psql
-- while collection is stopped, so the stamp falls in the message gap and
-- the boundary is unambiguous. Rows are data, not schema: no station
-- specifics are baked into this tracked file.

create table if not exists hardware_log (
    stamp timestamptz not null primary key,
    note  text        not null
);
