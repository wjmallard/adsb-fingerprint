-- Where the receiving station itself stood, per message. Sorts after
-- schema.sql, so the messages table already exists when init_db applies
-- this file. Null means the station's position is unknown for that message
-- (e.g. the antenna was in transit between spots).
--
-- The coordinates are data, not schema: backfill historical era ranges and
-- set column defaults for your current spot ad hoc via psql, so no real
-- station location is baked into this tracked file. With defaults set, the
-- collector needs no changes — its copy lists omit these columns, so every
-- new row picks up the current spot automatically. A future live-GPS reader
-- can instead write explicit per-message positions for mobile collection.

alter table messages add column if not exists receiver_latitude double precision;
alter table messages add column if not exists receiver_longitude double precision;

-- Generated like messages.geom in spatial.sql, so
-- st_distance(geom, receiver_geom) is the true per-message range.
alter table messages add column if not exists receiver_geom geography(point, 4326)
    generated always as (
        case
            when receiver_longitude is not null and receiver_latitude is not null
            then st_setsrid(st_makepoint(receiver_longitude, receiver_latitude), 4326)::geography
        end
    ) stored;
