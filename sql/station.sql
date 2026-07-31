-- Where the receiving station itself stood, per message. Sorts after
-- schema.sql, so the messages table already exists when init_db applies
-- this file. Null means the station's position is unknown for that message
-- (e.g. the antenna was in transit between spots, or a session ended before
-- enough traffic accumulated to estimate one).
--
-- The coordinates are data, not schema: adsb-collect stamps every row
-- explicitly — the traffic-derived station estimate, or the config receiver
-- position when the two agree — and adsb-geotag overwrites stamps from
-- logged GPS tracks where those exist. No real station location is baked
-- into this tracked file.

alter table messages add column if not exists receiver_latitude double precision;
alter table messages add column if not exists receiver_longitude double precision;

-- Early deployments filled these columns via ad-hoc psql column defaults;
-- explicit stamping replaced that, so make sure no stale default lingers to
-- mislabel rows after a station move. (A no-op when none is set.)
alter table messages alter column receiver_latitude drop default;
alter table messages alter column receiver_longitude drop default;

-- Generated like messages.geom in spatial.sql, so
-- st_distance(geom, receiver_geom) is the true per-message range.
alter table messages add column if not exists receiver_geom geography(point, 4326)
    generated always as (
        case
            when receiver_longitude is not null and receiver_latitude is not null
            then st_setsrid(st_makepoint(receiver_longitude, receiver_latitude), 4326)::geography
        end
    ) stored;
