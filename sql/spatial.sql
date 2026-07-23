-- PostGIS spatial layer over the messages index. Sorts after schema.sql,
-- so the messages table already exists when init_db applies this file.
create extension if not exists postgis;

-- Geography point derived from the decoded position. Generated, so Postgres
-- keeps it in sync automatically — for existing rows at ALTER time and for
-- every future insert — and the collector/indexer need no changes.
alter table messages add column if not exists geom geography(point, 4326)
    generated always as (
        case
            when longitude is not null and latitude is not null
            then st_setsrid(st_makepoint(longitude, latitude), 4326)::geography
        end
    ) stored;

create index if not exists messages_geom_idx on messages using gist (geom);
