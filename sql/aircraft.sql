-- Aircraft reference data. Both tables are rebuildable indexes over the
-- downloaded source files in data/ (FAA Releasable Aircraft; OpenSky CSV),
-- loaded by adsb-ingest-faa / adsb-ingest-opensky.

create table if not exists faa_aircraft (
    icao          char(6) primary key,   -- MODE S CODE HEX
    n_number      text,
    manufacturer  text,
    model         text,
    type_aircraft text,                   -- decoded (Rotorcraft, Fixed wing …)
    year_mfr      integer,
    owner         text,                   -- registrant name (no street address)
    owner_city    text,
    owner_state   text,
    registrant    text,                   -- decoded (Individual, Corporation …)
    status_code   text
);

create table if not exists opensky_aircraft (
    icao          char(6) primary key,   -- icao24
    registration  text,
    manufacturer  text,
    model         text,
    typecode      text,
    operator      text,
    owner         text,
    country       text,
    icao_class    text
);

-- Unified lookup: FAA (authoritative for US aircraft) takes precedence,
-- OpenSky fills the gaps for everything else.
create or replace view aircraft as
select
    coalesce(f.icao, o.icao)                 as icao,
    coalesce(f.n_number, o.registration)     as registration,
    coalesce(f.manufacturer, o.manufacturer) as manufacturer,
    coalesce(f.model, o.model)               as model,
    coalesce(f.type_aircraft, o.icao_class)  as type,
    o.typecode                               as typecode,
    coalesce(f.owner, o.owner, o.operator)   as owner,
    f.owner_city                             as owner_city,
    f.owner_state                            as owner_state,
    o.operator                               as operator,
    o.country                                as country,
    case when f.icao is not null then 'faa' else 'opensky' end as source
from faa_aircraft f
full outer join opensky_aircraft o
    on o.icao = f.icao;
