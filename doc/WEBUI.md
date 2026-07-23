# Live Map Web UI — Design

A real-time "what's overhead" view over the collector's Postgres index: a map
of current aircraft, a roster of recently heard airframes, and a detail panel
for the selected one. Companion to PLAN.md; its build track (W1–W5) is
independent of the model phases (P4–P6). Frontend patterns are lifted from
`~/Github/offline-maps` (MapLibre + vendored PMTiles, fully offline).

## Architecture (poll, don't push)

- The UI is just another *reader* of the `messages` index, like `dataset.py`
  and `stats.py` — the collector doesn't know it exists. No notify hook.
- The browser polls a small Flask JSON endpoint at 1 Hz; with `adsb-collect`
  flushing every ~1 s, the map runs ~2–3 s behind the air (dump1090-like).
- Flask + server-rendered Jinja + vanilla JS, no build step. psycopg
  connection per request (localhost, single viewer — no pooling).
- If sub-second latency is ever wanted: Postgres LISTEN/NOTIFY → Flask SSE →
  EventSource. Out of scope now; nothing here blocks adding it later.

## Freshness semantics

- Roster window: aircraft heard in the last 15 min (config), sorted by
  last-seen desc; live rows bright, stale rows dimmed.
- Map: markers fade after ~60 s without a message, drop after ~5 min (the
  aircraft stays rostered).
- The collect cap stores a micro-cluster (~3 back-to-back messages) per
  aircraft every 10 s, so each aircraft's map position refreshes on roughly
  that cadence. Optional polish: client dead reckoning (project along `track`
  at `ground_speed`).
- Sessions collected before decode existed have no lat/lon and simply never
  appear on the map (the window is recent anyway). A snippet re-decode
  backfill is possible later if ever wanted.
- A header chip shows "last message Ns ago" — doubles as a collector-health
  indicator.

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│                        MAP (MapLibre, dark)                  │
│   ⌂ receiver + range rings      ✈ planes rotated by track,   │
│   labeled ICAO/callsign, trail on selection, staleness fade  │
├───────────────────────────────┬──────────────────────────────┤
│ ROSTER (last-seen ↓, 15 min)  │ SELECTED: A1B2C3 · UAL123    │
│ ● A1B2C3 UAL123 2s 34000 -12  │ registry: N-number, make /   │
│ ● 4CA123        5s 12500 -18  │   model, type, owner, source │
│ ○ AB44EF N123AB 4m  8000 -25  │ live: alt, gs, trk, v/r,     │
│                               │   position, dist + bearing   │
│                               │ radio: msgs, rate, sessions  │
│                               │   seen, RSSI + sparkline     │
└───────────────────────────────┴──────────────────────────────┘
```

- **Map**: receiver marker + range rings; aircraft as a GeoJSON symbol layer
  with `icon-rotate` from `track`; the plane glyph is registered at runtime
  via `map.addImage()` from an inline SVG (no sprite-sheet edits). Selected
  aircraft gets a highlight + trail line.
- **Roster**: one line per airframe — live-dot · ICAO · callsign · age ·
  altitude · RSSI. Click row ↔ click plane (two-way selection).
- **Detail**, three blocks:
  - *registry*: join against the `aircraft` view — registration,
    manufacturer/model, type, owner/operator, city/state/country, faa|opensky.
  - *live*: callsign, altitude, ground speed, track, vertical rate, position,
    distance + bearing from the receiver.
  - *radio*: message count + rate, sessions seen (the fingerprinting
    currency), latest RSSI + 10-min sparkline on a small `<canvas>`.

## Endpoints

- `GET /` — the page.
- `GET /api/aircraft?minutes=15` — the 1 Hz poll. GeoJSON FeatureCollection,
  one feature per active ICAO (`geometry: null` until a position fix); one
  payload drives map, roster, and the detail panel's live block. A few KB.
- `GET /api/aircraft/<icao>` — once per selection: registry join + lifetime
  stats (total messages, sessions seen, first/last heard).
- `GET /api/aircraft/<icao>/history?minutes=60` — once per selection:
  per-message `{t, lat, lon, altitude_ft, rssi_db}` → trail + RSSI sparkline.
- `GET /api/overlay` — receiver point + range-ring polygons, server-generated.
- `GET /tiles/<file>` — the basemap `.pmtiles`, `conditional=True` for the
  range requests the pmtiles protocol needs (offline-maps pattern).
- `GET /embeddings` — index of training runs under `paths.models` with their
  `run.yaml` summaries (variant, classes, best balanced accuracy).
- `GET /embeddings/<run>` — that run's self-contained `adsb-embed` viewer
  (`embedding.html`, canvas 2D — no WebGL), served as-is.

Aircraft state is "latest non-null per field" — position, velocity, and ident
ride in different message types, so the newest row rarely has everything:

```sql
select
    icao,
    max(captured_at) as last_seen,
    count(*) as msg_count,
    (array_agg(latitude order by captured_at desc) filter (where latitude is not null))[1] as latitude
    -- same pattern: longitude, altitude_ft, callsign, ground_speed, track,
    -- vertical_rate, rssi_db
from messages
where captured_at > now() - %(minutes)s * interval '1 minute'
group by icao
order by last_seen desc
```

`messages_captured_at_idx` keeps this O(recent rows) as the table grows. If it
ever mattered (it won't at capped, localhost scale), the escape hatch is a
collector-maintained `aircraft_state` upsert table — noted, not planned.

## Spatial: PostGIS, not JS

No spatial geometry math in the browser or in Python. Per-aircraft
distance/bearing and the ring polygons come from the API — PostGIS
(`ST_Distance` / `ST_Azimuth` / `ST_Buffer` on `geography`), with the
receiver point parameterized from config until the GPS observing-location
table exists (then the queries join that instead). The JS only ever draws
coordinates it was handed.

## Basemap & dark style

- Tiles are shared, not duplicated: config points at
  `~/Github/offline-maps/data/basemap.pmtiles` (config.yaml is gitignored;
  `Path.expanduser()`).
- Vendored from offline-maps: `maplibre-gl.js/.css`, `pmtiles.js`, glyphs.
- Dark flavor requires **no tile re-download** — the `.pmtiles` is raw vector
  data; flavor is a regenerated style JSON (offline-maps'
  `scripts/build-style.mjs` with `dark`) plus that flavor's sprite assets
  (~4 small files from protomaps basemaps-assets). Panels get a matching dark
  palette in `style.css`.

## Module layout / config

```
src/adsb_fingerprint/web/
  app.py                 Flask app + routes (entry point: adsb-web)
  templates/index.html
  static/js/app.js, css/style.css
  static/vendor/, styles/, glyphs/, sprites/   (vendored)
```

- pyproject: add `flask`; `adsb-web = "adsb_fingerprint.web.app:main"`.
- config.yaml additions (map centers on the existing `receiver:` lat/lon):

```yaml
server:
  host: 127.0.0.1
  port: 5050                   # offline-maps owns 5000
  debug: true

map:
  tiles_file: ~/Github/offline-maps/data/basemap.pmtiles
  roster_minutes: 15
  rings_km: [50, 100, 150]
```

## Build phases (atomic chunks)

- W1 scaffold: flask dep + `web/` package + config + vendored map assets +
  dark style + `/tiles` route; page renders the basemap centered on the
  receiver, with receiver marker + rings from `/api/overlay`.
- W2 live: `/api/aircraft`, plane symbols + roster, 1 Hz poll, staleness
  fade/drop, "last message Ns ago" chip.
- W3 selection: two-way select, `/api/aircraft/<icao>`, detail panel
  (registry + live blocks).
- W4 history: `/api/aircraft/<icao>/history` → trail + RSSI sparkline; radio
  block complete.
- W5 polish (optional): dead reckoning, all-history roster toggle, SSE if
  ever needed.
