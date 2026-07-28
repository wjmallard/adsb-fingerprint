"""Flask app for the live aircraft map (doc/WEBUI.md), the /aircraft fleet table, and /embeddings."""

import json
import time

import yaml
from flask import (
    Flask,
    abort,
    render_template,
    request,
    send_file,
)

from adsb_fingerprint import (
    config,
    db,
)

app = Flask(__name__)

# One row per aircraft heard in the window. Position, velocity, and ident ride
# in different message types, so the newest row rarely has everything — each
# field is independently "latest non-null". Distance/bearing from the receiver
# are PostGIS geodesics over the latest fix; the receiver point is
# parameterized from config until a GPS observing-location table exists.
# The registry joins feed glyph_for() — which map symbol each airframe gets.
AIRCRAFT_SQL = """
    with latest as (
        select
            icao,
            extract(epoch from now() - last_seen) as age_s,
            callsign,
            latitude,
            longitude,
            altitude_ft,
            ground_speed,
            track,
            vertical_rate,
            rssi_db
        from live_state
        where last_seen > now() - %(minutes)s * interval '1 minute'
    ),
    stored as (
        select
            icao,
            count(*) as msg_count
        from messages
        where captured_at > now() - %(minutes)s * interval '1 minute'
        and crc_ok
        and icao is not null
        group by icao
    )
    select
        latest.*,
        coalesce(stored.msg_count, 0) as msg_count,
        faa_aircraft.type_aircraft,
        opensky_aircraft.icao_class,
        coalesce(faa_aircraft.manufacturer, opensky_aircraft.manufacturer) as manufacturer,
        st_distance(
            st_setsrid(st_makepoint(longitude, latitude), 4326)::geography,
            st_setsrid(st_makepoint(%(receiver_lon)s, %(receiver_lat)s), 4326)::geography
        ) / 1000.0 as distance_km,
        degrees(
            st_azimuth(
                st_setsrid(st_makepoint(%(receiver_lon)s, %(receiver_lat)s), 4326)::geography,
                st_setsrid(st_makepoint(longitude, latitude), 4326)::geography
            )
        ) as bearing_deg
    from latest
    left join stored on stored.icao = latest.icao
    left join faa_aircraft on faa_aircraft.icao = latest.icao
    left join opensky_aircraft on opensky_aircraft.icao = latest.icao
    order by age_s
"""

# CPR decoding against a fixed reference is only unambiguous within 180 NM;
# the rare rows beyond that are with_ref artifacts, not receptions, and must
# not set an airframe's range record.
WITH_REF_MAX_KM = 333

# adsb-train's --min-train default: airframes at or past this message count
# are eligible training classes.
TRAIN_MIN_MESSAGES = 50

# One row per airframe ever heard, joined against the registry. The date
# cast groups by the server's local timezone, matching the day-held-out
# bookkeeping elsewhere in the project. FAA N-numbers are stored without
# their leading "N" — restored here so display and sort order agree. The
# order-by slot is filled from FLEET_SORTS only, never from the request.
FLEET_SQL = """
    with per_message as (
        select
            icao,
            session,
            captured_at,
            callsign,
            st_distance(geom, receiver_geom) / 1000.0 as range_km
        from messages
        where crc_ok
        and icao is not null
    ),
    seen as (
        select
            icao,
            count(*) as msg_count,
            count(distinct session) as session_count,
            count(distinct captured_at::date) as day_count,
            min(captured_at) as first_heard,
            max(captured_at) as last_heard,
            (array_agg(callsign order by captured_at desc) filter (where callsign is not null))[1] as callsign,
            max(range_km) filter (where range_km <= %(max_range_km)s) as max_range_km
        from per_message
        group by icao
    )
    select
        seen.icao,
        seen.msg_count,
        seen.session_count,
        seen.day_count,
        seen.first_heard,
        seen.last_heard,
        seen.callsign,
        seen.max_range_km,
        case
            when aircraft.source = 'faa' and aircraft.registration is not null
            then 'N' || aircraft.registration
            else aircraft.registration
        end as registration,
        aircraft.manufacturer,
        aircraft.model,
        aircraft.typecode,
        aircraft.owner,
        aircraft.source
    from seen
    left join aircraft on aircraft.icao = seen.icao
    order by {order_by}
"""

FLEET_SORTS = {
    "aircraft": "manufacturer asc nulls last, model asc nulls last, icao asc",
    "callsign": "callsign asc nulls last, icao asc",
    "days": "day_count desc, msg_count desc",
    "first": "first_heard asc",
    "icao": "icao asc",
    "last": "last_heard desc",
    "msgs": "msg_count desc, icao asc",
    "owner": "owner asc nulls last, icao asc",
    "range": "max_range_km desc nulls last",
    "registration": "registration asc nulls last, icao asc",
    "sessions": "session_count desc, msg_count desc",
    "type": "typecode asc nulls last, icao asc",
}

REGISTRY_SQL = """
    select
        icao,
        registration,
        manufacturer,
        model,
        type,
        typecode,
        owner,
        owner_city,
        owner_state,
        operator,
        country,
        source
    from aircraft
    where icao = %(icao)s
"""

LIFETIME_SQL = """
    select
        count(*) as msg_count,
        count(distinct session) as session_count,
        min(captured_at) as first_heard,
        max(captured_at) as last_heard
    from messages
    where icao = %(icao)s
    and crc_ok
"""

HISTORY_SQL = """
    select
        extract(epoch from captured_at) as t,
        latitude,
        longitude,
        altitude_ft,
        rssi_db
    from messages
    where icao = %(icao)s
    and captured_at > now() - %(minutes)s * interval '1 minute'
    and crc_ok
    order by captured_at
"""

# /live: the identification table (adsb-predict output). A contact is an
# aircraft heard in the window; each of its recent messages voted for a
# nearest enrolled signature. The leading candidate names the contact
# outright at CONFIDENT_SHARE of the votes; everything at CANDIDATE_SHARE
# or better makes the ranked shortlist. Shares are shrunk by PSEUDO_VOTES
# phantom dissenters (votes / (n + PSEUDO_VOTES)) so a single-message
# contact reads ~25%, not a unanimous 100% — confidence has to be earned
# by accumulation, not by a small denominator.
LIVE_WINDOW_MINUTES = 10
LIVE_CONFIDENT_SHARE = 0.5
LIVE_CANDIDATE_SHARE = 0.2
LIVE_PSEUDO_VOTES = 3

LIVE_SQL = """
    select
        m.icao,
        extract(epoch from now() - m.captured_at) as age_s,
        p.predicted_icao,
        p.similarity,
        p.predicted_type
    from messages m
    join predictions p on p.message_id = m.id
    where m.captured_at > now() - %(minutes)s * interval '1 minute'
"""

SIGNATURES_SQL = """
    select
        icao,
        messages
    from signatures
"""

LIVE_REGISTRY_SQL = """
    select
        icao,
        case
            when source = 'faa' and registration is not null
            then 'N' || registration
            else registration
        end as registration,
        model,
        type
    from aircraft
    where icao = any(%(icaos)s)
"""

# Range rings as ST_Buffer circles on geography (quad_segs=32 -> 129-point
# rings), GeoJSON-encoded by PostGIS itself.
RINGS_SQL = """
    select
        radius_km,
        st_asgeojson(
            st_buffer(
                st_setsrid(st_makepoint(%(lon)s, %(lat)s), 4326)::geography,
                radius_km * 1000.0,
                'quad_segs=32'
            ),
            6
        ) as geojson
    from unnest(%(radii_km)s::float8[]) as radius_km
"""


@app.route("/")
def index():
    return render_template(
        "index.html",
        center=[config.RECEIVER_LON, config.RECEIVER_LAT],
        zoom=config.MAP_DEFAULT_ZOOM,
        roster_minutes=config.MAP_ROSTER_MINUTES,
    )


@app.route("/aircraft")
def aircraft():
    # Every airframe ever heard, server-rendered in one page — this is a
    # ~1500-row lifetime table loaded on demand, not a polled endpoint.
    sort = request.args.get("sort", default="last")
    if sort not in FLEET_SORTS:
        sort = "last"
    with db.connect() as conn:
        rows = conn.execute(
            FLEET_SQL.format(order_by=FLEET_SORTS[sort]),
            {
                "max_range_km": WITH_REF_MAX_KM,
            },
        ).fetchall()
    for row in rows:
        row["aircraft"] = " ".join(
            part
            for part in (row["manufacturer"], row["model"])
            if part
        )
        row["first_heard"] = row["first_heard"].astimezone().strftime("%Y-%m-%d %H:%M")
        row["last_heard"] = row["last_heard"].astimezone().strftime("%Y-%m-%d %H:%M")
    totals = {
        "airframes": f"{len(rows):,}",
        "messages": f"{sum(row['msg_count'] for row in rows):,}",
        "trainable": f"{sum(1 for row in rows if row['msg_count'] >= TRAIN_MIN_MESSAGES):,}",
    }
    return render_template(
        "aircraft.html",
        rows=rows,
        sort=sort,
        totals=totals,
        min_train=TRAIN_MIN_MESSAGES,
    )


@app.route("/live")
def live():
    return render_template(
        "live.html",
        minutes=LIVE_WINDOW_MINUTES,
        confident=LIVE_CONFIDENT_SHARE,
        candidate=LIVE_CANDIDATE_SHARE,
        pseudo=LIVE_PSEUDO_VOTES,
    )


@app.route("/api/live")
def api_live():
    # Per-contact aggregation of adsb-predict's per-message votes over the
    # window. Contacts sort newest-heard first, like the map roster.
    minutes = request.args.get(
        "minutes",
        default=LIVE_WINDOW_MINUTES,
        type=float,
    )
    with db.connect() as conn:
        rows = conn.execute(LIVE_SQL, {"minutes": minutes}).fetchall()
        enrolled = {
            row["icao"]: row["messages"]
            for row in conn.execute(SIGNATURES_SQL).fetchall()
        }
        contacts = {}
        for row in rows:
            contact = contacts.setdefault(
                row["icao"],
                {
                    "n": 0,
                    "age_s": row["age_s"],
                    "votes": {},
                    "sims": {},
                    "types": {},
                },
            )
            contact["n"] += 1
            contact["age_s"] = min(contact["age_s"], row["age_s"])
            contact["votes"][row["predicted_icao"]] = contact["votes"].get(row["predicted_icao"], 0) + 1
            contact["sims"][row["predicted_icao"]] = contact["sims"].get(row["predicted_icao"], 0.0) + row["similarity"]
            contact["types"][row["predicted_type"]] = contact["types"].get(row["predicted_type"], 0) + 1
        wanted = set(contacts)
        for contact in contacts.values():
            wanted.update(contact["votes"])
        registry = {}
        if wanted:
            registry = {
                row["icao"]: row
                for row in conn.execute(
                    LIVE_REGISTRY_SQL,
                    {"icaos": sorted(wanted)},
                ).fetchall()
            }

    def describe(icao):
        entry = registry.get(icao) or {}
        return {
            "icao": icao,
            "registration": entry.get("registration"),
            "model": entry.get("model") or entry.get("type"),
        }

    out = []
    for icao, contact in contacts.items():
        denominator = contact["n"] + LIVE_PSEUDO_VOTES
        candidates = [
            {
                **describe(candidate),
                "share": round(votes / denominator, 2),
                "similarity": round(contact["sims"][candidate] / votes, 3),
            }
            for candidate, votes in sorted(
                contact["votes"].items(),
                key=lambda item: -item[1],
            )
            if votes / denominator >= LIVE_CANDIDATE_SHARE
        ][:3]
        identified = (
            candidates[0]
            if candidates and candidates[0]["share"] >= LIVE_CONFIDENT_SHARE
            else None
        )
        type_label, type_votes = max(
            contact["types"].items(),
            key=lambda item: item[1],
        )
        out.append(
            {
                **describe(icao),
                "age_s": round(float(contact["age_s"]), 1),
                "msg_count": contact["n"],
                "enrolled": icao in enrolled,
                "signature_msgs": enrolled.get(icao),
                "radio_class": type_label,
                "radio_class_share": round(type_votes / denominator, 2),
                "candidates": candidates,
                "verdict": (
                    None
                    if identified is None
                    else ("match" if identified["icao"] == icao else "differ")
                ),
            }
        )
    out.sort(key=lambda contact: contact["age_s"])
    return {
        "contacts": out,
        "enrolled_total": len(enrolled),
        "minutes": minutes,
        "now": round(time.time(), 1),
    }


@app.route("/embeddings")
def embeddings():
    # Index of training runs under paths.models — the web app is just a
    # reader of the run dirs adsb-train/adsb-embed write.
    return render_template(
        "embeddings.html",
        runs=embedding_runs(),
    )


@app.route("/embeddings/<run>")
@app.route("/embeddings/<run>/held-out", defaults={"scope": "held-out"})
def embedding_viewer(run, scope=None):
    # The run segment is validated against the actual directory listing —
    # no path arithmetic on user input; the scope maps to a fixed filename.
    if not config.MODEL_DIR.is_dir():
        abort(404)
    run_dirs = {d.name for d in config.MODEL_DIR.iterdir() if d.is_dir()}
    if run not in run_dirs:
        abort(404)
    name = "embedding-held-out.html" if scope == "held-out" else "embedding.html"
    viewer = config.MODEL_DIR / run / name
    if not viewer.is_file():
        abort(404)
    return send_file(viewer, conditional=True)


def embedding_runs():
    """Model run dirs, newest first, each with a run.yaml summary + viewer flag."""
    if not config.MODEL_DIR.is_dir():
        return []
    runs = []
    for run_dir in sorted(config.MODEL_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary = {
            "name": run_dir.name,
            "has_viewer": (run_dir / "embedding.html").is_file(),
            "has_held_out": (run_dir / "embedding-held-out.html").is_file(),
            "variant": None,
            "n_classes": None,
            "epochs": None,
            "best_balanced": None,
        }
        run_yaml = run_dir / "run.yaml"
        if run_yaml.is_file():
            try:
                run = yaml.safe_load(run_yaml.read_text())
                summary["variant"] = run.get("args", {}).get("variant")
                summary["n_classes"] = len(run.get("classes", []))
                epochs = run.get("epochs", [])
                summary["epochs"] = len(epochs)
                balanced = [
                    epoch["balanced_accuracy"]
                    for epoch in epochs
                    if "balanced_accuracy" in epoch
                ]
                if balanced:
                    summary["best_balanced"] = max(balanced)
            except Exception:
                # adsb-train rewrites run.yaml as it goes — a mid-write or
                # malformed sidecar shouldn't take down the index.
                pass
        runs.append(summary)
    return runs


@app.route("/tiles/<path:filename>")
def tiles(filename):
    # Serve only the configured basemap. conditional=True enables the HTTP
    # range requests the pmtiles protocol relies on to read slices of the
    # archive instead of downloading all of it.
    if filename != config.MAP_TILES_FILE:
        abort(404)
    return send_file(config.MAP_TILES_PATH, conditional=True)


def glyph_for(faa_type, icao_class, manufacturer):
    """Map-symbol class for an airframe: heli, small (single-engine and
    other hobby craft), or plane. FAA is authoritative when present (OpenSky
    has junk rows — e.g. a 737-9 classed H2T); OpenSky's leading class
    letters fill the gaps for foreign registrations; and when both class
    fields are empty (registrations newer than the FAA snapshot, e.g.
    N1984S), a manufacturer that calls itself a helicopter maker — Airbus
    Helicopters, Bell Helicopter, Robinson Helicopter — settles heli."""
    if faa_type:
        if faa_type in ("Rotorcraft", "Gyroplane"):
            return "heli"
        if faa_type in (
            "Fixed wing single-engine",
            "Glider",
            "Powered parachute",
            "Weight-shift-control",
        ):
            return "small"
        return "plane"
    if icao_class:
        if icao_class.startswith(("H", "G")):
            return "heli"
        if icao_class.startswith("L1"):
            return "small"
    if manufacturer and "helicopter" in manufacturer.lower():
        return "heli"
    return "plane"


@app.route("/api/aircraft")
def api_aircraft():
    # The 1 Hz poll: GeoJSON FeatureCollection, one feature per active ICAO,
    # geometry null until a position fix. One payload drives the map, the
    # roster, and the detail panel's live block.
    minutes = request.args.get(
        "minutes",
        default=config.MAP_ROSTER_MINUTES,
        type=float,
    )
    with db.connect() as conn:
        rows = conn.execute(
            AIRCRAFT_SQL,
            {
                "minutes": minutes,
                "receiver_lat": config.RECEIVER_LAT,
                "receiver_lon": config.RECEIVER_LON,
            },
        ).fetchall()
    features = []
    for row in rows:
        geometry = None
        if row["latitude"] is not None and row["longitude"] is not None:
            geometry = {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            }
        distance_km = row["distance_km"]
        if distance_km is not None:
            distance_km = round(distance_km, 1)
        bearing_deg = row["bearing_deg"]
        if bearing_deg is not None:
            bearing_deg = round(bearing_deg) % 360
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "icao": row["icao"],
                    "callsign": row["callsign"],
                    "glyph": glyph_for(
                        row["type_aircraft"],
                        row["icao_class"],
                        row["manufacturer"],
                    ),
                    "age_s": round(float(row["age_s"]), 1),
                    "msg_count": row["msg_count"],
                    "altitude_ft": row["altitude_ft"],
                    "ground_speed": row["ground_speed"],
                    "track": row["track"],
                    "vertical_rate": row["vertical_rate"],
                    "rssi_db": row["rssi_db"],
                    "distance_km": distance_km,
                    "bearing_deg": bearing_deg,
                },
            },
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        # Server clock, so the client can place appended live samples on the
        # same time axis as /history's per-message t values (t = now - age_s).
        "now": round(time.time(), 1),
    }


@app.route("/api/aircraft/<icao>")
def api_aircraft_one(icao):
    # Fetched once per selection: registry join + lifetime stats. FAA
    # N-numbers are stored without their leading "N" — restore it here.
    icao = icao.upper()
    with db.connect() as conn:
        registry = conn.execute(REGISTRY_SQL, {"icao": icao}).fetchone()
        lifetime = conn.execute(LIFETIME_SQL, {"icao": icao}).fetchone()
    if registry is None and lifetime["msg_count"] == 0:
        abort(404)
    if registry and registry["source"] == "faa" and registry["registration"]:
        registry["registration"] = f"N{registry['registration']}"
    return {
        "icao": icao,
        "registry": registry,
        "msg_count": lifetime["msg_count"],
        "session_count": lifetime["session_count"],
        "first_heard": lifetime["first_heard"] and lifetime["first_heard"].isoformat(),
        "last_heard": lifetime["last_heard"] and lifetime["last_heard"].isoformat(),
    }


@app.route("/api/aircraft/<icao>/history")
def api_aircraft_history(icao):
    # Fetched once per selection: per-message points for the selected
    # aircraft's trail and RSSI sparkline. The client appends live samples
    # from the 1 Hz poll, so this is never re-fetched while selected.
    minutes = request.args.get(
        "minutes",
        default=60,
        type=float,
    )
    with db.connect() as conn:
        rows = conn.execute(
            HISTORY_SQL,
            {
                "icao": icao.upper(),
                "minutes": minutes,
            },
        ).fetchall()
    return {
        "icao": icao.upper(),
        "points": [
            {
                "t": round(float(row["t"]), 1),
                "lat": row["latitude"],
                "lon": row["longitude"],
                "altitude_ft": row["altitude_ft"],
                "rssi_db": row["rssi_db"],
            }
            for row in rows
        ],
    }


@app.route("/api/overlay")
def api_overlay():
    # Receiver point + range-ring polygons, all PostGIS-generated server-side
    # — the JS only ever draws coordinates it was handed.
    with db.connect() as conn:
        rows = conn.execute(
            RINGS_SQL,
            {
                "lat": config.RECEIVER_LAT,
                "lon": config.RECEIVER_LON,
                "radii_km": list(config.MAP_RINGS_KM),
            },
        ).fetchall()
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [config.RECEIVER_LON, config.RECEIVER_LAT],
            },
            "properties": {
                "kind": "receiver",
            },
        },
    ]
    for row in rows:
        features.append(
            {
                "type": "Feature",
                "geometry": json.loads(row["geojson"]),
                "properties": {
                    "kind": "ring",
                    "label": f"{row['radius_km']:g} km",
                    "radius_km": row["radius_km"],
                },
            },
        )
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def main():
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.SERVER_DEBUG,
    )


if __name__ == "__main__":
    main()
