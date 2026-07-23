"""Flask app for the live aircraft map (doc/WEBUI.md) — W1 basemap, W2 live, W3 detail."""

import json

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
AIRCRAFT_SQL = """
    with latest as (
        select
            icao,
            count(*) as msg_count,
            extract(epoch from now() - max(captured_at)) as age_s,
            (array_agg(callsign order by captured_at desc) filter (where callsign is not null))[1] as callsign,
            (array_agg(latitude order by captured_at desc) filter (where latitude is not null))[1] as latitude,
            (array_agg(longitude order by captured_at desc) filter (where longitude is not null))[1] as longitude,
            (array_agg(altitude_ft order by captured_at desc) filter (where altitude_ft is not null))[1] as altitude_ft,
            (array_agg(ground_speed order by captured_at desc) filter (where ground_speed is not null))[1] as ground_speed,
            (array_agg(track order by captured_at desc) filter (where track is not null))[1] as track,
            (array_agg(vertical_rate order by captured_at desc) filter (where vertical_rate is not null))[1] as vertical_rate,
            (array_agg(rssi_db order by captured_at desc) filter (where rssi_db is not null))[1] as rssi_db
        from messages
        where captured_at > now() - %(minutes)s * interval '1 minute'
        and crc_ok
        and icao is not null
        group by icao
    )
    select
        latest.*,
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
    order by age_s
"""

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


@app.route("/tiles/<path:filename>")
def tiles(filename):
    # Serve only the configured basemap. conditional=True enables the HTTP
    # range requests the pmtiles protocol relies on to read slices of the
    # archive instead of downloading all of it.
    if filename != config.MAP_TILES_FILE:
        abort(404)
    return send_file(config.MAP_TILES_PATH, conditional=True)


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
