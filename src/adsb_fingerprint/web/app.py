"""Flask app for the live aircraft map (doc/WEBUI.md) — W1 basemap, W2 live."""

from math import (
    asin,
    atan2,
    cos,
    degrees,
    pi,
    radians,
    sin,
)

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

EARTH_RADIUS_KM = 6371.0088

# One row per aircraft heard in the window. Position, velocity, and ident ride
# in different message types, so the newest row rarely has everything — each
# field is independently "latest non-null".
AIRCRAFT_SQL = """
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
    order by max(captured_at) desc
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
    # roster, and (W3) the detail panel's live block.
    minutes = request.args.get(
        "minutes",
        default=config.MAP_ROSTER_MINUTES,
        type=float,
    )
    with db.connect() as conn:
        rows = conn.execute(AIRCRAFT_SQL, {"minutes": minutes}).fetchall()
    features = []
    for row in rows:
        if row["latitude"] is not None and row["longitude"] is not None:
            geometry = {
                "type": "Point",
                "coordinates": [row["longitude"], row["latitude"]],
            }
        else:
            geometry = None
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
                },
            },
        )
    return {
        "type": "FeatureCollection",
        "features": features,
    }


@app.route("/api/overlay")
def api_overlay():
    # Receiver point + range-ring polygons, server-generated so the JS only
    # ever draws coordinates it was handed (spherical trig here; PostGIS
    # ST_Buffer is the W5 swap).
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
    for radius_km in config.MAP_RINGS_KM:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": ring_polygon(
                        config.RECEIVER_LAT,
                        config.RECEIVER_LON,
                        radius_km,
                    ),
                },
                "properties": {
                    "kind": "ring",
                    "label": f"{radius_km:g} km",
                    "radius_km": radius_km,
                },
            },
        )
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def ring_polygon(lat_deg, lon_deg, radius_km, points=128):
    """Closed circle of geodesic radius around (lat, lon), as polygon coords."""
    lat1 = radians(lat_deg)
    lon1 = radians(lon_deg)
    delta = radius_km / EARTH_RADIUS_KM
    coords = []
    for i in range(points + 1):
        bearing = -2 * pi * i / points  # negative: counterclockwise exterior ring
        lat2 = asin(
            sin(lat1) * cos(delta) + cos(lat1) * sin(delta) * cos(bearing),
        )
        lon2 = lon1 + atan2(
            sin(bearing) * sin(delta) * cos(lat1),
            cos(delta) - sin(lat1) * sin(lat2),
        )
        coords.append([round(degrees(lon2), 6), round(degrees(lat2), 6)])
    return [coords]


def main():
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.SERVER_DEBUG,
    )


if __name__ == "__main__":
    main()
