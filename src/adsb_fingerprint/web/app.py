"""Flask app for the live aircraft map (doc/WEBUI.md) — W1: basemap + overlay."""

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
    send_file,
)

from adsb_fingerprint import config

app = Flask(__name__)

EARTH_RADIUS_KM = 6371.0088


@app.route("/")
def index():
    return render_template(
        "index.html",
        center=[config.RECEIVER_LON, config.RECEIVER_LAT],
        zoom=config.MAP_DEFAULT_ZOOM,
    )


@app.route("/tiles/<path:filename>")
def tiles(filename):
    # Serve only the configured basemap. conditional=True enables the HTTP
    # range requests the pmtiles protocol relies on to read slices of the
    # archive instead of downloading all of it.
    if filename != config.MAP_TILES_FILE:
        abort(404)
    return send_file(config.MAP_TILES_PATH, conditional=True)


@app.route("/api/overlay")
def api_overlay():
    # Receiver point + range-ring polygons, server-generated so the JS only
    # ever draws coordinates it was handed (spherical trig here; PostGIS
    # ST_Buffer is the W5 swap if ever wanted).
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
