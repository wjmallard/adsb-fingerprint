// Live ADS-B map (W1 scaffold): vendored dark PMTiles basemap centered on the
// receiver, receiver marker + range rings from /api/overlay. All geometry is
// server-generated — this file only draws coordinates it was handed.
(function () {
    const cfg = window.MAP_CONFIG;

    // Teach MapLibre to read the vendored single-file .pmtiles archive.
    const protocol = new pmtiles.Protocol();
    maplibregl.addProtocol("pmtiles", protocol.tile);

    const map = new maplibregl.Map({
        container: "map",
        style: cfg.styleUrl,
        center: cfg.center,
        zoom: cfg.zoom,
    });

    map.addControl(
        new maplibregl.NavigationControl(),
        "top-right",
    );
    map.addControl(new maplibregl.ScaleControl());

    map.on("load", () => setupOverlay(map));

    async function setupOverlay(map) {
        let overlay;
        try {
            const resp = await fetch("/api/overlay");
            if (!resp.ok) return;
            overlay = await resp.json();
        } catch (err) {
            console.error("overlay load failed", err);
            return;
        }

        map.addSource("overlay", {
            type: "geojson",
            data: overlay,
        });

        map.addLayer({
            id: "range-rings",
            type: "line",
            source: "overlay",
            filter: ["==", ["get", "kind"], "ring"],
            paint: {
                "line-color": "#5b7ea8",
                "line-opacity": 0.6,
                "line-width": 1,
            },
        });

        map.addLayer({
            id: "range-ring-labels",
            type: "symbol",
            source: "overlay",
            filter: ["==", ["get", "kind"], "ring"],
            layout: {
                "symbol-placement": "line",
                "text-field": ["get", "label"],
                "text-font": ["Noto Sans Regular"],
                "text-size": 11,
            },
            paint: {
                "text-color": "#8fa8c4",
                "text-halo-color": "#34373d",
                "text-halo-width": 1.5,
            },
        });

        map.addLayer({
            id: "receiver",
            type: "circle",
            source: "overlay",
            filter: ["==", ["get", "kind"], "receiver"],
            paint: {
                "circle-color": "#f0a13a",
                "circle-radius": 5,
                "circle-stroke-color": "#1c1e22",
                "circle-stroke-width": 2,
            },
        });
    }
})();
