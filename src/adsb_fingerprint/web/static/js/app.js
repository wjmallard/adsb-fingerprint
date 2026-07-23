// Live ADS-B map: vendored dark PMTiles basemap centered on the receiver,
// receiver + range rings from /api/overlay (W1), a 1 Hz poll of /api/aircraft
// driving the plane symbols, the roster, and the status chip (W2), and
// two-way selection with a registry + live detail panel (W3). All geometry is
// server-generated — this file only draws coordinates it was handed.
(function () {
    const cfg = window.MAP_CONFIG;

    const POLL_MS = 1000;
    const STALE_S = 60;   // markers start fading, roster rows dim
    const DROP_S = 300;   // markers leave the map (the aircraft stays rostered)

    // Nose-up plane glyph (Material Symbols "flight"), rasterized at 2x and
    // registered at runtime — no sprite-sheet edits.
    const PLANE_SVG =
        '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24">'
        + '<path fill="#ffd24d" stroke="#1c1e22" stroke-width="0.8" d="M21,16v-2l-8-5V3.5'
        + 'C13,2.67,12.33,2,11.5,2S10,2.67,10,3.5V9l-8,5v2l8-2.5V19l-2,1.5V22l3.5-1l3.5,1'
        + 'v-1.5L13,19v-5.5L21,16z"/></svg>';

    const fadeByAge = [
        "interpolate",
        ["linear"],
        ["get", "age_s"],
        STALE_S, 1,
        DROP_S, 0.3,
    ];

    let selectedIcao = null;
    let lastFeatures = [];

    // The map needs WebGL; the roster, chip, and detail panel don't. If the
    // map can't come up (e.g. graphics acceleration disabled), keep polling
    // anyway so the page degrades to a live roster over a blank map area.
    let map = null;
    try {
        const protocol = new pmtiles.Protocol();
        maplibregl.addProtocol("pmtiles", protocol.tile);

        map = new maplibregl.Map({
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

        map.on("load", async () => {
            await setupOverlay(map);
            setupAircraftLayers(map);
        });
    } catch (err) {
        console.error("map init failed (WebGL unavailable?) — roster still live", err);
    }

    document.getElementById("roster-rows").addEventListener("click", (e) => {
        const row = e.target.closest(".row[data-icao]");
        if (!row) return;
        if (row.dataset.icao === selectedIcao) deselect();
        else selectAircraft(row.dataset.icao);
    });

    pollAircraft();
    setInterval(() => {
        if (!document.hidden) pollAircraft();
    }, POLL_MS);

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

    function setupAircraftLayers(map) {
        map.addSource("aircraft", {
            type: "geojson",
            data: {
                type: "FeatureCollection",
                features: [],
            },
        });

        const img = new Image(48, 48);
        img.onload = () => {
            map.addImage("plane", img, { pixelRatio: 2 });

            map.addLayer({
                id: "aircraft-selected",
                type: "circle",
                source: "aircraft",
                filter: selectionFilter(),
                paint: {
                    "circle-radius": 14,
                    "circle-color": "rgba(0, 0, 0, 0)",
                    "circle-stroke-color": "#5ad1e6",
                    "circle-stroke-width": 2,
                },
            });

            map.addLayer({
                id: "aircraft",
                type: "symbol",
                source: "aircraft",
                filter: ["<=", ["get", "age_s"], DROP_S],
                layout: {
                    "icon-image": "plane",
                    "icon-size": 0.9,
                    "icon-rotate": ["coalesce", ["get", "track"], 0],
                    "icon-rotation-alignment": "map",
                    "icon-allow-overlap": true,
                    "text-field": ["coalesce", ["get", "callsign"], ["get", "icao"]],
                    "text-font": ["Noto Sans Regular"],
                    "text-size": 10,
                    "text-offset": [0, 1.6],
                    "text-anchor": "top",
                    "text-optional": true,
                },
                paint: {
                    "icon-opacity": fadeByAge,
                    "text-color": "#c8cdd4",
                    "text-halo-color": "#1c1e22",
                    "text-halo-width": 1.5,
                    "text-opacity": fadeByAge,
                },
            });

            // Two-way selection, map side: click a plane to select it,
            // click empty map to deselect.
            map.on("click", (e) => {
                const hits = map.queryRenderedFeatures(e.point, {
                    layers: ["aircraft"],
                });
                if (hits.length) selectAircraft(hits[0].properties.icao);
                else deselect();
            });
            map.on("mouseenter", "aircraft", () => {
                map.getCanvas().style.cursor = "pointer";
            });
            map.on("mouseleave", "aircraft", () => {
                map.getCanvas().style.cursor = "";
            });
        };
        img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(PLANE_SVG);
    }

    async function pollAircraft() {
        let collection;
        try {
            const resp = await fetch(`/api/aircraft?minutes=${cfg.rosterMinutes}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            collection = await resp.json();
        } catch (err) {
            console.error("aircraft poll failed", err);
            renderChip(null);
            return;
        }
        lastFeatures = collection.features;
        const source = map && map.getSource("aircraft");
        if (source) source.setData(collection);
        renderRoster(lastFeatures);
        renderChip(lastFeatures);
        renderDetailLive();
    }

    function renderRoster(features) {
        const rows = features.map((f) => {
            const p = f.properties;
            const stale = p.age_s > STALE_S;
            const selected = p.icao === selectedIcao;
            return `<div class="row${stale ? " stale" : ""}${selected ? " selected" : ""}"`
                + ` data-icao="${escapeHtml(p.icao)}">`
                + `<span class="dot">${stale ? "○" : "●"}</span>`
                + `<span class="icao">${escapeHtml(p.icao)}</span>`
                + `<span class="callsign">${escapeHtml(p.callsign ?? "")}</span>`
                + `<span class="age">${fmtAge(p.age_s)}</span>`
                + `<span class="alt">${p.altitude_ft ?? ""}</span>`
                + `<span class="rssi">${p.rssi_db == null ? "" : Math.round(p.rssi_db)}</span>`
                + `</div>`;
        });
        document.getElementById("roster-rows").innerHTML = rows.join("");
        document.getElementById("roster-count").textContent =
            `— ${features.length} aircraft · last ${cfg.rosterMinutes} min`;
    }

    // The header chip doubles as a collector-health indicator: with the
    // collector flushing every ~1 s, a growing age means it stopped.
    function renderChip(features) {
        const chip = document.getElementById("status-chip");
        if (features === null) {
            chip.textContent = "API unreachable";
            chip.classList.add("warn");
            return;
        }
        if (!features.length) {
            chip.textContent = `no messages in ${cfg.rosterMinutes} min`;
            chip.classList.add("warn");
            return;
        }
        const age = features[0].properties.age_s;
        chip.textContent = `last message ${fmtAge(age)} ago`;
        chip.classList.toggle("warn", age > STALE_S);
    }

    // ---- selection + detail panel -------------------------------------

    async function selectAircraft(icao) {
        if (icao === selectedIcao) return;
        selectedIcao = icao;
        highlightSelection();
        const body = document.getElementById("detail-body");
        body.className = "";
        body.innerHTML = "";
        let info = null;
        try {
            const resp = await fetch(`/api/aircraft/${icao}`);
            if (resp.ok) info = await resp.json();
        } catch (err) {
            console.error("detail load failed", err);
        }
        if (icao !== selectedIcao) return;   // reselected while fetching
        renderDetailStatic(info);
        renderDetailLive();
    }

    function deselect() {
        if (selectedIcao === null) return;
        selectedIcao = null;
        highlightSelection();
        document.getElementById("detail-head").textContent = "selected";
        const body = document.getElementById("detail-body");
        body.className = "placeholder";
        body.textContent = "nothing selected yet";
    }

    function selectionFilter() {
        return [
            "all",
            ["==", ["get", "icao"], selectedIcao ?? ""],
            ["<=", ["get", "age_s"], DROP_S],
        ];
    }

    function highlightSelection() {
        if (map && map.getLayer("aircraft-selected")) {
            map.setFilter("aircraft-selected", selectionFilter());
        }
        renderRoster(lastFeatures);
    }

    // Registry + lifetime, fetched once per selection.
    function renderDetailStatic(info) {
        const body = document.getElementById("detail-body");
        if (!info) {
            body.innerHTML = '<div class="detail-note">no data for this aircraft</div>';
            return;
        }
        const blocks = [];
        const r = info.registry;
        if (r) {
            const fields = [
                ["reg", r.registration],
                ["aircraft", joinTruthy([r.manufacturer, r.model], " ")],
                ["type", joinTruthy([r.type, r.typecode && `(${r.typecode})`], " ")],
                ["owner", r.owner],
                ["operator", r.operator !== r.owner ? r.operator : null],
                ["based", joinTruthy([r.owner_city, r.owner_state], ", ") || r.country],
                ["source", r.source],
            ];
            blocks.push(block("registry", fields));
        } else {
            blocks.push(
                '<div class="block"><div class="block-title">registry</div>'
                + '<div class="detail-note">not in registry</div></div>',
            );
        }
        blocks.push('<div class="block"><div class="block-title">live</div><div id="detail-live"></div></div>');
        blocks.push(
            `<div class="detail-note">${info.msg_count.toLocaleString()} messages · `
            + `${info.session_count} session${info.session_count === 1 ? "" : "s"} · `
            + `first heard ${(info.first_heard ?? "").slice(0, 10) || "never"}</div>`,
        );
        body.innerHTML = blocks.join("");
    }

    // Live block, re-rendered from every poll's payload.
    function renderDetailLive() {
        if (!selectedIcao) return;
        const feature = lastFeatures.find((f) => f.properties.icao === selectedIcao);
        const p = feature?.properties;
        document.getElementById("detail-head").textContent =
            `selected — ${selectedIcao}${p?.callsign ? " · " + p.callsign : ""}`;
        const slot = document.getElementById("detail-live");
        if (!slot) return;
        if (!feature) {
            slot.innerHTML = `<div class="detail-note">not heard in the last ${cfg.rosterMinutes} min</div>`;
            return;
        }
        const position = feature.geometry
            ? `${feature.geometry.coordinates[1].toFixed(4)}, ${feature.geometry.coordinates[0].toFixed(4)}`
            : null;
        const fields = [
            ["altitude", p.altitude_ft != null ? `${p.altitude_ft.toLocaleString()} ft` : null],
            ["speed", p.ground_speed != null ? `${Math.round(p.ground_speed)} kt` : null],
            ["track", p.track != null ? `${Math.round(p.track)}°` : null],
            ["v/rate", p.vertical_rate != null ? `${p.vertical_rate} fpm` : null],
            ["position", position],
            ["range", p.distance_km != null ? `${p.distance_km} km @ ${p.bearing_deg}°` : null],
        ];
        slot.innerHTML = fieldsHtml(fields) || '<div class="detail-note">no live fields yet</div>';
    }

    function block(title, fields) {
        return `<div class="block"><div class="block-title">${title}</div>${fieldsHtml(fields)}</div>`;
    }

    function fieldsHtml(fields) {
        return fields
            .filter(([, value]) => value)
            .map(
                ([label, value]) =>
                    `<div class="field"><span class="label">${escapeHtml(label)}</span>`
                    + `<span class="value">${escapeHtml(String(value))}</span></div>`,
            )
            .join("");
    }

    function joinTruthy(parts, separator) {
        return parts.filter(Boolean).join(separator);
    }

    function fmtAge(seconds) {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        return `${Math.floor(seconds / 3600)}h`;
    }

    function escapeHtml(s) {
        return s.replace(/[&<>"']/g, (c) => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
        })[c]);
    }
})();
