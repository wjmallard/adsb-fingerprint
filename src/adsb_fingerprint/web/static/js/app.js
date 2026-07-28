// Live ADS-B map: vendored dark PMTiles basemap centered on the receiver,
// receiver + range rings from /api/overlay (W1), a 1 Hz poll of /api/aircraft
// driving the plane symbols, the roster, and the status chip (W2), two-way
// selection with a registry + live detail panel (W3), and per-selection
// history: position trail + RSSI sparkline + radio block (W4). All geometry
// is server-generated — this file only draws coordinates it was handed.
(function () {
    const cfg = window.MAP_CONFIG;

    const POLL_MS = 1000;
    const STALE_S = 60;      // markers start fading, roster rows dim
    const DROP_S = 300;      // markers leave the map (the aircraft stays rostered)
    const SPARK_S = 600;     // RSSI sparkline window
    const RATE_S = 60;       // message-rate window in the radio block

    // Nose-up glyphs, rasterized at 2x and registered at runtime — no
    // sprite-sheet edits. /api/aircraft's per-airframe glyph property picks
    // one (registry-driven: rotorcraft get the heli, single-engine GA gets
    // small). plane is Material Symbols "flight"; heli is hand-drawn —
    // rotor X, cabin, tail boom; small is a hand-drawn high-wing trainer —
    // straight square-tipped wing, boxy stabilizer, smaller footprint.
    const GLYPH_SVGS = {
        plane:
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24">'
            + '<path fill="#ffd24d" stroke="#1c1e22" stroke-width="0.8" d="M21,16v-2l-8-5V3.5'
            + 'C13,2.67,12.33,2,11.5,2S10,2.67,10,3.5V9l-8,5v2l8-2.5V19l-2,1.5V22l3.5-1l3.5,1'
            + 'v-1.5L13,19v-5.5L21,16z"/></svg>',
        small:
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24">'
            + '<path fill="#ffd24d" stroke="#1c1e22" stroke-width="0.8" '
            + 'd="M11,5.6L12,3.9L13,5.6V19H11Z'
            + 'M4.6,8.8h14.8v2.3H4.6Z'
            + 'M9.3,17.4h5.4v1.6H9.3Z"/></svg>',
        heli:
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24">'
            + '<g stroke="#1c1e22" stroke-width="2.6" stroke-linecap="round">'
            + '<line x1="5.8" y1="4.3" x2="18.2" y2="16.7"/>'
            + '<line x1="18.2" y1="4.3" x2="5.8" y2="16.7"/></g>'
            + '<g stroke="#ffd24d" stroke-width="1.3" stroke-linecap="round">'
            + '<line x1="5.8" y1="4.3" x2="18.2" y2="16.7"/>'
            + '<line x1="18.2" y1="4.3" x2="5.8" y2="16.7"/></g>'
            + '<path fill="#ffd24d" stroke="#1c1e22" stroke-width="0.8" '
            + 'd="M12.65,13.6v4.9h1.9v1.4h-5.1v-1.4h1.9v-4.9z"/>'
            + '<ellipse cx="12" cy="10.5" rx="2.7" ry="3.5" fill="#ffd24d" '
            + 'stroke="#1c1e22" stroke-width="0.8"/></svg>',
    };

    const fadeByAge = [
        "interpolate",
        ["linear"],
        ["get", "age_s"],
        STALE_S, 1,
        DROP_S, 0.3,
    ];

    let selectedIcao = null;
    let lastFeatures = [];
    let lastServerNow = null;

    // Per-selection buffers: seeded from /history, extended by the 1 Hz poll.
    let selection = null;   // { msgTimes, rssiPoints: [{t, rssi}], trailCoords, lastT }

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
            await setupAircraftLayers(map);
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

    function loadGlyph(map, name, svg) {
        return new Promise((resolve) => {
            const img = new Image(48, 48);
            img.onload = () => {
                map.addImage(name, img, { pixelRatio: 2 });
                resolve();
            };
            img.onerror = (err) => {
                // A bad glyph shouldn't sink every layer — those aircraft
                // just render label-only until it's fixed.
                console.error(`glyph "${name}" failed to load`, err);
                resolve();
            };
            img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
        });
    }

    async function setupAircraftLayers(map) {
        map.addSource("aircraft", {
            type: "geojson",
            data: emptyCollection(),
        });
        map.addSource("trail", {
            type: "geojson",
            data: emptyCollection(),
        });

        await Promise.all(
            Object.entries(GLYPH_SVGS).map(
                ([name, svg]) => loadGlyph(map, name, svg),
            ),
        );

        map.addLayer({
            id: "trail",
            type: "line",
            source: "trail",
            paint: {
                "line-color": "#5ad1e6",
                "line-opacity": 0.65,
                "line-width": 1.5,
            },
        });

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
                "icon-image": ["coalesce", ["get", "glyph"], "plane"],
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
        lastServerNow = collection.now;
        const source = map && map.getSource("aircraft");
        if (source) source.setData(collection);
        renderRoster(lastFeatures);
        renderChip(lastFeatures);
        renderDetailLive();
        appendLiveSample();
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
        selection = null;
        highlightSelection();
        setTrail([]);
        const body = document.getElementById("detail-body");
        body.className = "";
        body.innerHTML = "";
        let info = null;
        let history = null;
        try {
            const [infoResp, historyResp] = await Promise.all([
                fetch(`/api/aircraft/${icao}`),
                fetch(`/api/aircraft/${icao}/history`),
            ]);
            if (infoResp.ok) info = await infoResp.json();
            if (historyResp.ok) history = await historyResp.json();
        } catch (err) {
            console.error("detail load failed", err);
        }
        if (icao !== selectedIcao) return;   // reselected while fetching
        seedSelection(history);
        renderDetailStatic(info);
        renderDetailLive();
        updateRadio();
        setTrail(selection.trailCoords);
    }

    function deselect() {
        if (selectedIcao === null) return;
        selectedIcao = null;
        selection = null;
        highlightSelection();
        setTrail([]);
        document.getElementById("detail-head").textContent = "selected";
        const body = document.getElementById("detail-body");
        body.className = "placeholder";
        body.textContent = "nothing selected yet";
    }

    function seedSelection(history) {
        selection = {
            msgTimes: [],
            rssiPoints: [],
            trailCoords: [],
            lastT: 0,
        };
        for (const point of history?.points ?? []) {
            selection.msgTimes.push(point.t);
            selection.lastT = point.t;
            if (point.rssi_db != null) {
                selection.rssiPoints.push({ t: point.t, rssi: point.rssi_db });
            }
            if (point.lat != null && point.lon != null) {
                selection.trailCoords.push([point.lon, point.lat]);
            }
        }
    }

    // Extend the selection's buffers from the 1 Hz payload — the newest
    // message's server-side timestamp is (now - age_s), so history and live
    // samples share one time axis and nothing is ever re-fetched.
    function appendLiveSample() {
        if (!selection || lastServerNow == null) return;
        const feature = lastFeatures.find((f) => f.properties.icao === selectedIcao);
        if (!feature) return;
        const p = feature.properties;
        const t = lastServerNow - p.age_s;
        if (t <= selection.lastT + 0.5) return;   // same message as last poll
        selection.lastT = t;
        selection.msgTimes.push(t);
        if (p.rssi_db != null) {
            selection.rssiPoints.push({ t: t, rssi: p.rssi_db });
        }
        if (feature.geometry) {
            const coord = feature.geometry.coordinates;
            const last = selection.trailCoords[selection.trailCoords.length - 1];
            if (!last || last[0] !== coord[0] || last[1] !== coord[1]) {
                selection.trailCoords.push(coord);
                setTrail(selection.trailCoords);
            }
        }
        updateRadio();
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

    function setTrail(coords) {
        const source = map && map.getSource("trail");
        if (!source) return;
        if (coords.length < 2) {
            source.setData(emptyCollection());
            return;
        }
        source.setData({
            type: "Feature",
            geometry: {
                type: "LineString",
                coordinates: coords,
            },
            properties: {},
        });
    }

    // Registry + lifetime + radio skeleton, rendered once per selection.
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
            '<div class="block"><div class="block-title">radio</div>'
            + fieldsHtml([["messages", info.msg_count.toLocaleString()]])
            + '<div class="field"><span class="label">rate</span><span class="value" id="radio-rate"></span></div>'
            + fieldsHtml([["sessions", String(info.session_count)]])
            + '<div class="field"><span class="label">rssi</span><span class="value" id="radio-rssi"></span></div>'
            + '<canvas id="rssi-spark" class="spark"></canvas>'
            + '</div>',
        );
        blocks.push(
            `<div class="detail-note">first heard ${(info.first_heard ?? "").slice(0, 10) || "never"}</div>`,
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
            ["range", p.distance_km != null
                ? `${p.distance_km} km${p.bearing_deg != null ? ` @ ${p.bearing_deg}°` : ""}`
                : null],
        ];
        slot.innerHTML = fieldsHtml(fields) || '<div class="detail-note">no live fields yet</div>';
    }

    // Radio block: rate + latest RSSI + sparkline, from the selection buffers.
    function updateRadio() {
        if (!selection) return;
        const rate = document.getElementById("radio-rate");
        const rssi = document.getElementById("radio-rssi");
        const spark = document.getElementById("rssi-spark");
        if (!rate || !rssi || !spark) return;
        const tMax = selection.lastT;
        const recent = selection.msgTimes.filter((t) => t >= tMax - RATE_S).length;
        rate.textContent = `${recent} msg/min`;
        const latest = selection.rssiPoints[selection.rssiPoints.length - 1];
        rssi.textContent = latest ? `${latest.rssi.toFixed(1)} dB` : "–";
        drawSparkline(spark, selection.rssiPoints, tMax);
    }

    // Last SPARK_S seconds of per-message RSSI as dots on a 2D canvas.
    function drawSparkline(canvas, points, tMax) {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);
        const recent = points.filter((p) => p.t >= tMax - SPARK_S);
        if (!recent.length) return;
        const values = recent.map((p) => p.rssi);
        let lo = Math.min(...values);
        let hi = Math.max(...values);
        if (hi - lo < 4) {   // keep a sane vertical scale for quiet traces
            const mid = (hi + lo) / 2;
            lo = mid - 2;
            hi = mid + 2;
        }
        ctx.fillStyle = "#5ad1e6";
        for (const p of recent) {
            const x = ((p.t - (tMax - SPARK_S)) / SPARK_S) * width;
            const y = height - 3 - ((p.rssi - lo) / (hi - lo)) * (height - 6);
            ctx.fillRect(x - 1, y - 1, 2, 2);
        }
        ctx.fillStyle = "#5c636e";
        ctx.font = "9px ui-monospace, monospace";
        ctx.fillText(`${Math.round(hi)}`, 2, 9);
        ctx.fillText(`${Math.round(lo)}`, 2, height - 3);
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

    function emptyCollection() {
        return {
            type: "FeatureCollection",
            features: [],
        };
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
