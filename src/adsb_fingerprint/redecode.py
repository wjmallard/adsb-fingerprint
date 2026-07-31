"""adsb-redecode: re-decode stored messages' positions from their raw hex.

Positions written live before the self-locating decode were resolved
against a fixed receiver reference — only unambiguous within 180 NM — so
rows collected far from that reference carry plausible-looking aliases.
Every message keeps its hex, so this replays whole sessions through the
same reference-free pipeline the collector now uses (batch mode, which
also retro-fills fixes held during each aircraft's bootstrap) and
rewrites latitude/longitude with what actually resolves: true positions
where a track validates, honest nulls where nothing does.

The per-row receiver stamp is redone per --receiver: a stationary
session gets the traffic-median station estimate (config-verified, like
the live collector); a mobile session's stamps are nulled as unknown
until GPS tracks (adsb-geotag) or a future receiver-track solver supply
real ones; --receiver keep leaves stamps alone.
"""

import argparse

from pyModeS.util import icao as icao_of

from adsb_fingerprint import db, modes
from adsb_fingerprint.collect import (
    STATION_MIN_AIRCRAFT,
    StationEstimator,
    resolve_station,
)

POSITION_UPDATE_SQL = """
    update messages
    set latitude = u.latitude,
        longitude = u.longitude
    from redecode_updates u
    where messages.id = u.id
"""

RECEIVER_UPDATE_SQL = """
    update messages
    set receiver_latitude = %(latitude)s,
        receiver_longitude = %(longitude)s
    where session = %(session)s
"""


def redecode_session(conn, session, receiver, dry_run=False):
    rows = conn.execute(
        """
        select
            id,
            hex,
            captured_at,
            latitude
        from messages
        where session = %(session)s
        order by captured_at
        """,
        {"session": session},
    ).fetchall()
    if not rows:
        print(f"{session}: no messages")
        return

    decoded = modes.decode_batch(
        [row["hex"] for row in rows],
        [row["captured_at"].timestamp() for row in rows],
    )

    estimator = StationEstimator()
    updates = []
    had = 0
    have = 0
    for row, fields in zip(rows, decoded):
        latitude = fields["latitude"]
        longitude = fields["longitude"]
        if row["latitude"] is not None:
            had += 1
        if latitude is not None:
            have += 1
            estimator.add(icao_of(row["hex"]).upper(), latitude, longitude)
        updates.append((row["id"], latitude, longitude))

    print(f"{session}: {len(rows)} messages — positions {had} stored -> {have} resolved")

    stamp = None
    if receiver == "estimate":
        if estimator.n >= STATION_MIN_AIRCRAFT:
            stamp = resolve_station(estimator)
        else:
            print(
                f"  receiver: unresolved ({estimator.n} aircraft, "
                f"need {STATION_MIN_AIRCRAFT}) — stamps untouched"
            )

    if dry_run:
        print("  dry run — database unchanged")
        return

    conn.execute(
        """
        create temp table redecode_updates (
            id bigint primary key,
            latitude double precision,
            longitude double precision
        ) on commit drop
        """
    )
    db.copy_rows(
        conn,
        "copy redecode_updates (id, latitude, longitude) from stdin",
        updates,
        total=len(updates),
        desc="load",
    )
    updated = conn.execute(POSITION_UPDATE_SQL).rowcount
    if stamp is not None:
        conn.execute(
            RECEIVER_UPDATE_SQL,
            {
                "latitude": stamp[0],
                "longitude": stamp[1],
                "session": session,
            },
        )
        print(f"  receiver stamps -> {stamp[0]:.4f}, {stamp[1]:.4f} ({stamp[2]})")
    elif receiver == "unknown":
        conn.execute(
            RECEIVER_UPDATE_SQL,
            {
                "latitude": None,
                "longitude": None,
                "session": session,
            },
        )
        print("  receiver stamps -> null (unknown)")
    conn.commit()
    print(f"  updated {updated} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Re-decode stored messages' positions from their raw hex.",
    )
    parser.add_argument(
        "sessions",
        nargs="+",
        help="Session names to re-decode.",
    )
    parser.add_argument(
        "--receiver",
        choices=(
            "estimate",
            "unknown",
            "keep",
        ),
        default="estimate",
        help=(
            "Receiver stamp policy: traffic-median station estimate "
            "(stationary sessions, default), null for a mobile/unknown "
            "receiver, or keep existing stamps."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without updating the database.",
    )
    args = parser.parse_args()

    with db.connect() as conn:
        for session in args.sessions:
            redecode_session(
                conn,
                session,
                args.receiver,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
