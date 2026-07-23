"""adsb-stats: summarize what's in the database — corpus size, cross-session
coverage (the fingerprinting-readiness metric), aircraft types, and storage.
"""

import argparse

from adsb_fingerprint import config, db


def _sz(nbytes):
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"


def _iq_bytes(conn):
    total = 0
    rows = conn.execute("select distinct capture_file from messages").fetchall()
    for row in rows:
        path = config.CAPTURE_DIR / row["capture_file"]
        if path.exists():
            total += path.stat().st_size
    return total


def main():
    argparse.ArgumentParser(description="Summarize the ADS-B database.").parse_args()

    with db.connect() as conn:
        overview = conn.execute(
            """
            select
                count(*) as msgs,
                count(distinct icao) as aircraft,
                count(distinct session) as sessions,
                min(captured_at) as first_seen,
                max(captured_at) as last_seen
            from messages
            where crc_ok
            """
        ).fetchone()

        if not overview["msgs"]:
            print("No messages yet — run adsb-collect / adsb-index first.")
            return

        matched = conn.execute(
            """
            select count(distinct m.icao) as n
            from messages m
            join aircraft a using (icao)
            where m.crc_ok
            """
        ).fetchone()["n"]

        print("=== corpus ===")
        print(f"messages : {overview['msgs']:,}")
        print(f"aircraft : {overview['aircraft']:,}  ({matched:,} matched to registry)")
        print(f"sessions : {overview['sessions']:,}")
        print(f"span     : {overview['first_seen']:%Y-%m-%d %H:%M} .. {overview['last_seen']:%Y-%m-%d %H:%M} UTC")

        print("\n=== storage ===")
        for row in conn.execute(
            """
            select
                relname as name,
                pg_total_relation_size(c.oid) as bytes
            from pg_class c
            join pg_namespace n on n.oid = c.relnamespace
            where n.nspname = 'public'
                and relkind = 'r'
            order by bytes desc
            """
        ).fetchall():
            print(f"  db/{row['name']:<18} {_sz(row['bytes'])}")
        print(f"  iq snippets/captures {_sz(_iq_bytes(conn))}")

        print("\n=== cross-session coverage (aircraft by # sessions seen) ===")
        coverage = conn.execute(
            """
            select
                sessions_seen,
                count(*) as aircraft
            from (
                select
                    icao,
                    count(distinct session) as sessions_seen
                from messages
                where crc_ok
                group by icao
            ) t
            group by sessions_seen
            order by sessions_seen
            """
        ).fetchall()
        for row in coverage:
            print(f"  {row['sessions_seen']:>2} session(s): {row['aircraft']} aircraft")
        multi = sum(r["aircraft"] for r in coverage if r["sessions_seen"] >= 2)
        print(f"  -> {multi} aircraft in >= 2 sessions (usable for cross-session eval)")

        print("\n=== per session ===")
        for row in conn.execute(
            """
            select
                session,
                count(*) as msgs,
                count(distinct icao) as aircraft,
                to_char(min(captured_at), 'MM-DD HH24:MI') as first,
                to_char(max(captured_at), 'HH24:MI') as last
            from messages
            where crc_ok
            group by session
            order by min(captured_at)
            """
        ).fetchall():
            print(f"  {row['session']:<22} {row['msgs']:>6} msgs  {row['aircraft']:>3} ac  {row['first']}-{row['last']}")

        print("\n=== aircraft types ===")
        for row in conn.execute(
            """
            select
                coalesce(a.type, '(unmatched)') as type,
                count(distinct m.icao) as aircraft,
                count(*) as msgs
            from messages m
            left join aircraft a using (icao)
            where m.crc_ok
            group by a.type
            order by aircraft desc
            """
        ).fetchall():
            print(f"  {row['type']:<26} {row['aircraft']:>4} ac  {row['msgs']:>7} msgs")

        print("\n=== most-heard aircraft ===")
        for row in conn.execute(
            """
            select
                m.icao,
                count(*) as msgs,
                count(distinct m.session) as sessions,
                a.registration,
                a.manufacturer,
                a.model
            from messages m
            left join aircraft a using (icao)
            where m.crc_ok
            group by m.icao, a.registration, a.manufacturer, a.model
            order by msgs desc
            limit 12
            """
        ).fetchall():
            label = " ".join(p for p in (row["registration"], row["manufacturer"], row["model"]) if p)
            print(f"  {row['icao']}  {row['msgs']:>6} msgs  {row['sessions']:>2} sess   {label}")


if __name__ == "__main__":
    main()
