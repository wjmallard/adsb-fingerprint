"""adsb-ingest-opensky: load the OpenSky aircraft database into opensky_aircraft.

The OpenSky CSV is single-quote quoted with a mix of quoted and bare fields;
read_csv(quotechar="'") handles the dialect.
"""

import argparse
from pathlib import Path

import pandas as pd

from adsb_fingerprint import config, db

OPENSKY_COLS = [
    "icao24",
    "registration",
    "manufacturerName",
    "model",
    "typecode",
    "operator",
    "owner",
    "country",
    "icaoAircraftClass",
]

COPY_SQL = """
    copy opensky_aircraft (
        icao,
        registration,
        manufacturer,
        model,
        typecode,
        operator,
        owner,
        country,
        icao_class
    )
    from stdin
"""


def _default_csv():
    hits = sorted((config.PROJECT_ROOT / "data").glob("aircraft-database-complete-*.csv"))
    return hits[-1] if hits else None


def ingest(csv_path):
    csv_path = Path(csv_path).expanduser()
    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        quotechar="'",
        usecols=OPENSKY_COLS,
    )
    df = df.apply(lambda column: column.str.strip())
    df["icao"] = df["icao24"].str.upper()
    df = (
        df[df["icao"].str.fullmatch(r"[0-9A-F]{6}") & (df["icao"] != "000000")]
        .drop_duplicates("icao", keep="first")
        .reset_index(drop=True)
    )

    out = pd.DataFrame(
        {
            "icao": df["icao"].values,
            "registration": df["registration"].values,
            "manufacturer": df["manufacturerName"].values,
            "model": df["model"].values,
            "typecode": df["typecode"].values,
            "operator": df["operator"].values,
            "owner": df["owner"].values,
            "country": df["country"].values,
            "icao_class": df["icaoAircraftClass"].values,
        }
    )
    out = out.replace({"": None})
    out = out.astype(object).where(pd.notna(out), None)

    with db.connect() as conn:
        conn.execute("truncate opensky_aircraft")
        n = db.copy_rows(
            conn,
            COPY_SQL,
            out.itertuples(index=False, name=None),
            total=len(out),
            desc="opensky_aircraft",
        )
        conn.commit()
    print(f"loaded {n} aircraft into opensky_aircraft")


def main():
    parser = argparse.ArgumentParser(
        description="Load the OpenSky aircraft database CSV into opensky_aircraft.",
    )
    parser.add_argument(
        "--csv",
        default=_default_csv(),
        type=Path,
        help="OpenSky aircraft-database CSV (default: newest in data/).",
    )
    args = parser.parse_args()
    if args.csv is None:
        raise SystemExit("No OpenSky CSV found in data/ — pass --csv PATH.")
    ingest(args.csv)


if __name__ == "__main__":
    main()
