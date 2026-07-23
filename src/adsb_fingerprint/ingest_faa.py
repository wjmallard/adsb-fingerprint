"""adsb-ingest-faa: load the FAA Releasable Aircraft registry into faa_aircraft.

MASTER.txt (registrations, keyed by MODE S CODE HEX) is joined to ACFTREF.txt
(make/model, keyed by MFR MDL CODE). The files are comma-delimited with
space-padded fields and an identical field count per record, so read_csv
parses them cleanly.
"""

import argparse
from pathlib import Path

import pandas as pd

from adsb_fingerprint import config, db

# Code tables from ardata.pdf.
TYPE_AIRCRAFT = {
    "1": "Glider",
    "2": "Balloon",
    "3": "Blimp/Dirigible",
    "4": "Fixed wing single-engine",
    "5": "Fixed wing multi-engine",
    "6": "Rotorcraft",
    "7": "Weight-shift-control",
    "8": "Powered parachute",
    "9": "Gyroplane",
    "H": "Hybrid lift",
    "O": "Other",
}

REGISTRANT_TYPE = {
    "1": "Individual",
    "2": "Partnership",
    "3": "Corporation",
    "4": "Co-owned",
    "5": "Government",
    "7": "LLC",
    "8": "Non-citizen corporation",
    "9": "Non-citizen co-owned",
}

MASTER_COLS = [
    "N-NUMBER",
    "MFR MDL CODE",
    "YEAR MFR",
    "TYPE REGISTRANT",
    "NAME",
    "CITY",
    "STATE",
    "TYPE AIRCRAFT",
    "STATUS CODE",
    "MODE S CODE HEX",
]

COPY_SQL = """
    copy faa_aircraft (
        icao,
        n_number,
        manufacturer,
        model,
        type_aircraft,
        year_mfr,
        owner,
        owner_city,
        owner_state,
        registrant,
        status_code
    )
    from stdin
"""


def ingest(faa_dir):
    faa_dir = Path(faa_dir).expanduser()

    ref = pd.read_csv(
        faa_dir / "ACFTREF.txt",
        dtype=str,
        keep_default_na=False,
        usecols=["CODE", "MFR", "MODEL"],
    )
    ref = ref.apply(lambda column: column.str.strip())
    ref_map = {
        code: (manufacturer or None, model or None)
        for code, manufacturer, model in ref.itertuples(index=False, name=None)
    }
    print(f"loaded {len(ref_map)} make/model codes from ACFTREF.txt")

    df = pd.read_csv(
        faa_dir / "MASTER.txt",
        dtype=str,
        keep_default_na=False,
        usecols=MASTER_COLS,
    )
    df = df.apply(lambda column: column.str.strip())
    df["icao"] = df["MODE S CODE HEX"].str.upper()
    df = (
        df[df["icao"].str.len() == 6]
        .drop_duplicates("icao", keep="first")
        .reset_index(drop=True)
    )

    codes = df["MFR MDL CODE"]
    out = pd.DataFrame(
        {
            "icao": df["icao"].values,
            "n_number": df["N-NUMBER"].values,
            "manufacturer": [ref_map.get(c, (None, None))[0] for c in codes],
            "model": [ref_map.get(c, (None, None))[1] for c in codes],
            "type_aircraft": [TYPE_AIRCRAFT.get(v) for v in df["TYPE AIRCRAFT"]],
            "year_mfr": [y if y.isdigit() else None for y in df["YEAR MFR"]],
            "owner": df["NAME"].values,
            "owner_city": df["CITY"].values,
            "owner_state": df["STATE"].values,
            "registrant": [REGISTRANT_TYPE.get(v) for v in df["TYPE REGISTRANT"]],
            "status_code": df["STATUS CODE"].values,
        }
    )
    out = out.replace({"": None})
    out = out.astype(object).where(pd.notna(out), None)

    with db.connect() as conn:
        conn.execute("truncate faa_aircraft")
        n = db.copy_rows(
            conn,
            COPY_SQL,
            out.itertuples(index=False, name=None),
            total=len(out),
            desc="faa_aircraft",
        )
        conn.commit()
    print(f"loaded {n} US aircraft into faa_aircraft")


def main():
    parser = argparse.ArgumentParser(
        description="Load the FAA Releasable Aircraft registry into faa_aircraft.",
    )
    parser.add_argument(
        "--dir",
        default=config.PROJECT_ROOT / "data" / "ReleasableAircraft",
        type=Path,
        help="Directory with MASTER.txt and ACFTREF.txt (default: data/ReleasableAircraft).",
    )
    args = parser.parse_args()
    ingest(args.dir)


if __name__ == "__main__":
    main()
