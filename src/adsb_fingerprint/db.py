"""PostgreSQL access, schema initialisation, and bulk-load helper."""

import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm

from adsb_fingerprint import config


def connect():
    return psycopg.connect(
        dbname=config.DBNAME,
        row_factory=dict_row,
    )


def _statements(schema_sql):
    # Strip line comments first, then split, so a ';' inside a `-- comment`
    # can't be mistaken for a statement terminator. Assumes our own DDL:
    # no dollar-quoting and no '--' or ';' inside string literals.
    code = "\n".join(
        line.split("--", 1)[0]
        for line in schema_sql.splitlines()
    )
    return [
        statement
        for statement in (chunk.strip() for chunk in code.split(";"))
        if statement
    ]


def init_db():
    sql_dir = config.PROJECT_ROOT / "sql"
    applied = []
    with connect() as conn:
        for path in sorted(sql_dir.glob("*.sql")):
            for statement in _statements(path.read_text()):
                conn.execute(statement)
            applied.append(path.name)
        conn.commit()
    return applied


def copy_rows(conn, copy_sql, rows, total=None, desc="copy"):
    n = 0
    with conn.cursor() as cur, cur.copy(copy_sql) as copy:
        for row in tqdm(rows, total=total, unit=" rows", desc=desc):
            copy.write_row(row)
            n += 1
    return n


def main():
    applied = init_db()
    print(f"Applied {', '.join(applied)} to database {config.DBNAME!r}.")
