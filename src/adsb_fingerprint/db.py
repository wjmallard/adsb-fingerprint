"""PostgreSQL access and schema initialisation."""

import psycopg
from psycopg.rows import dict_row

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
    schema_sql = (config.PROJECT_ROOT / "sql" / "schema.sql").read_text()
    with connect() as conn:
        for statement in _statements(schema_sql):
            conn.execute(statement)
        conn.commit()


def main():
    init_db()
    print(f"Schema applied to database {config.DBNAME!r}.")
