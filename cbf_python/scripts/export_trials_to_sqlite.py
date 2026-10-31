"""Export Optuna studies and trials from PostgreSQL to a standalone SQLite database.

Usage:
    python -m cbf_python.scripts.export_trials_to_sqlite [--output optuna_gpr_grid.db] [--pg-url postgresql://...]
"""

from __future__ import annotations
import argparse
import os
import sqlite3
import time
import psycopg2
import optuna


TABLES = [
    "studies",
    "study_directions",
    "study_user_attributes",
    "study_system_attributes",
    "trials",
    "trial_user_attributes",
    "trial_system_attributes",
    "trial_params",
    "trial_values",
    "trial_intermediate_values",
    "trial_heartbeats",
]


def export_to_sqlite(
    pg_host: str = "127.0.0.1",
    pg_port: int = 5432,
    pg_user: str = "optuna",
    pg_password: str = "optuna_pw",
    pg_db: str = "optuna_db",
    output_db: str = "optuna_gpr_grid.db",
) -> None:
    t0 = time.time()
    abs_output = os.path.abspath(output_db)
    if os.path.exists(abs_output):
        print(f"Removing existing file: {abs_output}")
        os.remove(abs_output)

    print(f"Initializing Optuna schema in SQLite: {abs_output}")
    # Initialize schema using Optuna RDBStorage
    _ = optuna.storages.RDBStorage(f"sqlite:///{abs_output}")

    print(f"Connecting to PostgreSQL ({pg_host}:{pg_port}/{pg_db})...")
    pg_conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        dbname=pg_db,
    )
    sl_conn = sqlite3.connect(abs_output)

    sl_cursor = sl_conn.cursor()
    pg_cursor = pg_conn.cursor()

    # Fast bulk insertion settings
    sl_cursor.execute("PRAGMA synchronous = OFF")
    sl_cursor.execute("PRAGMA journal_mode = MEMORY")

    for tbl in TABLES:
        t_start = time.time()
        pg_cursor.execute(f"SELECT * FROM {tbl}")
        rows = pg_cursor.fetchall()
        if not rows:
            print(f"  - {tbl}: 0 rows (skipped)")
            continue

        col_names = [desc[0] for desc in pg_cursor.description]
        placeholders = ",".join(["?"] * len(col_names))
        cols = ",".join(col_names)

        sl_cursor.execute(f"DELETE FROM {tbl}")
        sl_cursor.executemany(f"INSERT INTO {tbl} ({cols}) VALUES ({placeholders})", rows)
        sl_conn.commit()
        print(f"  - {tbl}: {len(rows)} rows copied ({time.time() - t_start:.2f}s)")

    sl_cursor.execute("PRAGMA journal_mode = DELETE")
    sl_cursor.execute("PRAGMA synchronous = NORMAL")
    sl_cursor.execute("VACUUM")
    sl_cursor.execute("ANALYZE")

    sl_conn.close()
    pg_conn.close()

    file_size_mb = os.path.getsize(abs_output) / (1024 * 1024)
    print(f"\nSuccessfully exported all trials to SQLite: {abs_output}")
    print(f"File size: {file_size_mb:.1f} MB in {time.time() - t0:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Optuna PostgreSQL database to SQLite.")
    parser.add_argument("--output", "-o", type=str, default="optuna_gpr_grid.db", help="Target SQLite file path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--user", type=str, default="optuna", help="PostgreSQL user")
    parser.add_argument("--password", type=str, default="optuna_pw", help="PostgreSQL password")
    parser.add_argument("--db", type=str, default="optuna_db", help="PostgreSQL database name")

    args = parser.parse_args()
    export_to_sqlite(
        pg_host=args.host,
        pg_port=args.port,
        pg_user=args.user,
        pg_password=args.password,
        pg_db=args.db,
        output_db=args.output,
    )


if __name__ == "__main__":
    main()

