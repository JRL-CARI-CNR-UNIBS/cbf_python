"""Export Optuna studies and trials from PostgreSQL to a standalone SQLite database.

Usage:
    python -m cbf_python.scripts.export_trials_to_sqlite [--output optuna_exported.db] [--prefix gpr_slice] [--study-name my_study]
"""

from __future__ import annotations
import argparse
import os
import sqlite3
import time
from typing import Optional
import psycopg2
import optuna


STUDY_TABLES = [
    "study_directions",
    "study_user_attributes",
    "study_system_attributes",
]

TRIAL_TABLES = [
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
    output_db: str = "optimal_controller_opt_result.db",
    prefix: Optional[str] = "params_optimal_20261101",
    study_name: Optional[str] = None,
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

    # Build study filter
    study_conds = []
    study_params = []
    if study_name:
        study_conds.append("study_name = %s")
        study_params.append(study_name)
    elif prefix:
        study_conds.append("study_name LIKE %s")
        study_params.append(f"{prefix}%")

    study_where = f"WHERE {' AND '.join(study_conds)}" if study_conds else ""

    # Check matching studies
    pg_cursor.execute(f"SELECT study_id, study_name FROM studies {study_where} ORDER BY study_id", study_params)
    matching_studies = pg_cursor.fetchall()
    if not matching_studies:
        filter_desc = f"study_name='{study_name}'" if study_name else f"prefix='{prefix}'"
        print(f"No studies found matching filter ({filter_desc}). Aborting.")
        sl_conn.close()
        pg_conn.close()
        if os.path.exists(abs_output):
            os.remove(abs_output)
        return

    study_ids = [row[0] for row in matching_studies]
    print(f"Found {len(matching_studies)} study/studies to export:")
    for s_id, s_name in matching_studies[:10]:
        print(f"  - [{s_id}] {s_name}")
    if len(matching_studies) > 10:
        print(f"  ... and {len(matching_studies) - 10} more.")

    # 1. Export studies table
    pg_cursor.execute(f"SELECT * FROM studies {study_where}", study_params)
    rows = pg_cursor.fetchall()
    col_names = [desc[0] for desc in pg_cursor.description]
    placeholders = ",".join(["?"] * len(col_names))
    cols = ",".join(col_names)
    sl_cursor.execute("DELETE FROM studies")
    sl_cursor.executemany(f"INSERT INTO studies ({cols}) VALUES ({placeholders})", rows)
    sl_conn.commit()
    print(f"  - studies: {len(rows)} rows copied")

    # 2. Export study-level tables (study_id filter)
    for tbl in STUDY_TABLES:
        t_start = time.time()
        pg_cursor.execute(
            f"SELECT * FROM {tbl} WHERE study_id = ANY(%s)",
            (study_ids,)
        )
        rows = pg_cursor.fetchall()
        if rows:
            col_names = [desc[0] for desc in pg_cursor.description]
            placeholders = ",".join(["?"] * len(col_names))
            cols = ",".join(col_names)
            sl_cursor.execute(f"DELETE FROM {tbl}")
            sl_cursor.executemany(f"INSERT INTO {tbl} ({cols}) VALUES ({placeholders})", rows)
            sl_conn.commit()
            print(f"  - {tbl}: {len(rows)} rows copied ({time.time() - t_start:.2f}s)")
        else:
            print(f"  - {tbl}: 0 rows (skipped)")

    # 3. Export trials table
    t_start = time.time()
    pg_cursor.execute(
        "SELECT * FROM trials WHERE study_id = ANY(%s) ORDER BY trial_id",
        (study_ids,)
    )
    trial_rows = pg_cursor.fetchall()
    if trial_rows:
        col_names = [desc[0] for desc in pg_cursor.description]
        placeholders = ",".join(["?"] * len(col_names))
        cols = ",".join(col_names)
        sl_cursor.execute("DELETE FROM trials")
        sl_cursor.executemany(f"INSERT INTO trials ({cols}) VALUES ({placeholders})", trial_rows)
        sl_conn.commit()
        print(f"  - trials: {len(trial_rows)} rows copied ({time.time() - t_start:.2f}s)")

        # 4. Export trial-level tables using subquery or IN condition
        for tbl in TRIAL_TABLES:
            t_start = time.time()
            pg_cursor.execute(
                f"""
                SELECT t.* FROM {tbl} t
                INNER JOIN trials tr ON t.trial_id = tr.trial_id
                WHERE tr.study_id = ANY(%s)
                """,
                (study_ids,)
            )
            rows = pg_cursor.fetchall()
            if rows:
                col_names = [desc[0] for desc in pg_cursor.description]
                placeholders = ",".join(["?"] * len(col_names))
                cols = ",".join(col_names)
                sl_cursor.execute(f"DELETE FROM {tbl}")
                sl_cursor.executemany(f"INSERT INTO {tbl} ({cols}) VALUES ({placeholders})", rows)
                sl_conn.commit()
                print(f"  - {tbl}: {len(rows)} rows copied ({time.time() - t_start:.2f}s)")
            else:
                print(f"  - {tbl}: 0 rows (skipped)")
    else:
        print("  - trials: 0 rows found for selected study/studies")

    sl_cursor.execute("PRAGMA journal_mode = DELETE")
    sl_cursor.execute("PRAGMA synchronous = NORMAL")
    sl_cursor.execute("VACUUM")
    sl_cursor.execute("ANALYZE")

    sl_conn.close()
    pg_conn.close()

    file_size_mb = os.path.getsize(abs_output) / (1024 * 1024)
    print(f"\nSuccessfully exported to SQLite: {abs_output}")
    print(f"File size: {file_size_mb:.2f} MB in {time.time() - t0:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Optuna PostgreSQL database to SQLite.")
    parser.add_argument("--output", "-o", type=str, default="optimal_controller_opt_result.db", help="Target SQLite file path")
    parser.add_argument("--prefix", "-p", type=str, default=None, help="Filter studies by name prefix (e.g. 'gpr_slice' or 'params_optimal')")
    parser.add_argument("--study-name", "-s", type=str, default=None, help="Export a single specific study by exact name")
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
        prefix=args.prefix,
        study_name=args.study_name,
    )


if __name__ == "__main__":
    main()


