"""Backfill production JSONL logs into PostgreSQL api_calls table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from credit_scoring.storage import PostgresStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill JSONL logs into PostgreSQL")
    parser.add_argument(
        "--input",
        type=str,
        default="data/production_logs.jsonl",
        help="Path to JSONL log file",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv("DATABASE_URL", ""),
        help="PostgreSQL connection URL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    storage = PostgresStorage(args.database_url)
    if not storage.is_available():
        raise RuntimeError("PostgreSQL indisponible. Vérifie DATABASE_URL et psycopg.")

    storage.ensure_schema()

    inserted = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            storage.insert_event(event)
            inserted += 1

    print(f"Backfill terminé: {inserted} lignes insérées depuis {input_path}")


if __name__ == "__main__":
    main()
