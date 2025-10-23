from __future__ import annotations

from pathlib import Path
import sys
# Ensure project root is on sys.path so `app/...` imports work when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.build_index import build_and_save_index


def main() -> None:
    build_and_save_index()


if __name__ == "__main__":
    main()
