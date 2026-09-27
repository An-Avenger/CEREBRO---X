"""
scripts/run_api.py
-------------------
Launch the Cerebro X Research API server.

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000 --reload

Then visit:
    http://localhost:8000/docs   — Interactive Swagger UI
    http://localhost:8000/redoc  — ReDoc documentation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Cerebro X Research API")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on file changes (dev mode)")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    import uvicorn
    print("=" * 60)
    print("  Cerebro X Research API")
    print("=" * 60)
    print(f"  URL:  http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    uvicorn.run(
        "cerebro_x.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        app_dir=str(Path(__file__).parent.parent / "src"),
    )


if __name__ == "__main__":
    main()
