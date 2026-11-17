"""Command-line entry point for the local MVP application."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .application import LocalAppConfig, create_local_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentic-cv")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="start the local MVP API")
    serve.add_argument("--data-dir", type=Path, default=Path(".agentic-cv"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run(create_local_app(LocalAppConfig(data_dir=args.data_dir)), host=args.host, port=args.port)