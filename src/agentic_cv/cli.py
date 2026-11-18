"""Command-line entry point for the local MVP application."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .application import LocalAppConfig, create_local_app
from .doctor import doctor_exit_code, format_report, report_as_json, run_doctor
from .observability import configure_logging
from .worker_runtime import create_export_worker


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentic-cv")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="start the local MVP API")
    serve.add_argument("--data-dir", type=Path, default=Path(".agentic-cv"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    worker = subcommands.add_parser("worker", help="run a local durable export worker")
    worker.add_argument("--data-dir", type=Path, default=Path(".agentic-cv"))
    worker.add_argument("--worker-id", default="export-worker")
    worker.add_argument("--poll-interval", type=float, default=1.0)
    doctor = subcommands.add_parser("doctor", help="check the local environment")
    doctor.add_argument("--data-dir", type=Path, default=Path(".agentic-cv"))
    doctor.add_argument("--no-network", action="store_true", help="skip Hugging Face access checks")
    doctor.add_argument("--json", action="store_true", help="print machine-readable output")
    doctor.add_argument("--ollama-url", default="http://localhost:11434")
    doctor.add_argument("--ollama-model", default="qwen3-vl:4b")
    doctor.add_argument("--host", default="127.0.0.1")
    doctor.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "doctor":
        checks = run_doctor(
            data_dir=args.data_dir,
            network=not args.no_network,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            host=args.host,
            port=args.port,
        )
        print(report_as_json(checks) if args.json else format_report(checks))
        raise SystemExit(doctor_exit_code(checks))
    if args.command == "serve":
        configure_logging()
        uvicorn.run(create_local_app(LocalAppConfig.from_environment(args.data_dir)), host=args.host, port=args.port)
    elif args.command == "worker":
        configure_logging()
        runtime = create_export_worker(args.data_dir)
        try:
            runtime.worker.run_forever(
                worker_id=args.worker_id, poll_interval=args.poll_interval
            )
        except KeyboardInterrupt:
            pass
        finally:
            runtime.close()