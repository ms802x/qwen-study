#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Start static localhost server for research pages.")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    pid_path = logs / f"research_page_{args.port}.pid"
    log_path = logs / f"research_page_{args.port}.log"

    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and process_running(pid):
            print(f"already running pid={pid}")
            print(f"url=http://{args.host}:{args.port}/research_pages/qwen_text_failure/")
            return
        pid_path.unlink(missing_ok=True)

    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            ["python", "-m", "http.server", str(args.port), "--bind", args.host],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_path.write_text(str(proc.pid))
    time.sleep(0.8)
    if not process_running(proc.pid):
        raise SystemExit(f"server exited immediately; see {log_path}")

    url = f"http://{args.host}:{args.port}/research_pages/qwen_text_failure/"
    with urllib.request.urlopen(url, timeout=10) as response:
        status = response.status
    print(f"started pid={proc.pid}")
    print(f"url={url}")
    print(f"status={status}")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
