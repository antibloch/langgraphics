"""Serve a saved, actual trace without executing an agent or calling a model."""

import argparse
import asyncio
import json
import webbrowser
from pathlib import Path

from .broadcaster import Broadcaster
from .watch import start_http_server, start_ws_server


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--ws-port", type=int, default=8775)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    if trace.get("format") != "langgraphics.trace.v1":
        parser.error("Unsupported trace format")
    broadcaster = Broadcaster(trace["graph"])
    for event in trace["events"]:
        if event["type"] == "run_start":
            event = {**event, "recorded": True}
        broadcaster.record(json.dumps(event))
    http = start_http_server("localhost", args.port)
    start_ws_server(broadcaster, "localhost", args.ws_port)
    url = f"http://localhost:{args.port}?ws_port={args.ws_port}&theme=dark"
    print(f"Recorded execution (no model calls): {url}", flush=True)
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        await asyncio.Event().wait()
    finally:
        await broadcaster.shutdown()
        await asyncio.to_thread(http.shutdown)
        http.server_close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
