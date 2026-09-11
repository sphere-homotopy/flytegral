"""Local-only HTTP bridge between the Flytegral browser UI and MaleCNS runtime."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from brain_runtime.runtime import describe_runtime, runtime_from_paths

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOOMFLY = ROOT / ".vendor" / "doomfly"
DEFAULT_GRAPH = DEFAULT_DOOMFLY / "outputs" / "doom" / "malecns_v1" / "graph.npz"
DEFAULT_READOUT = ROOT / "brain_runtime" / "readout.npz"


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FlytegralMaleCNS/0.1"

        def _headers(self, status=200):
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("access-control-allow-origin", "*")
            self.send_header("access-control-allow-headers", "content-type")
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.end_headers()

        def _json(self, payload, status=200):
            self._headers(status)
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def do_OPTIONS(self):
            self._headers(204)

        def do_GET(self):
            if self.path == "/health":
                self._json({"ok": True, "runtime": json.loads(describe_runtime(runtime))})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/estimate":
                self._json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                if body.get("schema") != 1:
                    raise ValueError("unsupported request schema")
                result = runtime.estimate(body["stimulus"])
                self._json(result)
            except (KeyError, TypeError, ValueError) as error:
                self._json({"error": str(error)}, 400)
            except Exception as error:
                self._json({"error": f"runtime failure: {error}"}, 500)

        def log_message(self, fmt, *args):
            print(f"[malecns] {self.address_string()} {fmt % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doomfly-root", type=Path, default=DEFAULT_DOOMFLY)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--readout", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()

    runtime = runtime_from_paths(args.doomfly_root, args.graph, args.readout)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runtime))
    print(f"Flytegral MaleCNS bridge http://127.0.0.1:{args.port} {describe_runtime(runtime)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
