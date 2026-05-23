#!/usr/bin/env python3
"""
VGA SVI Persistent Inference Server — runs inside svi_wan22 conda env.

Loads the Wan2.2 FP8 pipeline ONCE on startup, then serves inference requests
via HTTP. This eliminates the 3-5 minute cold model loading overhead that occurs
when SVIWrapper spawns a new subprocess per segment.

Architecture
------------
  SVIWrapper (vga env) ──POST /infer──► this server (svi_wan22 env)
                        ◄── JSON result ──

  SVIWrapper starts this server on the first segment call (detached background
  process), waits up to 600 s for /health to report "ready", then sends all
  subsequent segment requests to the already-warm pipeline.

Usage (auto-managed by SVIWrapper; manual start for debugging):
    /workspace/miniconda3/envs/svi_wan22/bin/python scripts/vga_svi_server.py \
        --lora-path-high /workspace/loras/svi/SVI_...high....safetensors \
        --lora-path-low  /workspace/loras/svi/SVI_...low....safetensors \
        --port 8765

Endpoints
---------
  GET  /health  → {"status": "ready"|"loading"}
  POST /infer   → JSON config body → {"status": "ok", "output_path": "..."}
                                   → {"status": "error", "error": "..."}
  GET  /shutdown → graceful shutdown (used by SVIWrapper when pod stops)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Resolve script directory so we can import from the same scripts/ folder.
SCRIPT_DIR = str(Path(__file__).parent.resolve())
SVI_REPO = "/workspace/Stable-Video-Infinity"
for _p in (SVI_REPO, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402 — after sys.path setup

# Shared state: set by the loader thread once the pipeline is ready.
_pipeline = None
_pipeline_lock = threading.Lock()
_server_ready = threading.Event()
_shutdown_event = threading.Event()


class SVIRequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — no external dependencies required."""

    def log_message(self, fmt: str, *args) -> None:  # suppress default logging
        print(f"[SVI-SERVER] {fmt % args}", flush=True)

    # ------------------------------------------------------------------ GET --
    def do_GET(self) -> None:
        if self.path == "/health":
            status = "ready" if _server_ready.is_set() else "loading"
            body = json.dumps({"status": status}).encode()
            self._send_json(200, body)

        elif self.path == "/shutdown":
            body = json.dumps({"status": "shutting_down"}).encode()
            self._send_json(200, body)
            print("[SVI-SERVER] Shutdown requested.", flush=True)
            _shutdown_event.set()
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        else:
            self._send_json(404, json.dumps({"error": "not found"}).encode())

    # ----------------------------------------------------------------- POST --
    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json(404, json.dumps({"error": "not found"}).encode())
            return

        if not _server_ready.is_set():
            body = json.dumps({"status": "error", "error": "pipeline not ready yet"}).encode()
            self._send_json(503, body)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        config = json.loads(raw)

        print(
            f"[SVI-SERVER] /infer: segment → {Path(config.get('output_path', '?')).name}",
            flush=True,
        )

        # Import run_inference from the inference bridge (same scripts/ dir).
        from vga_svi_inference import run_inference  # noqa: PLC0415

        with _pipeline_lock:
            result = run_inference(_pipeline, config)

        code = 200 if result.get("status") == "ok" else 500
        self._send_json(code, json.dumps(result).encode())

    # ------------------------------------------------------------ helpers ---
    def _send_json(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _load_pipeline(lora_path_high: str, lora_path_low: str, device: str) -> None:
    """Load the pipeline in a background thread and signal readiness."""
    global _pipeline

    from vga_svi_inference import build_pipeline, verify_and_configure_attention  # noqa: PLC0415

    print("[SVI-SERVER] Verifying FlashAttention-2 and SDPA backend...", flush=True)
    verify_and_configure_attention()

    print("[SVI-SERVER] Loading pipeline (Wan2.2 FP8 + T5 + VAE)...", flush=True)
    print("[SVI-SERVER] This takes 3-5 minutes. Server will accept requests once ready.",
          flush=True)

    try:
        _pipeline = build_pipeline(lora_path_high, lora_path_low, device)
        _server_ready.set()
        print("[SVI-SERVER] Pipeline ready — accepting inference requests.", flush=True)
    except Exception as exc:
        import traceback
        print(f"[SVI-SERVER] FATAL: pipeline load failed: {exc}", flush=True)
        traceback.print_exc()
        # Don't set _server_ready; health endpoint will keep returning "loading"
        # until the process exits or the pod is recycled.
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="VGA SVI Persistent Inference Server")
    parser.add_argument("--lora-path-high", required=True)
    parser.add_argument("--lora-path-low", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[SVI-SERVER] Starting on localhost:{args.port}", flush=True)

    # Start pipeline loader in background so the HTTP server can respond to /health
    # immediately (returning "loading") while the model is being read from disk.
    loader = threading.Thread(
        target=_load_pipeline,
        args=(args.lora_path_high, args.lora_path_low, args.device),
        daemon=True,
    )
    loader.start()

    server = HTTPServer(("localhost", args.port), SVIRequestHandler)
    print(f"[SVI-SERVER] HTTP server listening on localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
