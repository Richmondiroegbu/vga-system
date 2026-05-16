#!/usr/bin/env python3
"""
session_controller.py — VGA v17.2 / RunPod Session Controller v7.1
Spec:  Session Controller Upgrade Specification v7.0 (Parts 1–3) + v7.1 Patch
Arch:  Controller → Bootstrap (v6.5) → VGA Runtime (v17.2) → execute_stage()

Responsibilities (STRICT — v7.1):
  ✔  Start / stop existing pod via RunPod GraphQL (Resume / Stop only)
  ✔  SSH connection + service rehydration after every pod resume
  ✔  Auto-detect bootstrap completion via /workspace/state/run_manifest.json
  ✔  Invoke bootstrap via single entrypoint: python3 /workspace/bootstrap_pipeline.py
  ✔  Explicitly inject required env vars into bootstrap invocation (v7.1 FIX)
  ✔  Validate manifest content (not just existence) before skipping bootstrap (v7.1 FIX)
  ✔  Start VGA API (uvicorn) and verify /health with retry-safe restart loop (v7.1 FIX)
  ✔  Drive pipeline exclusively through HTTP endpoints
  ✔  Handle HRG pause → human review → pod resume → API restart → /system/resume
  ✔  Structured JSON event logging + run correlation ID
  ✔  Exponential-backoff retry for transient failures
  ✔  Bootstrap timeout guard (v7.1 FIX)

Absolute constraints (enforced):
  ❌  Does NOT create pods
  ❌  Does NOT terminate pods
  ❌  Does NOT install any dependencies
  ❌  Does NOT write /workspace/.env_vga (bootstrap owns that)
  ❌  Does NOT download models or manage asset registry
  ❌  Does NOT call VGA code directly — API only
  ❌  Does NOT source .env_vga before bootstrap completes (v7.1 FIX)

v7.1 Changes (from assessment patch):
  ✅  ISSUE 1 FIX — Explicit env injection into bootstrap invocation (no implicit SSH env)
  ✅  ISSUE 2 FIX — Manifest validation checks content, not just file existence
  ✅  ISSUE 3 FIX — Controller never sources .env_vga before bootstrap; env passed explicitly
  ✅  ISSUE 4 FIX — API startup is retry-safe with per-attempt restart + backoff
  ✅  OPTIONAL  — Bootstrap timeout guard (1-hour hard limit)

v7.1.1 Fixes (SVI LoRA filename alignment):
  ✅  build_bootstrap_env_exports() updated: SVI_HIGH_NOISE_FILE and SVI_LOW_NOISE_FILE
      now inject the CORRECT verified filenames with version-2.0/ prefix and full
      model name (aligned with vita-video-gen/svi-model on HuggingFace).
  ✅  CLI docstring updated to document correct SVI env var values.
  ✅  Pre-flight check includes SVI env var defaults documentation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
import uuid
from typing import Optional

import paramiko
import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# RUN CORRELATION ID
# Generated once per controller invocation; attached to all log lines and
# outgoing API headers so logs across layers can be correlated.
# ─────────────────────────────────────────────────────────────────────────────
RUN_ID = uuid.uuid4().hex


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
class _JsonFormatter(logging.Formatter):
    """Emit structured JSON lines for machine-parseable observability."""
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level":   record.levelname,
            "run_id":  RUN_ID,
            "event":   record.getMessage(),
        }
        # Allow callers to inject extra keys via `extra={"event": ...}`
        for key in ("event_type", "checkpoint", "state", "stage"):
            if hasattr(record, key):
                base[key] = getattr(record, key)
        return json.dumps(base)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("session_controller")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    # Console — human-readable
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [SessionController] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(sh)

    # File — structured JSON
    os.makedirs(os.path.dirname("session_controller.log"), exist_ok=True)
    fh = logging.FileHandler("session_controller.log", mode="a")
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    return logger


log = _build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (read from .env or shell — controller never writes env)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

RUNPOD_API_KEY   = os.getenv("RUNPOD_API_KEY",           "")
POD_ID           = os.getenv("RUNPOD_POD_ID",            "")
POD_IP           = os.getenv("POD_IP",                   "")
POD_PORT         = int(os.getenv("POD_PORT",             "22"))
SSH_USER         = os.getenv("SSH_USER",                 "root")
SSH_KEY_PATH     = os.path.expanduser(os.getenv("SSH_PRIVATE_KEY_PATH", "~/.ssh/id_rsa"))
POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL",        "10"))
API_BASE         = f"http://{POD_IP}:8000"

# Retry / backoff constants (spec Part 3 §6.2)
MAX_RETRIES      = 3
BACKOFF          = [5, 15, 45]   # seconds

# Minimum model count used for optional deep bootstrap check (spec Part 3 §3.2)
MIN_MODEL_COUNT  = int(os.getenv("MIN_MODEL_COUNT",      "5"))

# Bootstrap timeout guard (v7.1 — 1 hour hard limit)
BOOTSTRAP_TIMEOUT = 60 * 60  # seconds

# Remote paths — controller only reads / checks these, never creates them
BOOTSTRAP_ENTRYPOINT = "/workspace/bootstrap_pipeline.py"
RUN_MANIFEST_PATH    = "/workspace/state/run_manifest.json"
MODELS_DIR           = "/workspace/models"
ENV_FILE             = "/workspace/.env_vga"
API_LOG              = "/workspace/logs/api.log"

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT: fail fast on missing critical config
# ─────────────────────────────────────────────────────────────────────────────
def _require_env() -> None:
    """Raise immediately if any critical variable is unset."""
    missing = [
        var for var, val in [
            ("RUNPOD_API_KEY",       RUNPOD_API_KEY),
            ("RUNPOD_POD_ID",        POD_ID),
            ("POD_IP",               POD_IP),
            ("SSH_PRIVATE_KEY_PATH", SSH_KEY_PATH),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}\n"
            "Set them in .env or the shell before running session_controller.py"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RUNPOD CONTROL  (Resume / Stop only — NO create / terminate)
# ─────────────────────────────────────────────────────────────────────────────
RUNPOD_GQL = "https://api.runpod.io/graphql"


def _runpod_action(action: str) -> None:
    """
    Issue a podResume or podStop GraphQL mutation.
    `action` must be exactly 'Resume' or 'Stop'.
    """
    assert action in ("Stop", "Resume"), f"Disallowed pod action: {action!r}"
    query = textwrap.dedent(f"""
        mutation {{
            pod{action}(input: {{ podId: "{POD_ID}" }}) {{
                id
                desiredStatus
            }}
        }}
    """)
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type":  "application/json",
    }
    resp = requests.post(
        RUNPOD_GQL, json={"query": query}, headers=headers, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"RunPod API error: {data['errors']}")
    log.info(f"[runpod] pod{action} accepted — {data}")


def stop_pod() -> None:
    log.info("[runpod] Requesting pod STOP …")
    _runpod_action("Stop")


def start_pod() -> None:
    log.info("[runpod] Requesting pod RESUME …")
    _runpod_action("Resume")


# ─────────────────────────────────────────────────────────────────────────────
# SSH
# ─────────────────────────────────────────────────────────────────────────────
def ssh_connect(retries: int = MAX_RETRIES) -> paramiko.SSHClient:
    """
    Open an SSH connection with exponential-backoff retry.
    Pod may need time to boot after a Resume call.
    """
    # Auto-detect key type from file to support both RSA and Ed25519 keys.
    # RunPod pods may be configured with either key type.
    try:
        key = paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
    except paramiko.ssh_exception.SSHException:
        key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY_PATH)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            client.connect(
                POD_IP, port=POD_PORT, username=SSH_USER,
                pkey=key, timeout=30,
            )
            log.info(f"[ssh] Connected to {SSH_USER}@{POD_IP}:{POD_PORT}")
            return client
        except Exception as exc:
            last_exc = exc
            delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            log.warning(
                f"[ssh] Attempt {attempt}/{retries} failed: {exc} "
                f"— retrying in {delay}s"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"[ssh] Could not connect after {retries} attempts. "
        f"Last error: {last_exc}"
    )


def run_remote(
    client: paramiko.SSHClient,
    command: str,
    raise_on_error: bool = True,
) -> tuple[str, str, int]:
    """
    Execute a command over SSH.
    Returns (stdout_text, stderr_text, exit_code).
    Streams output to local log in real time.
    """
    log.info(f"[ssh] $ {command[:200]}")
    transport = client.get_transport()
    channel = transport.open_session()
    channel.set_combine_stderr(False)
    channel.exec_command(command)

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    while True:
        if channel.recv_ready():
            chunk = channel.recv(4096).decode("utf-8", errors="replace")
            stdout_chunks.append(chunk)
            for line in chunk.splitlines():
                log.info(f"[remote] {line}")
        if channel.recv_stderr_ready():
            chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
            stderr_chunks.append(chunk)
            for line in chunk.splitlines():
                log.warning(f"[remote:err] {line}")
        if channel.exit_status_ready():
            # drain remaining buffers
            while channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                stdout_chunks.append(chunk)
            while channel.recv_stderr_ready():
                chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                stderr_chunks.append(chunk)
            break
        time.sleep(0.2)

    exit_code    = channel.recv_exit_status()
    stdout_text  = "".join(stdout_chunks)
    stderr_text  = "".join(stderr_chunks)

    if raise_on_error and exit_code != 0:
        raise RuntimeError(
            f"[ssh] Command exited {exit_code}:\n"
            f"  cmd={command[:200]}\n"
            f"  stderr={stderr_text[:500]}"
        )
    return stdout_text, stderr_text, exit_code


def remote_exists(client: paramiko.SSHClient, path: str) -> bool:
    """Return True if `path` exists on the remote pod."""
    out, _, _ = run_remote(
        client, f'test -e "{path}" && echo 1 || echo 0',
        raise_on_error=False,
    )
    return out.strip() == "1"


def remote_file_count(client: paramiko.SSHClient, path: str) -> int:
    """Return the number of regular files beneath `path` on the remote pod."""
    out, _, rc = run_remote(
        client,
        f'find "{path}" -type f 2>/dev/null | wc -l',
        raise_on_error=False,
    )
    try:
        return int(out.strip())
    except ValueError:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# v7.1 FIX #1 — EXPLICIT ENVIRONMENT INJECTION
# Controller NEVER relies on SSH inheriting env vars implicitly.
# Paramiko exec_command() does NOT guarantee environment inheritance.
# All required vars are explicitly injected into the bootstrap shell command.
# ─────────────────────────────────────────────────────────────────────────────
def build_bootstrap_env_exports() -> str:
    """
    Build an explicit 'export KEY="VALUE"' block for all variables that
    bootstrap_pipeline.py needs at runtime.

    v7.1.1 NOTE — SVI LoRA filenames:
      SVI_HIGH_NOISE_FILE and SVI_LOW_NOISE_FILE must match the ACTUAL filenames
      in vita-video-gen/svi-model on HuggingFace. They include the version-2.0/
      subfolder prefix and the full model identifier in the filename.

      Correct values (set in .env or override here):
        SVI_HIGH_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
        SVI_LOW_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors

      If not set in the environment, bootstrap_pipeline.py uses these same defaults.

    Returns a shell snippet that can be prepended to the bootstrap command:
        export K1="V1" && export K2="V2" && ...
    """
    # These defaults match bootstrap_pipeline.py defaults exactly.
    # If the operator has set different values in .env, those take precedence.
    SVI_HIGH_NOISE_DEFAULT = (
        "version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors"
    )
    SVI_LOW_NOISE_DEFAULT = (
        "version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors"
    )

    keys_with_defaults = [
        ("HUGGING_FACE_HUB_TOKEN",  os.getenv("HUGGING_FACE_HUB_TOKEN", "")),
        ("LORA_IDENTITY_REPO",      os.getenv("LORA_IDENTITY_REPO", "")),
        ("LORA_STYLE_REPO",         os.getenv("LORA_STYLE_REPO", "")),
        ("SVI_LORA_REPO",           os.getenv("SVI_LORA_REPO", "vita-video-gen/svi-model")),
        ("SVI_HIGH_NOISE_FILE",     os.getenv("SVI_HIGH_NOISE_FILE", SVI_HIGH_NOISE_DEFAULT)),
        ("SVI_LOW_NOISE_FILE",      os.getenv("SVI_LOW_NOISE_FILE",  SVI_LOW_NOISE_DEFAULT)),
        ("DOWNLOAD_MIN_FREE_GB",    os.getenv("DOWNLOAD_MIN_FREE_GB", "15")),
    ]

    exports = []
    for k, v in keys_with_defaults:
        if v:
            # Escape any embedded double-quotes in the value for safety
            v_escaped = v.replace('"', '\\"')
            exports.append(f'export {k}="{v_escaped}"')
        else:
            if k == "HUGGING_FACE_HUB_TOKEN":
                log.warning(
                    "[bootstrap] HUGGING_FACE_HUB_TOKEN is empty — "
                    "gated model downloads (LatentSync, SVI, FLUX.2, etc.) will fail. "
                    "Set HUGGING_FACE_HUB_TOKEN in your .env before running."
                )

    if not exports:
        log.warning(
            "[bootstrap] No env vars found for injection — "
            "bootstrap may fail on gated model downloads."
        )
        return "true"  # shell no-op — still valid syntax

    return " && ".join(exports)


# ─────────────────────────────────────────────────────────────────────────────
# v7.1 FIX #2 — MANIFEST CONTENT VALIDATION
# Manifest existence alone is insufficient — it may exist from a partial run.
# We validate that the manifest carries the required fields and passed status.
# ─────────────────────────────────────────────────────────────────────────────
def read_remote_json(client: paramiko.SSHClient, path: str) -> Optional[dict]:
    """
    Read and parse a JSON file from the remote pod.
    Returns the parsed dict, or None if the file is missing or malformed.
    """
    out, _, rc = run_remote(client, f'cat "{path}"', raise_on_error=False)
    if rc != 0 or not out.strip():
        return None
    try:
        return json.loads(out)
    except Exception as exc:
        log.warning(f"[bootstrap] Failed to parse remote JSON at {path}: {exc}")
        return None


def manifest_valid(client: paramiko.SSHClient) -> bool:
    """
    Validate manifest content — not just existence.

    A valid manifest must have:
      - validation == "passed"   (bootstrap ran to completion without errors)
      - "models" key present     (at least one model was processed)
      - "torch_version" present  (CUDA stack was installed and validated)

    This prevents treating a manifest from a crashed/partial bootstrap as valid.
    """
    data = read_remote_json(client, RUN_MANIFEST_PATH)
    if not data:
        log.info("[bootstrap] Manifest missing or unreadable.")
        return False

    validation_ok  = data.get("validation") == "passed"
    has_models     = bool(data.get("models"))
    has_torch      = bool(data.get("torch_version"))

    if not validation_ok:
        log.warning(
            f"[bootstrap] Manifest validation field is '{data.get('validation')}' "
            "— expected 'passed'."
        )
    if not has_models:
        log.warning("[bootstrap] Manifest 'models' list is empty or missing.")
    if not has_torch:
        log.warning("[bootstrap] Manifest 'torch_version' field is missing.")

    # Log SVI LoRA filenames from manifest for auditability
    if data.get("svi_high_noise_file"):
        log.info(f"[bootstrap] SVI high noise file: {data['svi_high_noise_file']}")
    if data.get("svi_low_noise_file"):
        log.info(f"[bootstrap] SVI low noise file:  {data['svi_low_noise_file']}")

    return validation_ok and has_models and has_torch


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP DETECTION  (spec Part 3 §3 + v7.1 upgrades)
# Controller NEVER re-implements bootstrap logic — it only checks the manifest.
# ─────────────────────────────────────────────────────────────────────────────
def models_ok(client: paramiko.SSHClient) -> bool:
    """
    Deep check: ensure /workspace/models/ exists and contains at least
    MIN_MODEL_COUNT files.  Secondary signal when manifest is present
    but models directory looks empty (e.g. after a failed partial run).
    """
    if not remote_exists(client, MODELS_DIR):
        return False
    count = remote_file_count(client, MODELS_DIR)
    return count >= MIN_MODEL_COUNT


def should_run_bootstrap(client: paramiko.SSHClient) -> bool:
    """
    v7.1 — Unified bootstrap decision gate.

    Checks manifest existence, content validity, AND model completeness.
    Any failure in the chain forces a full re-bootstrap.

    Decision logic:
      1. If manifest file does not exist → bootstrap required
      2. If manifest exists but content is invalid (partial/crashed run) → re-bootstrap
      3. If manifest is valid but models are incomplete → re-bootstrap
      4. All checks pass → skip bootstrap
    """
    if not remote_exists(client, RUN_MANIFEST_PATH):
        log.info("[bootstrap] Manifest file not found — bootstrap required.")
        return True

    if not manifest_valid(client):
        log.warning("[bootstrap] Manifest invalid or incomplete — forcing re-bootstrap.")
        return True

    if not models_ok(client):
        log.warning(
            f"[bootstrap] Models directory has fewer than {MIN_MODEL_COUNT} files "
            "— forcing re-bootstrap."
        )
        return True

    log.info("[bootstrap] Valid manifest + model check passed — bootstrap not required.")
    return False


def ensure_bootstrap(client: paramiko.SSHClient) -> None:
    """
    v7.1 — Invoke the bootstrap entrypoint with explicit env injection.

    Changes from v7.0:
      - Uses should_run_bootstrap() (manifest content check, not just existence)
      - Injects all required env vars explicitly via build_bootstrap_env_exports()
      - Does NOT source .env_vga before bootstrap (it doesn't exist on first run)
      - Enforces a hard 1-hour timeout guard

    v7.1.1: SVI LoRA env vars injected with correct verified filenames.
    """
    if not should_run_bootstrap(client):
        log.info("[bootstrap] Bootstrap already complete — skipping.")
        return

    log.info(
        "[bootstrap] Running bootstrap entrypoint: "
        f"python3 {BOOTSTRAP_ENTRYPOINT}"
    )

    # v7.1 FIX #1 & #3: Inject env explicitly. Do NOT source .env_vga here —
    # that file is written BY bootstrap, so it cannot exist before it runs.
    env_block = build_bootstrap_env_exports()

    cmd = (
        f"{env_block} && "
        f"cd /workspace && "
        f"python3 {BOOTSTRAP_ENTRYPOINT} "
        f"2>&1 | tee -a /workspace/logs/bootstrap.log"
    )

    # v7.1 OPTIONAL: Bootstrap timeout guard
    start_time = time.time()
    log.info(
        f"[bootstrap] Timeout guard active — max {BOOTSTRAP_TIMEOUT // 60} minutes."
    )

    import threading

    result: dict = {"exit_code": None, "exc": None}

    def _run() -> None:
        try:
            _, _, ec = run_remote(client, cmd, raise_on_error=False)
            result["exit_code"] = ec
        except Exception as exc:
            result["exc"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=BOOTSTRAP_TIMEOUT)

    if thread.is_alive():
        log.error(
            f"[bootstrap] TIMEOUT — bootstrap exceeded {BOOTSTRAP_TIMEOUT // 60} minutes. "
            "Stopping pod."
        )
        stop_pod()
        raise RuntimeError(
            f"[bootstrap] bootstrap_pipeline.py timed out after "
            f"{BOOTSTRAP_TIMEOUT // 60} minutes."
        )

    if result["exc"] is not None:
        stop_pod()
        raise RuntimeError(
            f"[bootstrap] bootstrap_pipeline.py raised an exception: {result['exc']}"
        )

    exit_code = result["exit_code"]
    if exit_code != 0:
        stop_pod()
        raise RuntimeError(
            f"[bootstrap] bootstrap_pipeline.py exited with code {exit_code}. "
            "Check /workspace/logs/bootstrap.log on the pod."
        )

    elapsed = time.time() - start_time
    log.info(
        f"[bootstrap] Bootstrap completed successfully in {elapsed / 60:.1f} minutes."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE MANAGEMENT  (rehydration after every pod resume — spec Part 3 §4)
# ─────────────────────────────────────────────────────────────────────────────
def start_api_server(client: paramiko.SSHClient) -> None:
    """
    Kill any stale uvicorn process, then launch a fresh instance.
    Sources /workspace/.env_vga — this is SAFE here because start_api_server
    is only called AFTER ensure_bootstrap() has confirmed bootstrap completed
    successfully, meaning .env_vga is guaranteed to exist.
    Logs are written to /workspace/logs/api.log.
    """
    log.info("[api] Starting VGA API server (uvicorn) …")
    run_remote(client, "pkill -f uvicorn || true", raise_on_error=False)
    time.sleep(2)
    # .env_vga is sourced HERE (post-bootstrap) — never before bootstrap
    start_cmd = (
        f"source {ENV_FILE} && "
        "nohup uvicorn vga.api.main:app --host 0.0.0.0 --port 8000 "
        f"> {API_LOG} 2>&1 &"
    )
    run_remote(client, start_cmd)
    log.info("[api] uvicorn launch command sent.")


def wait_for_api(timeout: int = 300) -> None:
    """
    Block until GET /health returns HTTP 200 or `timeout` seconds elapse.
    Retries every 3 seconds with the run correlation ID in request headers.

    Note: This function checks health only; API restart on failure is handled
    by the caller (rehydrate_services) which owns the retry+restart loop.
    """
    log.info("[api] Waiting for VGA API to become healthy …")
    headers = {"X-Run-Id": RUN_ID}
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            r = requests.get(
                f"{API_BASE}/health", timeout=3, headers=headers
            )
            if r.status_code == 200:
                log.info(f"[api] API healthy: {r.json()}")
                return
        except requests.RequestException:
            pass
        time.sleep(3)

    raise RuntimeError(
        f"[api] API did not become healthy within {timeout}s. "
        f"Check {API_LOG} on the pod."
    )


def rehydrate_services(client: paramiko.SSHClient) -> None:
    """
    v7.1 — Full service rehydration with retry-safe API startup loop.

    Changes from v7.0:
      - Each failed health-check attempt triggers a fresh uvicorn restart
        before the next attempt (not just a passive wait)
      - After MAX_RETRIES failed attempts, pod is stopped and exception raised
      - Preserves structured logging per attempt

    Spec Part 3 §4: reconnect SSH → restart API → verify /health.
    SSH reconnection is the caller's responsibility (client already provided).
    """
    log.info("[session] Rehydrating services after pod resume …")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start_api_server(client)
            wait_for_api()
            log.info(f"[session] API ready (attempt {attempt}/{MAX_RETRIES}).")
            return

        except Exception as exc:
            log.warning(
                f"[api] Rehydration attempt {attempt}/{MAX_RETRIES} failed: {exc}"
            )

            if attempt == MAX_RETRIES:
                log.error(
                    "[api] API failed to start after all retry attempts — stopping pod."
                )
                stop_pod()
                raise RuntimeError(
                    f"[api] VGA API failed to become healthy after "
                    f"{MAX_RETRIES} attempts. Check {API_LOG} on the pod."
                ) from exc

            delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            log.info(f"[api] Retrying rehydration in {delay}s …")
            time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# VGA PIPELINE CONTROL  (HTTP-only — spec Part 3 §1.2)
# ─────────────────────────────────────────────────────────────────────────────
def _api_post(path: str, description: str) -> requests.Response:
    """POST to a VGA API endpoint with run-correlation header."""
    headers = {"X-Run-Id": RUN_ID, "Content-Type": "application/json"}
    log.info(f"[pipeline] POST {path} ({description})")
    r = requests.post(f"{API_BASE}{path}", headers=headers, timeout=30)
    r.raise_for_status()
    log.info(f"[pipeline] {path} → {r.status_code}: {r.text[:200]}")
    return r


def start_pipeline() -> None:
    _api_post("/system/start", "initiate VGA pipeline")


def resume_pipeline() -> None:
    _api_post("/system/resume", "resume after HRG")


def get_pipeline_status() -> Optional[dict]:
    """
    GET /system/status.  Returns the parsed JSON body or None on error.
    Expected schema: { state, stage, job_id, checkpoint? }
    """
    headers = {"X-Run-Id": RUN_ID}
    try:
        r = requests.get(
            f"{API_BASE}/system/status", headers=headers, timeout=10
        )
        if r.status_code == 200:
            return r.json()
        log.warning(f"[pipeline] /system/status returned {r.status_code}")
        return None
    except requests.RequestException as exc:
        log.warning(f"[pipeline] /system/status error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN-IN-THE-LOOP (HRG)  (spec Part 3 §5)
# ─────────────────────────────────────────────────────────────────────────────
def wait_for_human_approval(status: dict) -> None:
    """
    Block until the operator confirms the HRG checkpoint.
    Default mechanism: blocking ENTER prompt (CLI).
    File-flag alternative: create /workspace/state/hrg_approved.flag on the
    local machine and the loop will detect it automatically.
    """
    stage      = status.get("stage",      "UNKNOWN")
    checkpoint = status.get("checkpoint", "UNKNOWN")

    # Emit structured event
    log.info(
        json.dumps({
            "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event":      "HRG_PAUSE",
            "run_id":     RUN_ID,
            "stage":      stage,
            "checkpoint": checkpoint,
        })
    )

    print("\n" + "=" * 70)
    print("  ⚠️  HRG CHECKPOINT REACHED — HUMAN REVIEW REQUIRED")
    print("=" * 70)
    print(f"  Stage          : {stage}")
    print(f"  Checkpoint     : {checkpoint}")
    print(f"  Run ID         : {RUN_ID}")
    print(f"  Full status    : {json.dumps(status, indent=4)}")
    print("-" * 70)
    print("  Before pressing ENTER:")
    print("  1. Review output artefacts in /workspace/hrg/checkpoints/")
    print("  2. Check identity drift scores in the HRG approval log")
    print("  3. Inspect /workspace/logs/hrg.log for flagged issues")
    print("  4. If the output is acceptable, press ENTER to resume.")
    print("  5. To abort, press Ctrl+C.")
    print("=" * 70 + "\n")

    # File-flag alternative for unattended / webhook-driven setups
    flag = "hrg_approved.flag"
    print(
        f"  Alternatively, create the file '{flag}' in the current directory\n"
        "  to approve automatically (useful for webhook / CI integrations).\n"
    )
    while True:
        if os.path.exists(flag):
            log.info(f"[hrg] Approval flag detected: {flag} — continuing.")
            os.remove(flag)
            break
        # Non-blocking: try to read from stdin with a short timeout
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], 2)
        if rlist:
            sys.stdin.readline()
            break

    log.info(
        json.dumps({
            "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event":      "HRG_APPROVED",
            "run_id":     RUN_ID,
            "stage":      stage,
            "checkpoint": checkpoint,
        })
    )


def handle_hrg(status: dict) -> paramiko.SSHClient:
    """
    Full HRG pause/resume sequence (spec Part 3 §5.2):
      1. stop pod
      2. wait for human approval (blocking)
      3. start pod
      4. rehydrate services (SSH + API restart + health)
      5. POST /system/resume
    Returns new SSH client with API healthy and pipeline resumed.
    """
    log.info("[hrg] HRG checkpoint reached — stopping pod for human review.")
    stop_pod()
    wait_for_human_approval(status)

    log.info("[hrg] Restarting pod after human approval …")
    start_pod()
    # Allow the pod to come up before attempting SSH
    time.sleep(15)

    new_client = ssh_connect()
    rehydrate_services(new_client)
    resume_pipeline()

    log.info(
        json.dumps({
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event":  "HRG_RESUMED",
            "run_id": RUN_ID,
            "stage":  status.get("stage"),
        })
    )
    # Return new client to caller so the monitoring loop can use it
    return new_client


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE MONITOR  (spec Part 3 §7)
# ─────────────────────────────────────────────────────────────────────────────
def monitor_pipeline(client: paramiko.SSHClient) -> None:
    """
    Poll /system/status in a loop.
    Handles: RUNNING, AWAITING_HRG, COMPLETED, FAILED, and transient errors.
    Mutates `client` in-place when a new SSH connection is established after HRG.
    """
    log.info("[session] Monitoring pipeline …")
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5

    while True:
        status = get_pipeline_status()

        if status is None:
            consecutive_errors += 1
            log.warning(
                f"[session] No status response "
                f"({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}) …"
            )
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                stop_pod()
                raise RuntimeError(
                    "[session] Pipeline status unreachable — stopping pod."
                )
            time.sleep(POLL_INTERVAL)
            continue

        consecutive_errors = 0
        state = status.get("state", "UNKNOWN")
        stage = status.get("stage", "")
        log.info(f"[session] state={state}  stage={stage}  full={status}")

        if state == "RUNNING":
            time.sleep(POLL_INTERVAL)

        elif state == "AWAITING_HRG":
            new_client = handle_hrg(status)
            # Update the client reference for future run_remote calls if needed
            client = new_client  # noqa: F841 — caller does not need the return

        elif state == "COMPLETED":
            log.info("[session] ✅ Pipeline COMPLETED successfully.")
            log.info(
                json.dumps({
                    "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event":  "PIPELINE_COMPLETED",
                    "run_id": RUN_ID,
                })
            )
            stop_pod()
            break

        elif state == "FAILED":
            log.error("[session] ❌ Pipeline FAILED — stopping pod.")
            log.info(
                json.dumps({
                    "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event":  "PIPELINE_FAILED",
                    "run_id": RUN_ID,
                    "status": status,
                })
            )
            stop_pod()
            raise RuntimeError(
                "[session] VGA pipeline reported FAILED state.\n"
                f"Full status: {json.dumps(status, indent=2)}"
            )

        else:
            # Unknown / transitional states — continue polling
            log.warning(
                f"[session] Unknown pipeline state '{state}' — continuing to poll …"
            )
            time.sleep(POLL_INTERVAL)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SESSION ORCHESTRATION  (spec Part 3 §6)
# ─────────────────────────────────────────────────────────────────────────────
def run_session() -> None:
    """
    Full session flow:
      podResume → SSH → bootstrap (auto-detect + manifest validation) →
      API start (retry-safe) → /system/start → monitor loop →
      (HRG if needed) → /system/resume → COMPLETED → podStop

    v7.1 guarantees:
      - Env vars always explicitly injected into bootstrap (no implicit SSH env)
      - Bootstrap skipped only when manifest is content-valid AND models are present
      - .env_vga sourced only after bootstrap confirms completion
      - API rehydration retries with fresh uvicorn restarts before failing

    v7.1.1 guarantees:
      - SVI LoRA env vars injected with correct verified filenames
        (version-2.0/ prefix + full model identifier in filename)
    """
    _require_env()

    log.info("[session] ══════════════════════════════════════════════")
    log.info("[session]  VGA v17.2  Session Controller — Spec v7.1.1  ")
    log.info("[session] ══════════════════════════════════════════════")
    log.info(f"[session]  Run ID   : {RUN_ID}")
    log.info(f"[session]  Pod ID   : {POD_ID}")
    log.info(f"[session]  Pod IP   : {POD_IP}:{POD_PORT}")
    log.info(f"[session]  API base : {API_BASE}")
    log.info("[session] ══════════════════════════════════════════════")

    # ── Step 1: Resume pod ────────────────────────────────────────────────
    start_pod()
    log.info("[session] Waiting 15s for pod to become available …")
    time.sleep(15)

    # ── Step 2: SSH connect ───────────────────────────────────────────────
    client = ssh_connect()

    try:
        # ── Step 3: Bootstrap (v7.1 — manifest validation + env injection) ─
        ensure_bootstrap(client)

        # ── Step 4: Rehydrate services (v7.1 — retry-safe) ───────────────
        rehydrate_services(client)

        # ── Step 5: Start pipeline ────────────────────────────────────────
        start_pipeline()

        # ── Step 6: Monitor until COMPLETED or FAILED ─────────────────────
        monitor_pipeline(client)

    finally:
        try:
            client.close()
        except Exception:
            pass

    log.info("[session] Session complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VGA v17.2 / RunPod Session Controller — Spec v7.1.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Full run (bootstrap auto-detected via manifest):
          python session_controller.py

          # Stop the pod immediately and exit:
          python session_controller.py --stop-pod

        Bootstrap detection (v7.1):
          The controller checks /workspace/state/run_manifest.json for both
          existence AND content validity (validation==passed, models present,
          torch_version present). A manifest from a crashed run is rejected.
          To force a re-bootstrap, delete the manifest from the pod:
            ssh root@<POD_IP> "rm /workspace/state/run_manifest.json"

        Environment injection (v7.1):
          Bootstrap receives all required env vars via explicit export statements
          prepended to the bootstrap command. Paramiko does NOT guarantee SSH
          environment inheritance, so this controller never relies on it.

          Required vars and their correct values:

          HUGGING_FACE_HUB_TOKEN   — your HuggingFace token (gated models)
          LORA_IDENTITY_REPO       — your-username/your-identity-lora
          LORA_STYLE_REPO          — your-username/your-style-lora
          SVI_LORA_REPO            — vita-video-gen/svi-model (default)

          SVI LoRA filenames (v7.1.1 CORRECTED — include version-2.0/ prefix):
          SVI_HIGH_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
          SVI_LOW_NOISE_FILE=version-2.0/SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro.safetensors

          DOWNLOAD_MIN_FREE_GB     — minimum free disk headroom (default: 15)

        SVI LoRA Note (v7.1.1):
          The correct SVI LoRA filenames include the full model identifier:
            SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro.safetensors
          NOT the shortened form previously documented:
            SVI_Wan2.2_high_noise_v2.0_pro.safetensors (WRONG — will FileNotFoundError)
          Files live inside the version-2.0/ subfolder of vita-video-gen/svi-model.
          Source: https://huggingface.co/vita-video-gen/svi-model/tree/main/version-2.0
        """),
    )
    parser.add_argument(
        "--stop-pod",
        action="store_true",
        default=False,
        help="Stop the pod immediately and exit (no pipeline or bootstrap).",
    )
    args = parser.parse_args()

    load_dotenv()

    if args.stop_pod:
        _require_env()
        stop_pod()
        sys.exit(0)

    run_session()
