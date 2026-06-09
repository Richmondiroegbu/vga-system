#!/usr/bin/env python3
"""
download_test_outputs.py — Download test outputs from RunPod pod to local machine.

Run this script LOCALLY (not on the pod) after test_image_video.py finishes.
It reads the same .env / environment variables as session_controller.py.

Usage:
    # Download the most recent test run
    python3 scripts/download_test_outputs.py

    # Download a specific run by ID (timestamp e.g. 20260610_143022)
    python3 scripts/download_test_outputs.py --run-id 20260610_143022

    # Download to a custom local directory
    python3 scripts/download_test_outputs.py --local-dir C:/Downloads/vga_test

    # List available runs on the pod without downloading
    python3 scripts/download_test_outputs.py --list

Required environment variables (same as session_controller.py — set in .env):
    POD_IP               RunPod pod public IP
    POD_PORT             SSH port (default: 22)
    SSH_USER             SSH username (default: root)
    SSH_PRIVATE_KEY_PATH Path to SSH private key (default: ~/.ssh/id_rsa)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("ERROR: paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

# ── Configuration (mirrors session_controller.py exactly) ─────────────────────
load_dotenv()

POD_IP           = os.getenv("POD_IP", "")
POD_PORT         = int(os.getenv("POD_PORT", "22"))
SSH_USER         = os.getenv("SSH_USER", "root")
SSH_KEY_PATH     = os.path.expanduser(os.getenv("SSH_PRIVATE_KEY_PATH", "~/.ssh/id_rsa"))

REMOTE_TEST_DIR  = "/workspace/output/test"
DEFAULT_LOCAL_DIR = Path(__file__).parent.parent / "test_outputs"


# ── SSH helpers ────────────────────────────────────────────────────────────────

def connect() -> paramiko.SSHClient:
    """Connect to the RunPod pod using the same method as session_controller.py."""
    if not POD_IP:
        print("ERROR: POD_IP not set. Add POD_IP=<your_pod_ip> to .env")
        sys.exit(1)

    try:
        key = paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
    except paramiko.ssh_exception.SSHException:
        key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY_PATH)
    except FileNotFoundError:
        print(f"ERROR: SSH key not found: {SSH_KEY_PATH}")
        print("  Set SSH_PRIVATE_KEY_PATH in .env to the correct path.")
        sys.exit(1)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {SSH_USER}@{POD_IP}:{POD_PORT} ...")
    client.connect(POD_IP, port=POD_PORT, username=SSH_USER, pkey=key, timeout=30)
    print("Connected.")
    return client


def list_runs(client: paramiko.SSHClient) -> list[str]:
    """Return a list of run IDs (subdirectory names) in /workspace/output/test/."""
    _, stdout, _ = client.exec_command(
        f"ls -1t {REMOTE_TEST_DIR}/ 2>/dev/null || echo '__empty__'"
    )
    lines = stdout.read().decode().strip().splitlines()
    if lines == ["__empty__"] or not lines:
        return []
    return [l.strip() for l in lines if l.strip()]


def sftp_download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> list[str]:
    """Download all files from remote_dir into local_dir. Returns list of downloaded filenames."""
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    try:
        entries = sftp.listdir_attr(remote_dir)
    except FileNotFoundError:
        print(f"  ERROR: Remote directory not found: {remote_dir}")
        return []

    if not entries:
        print(f"  No files found in {remote_dir}")
        return []

    for entry in sorted(entries, key=lambda e: e.filename):
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path  = local_dir / entry.filename
        size_mb     = (entry.st_size or 0) / 1e6

        print(f"  ↓ {entry.filename:<45} {size_mb:>8.1f} MB", end="", flush=True)
        t0 = time.time()

        def _progress(transferred: int, total: int) -> None:
            pct = int(transferred / total * 100) if total else 0
            print(f"\r  ↓ {entry.filename:<45} {size_mb:>8.1f} MB  {pct:3d}%", end="", flush=True)

        sftp.get(remote_path, str(local_path), callback=_progress)
        elapsed = time.time() - t0
        speed_mb = size_mb / elapsed if elapsed > 0 else 0
        print(f"\r  ✓ {entry.filename:<45} {size_mb:>8.1f} MB  done ({elapsed:.1f}s, {speed_mb:.1f} MB/s)")
        downloaded.append(entry.filename)

    return downloaded


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download VGA test outputs from RunPod to local machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Specific run ID to download (e.g. 20260610_143022). "
            "If omitted, downloads the most recent run."
        ),
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help=f"Local directory to save outputs. Default: {DEFAULT_LOCAL_DIR}",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available runs on the pod without downloading.",
    )
    args = parser.parse_args()

    client = connect()

    try:
        runs = list_runs(client)

        if not runs:
            print(f"\nNo test runs found in {REMOTE_TEST_DIR}/")
            print("Run test_image_video.py on the pod first.")
            return 1

        # ── --list mode ────────────────────────────────────────────────────────
        if args.list:
            print(f"\nAvailable test runs in {REMOTE_TEST_DIR}/:")
            for i, run_id in enumerate(runs):
                marker = "  ← most recent" if i == 0 else ""
                print(f"  {run_id}{marker}")
            print(f"\nTotal: {len(runs)} run(s)")
            print("\nTo download most recent:  python3 scripts/download_test_outputs.py")
            print("To download specific run:  python3 scripts/download_test_outputs.py --run-id <ID>")
            return 0

        # ── Select run ─────────────────────────────────────────────────────────
        if args.run_id:
            if args.run_id not in runs:
                print(f"\nERROR: Run ID '{args.run_id}' not found on pod.")
                print(f"Available runs: {', '.join(runs)}")
                return 1
            run_id = args.run_id
        else:
            run_id = runs[0]  # most recent (ls -1t = newest first)
            print(f"\nNo --run-id specified. Using most recent run: {run_id}")

        remote_run_dir = f"{REMOTE_TEST_DIR}/{run_id}"
        local_run_dir  = Path(args.local_dir) / run_id

        print(f"\nDownloading run: {run_id}")
        print(f"  Remote: {remote_run_dir}/")
        print(f"  Local:  {local_run_dir}/")
        print()

        sftp = client.open_sftp()
        downloaded = sftp_download_dir(sftp, remote_run_dir, local_run_dir)
        sftp.close()

        if not downloaded:
            print("\nNothing was downloaded.")
            return 1

        # ── Summary ────────────────────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"Download complete — {len(downloaded)} file(s) → {local_run_dir}/")
        print(f"{'=' * 60}")

        # Group by stage for a cleaner summary
        stages = {
            "S-05 (FLUX base image)":    [f for f in downloaded if f.startswith("S05")],
            "S-07 (Z-Image refined)":    [f for f in downloaded if f.startswith("S07")],
            "S-08 (WAN2.2 first seg)":   [f for f in downloaded if f.startswith("S08")],
            "S-09 (SVI continuation)":   [f for f in downloaded if f.startswith("S09")],
            "Report":                    [f for f in downloaded if f.endswith(".txt")],
        }
        for stage, files in stages.items():
            if files:
                print(f"  {stage}:")
                for f in files:
                    fp = local_run_dir / f
                    mb = fp.stat().st_size / 1e6 if fp.exists() else 0
                    print(f"    {f}  ({mb:.1f} MB)  →  {fp}")

        print(f"\nOpen folder: {local_run_dir}")
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
