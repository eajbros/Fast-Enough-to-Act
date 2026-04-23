"""
gpu_guard — a tiny helper for bench scripts to check whether the GPU has been
properly claimed via `240d-gpu claim` before running CUDA work.

The motivation: running `ncu` or latency-sensitive training on cortex without
first stopping the vault AI stack (llama-server + ask-api) produces garbage
numbers because llama-server's KV cache + occasional inference kernels
interleave with yours. The `240d-gpu` coordination script writes a lock at
/run/lock/240d-gpu.lock when someone claims the GPU; this helper reads it.

Usage in bench scripts (one line at the top of main()):

    from src.gpu_guard import check_claim_or_warn
    check_claim_or_warn()

That emits a prominent warning (but does NOT exit) if:
  - No one has claimed the GPU
  - Someone ELSE has claimed it (you're stepping on their toes)

If the user wants hard-fail behavior, they pass `require=True`:

    check_claim_or_warn(require=True)
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

LOCK_FILE = Path("/run/lock/240d-gpu.lock")


def _lock_state() -> tuple[str, str, str] | None:
    """Return (user, pid, note) from the lock file, or None if no lock."""
    if not LOCK_FILE.exists():
        return None
    try:
        parts = LOCK_FILE.read_text().strip().split("\t")
        if len(parts) < 4:
            return None
        user, pid, _start, note = parts[0], parts[1], parts[2], parts[3]
        return (user, pid, note)
    except OSError:
        return None


def check_claim_or_warn(require: bool = False) -> bool:
    """Check whether current user has claimed the GPU.

    Returns True if OK to proceed, False otherwise. If `require=True`, exits
    the process on mismatch.

    The function is a no-op outside cortex (it won't see the lock file and
    silently passes) so it's safe to call unconditionally from bench scripts
    that might run on a dev laptop for quick CPU-only sanity.
    """
    # Safe to skip entirely if we're not on a GPU system
    try:
        import torch
        if not torch.cuda.is_available():
            return True
    except ImportError:
        return True

    # Also skip if not on cortex (lock file path is cortex-specific)
    # Heuristic: the /run/lock path exists on Linux, and cortex specifically
    # is where /srv/240d/ lives.
    if not Path("/srv/240d").exists():
        return True

    state = _lock_state()
    me = getpass.getuser()

    if state is None:
        _emit_warning(
            "GPU is UNCLAIMED.",
            (
                f"Nothing in {LOCK_FILE} indicates anyone has coordinated exclusive "
                "GPU access. Your results will share VRAM with llama-server (vault "
                "AI, ~5.5 GB) and ncu numbers will be unreliable."
            ),
            "Run `240d-gpu claim` before this script to stop the vault AI stack.",
        )
        if require:
            sys.exit(1)
        return False

    user, pid, note = state
    if user != me:
        _emit_warning(
            f"GPU is CLAIMED BY {user.upper()} (pid {pid}).",
            (
                f"Your note should be: {note!r}" if note else ""
            ) + "\nRunning CUDA work now will interfere with their measurement.",
            f"DM {user} before continuing, or `240d-gpu release --force` if they're "
            "unreachable and you've waited a reasonable time.",
        )
        if require:
            sys.exit(1)
        return False

    # claimed by current user — OK
    note_suffix = f" (note: {note})" if note else ""
    print(f"[gpu_guard] ✓ GPU claimed by you ({me}){note_suffix}", file=sys.stderr)
    return True


def _emit_warning(title: str, detail: str, action: str) -> None:
    # Box the warning so it's hard to miss even when scrolled past
    line = "=" * 72
    print(f"\n{line}", file=sys.stderr)
    print(f"[gpu_guard] WARNING: {title}", file=sys.stderr)
    if detail:
        print(f"[gpu_guard] {detail}", file=sys.stderr)
    print(f"[gpu_guard] → {action}", file=sys.stderr)
    print(f"{line}\n", file=sys.stderr)


if __name__ == "__main__":
    # CLI mode — exit 0 if claimed by current user, 1 otherwise.
    # Useful in Makefiles: `@python -m src.gpu_guard || (echo 'claim first'; exit 1)`
    require = "--require" in sys.argv or "-r" in sys.argv
    ok = check_claim_or_warn(require=require)
    sys.exit(0 if ok else 1)
