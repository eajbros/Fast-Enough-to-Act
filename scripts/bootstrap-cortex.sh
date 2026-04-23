#!/usr/bin/env bash
# bootstrap-cortex.sh — one-shot idempotent onboarding for a cse240d teammate.
#
# Run this on cortex AFTER Purush has granted Tailscale access + appended your
# SSH key to /home/<you>/.ssh/authorized_keys. Run this from /srv/240d/ or your
# home — either works.
#
# What it does (all idempotent — safe to re-run):
#   1. Check you're in group cse240d (you must be for /etc/profile.d/240d.sh
#      to auto-export HF_HUB_CACHE etc.)
#   2. Install miniforge if conda isn't on your PATH
#   3. Clone the team repo to ~/Fast-Enough-to-Act (if not already there)
#   4. Create/update your role's conda env from envs/<role>.yml
#   5. Run a smoke test: import torch, confirm CUDA available, print version
#
# Usage:
#   bash bootstrap-cortex.sh --role openvla        # Purush
#   bash bootstrap-cortex.sh --role diffusion      # Yuva
#   bash bootstrap-cortex.sh --role profiling      # Ethan
#
# Or run without --role and the script will infer from $USER when possible
# and otherwise prompt.

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────
R="\033[31m"; G="\033[32m"; Y="\033[33m"; B="\033[1m"; N="\033[0m"
die()  { printf "${R}error:${N} %s\n" "$*" >&2; exit 1; }
info() { printf "${B}==>${N} %s\n" "$*"; }
ok()   { printf "${G}✓${N} %s\n" "$*"; }
warn() { printf "${Y}⚠${N}  %s\n" "$*"; }

# ─── Args ────────────────────────────────────────────────────────────────
ROLE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --role) ROLE="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '3,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) die "unknown flag: $1 (try --help)" ;;
    esac
done

# Infer role from username if not given
if [ -z "$ROLE" ]; then
    case "$USER" in
        pooshman|purush) ROLE="openvla" ;;
        yuva)            ROLE="diffusion" ;;
        ethan)           ROLE="profiling" ;;
        *)
            warn "couldn't infer role from username '$USER'"
            echo "Choose your role:"
            echo "  1) openvla    (OpenVLA autoregressive-VLA track)"
            echo "  2) diffusion  (Diffusion Policy track)"
            echo "  3) profiling  (infrastructure + plotting)"
            read -rp "Enter 1/2/3: " choice
            case "$choice" in
                1) ROLE="openvla" ;;
                2) ROLE="diffusion" ;;
                3) ROLE="profiling" ;;
                *) die "invalid choice" ;;
            esac
            ;;
    esac
fi

case "$ROLE" in
    openvla|diffusion|profiling) ;;
    *) die "unknown role '$ROLE' (expected: openvla, diffusion, profiling)" ;;
esac

info "bootstrapping for role: $ROLE (user: $USER)"

# ─── Step 1: group membership ───────────────────────────────────────────
info "step 1/5 — group membership"
if ! id -Gn | grep -qw cse240d; then
    die "you are NOT in group cse240d. Ask Purush to 'sudo usermod -aG cse240d $USER' and log out/in. Without this, HF_HUB_CACHE and friends won't auto-set."
fi
ok "you're in group cse240d"

# Verify the env vars are actually set (if not, either you just joined and
# haven't re-logged in, or profile.d didn't fire)
if [ -z "${HF_HUB_CACHE:-}" ]; then
    warn "HF_HUB_CACHE is not set in this shell. Expected /srv/240d/checkpoints/hf."
    warn "Either log out and back in, or \`source /etc/profile.d/240d.sh\` in this shell."
fi

# ─── Step 2: miniforge ──────────────────────────────────────────────────
info "step 2/5 — miniforge"
if command -v conda >/dev/null 2>&1; then
    ok "conda already installed: $(conda --version)"
else
    info "installing miniforge to $HOME/miniforge3 (~600 MB download, ~6 GB installed)..."
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm /tmp/miniforge.sh
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda init bash >/dev/null
    ok "miniforge installed — future shells will have conda on PATH"
fi

# Make conda callable in *this* non-interactive shell if it wasn't already.
# shellcheck disable=SC1091
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi

# ─── Step 3: clone team repo ────────────────────────────────────────────
info "step 3/5 — team repo"
REPO_DIR="$HOME/Fast-Enough-to-Act"
if [ -d "$REPO_DIR/.git" ]; then
    ok "repo already cloned at $REPO_DIR"
    info "fetching latest..."
    ( cd "$REPO_DIR" && git fetch --quiet )
else
    info "cloning github.com/eajbros/Fast-Enough-to-Act to $REPO_DIR..."
    if ! git clone https://github.com/eajbros/Fast-Enough-to-Act.git "$REPO_DIR" 2>/dev/null; then
        warn "HTTPS clone failed — probably a gated or auth-required repo."
        warn "Set up a GitHub PAT (github.com/settings/tokens, scope: repo)"
        warn "then re-run this script, or clone manually:"
        warn "  git clone https://<token>@github.com/eajbros/Fast-Enough-to-Act.git $REPO_DIR"
        die "cannot proceed without repo"
    fi
    ok "cloned"
fi

# ─── Step 4: conda env ──────────────────────────────────────────────────
info "step 4/5 — conda env '$ROLE'"
ENV_FILE="$REPO_DIR/envs/${ROLE}.yml"
[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"

if conda env list | awk '{print $1}' | grep -qx "$ROLE"; then
    info "env '$ROLE' exists — updating from $ENV_FILE..."
    conda env update -n "$ROLE" -f "$ENV_FILE"
else
    info "creating env '$ROLE' from $ENV_FILE (this takes 5-15 min)..."
    conda env create -n "$ROLE" -f "$ENV_FILE"
fi
ok "env '$ROLE' ready"

# ─── Step 5: smoke test ─────────────────────────────────────────────────
info "step 5/5 — smoke test"
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROLE"

python - <<'PYEOF' || die "smoke test failed — see error above"
import sys, torch
print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:  {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("⚠  CUDA not available — check that you've claimed the GPU or that it's not 100% used by llama-server")
PYEOF

ok "smoke test passed"

echo ""
info "BOOTSTRAP COMPLETE"
echo ""
echo "Next steps:"
echo "  1) Read /srv/240d/WORKSPACE-GUIDE.md for day-to-day patterns"
echo "  2) Read /srv/240d/TEAMMATES.md § 'Your Week 1' for your role-specific first task"
echo "  3) Try a claim/release cycle:"
echo "       240d-gpu status"
echo "       240d-gpu claim --note \"first test claim\""
echo "       240d-gpu release"
echo "  4) Activate your env in any new shell with:"
echo "       conda activate $ROLE"
echo ""
