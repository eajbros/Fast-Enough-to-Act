# Fast-Enough-to-Act

Roofline study of two robot-action paradigms on the same GPU. CSE 240D final project, Spring 2026.

> *"Fast Enough to Act: A Roofline Study of Autoregressive VLA vs. Diffusion Policy Inference on a Fixed GPU"* — proposal submitted Apr 19, 2026.

## Why this project exists

Modern robot-control models come in two shapes:

- **Autoregressive VLAs** (OpenVLA) — a 7B language model emits one action token at a time. Each token reads a growing KV cache. The kernel that dominates latency is a vector-matrix multiply whose bottleneck is reading weights from VRAM — **memory-bandwidth-bound**.
- **Diffusion policies** (Diffusion Policy) — a small dense net denoises action sequences over K iterative steps. Each step is a matrix-matrix product through the same layers — **compute-bound**.

Both want the same thing (actions at ~30 Hz on an 8 GB GPU) but their bottlenecks sit on **opposite sides of the GPU roofline**. Nobody has published a hardware comparison of these two paradigms at batch-of-one (the only regime that matters for real-time robot control). We're building one.

The central figure plots both paradigms' dominant kernels on one roofline. The second finding: an optimization like INT-quantization moves each paradigm *differently* because they started on different sides of the ridge — which tells hybrid-architecture designers where to spend compute and where to spend bandwidth.

Canonical plan + research framing: **`Classes/CSE-240D/240D Project.md`** in the vault. This README is the repo-level pointer; the vault is the source of truth.

## Team

Purush, Yuva, Ethan — all three enrolled in 240D. Azfar handles the robot hardware (separate course, separate repo at `bistable-vlm`).

High-level tracks:

- **OpenVLA** — autoregressive-VLA benchmarks and optimizations
- **Diffusion Policy** — diffusion-side benchmarks and optimizations
- **Infrastructure** — timing harness, plotting, Pareto frontiers, paper figures

> ⚠️ **Per-person deliverables are being revised by Purush (2026-04-23).** Until the revision lands, treat the *tracks* above as stable and specific tasks as in flux.

## Repo layout

```
envs/            conda env recipes (one per track)
scripts/         runnable benchmark entrypoints (bench_*.py, bootstrap-cortex.sh)
src/             shared utilities: timing.py, plot_roofline.py, gpu_guard.py
results/         .gitkeep only — real artifacts live on cortex at /srv/240d/results/
Makefile         make bench-openvla / bench-dp / roofline / status
CLAUDE.md        context auto-loaded by Claude Code sessions in this repo
```

Source code lives per-user in each teammate's home clone. **Data + checkpoints + results live on cortex at `/srv/240d/`** — shared so we don't download 34 GB of weights three times, and so anyone can pick up anyone else's artifacts. Read `/srv/240d/README.md` on cortex for the workspace map.

## Getting started

First time on cortex, after Purush grants access:

```bash
ssh <you>@cortex
bash <(curl -fsSL https://raw.githubusercontent.com/eajbros/Fast-Enough-to-Act/main/scripts/bootstrap-cortex.sh) --role <openvla|diffusion|profiling>
```

Idempotent — installs miniforge, clones this repo, creates your conda env, runs a torch+CUDA smoke test. See `/srv/240d/TEAMMATES.md` for the step-by-step.

After that, day-to-day flow:

```bash
240d-gpu status                              # see the card
240d-gpu claim --note "what you're doing"    # stop vault AI, lock GPU
make bench-openvla                           # or bench-dp, roofline
240d-gpu release                             # restores vault AI
```

`make bench-*` refuses to run unless the GPU is claimed by you (via `src/gpu_guard.py`), so forgetting to claim fails loud with a pointer, not silently-wrong numbers.

For the why, how, and troubleshooting of any of the above: **`/srv/240d/WORKSPACE-GUIDE.md`** on cortex.

## Related

- Canonical plan: vault `Classes/CSE-240D/240D Project.md`
- Workspace map: `/srv/240d/README.md` on cortex
- Day-to-day patterns: `/srv/240d/WORKSPACE-GUIDE.md`
- Onboarding walk-through: `/srv/240d/TEAMMATES.md`
- Robot hardware (separate repo): `bistable-vlm` on the Pi (`ssh bistable`)
