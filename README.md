# Fast-Enough-to-Act

> "Fast Enough to Act: A Roofline Study of Autoregressive VLA vs Diffusion Policy Inference on a Fixed GPU"
> CSE 240D, Spring 2026. Submitted Apr 19.

Roofline study comparing **OpenVLA** (autoregressive, memory-bandwidth-bound) against **Diffusion Policy** (iterative denoising, compute-bound) on the **same RTX 4060 Laptop GPU**, measuring how each paradigm responds to paradigm-specific and shared optimizations.

The paper's central figure: both paradigms' dominant kernels plotted on the same roofline, showing they occupy opposite regions.

## Team

| Person | Track | Owns |
|---|---|---|
| Purush | OpenVLA | `scripts/bench_openvla.py`, VLA optimizations (INT4, FA-2, KV compression, OFT), roofline dots for memory-bound decode |
| Yuva   | Diffusion Policy | `scripts/bench_dp.py`, DDIM step reduction + FP16 + torch.compile + INT8, roofline dots for compute-bound denoise |
| Ethan  | Infrastructure | `src/` (timing + plotting utilities), data collection pipeline, Pareto frontiers, paper figures |
| Azfar  | Robot | Physical bot chassis + PID (stretch demo; not enrolled in 240D) |

## Repo layout

```
envs/                                conda env recipes, one per role
├── openvla.yml                      Purush
├── diffusion.yml                    Yuva
└── profiling.yml                    Ethan + analysis

scripts/                             runnable benchmark scripts
├── bench_openvla.py                 M1 (Purush) — OpenVLA baseline latency
└── bench_dp.py                      M1 (Yuva)   — DP baseline latency (DDPM-100 + DDIM-10)

src/                                 shared Python utilities
├── timing.py                        torch.cuda.Event → structured JSON (everyone uses this)
└── plot_roofline.py                 ncu CSV / JSON → PNG roofline plot (Ethan extends)

results/                             .gitkeep only — real artifacts land in /srv/240d/ on cortex
├── latency/                         Timer JSONs
├── nsys/                            Nsight Systems traces (*.nsys-rep)
├── ncu/                             Nsight Compute reports (*.ncu-rep)
└── roofline/                        PNG plots
```

## Environments

Cortex doesn't ship system-wide conda — each teammate installs miniforge in their home (see `/srv/240d/TEAMMATES.md`), then:

```bash
conda env create -f envs/openvla.yml      # Purush
conda env create -f envs/diffusion.yml    # Yuva
conda env create -f envs/profiling.yml    # Ethan

conda activate openvla
pip install flash-attn==2.6.3 --no-build-isolation   # Purush only; must be post-torch
```

All three envs use PyTorch 2.4.1 + bundled CUDA 12.4 runtime (works with cortex's driver 590).

## Shared data on cortex (not in the repo)

Pre-pulled by Purush so nobody downloads 14 GB three times:

| Path | What | Size |
|---|---|---|
| `/srv/240d/checkpoints/openvla-7b/` | OpenVLA base checkpoint | 15 GB |
| `/srv/240d/checkpoints/openvla-7b-libero-spatial/` | LIBERO-Spatial fine-tune (for task-quality eval) | ~14 GB |
| `/srv/240d/checkpoints/diffusion-policy-pusht/latest.ckpt` | DP Push-T CNN baseline | ~200 MB |
| `/srv/240d/repos/openvla/` | Upstream OpenVLA source (read-only reference) | 2 MB |
| `/srv/240d/repos/diffusion_policy/` | Upstream DP source | 32 MB |

Your Timer output lands under `/srv/240d/results/latency/<your-name>/<YYYY-MM-DD>/`.

## Smoke test — verify env

```bash
conda activate openvla     # or diffusion, or profiling
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 4060 Laptop GPU

gpu-check                  # runs /srv/240d/bin/gpu-check — shows GPU state + who's using it
```

## Run a baseline benchmark

```bash
# Purush (after conda activate openvla)
python scripts/bench_openvla.py --runs 10 --warmup 3
# → writes /srv/240d/results/latency/purush/YYYY-MM-DD/openvla-baseline.json

# Yuva (after conda activate diffusion)
python scripts/bench_dp.py --runs 10 --warmup 3
# → writes /srv/240d/results/latency/yuva/YYYY-MM-DD/dp-ddpm100.json + dp-ddim10.json
```

## Plot a roofline

```bash
# Ethan (after conda activate profiling)
python src/plot_roofline.py --dummy --output results/roofline/ethan/YYYY-MM-DD/dummy.png
# Verifies the plotting pipeline end-to-end with placeholder kernel data.

# Week 2+: real data from ncu
python src/plot_roofline.py --input /srv/240d/results/ncu/purush/.../openvla.csv --paradigm vla --output ...
```

## GPU etiquette

See `/srv/240d/README.md` on cortex. Short version:
1. `gpu-check` before launching.
2. For `ncu` profiling: `gpu-exclusive` first (stops cortex-ask + Ollama), do your run, `gpu-restore` when done.
3. Coordinate long jobs in Slack — the 4060 has 8 GB of VRAM, serialize heavy workloads.

## Related

- Proposal (canonical): Google Doc `240D Project Proposal` + vault `Classes/CSE-240D/240D Project.md`
- Technical decisions: vault `Classes/CSE-240D/240D Implementation.md`
- Team study curriculum: vault `Classes/CSE-240D/240D Study Plan.md`
- Robot hardware (Azfar's side): `bistable-vlm` repo, lives on the Pi (`ssh bistable`)
