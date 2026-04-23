"""
Baseline latency benchmark for Diffusion Policy (Yuva's Week 1 deliverable).

Loads the Push-T CNN checkpoint from /srv/240d/checkpoints/, runs one inference
for both DDPM-100 and DDIM-10 schedules, times observation encoding + denoising steps + action
extraction, writes structured JSON to /srv/240d/results/latency/yuva/<date>/.

Success metric (per submitted proposal Step 1):
    DDIM-10 latency within 20% of Chi 2023 Table 2 Push-T numbers

Usage:
    conda activate diffusion
    python scripts/bench_dp.py --runs 10 --warmup 3

NOTE (yuva): this scaffold uses the Diffusion Policy repo's checkpoint-loading pattern.
Exact API call may shift based on what loads cleanly — skim
`/srv/240d/repos/diffusion_policy/diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
for the canonical pattern, then adapt below.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/srv/240d/repos/diffusion_policy")
from src.timing import Timer

CHECKPOINT = "/srv/240d/checkpoints/diffusion-policy-pusht/latest.ckpt"
RESULTS_ROOT = Path("/srv/240d/results/latency")


def load_policy(checkpoint_path: str, num_inference_steps: int):
    # TODO(yuva): fill in using the DP repo's load pattern. Typical:
    #   from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
    #   workspace = TrainDiffusionUnetImageWorkspace.load_checkpoint(checkpoint_path)
    #   policy = workspace.model
    #   policy.num_inference_steps = num_inference_steps   # 100 for DDPM, 10 for DDIM
    #   return policy.to('cuda').eval()
    raise NotImplementedError("Wire up DP checkpoint loading. See /srv/240d/repos/diffusion_policy/")


def make_sample_observation():
    # Push-T observation is a dict: {'image': (C,H,W) uint8, 'agent_pos': (D,) float}.
    # For latency smoke-test, any valid-shape tensors work.
    return {
        "image": torch.zeros(1, 2, 3, 96, 96, dtype=torch.float32, device="cuda"),   # (batch, obs_horizon, C, H, W)
        "agent_pos": torch.zeros(1, 2, 2, dtype=torch.float32, device="cuda"),       # (batch, obs_horizon, agent_dim)
    }


def bench_schedule(policy, obs, schedule_name: str, n_steps: int, runs: int, warmup: int) -> Timer:
    policy.num_inference_steps = n_steps

    print(f"[*] Warmup ({warmup}) for {schedule_name} K={n_steps} ...")
    for _ in range(warmup):
        with torch.inference_mode():
            _ = policy.predict_action(obs)
    torch.cuda.synchronize()

    print(f"[*] Timed runs ({runs}) ...")
    detailed = None
    totals: list[float] = []
    for i in range(runs):
        t = Timer(
            paradigm="diffusion",
            model="diffusion-policy-pusht-cnn",
            config={"schedule": schedule_name, "K": n_steps, "batch": 1},
            warmup_runs=warmup,
        )
        # Phase-level split for Week 2. For Week 1 M1, end-to-end suffices.
        with t.phase("predict_action_end_to_end"):
            with torch.inference_mode():
                _ = policy.predict_action(obs)
        totals.append(t.total_ms)
        if i == runs // 2:
            detailed = t

    median_ms = sorted(totals)[runs // 2]
    print(f"[+] {schedule_name} K={n_steps}: median {median_ms:.1f} ms  ({1000/median_ms:.1f} Hz)")
    return detailed, median_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    today = date.today().isoformat()
    outdir = Path(args.output_dir) if args.output_dir else RESULTS_ROOT / "yuva" / today
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading DP checkpoint from {CHECKPOINT} ...")

    # DDPM-100 (full schedule — the "reference" expensive inference)
    policy = load_policy(CHECKPOINT, num_inference_steps=100)
    obs = make_sample_observation()
    ddpm_detail, ddpm_med = bench_schedule(policy, obs, "DDPM", 100, args.runs, args.warmup)
    data = ddpm_detail.finalize()
    data["median_ms_of_runs"] = ddpm_med
    import json
    (outdir / "dp-ddpm100.json").write_text(json.dumps(data, indent=2))

    # DDIM-10 (inference-time trick — this is the latency number the paper uses)
    policy = load_policy(CHECKPOINT, num_inference_steps=10)
    ddim_detail, ddim_med = bench_schedule(policy, obs, "DDIM", 10, args.runs, args.warmup)
    data = ddim_detail.finalize()
    data["median_ms_of_runs"] = ddim_med
    (outdir / "dp-ddim10.json").write_text(json.dumps(data, indent=2))

    speedup = ddpm_med / ddim_med
    print(f"[+] DDIM-10 speedup over DDPM-100: {speedup:.1f}x")
    print(f"[+] Output: {outdir}/")


if __name__ == "__main__":
    main()
