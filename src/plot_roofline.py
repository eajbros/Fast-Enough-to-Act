"""
Plot per-kernel arithmetic intensity vs achieved throughput on the RTX 4060 roofline.

Input is a list of dicts (one per kernel) typically parsed from Nsight Compute CSV export:
    {"name": "llama_decode_matmul", "arith_intensity": 0.4, "throughput_gflops": 95, "paradigm": "vla"}

The RTX 4060 peaks below are the physical ceilings. Kernel dots that hug the memory-bound line
(the diagonal on the left) are bandwidth-limited; dots near the flat compute ceiling are math-limited.
The paper's central claim is that VLA decode kernels cluster on the left and DP denoise kernels
cluster on the right.
"""
from pathlib import Path
import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np


# RTX 4060 Laptop GPU peaks. Verify on cortex via:
#   nvidia-smi --query-gpu=name,clocks.max.mem,clocks.max.graphics --format=csv
# and CUDA samples `bandwidthTest` for actual sustained bandwidth.
PEAK_FP32_TFLOPS = 11.7
PEAK_FP16_TFLOPS = 23.4
PEAK_MEM_BW_GB_S = 272.0


def roofline_ceiling(ai: np.ndarray, peak_gflops: float, peak_bw_gbs: float) -> np.ndarray:
    """Piecewise ceiling: min(peak_bw * ai, peak_compute). Returns GFLOPs/s."""
    return np.minimum(ai * peak_bw_gbs, peak_gflops)


def plot_roofline(
    kernels: list[dict],
    output_path: str | Path,
    precision: str = "fp16",
    title: str = "RTX 4060 Laptop Roofline",
) -> Path:
    peak_gflops = (PEAK_FP16_TFLOPS if precision == "fp16" else PEAK_FP32_TFLOPS) * 1000.0
    ridge_ai = peak_gflops / PEAK_MEM_BW_GB_S

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    ai_range = np.logspace(-2, 3, 200)
    roof = roofline_ceiling(ai_range, peak_gflops, PEAK_MEM_BW_GB_S)
    ax.plot(ai_range, roof, color="black", linewidth=2, label=f"Roofline ({precision})")
    ax.axvline(ridge_ai, color="gray", linestyle="--", alpha=0.6,
               label=f"Ridge @ AI = {ridge_ai:.1f} FLOPs/byte")

    styles = {
        "vla": dict(color="tab:red", marker="o", label="OpenVLA"),
        "diffusion": dict(color="tab:blue", marker="s", label="Diffusion Policy"),
    }
    seen_labels: set[str] = set()
    for k in kernels:
        style = styles.get(k.get("paradigm", ""), dict(color="tab:gray", marker="^", label="other"))
        label = style["label"] if style["label"] not in seen_labels else None
        seen_labels.add(style["label"])
        ax.scatter(k["arith_intensity"], k["throughput_gflops"],
                   s=80, edgecolor="black", zorder=5, label=label, **{kk: vv for kk, vv in style.items() if kk not in ("label",)})
        ax.annotate(k["name"], (k["arith_intensity"], k["throughput_gflops"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Throughput (GFLOPs / s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_ncu_csv(csv_path: str | Path, paradigm: str) -> list[dict]:
    """Parse an ncu CSV export. ncu column names depend on --section; this assumes you ran
    with at least SpeedOfLight_RooflineChart which produces Memory.Traffic + Compute.FLOPs metrics."""
    kernels = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            # These exact column names may vary by ncu version; adjust here once we have real output.
            kernels.append({
                "name": row.get("Kernel Name", "unknown")[:40],
                "arith_intensity": float(row.get("Arithmetic Intensity [FLOP/byte]", 0) or 0),
                "throughput_gflops": float(row.get("Achieved Performance [GFLOP/s]", 0) or 0),
                "paradigm": paradigm,
            })
    return kernels


def main():
    ap = argparse.ArgumentParser(description="Plot RTX 4060 roofline from kernel data (JSON or CSV).")
    ap.add_argument("--input", help="JSON file (list of kernel dicts) or CSV from ncu")
    ap.add_argument("--paradigm", choices=["vla", "diffusion"],
                    help="If loading from ncu CSV, label all kernels with this paradigm")
    ap.add_argument("--output", default="results/roofline/roofline.png")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--dummy", action="store_true", help="Plot placeholder data to verify pipeline")
    args = ap.parse_args()

    if args.dummy:
        kernels = [
            {"name": "vla_decode_matmul", "arith_intensity": 0.5, "throughput_gflops": 150, "paradigm": "vla"},
            {"name": "vla_kv_read", "arith_intensity": 0.2, "throughput_gflops": 55, "paradigm": "vla"},
            {"name": "vla_prefill_matmul", "arith_intensity": 30, "throughput_gflops": 8000, "paradigm": "vla"},
            {"name": "dp_unet_conv3x3", "arith_intensity": 85, "throughput_gflops": 18000, "paradigm": "diffusion"},
            {"name": "dp_unet_attn", "arith_intensity": 50, "throughput_gflops": 14000, "paradigm": "diffusion"},
            {"name": "dp_upsample", "arith_intensity": 12, "throughput_gflops": 3200, "paradigm": "diffusion"},
        ]
    elif args.input and args.input.endswith(".json"):
        kernels = json.loads(Path(args.input).read_text())
    elif args.input and args.input.endswith(".csv"):
        assert args.paradigm, "--paradigm required when loading from ncu CSV"
        kernels = load_ncu_csv(args.input, args.paradigm)
    else:
        ap.error("Provide --input <file.json|.csv> or --dummy")

    out = plot_roofline(kernels, args.output, precision=args.precision)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
