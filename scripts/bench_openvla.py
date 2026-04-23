"""
Baseline latency benchmark for OpenVLA (Purush's Week 1 deliverable).

Loads openvla-7b-libero-spatial from /srv/240d/checkpoints/, runs one inference
on a single LIBERO image, times each phase (vision encode, prefill, decode),
and writes a structured JSON to /srv/240d/results/latency/purush/<date>/.

Success metric (per submitted proposal Step 1):
    end-to-end latency within 20% of OpenVLA paper Table 7 (~250-350 ms @ INT4 on 4060-class GPU)

Usage:
    conda activate openvla
    python scripts/bench_openvla.py --runs 10 --warmup 3
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.timing import Timer

CHECKPOINT = "/srv/240d/checkpoints/openvla-7b-libero-spatial"
RESULTS_ROOT = Path("/srv/240d/results/latency")


def load_sample_image() -> Image.Image:
    # TODO(purush): replace with a real LIBERO eval image once you have the LIBERO repo installed.
    # For the first smoke-test, any 224x224 RGB image is fine — we only care about *latency*,
    # not task success. A blank image still exercises every kernel.
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="Timed runs (reported latency = median)")
    ap.add_argument("--warmup", type=int, default=3, help="Warmup runs before timing starts")
    ap.add_argument("--instruction", default="navigate to the red marker")
    ap.add_argument("--output", default=None, help="Output JSON path (default: auto-date)")
    args = ap.parse_args()

    print(f"[*] Loading OpenVLA from {CHECKPOINT} ...")
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,           # INT4 quantization (NF4 via bitsandbytes)
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.eval()
    print(f"[*] Model loaded. GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    image = load_sample_image()
    prompt = f"In: What action should the robot take to {args.instruction}?\nOut:"

    # Warmup — don't time, but forces CUDA graphs + kernel autotune to settle.
    print(f"[*] Warming up ({args.warmup} runs) ...")
    for _ in range(args.warmup):
        inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            _ = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
    torch.cuda.synchronize()

    # Timed runs. We record ONE run in detail (for the JSON) and use the rest to confirm variance.
    print(f"[*] Timed runs ({args.runs}) ...")
    totals_ms: list[float] = []
    detailed = None

    for i in range(args.runs):
        t = Timer(
            paradigm="vla",
            model="openvla-7b-libero-spatial",
            config={"precision": "int4", "batch": 1, "instruction": args.instruction},
            warmup_runs=args.warmup,
        )
        inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

        # Three phases of VLA inference. The decomposition below is conceptual;
        # OpenVLA's predict_action is a single call, so for phase-level numbers we'll need to
        # call vision / llm.forward / llm.generate separately in Week 2. For Week 1 M1, we
        # measure just end-to-end — still satisfies Step 1 ("baseline latency").
        with t.phase("predict_action_end_to_end"):
            with torch.inference_mode():
                _ = model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)

        totals_ms.append(t.total_ms)
        if i == args.runs // 2:                    # grab the median-ish run as the detailed record
            detailed = t

    median_ms = sorted(totals_ms)[len(totals_ms) // 2]
    p90_ms = sorted(totals_ms)[int(len(totals_ms) * 0.9)]
    print(f"[+] Median E2E: {median_ms:.1f} ms   P90: {p90_ms:.1f} ms   Hz: {1000/median_ms:.2f}")

    today = date.today().isoformat()
    output = Path(args.output) if args.output else RESULTS_ROOT / "purush" / today / "openvla-baseline.json"
    data = detailed.finalize()
    data["median_ms_of_runs"] = median_ms
    data["p90_ms_of_runs"] = p90_ms
    data["run_count"] = args.runs
    import json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2))
    print(f"[+] Wrote {output}")


if __name__ == "__main__":
    main()
