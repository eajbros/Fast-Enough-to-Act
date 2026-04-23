"""
Shared GPU timing utility for 240D benchmarks.

Uses torch.cuda.Event for accurate GPU-side timing with negligible overhead
(~0.1 microseconds per record). Outputs structured JSON that plot_roofline.py
and downstream analysis scripts can ingest uniformly across teammates.

Typical use:

    from src.timing import Timer

    t = Timer(paradigm="vla",
              model="openvla-7b-libero-spatial",
              config={"precision": "int4", "batch": 1, "seq_len": 295})

    with t.phase("vision_encode"):
        features = model.vision(image)
    with t.phase("prefill"):
        prefill = model.llm_prefill(features, prompt_tokens)
    with t.phase("decode"):
        actions = model.llm_decode(prefill, n_new_tokens=7)

    t.save("/srv/240d/results/latency/purush/2026-04-26/openvla-baseline.json")
    print(f"total: {t.total_ms:.2f} ms")
"""
from contextlib import contextmanager
from pathlib import Path
import json
import platform
import socket
import time
import torch


class Timer:
    def __init__(self, paradigm: str, model: str, config: dict | None = None, warmup_runs: int = 3):
        """
        paradigm: 'vla' | 'diffusion' (categorizes results for the roofline plot)
        model:    model identifier, e.g. 'openvla-7b-libero-spatial'
        config:   dict of knobs you want recorded with the run (precision, batch, DDIM steps, ...)
        warmup_runs: used by caller as guidance; this class records phases, not warmup policy
        """
        assert paradigm in ("vla", "diffusion"), f"paradigm must be 'vla' or 'diffusion', got {paradigm!r}"
        self.paradigm = paradigm
        self.model = model
        self.config = dict(config or {})
        self.warmup_runs = warmup_runs
        self.phases: list[dict] = []

    @contextmanager
    def phase(self, name: str):
        """Time everything inside the `with` block on the GPU. Synchronizes
        before start and after end so CPU-side code outside the block doesn't
        pollute the measurement."""
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            self.phases.append({"name": name, "elapsed_ms": start.elapsed_time(end)})

    @property
    def total_ms(self) -> float:
        return sum(p["elapsed_ms"] for p in self.phases)

    def finalize(self) -> dict:
        return {
            "paradigm": self.paradigm,
            "model": self.model,
            "config": self.config,
            "phases": self.phases,
            "total_ms": self.total_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "warmup_runs": self.warmup_runs,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.finalize(), indent=2))
        return path
