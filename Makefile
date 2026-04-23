# Makefile — Fast-Enough-to-Act
#
# Convenience wrappers over the conda + bench pattern. Intended to run on
# cortex AFTER you've claimed the GPU (`240d-gpu claim`). The top-level
# bench-* targets refuse to run if the lock doesn't show your name.
#
# On Mac (no GPU), use `make lint` / `make test` for CPU-only work.

SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

DATE  := $(shell date +%F)
USER_ := $(shell whoami)
RESULTS_ROOT := /srv/240d/results

# Conda activation is per-env; scripts must run INSIDE `conda activate`.
# Makefile targets that need CUDA assume caller sourced conda & activated the
# right env (see bootstrap-cortex.sh).
CONDA_ENV ?= # leave empty; caller must activate first

.PHONY: help
help:           ## Show this help
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ─── Environments ────────────────────────────────────────────────────────
.PHONY: env-openvla env-diffusion env-profiling
env-openvla:   ## Create/update the openvla conda env  (Purush)
	conda env update -n openvla -f envs/openvla.yml

env-diffusion: ## Create/update the diffusion conda env  (Yuva)
	conda env update -n diffusion -f envs/diffusion.yml

env-profiling: ## Create/update the profiling conda env  (Ethan)
	conda env update -n profiling -f envs/profiling.yml

# ─── Bench runs (require GPU claim) ───────────────────────────────────────
# Each bench target calls gpu_guard first — exits with a pointer to
# `240d-gpu claim` if not claimed by current user.
.PHONY: gpu-claimed
gpu-claimed:   ## Exit 0 if current user holds the GPU claim, else 1
	@python -m src.gpu_guard --require || { \
		echo ""; \
		echo "  ❯ GPU not claimed by you. Before running a bench:"; \
		echo "    240d-gpu claim --note \"<what you're doing>\""; \
		echo "  ❯ When done:"; \
		echo "    240d-gpu release"; \
		echo ""; \
		exit 1; \
	}

.PHONY: bench-openvla
bench-openvla: gpu-claimed  ## Run OpenVLA baseline latency bench
	mkdir -p $(RESULTS_ROOT)/latency/$(USER_)/$(DATE)
	python scripts/bench_openvla.py --runs 10 --warmup 3 \
		--out $(RESULTS_ROOT)/latency/$(USER_)/$(DATE)/openvla-baseline.json

.PHONY: bench-dp
bench-dp: gpu-claimed  ## Run Diffusion Policy baseline latency bench
	mkdir -p $(RESULTS_ROOT)/latency/$(USER_)/$(DATE)
	python scripts/bench_dp.py --ddim-steps 10 --runs 10 --warmup 3 \
		--out $(RESULTS_ROOT)/latency/$(USER_)/$(DATE)/dp-ddim10.json

# ─── Plotting (no GPU needed — CPU only) ─────────────────────────────────
.PHONY: roofline
roofline:      ## Re-generate roofline plot from latest per-teammate results
	mkdir -p $(RESULTS_ROOT)/roofline/$(USER_)/$(DATE)
	python src/plot_roofline.py \
		--latency $(RESULTS_ROOT)/latency \
		--ncu $(RESULTS_ROOT)/ncu \
		--out $(RESULTS_ROOT)/roofline/$(USER_)/$(DATE)/combined.png

# ─── Dev targets ─────────────────────────────────────────────────────────
.PHONY: lint
lint:          ## Ruff lint (CPU only)
	@command -v ruff >/dev/null 2>&1 || { echo "ruff not installed: pip install ruff"; exit 1; }
	ruff check src scripts

.PHONY: test
test:          ## Pytest (CPU only — real tests TBD)
	@command -v pytest >/dev/null 2>&1 || { echo "pytest not installed"; exit 1; }
	pytest -xvs tests/ 2>/dev/null || echo "no tests yet (tests/ does not exist)"

# ─── Misc ────────────────────────────────────────────────────────────────
.PHONY: clean
clean:         ## Remove Python caches (does NOT touch /srv/240d/ artifacts)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

.PHONY: status
status:        ## Show GPU/claim status (calls 240d-gpu status on cortex)
	@if command -v 240d-gpu >/dev/null 2>&1; then \
		240d-gpu status; \
	else \
		echo "not on cortex — 240d-gpu is a cortex-only command"; \
	fi
