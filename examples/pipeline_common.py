"""Shared helpers for the standalone pipeline case examples."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    import sys

    sys.path.insert(0, str(SRC_DIR))

from nsr_engine import NSREngine, ParetoFront


def small_engine(
    *,
    seed: int,
    n_iters: int,
    n_lambda: int,
    cache_dir: Path | None = None,
    **overrides: object,
) -> NSREngine:
    """Build a fast example engine with conservative CPU-sized defaults.

    Any default here can be replaced through ``overrides``.
    """
    defaults: dict[str, object] = {
        "batch_size": 24,
        "max_len": 7,
        "hidden_dim": 32,
        "embed_dim": 12,
        "prefilter_per_complexity": 4,
    }
    return NSREngine(
        n_lambda=n_lambda,
        n_iters=n_iters,
        random_state=seed,
        cache_dir=cache_dir,
        cache_prefix="pipeline_examples",
        **{**defaults, **overrides},
    )


def print_front(label: str, front: ParetoFront) -> None:
    print(f"\n[{label}] Pareto front")
    if len(front) == 0:
        print("No expressions found; increase --iters or --max-len for this case.")
        return
    print(front.to_frame().to_string(index=False))
    print(f"elbow: {front.elbow().equation}")


def add_common_args(parser) -> None:
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--lambdas", type=int, default=2)
