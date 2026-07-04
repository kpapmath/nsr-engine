"""Run the nsr-engine full pipeline from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from nsr_engine import NSREngine
from nsr_engine.pipeline import (
    evaluate_with_sympy,
    load_csv_dataset,
    make_dataset,
    train_test_validation_split,
    validate_split_fractions,
)


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _none_or_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
    disable_help_text: str,
) -> None:
    dashed = name.replace("_", "-")
    parser.add_argument(
        f"--{dashed}",
        dest=name,
        action="store_true",
        default=default,
        help=help_text,
    )
    parser.add_argument(
        f"--no-{dashed}",
        dest=name,
        action="store_false",
        help=disable_help_text,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    data = parser.add_argument_group("data")
    data.add_argument("--input-csv", type=Path, default=None)
    data.add_argument("--target-col", default=None)
    data.add_argument(
        "--feature-cols",
        type=_csv_list,
        default=None,
        help="Comma-separated feature columns. Defaults to all non-target columns.",
    )
    data.add_argument("--rows", type=int, default=3000)
    data.add_argument("--seed", type=int, default=7)

    split = parser.add_argument_group("split")
    split.add_argument("--train-frac", type=float, default=0.8)
    split.add_argument("--test-frac", type=float, default=0.2)
    split.add_argument("--validation-frac", type=float, default=None)

    engine = parser.add_argument_group("engine")
    engine.add_argument(
        "--lambda-grid",
        type=_float_grid,
        default=None,
        help="Comma-separated lambda values. Overrides n/lambda min/max.",
    )
    engine.add_argument("--lambdas", "--n-lambda", dest="n_lambda", type=int, default=10)
    engine.add_argument("--lambda-min", type=float, default=1e-4)
    engine.add_argument("--lambda-max", type=float, default=1e-1)
    engine.add_argument("--iters", "--n-iters", dest="n_iters", type=int, default=200)
    engine.add_argument("--batch-size", type=int, default=64)
    engine.add_argument("--max-len", type=int, default=15)
    engine.add_argument("--elite-frac", type=float, default=0.05)
    engine.add_argument("--entropy-weight", type=float, default=0.005)
    engine.add_argument("--hidden-dim", type=int, default=128)
    engine.add_argument("--embed-dim", type=int, default=32)
    engine.add_argument("--lr", type=float, default=1e-3)
    engine.add_argument("--random-state", type=int, default=None)
    engine.add_argument("--cache-dir", type=Path, default=None)
    engine.add_argument("--cache-prefix", default="full_pipeline")
    engine.add_argument("--binary-ops", type=_csv_list, default=None)
    engine.add_argument("--unary-ops", type=_csv_list, default=None)
    engine.add_argument("--const-tokens", type=_csv_list, default=None)
    engine.add_argument("--device", default="auto")
    engine.add_argument("--step-subsample-size", type=_none_or_int, default=None)
    _add_bool_arg(
        engine,
        "standardize",
        default=True,
        help_text="Z-score feature columns before training.",
        disable_help_text="Do not z-score feature columns before training.",
    )
    _add_bool_arg(
        engine,
        "affine_reward",
        default=True,
        help_text="Score expressions after least-squares affine fitting.",
        disable_help_text="Score raw expression predictions without affine fitting.",
    )
    engine.add_argument(
        "--metric",
        "--score-metric",
        dest="score_metric",
        default="mse",
    )
    engine.add_argument("--prefilter-per-complexity", type=int, default=16)

    args = parser.parse_args()

    if args.input_csv is not None and args.target_col is None:
        parser.error("--target-col is required when --input-csv is provided")

    try:
        validate_split_fractions(args.train_frac, args.test_frac, args.validation_frac)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main() -> None:
    args = parse_args()

    if args.input_csv is None:
        X, y = make_dataset(args.rows, args.seed)
    else:
        X, y = load_csv_dataset(
            str(args.input_csv),
            target_col=args.target_col,
            feature_cols=args.feature_cols,
        )

    X_train, X_test, X_validation, y_train, y_test, y_validation = (
        train_test_validation_split(
            X,
            y,
            train_frac=args.train_frac,
            test_frac=args.test_frac,
            validation_frac=args.validation_frac,
            seed=args.seed + 1,
        )
    )

    engine = NSREngine(
        lambda_grid=args.lambda_grid,
        n_lambda=args.n_lambda,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        max_len=args.max_len,
        elite_frac=args.elite_frac,
        entropy_weight=args.entropy_weight,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        lr=args.lr,
        random_state=args.seed if args.random_state is None else args.random_state,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        binary_ops=args.binary_ops,
        unary_ops=args.unary_ops,
        const_tokens=args.const_tokens,
        device=args.device,
        step_subsample_size=args.step_subsample_size,
        standardize=args.standardize,
        affine_reward=args.affine_reward,
        score_metric=args.score_metric,
        prefilter_per_complexity=args.prefilter_per_complexity,
    )

    front = engine.fit(X_train, y_train)
    if len(front) == 0:
        raise RuntimeError(
            "No valid expressions were discovered. Increase --iters or --max-len."
        )

    frame = front.to_frame()
    print("\nPareto front")
    print(frame.to_string(index=False))

    selected = front.elbow()
    print("\nSelected elbow formula")
    print(selected.equation)
    print(f"complexity={selected.complexity} score_metric={selected.score_metric}")

    if X_validation is not None and y_validation is not None:
        validation_rmse = evaluate_with_sympy(
            selected.sympy_expr, X_validation, y_validation
        )
        if validation_rmse is not None:
            print(f"validation_rmse={validation_rmse:.6f}")

    test_rmse = evaluate_with_sympy(selected.sympy_expr, X_test, y_test)
    if test_rmse is not None:
        print(f"test_rmse={test_rmse:.6f}")


if __name__ == "__main__":
    main()
