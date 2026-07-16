"""Run the nsr-engine full pipeline from the command line."""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

from nsr_engine import (
    NSREngine,
    ParetoFront,
    ParetoPoint,
    ResidualBoostedNSR,
    joint_refit_prune,
    optimize_constants,
    optimize_front,
)
from nsr_engine.pipeline import (
    blocked_time_series_splits,
    evaluate_with_sympy,
    expanding_window_splits,
    k_fold_splits,
    load_csv_dataset,
    make_dataset,
    train_test_validation_split,
    validate_split_fractions,
    walk_forward_splits,
)


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _none_or_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _validation_mode(value: str) -> str:
    normalized = value.lower().replace("_", "-")
    aliases = {
        "kfold": "k-fold",
        "k-fold": "k-fold",
        "blocked": "blocked-time-series",
        "blocked-time-series": "blocked-time-series",
        "blocked-timeseries": "blocked-time-series",
        "blocked-time": "blocked-time-series",
        "expanding": "expanding-window",
        "expanding-window": "expanding-window",
        "expandingwindow": "expanding-window",
        "walkforward": "walk-forward",
        "walk-forward": "walk-forward",
        "none": "none",
        "holdout": "holdout",
        "sequential": "sequential",
        "sequential-train-test": "sequential",
    }
    if normalized not in aliases:
        valid = ", ".join(
            [
                "none",
                "sequential",
                "holdout",
                "k-fold",
                "expanding-window",
                "walk-forward",
                "blocked-time-series",
            ]
        )
        raise argparse.ArgumentTypeError(f"invalid validation mode: {value}. Use {valid}.")
    return aliases[normalized]


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
    split.add_argument(
        "--validation-mode",
        type=_validation_mode,
        default="none",
        help=(
            "Validation strategy: none, sequential, holdout, k-fold, "
            "expanding-window, walk-forward, or blocked-time-series. "
            "Default uses the entire dataset for the final fit."
        ),
    )
    split.add_argument("--train-frac", type=float, default=0.8)
    split.add_argument("--test-frac", type=float, default=0.2)
    split.add_argument("--validation-frac", type=float, default=None)
    split.add_argument(
        "--folds",
        type=int,
        default=5,
        help=(
            "Number of folds for k-fold, expanding-window, walk-forward, "
            "or blocked-time-series validation."
        ),
    )
    _add_bool_arg(
        split,
        "shuffle",
        default=False,
        help_text="Shuffle rows before splitting.",
        disable_help_text="Preserve row order when splitting.",
    )

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

    layers = parser.add_argument_group("accuracy layers")
    _add_bool_arg(
        layers,
        "boosting",
        default=False,
        help_text=(
            "Fit additive terms by running the engine on the residual of the "
            "previous rounds (accuracy layer 1)."
        ),
        disable_help_text="Run a single engine fit (no residual boosting).",
    )
    layers.add_argument("--boosting-max-rounds", type=int, default=3)
    layers.add_argument("--boosting-min-gain", type=float, default=0.02)
    layers.add_argument(
        "--term-selection",
        choices=("elbow", "min_mse"),
        default="elbow",
        help="Which point of each boosting round's front becomes the term.",
    )
    _add_bool_arg(
        layers,
        "constant_opt",
        default=False,
        help_text="Refit float constants by least squares (accuracy layer 2).",
        disable_help_text="Leave discovered constants as the search found them.",
    )
    layers.add_argument(
        "--max-free-consts",
        type=int,
        default=12,
        help="Skip constant optimization for expressions with more free constants.",
    )
    layers.add_argument(
        "--max-nfev",
        type=int,
        default=200,
        help="Maximum residual evaluations per constant-optimization solve.",
    )
    _add_bool_arg(
        layers,
        "joint_refit",
        default=False,
        help_text=(
            "Re-weight the boosted terms jointly and prune redundant ones "
            "(accuracy layer 3; requires --boosting)."
        ),
        disable_help_text="Keep the boosted terms as greedily fitted.",
    )
    layers.add_argument(
        "--joint-refit-estimator",
        choices=("lasso_cv", "ols"),
        default="lasso_cv",
    )
    layers.add_argument("--coef-rel-tol", type=float, default=1e-3)
    # Shared by layers 2 and 3: both cap the rows they fit on.
    layers.add_argument(
        "--fit-subsample",
        type=int,
        default=8000,
        help=(
            "Maximum rows the constant fit (layer 2) and the joint refit "
            "(layer 3) see. 0 uses every row. Reported scores always use "
            "every row."
        ),
    )

    args = parser.parse_args()

    if args.input_csv is not None and args.target_col is None:
        parser.error("--target-col is required when --input-csv is provided")

    if args.joint_refit and not args.boosting:
        parser.error("--joint-refit requires --boosting (it consumes the boosted terms)")

    if args.boosting and args.boosting_max_rounds < 1:
        parser.error("--boosting-max-rounds must be at least 1")

    if args.validation_mode in {"holdout", "sequential"}:
        try:
            validate_split_fractions(
                args.train_frac, args.test_frac, args.validation_frac
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.validation_frac is not None:
        parser.error(
            "--validation-frac is only used with --validation-mode sequential or holdout"
        )

    if (
        args.validation_mode
        in {"k-fold", "expanding-window", "walk-forward", "blocked-time-series"}
        and args.folds < 2
    ):
        parser.error(
            "--folds must be at least 2 for k-fold, expanding-window, "
            "walk-forward, or blocked-time-series validation"
        )

    return args


def _build_engine(args: argparse.Namespace, *, cache_prefix: str | None = None) -> NSREngine:
    return NSREngine(
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
        cache_prefix=args.cache_prefix if cache_prefix is None else cache_prefix,
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


def _fit_front(args: argparse.Namespace, X, y, *, cache_prefix: str | None = None):
    """Fit the engine, applying whichever accuracy layers are enabled."""
    const_opt_kwargs = {
        "max_nfev": args.max_nfev,
        "seed": args.seed,
        "max_free_consts": args.max_free_consts,
        "fit_subsample": args.fit_subsample,
    }
    refiner = (
        functools.partial(optimize_constants, **const_opt_kwargs)
        if args.constant_opt
        else None
    )

    if not args.boosting:
        front = _build_engine(args, cache_prefix=cache_prefix).fit(X, y)
        if not args.constant_opt:
            return front
        return optimize_front(front, X, y, **const_opt_kwargs)

    if args.score_metric != "mse":
        print(
            f"[nsr] note: boosted front points are scored in MSE, not "
            f"{args.score_metric!r} (terms are still selected by "
            f"{args.score_metric!r})."
        )

    def engine_factory(round_idx: int) -> NSREngine:
        prefix = cache_prefix if cache_prefix is not None else args.cache_prefix
        engine = _build_engine(args, cache_prefix=f"{prefix}_round{round_idx}")
        # A fresh engine per round, with a seed that moves, so rounds do not
        # repeat the same search on the residual.
        engine.random_state += round_idx
        return engine

    booster = ResidualBoostedNSR(
        engine_factory,
        max_rounds=args.boosting_max_rounds,
        min_gain=args.boosting_min_gain,
        term_refiner=refiner,
        term_selection=args.term_selection,
    )
    front = booster.fit(X, y)
    for record in booster.rounds_:
        print(
            f"[nsr] boost round {record['round']}: added={record['added']} "
            f"gain={record['gain']:.4f} cum_mse={record['cum_mse']:.6f} "
            f"({record['reason']})"
        )

    if not args.joint_refit or not booster.terms_:
        return front

    refined = joint_refit_prune(
        booster.terms_,
        X,
        y,
        coef_rel_tol=args.coef_rel_tol,
        seed=args.seed,
        estimator=args.joint_refit_estimator,
        fit_subsample=args.fit_subsample,
        polish=refiner,
    )
    if refined is None:
        print("[nsr] joint refit produced no usable model — keeping boosted front")
        return front

    expr, complexity, mse = refined
    # The polished sum is flattened, so its sympy structure no longer reveals
    # how many basis terms survived; complexity does, since both sides use the
    # boosting convention.
    summed = sum(c for _, c in booster.terms_) + (len(booster.terms_) - 1)
    if complexity < summed:
        print(f"[nsr] joint refit pruned terms (complexity {summed} -> {complexity})")
    points = list(front.points)
    points.append(
        ParetoPoint(equation=str(expr), sympy_expr=expr, complexity=complexity, mse=mse)
    )
    return ParetoFront(points).dominance_filter()


def _print_selected_front(front, *, title: str) -> object:
    if len(front) == 0:
        raise RuntimeError(
            "No valid expressions were discovered. Increase --iters or --max-len."
        )

    frame = front.to_frame()
    print(f"\n{title}")
    print(frame.to_string(index=False))

    selected = front.elbow()
    print("\nSelected elbow formula")
    print(selected.equation)
    print(f"complexity={selected.complexity} score_metric={selected.score_metric}")
    return selected


def _run_validation_folds(args: argparse.Namespace, X, y) -> None:
    if args.validation_mode == "k-fold":
        folds = list(
            k_fold_splits(
                X,
                y,
                n_splits=args.folds,
                seed=args.seed + 1,
                shuffle=args.shuffle,
            )
        )
    elif args.validation_mode == "expanding-window":
        folds = list(expanding_window_splits(X, y, n_splits=args.folds))
    elif args.validation_mode == "walk-forward":
        folds = list(walk_forward_splits(X, y, n_splits=args.folds))
    elif args.validation_mode == "blocked-time-series":
        folds = list(blocked_time_series_splits(X, y, n_splits=args.folds))
    else:
        return

    fold_rmses: list[float] = []
    print(f"\n{args.validation_mode} validation")
    for fold in folds:
        front = _fit_front(
            args,
            fold.X_train,
            fold.y_train,
            cache_prefix=f"{args.cache_prefix}_{fold.name}",
        )
        if len(front) == 0:
            print(f"{fold.name}_rmse=nan")
            continue
        selected = front.elbow()
        fold_rmse = evaluate_with_sympy(selected.sympy_expr, fold.X_eval, fold.y_eval)
        if fold_rmse is not None:
            fold_rmses.append(fold_rmse)
            print(f"{fold.name}_rmse={fold_rmse:.6f}")

    if fold_rmses:
        mean_rmse = sum(fold_rmses) / len(fold_rmses)
        print(f"mean_validation_rmse={mean_rmse:.6f}")


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

    if args.validation_mode in {"holdout", "sequential"}:
        split_shuffle = args.shuffle and args.validation_mode == "holdout"
        X_train, X_test, X_validation, y_train, y_test, y_validation = (
            train_test_validation_split(
                X,
                y,
                train_frac=args.train_frac,
                test_frac=args.test_frac,
                validation_frac=args.validation_frac,
                seed=args.seed + 1,
                shuffle=split_shuffle,
            )
        )
        front = _fit_front(args, X_train, y_train)
        selected = _print_selected_front(front, title="Pareto front")

        if X_validation is not None and y_validation is not None:
            validation_rmse = evaluate_with_sympy(
                selected.sympy_expr, X_validation, y_validation
            )
            if validation_rmse is not None:
                print(f"validation_rmse={validation_rmse:.6f}")

        test_rmse = evaluate_with_sympy(selected.sympy_expr, X_test, y_test)
        if test_rmse is not None:
            print(f"test_rmse={test_rmse:.6f}")
        return

    _run_validation_folds(args, X, y)

    front = _fit_front(args, X, y)
    _print_selected_front(front, title="Pareto front")


if __name__ == "__main__":
    main()
