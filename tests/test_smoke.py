"""Smoke tests: import, fit on a tiny synthetic dataset, check front shape."""

import numpy as np
import pandas as pd
import pytest

from nsr_engine import NSREngine, ParetoFront
from nsr_engine.pipeline import (
    blocked_time_series_splits,
    expanding_window_splits,
    k_fold_splits,
    train_test_validation_split,
    validate_split_fractions,
    walk_forward_splits,
)
from nsr_engine.main import parse_args


def _make_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    X = pd.DataFrame({"a": a, "b": b})
    y = pd.Series(0.5 * a + 0.3 * b + 0.1 * rng.standard_normal(n))
    return X, y


def test_import():
    from nsr_engine import NSREngine, ParetoFront, ParetoPoint, SREngine  # noqa: F401


def test_fit_returns_pareto_front():
    X, y = _make_data()
    engine = NSREngine(
        n_lambda=2,
        n_iters=5,
        batch_size=16,
        max_len=5,
        random_state=0,
        standardize=True,
        affine_reward=True,
        boosting=False,  # the single-fit contract; the boosted path has its own tests
    )
    front = engine.fit(X, y)
    assert isinstance(front, ParetoFront)
    assert len(front) >= 0  # may be empty for very short runs


def test_score_metric_mae():
    engine = NSREngine(score_metric="mae", affine_reward=False)
    pred = np.array([1.0, 3.0, 6.0])
    y = np.array([0.0, 4.0, 3.0])

    assert engine._score(pred, y) == pytest.approx(5.0 / 3.0)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("mape", (0.1 + 0.1 + 0.1) / 3.0 * 100.0),
        ("mbd", abs((11.0 + 18.0 + 33.0 - 60.0) / 3.0)),
        ("r2", 0.93),
        ("adjusted_r2", 0.86),
    ],
)
def test_additional_score_metrics(metric, expected):
    engine = NSREngine(score_metric=metric, affine_reward=False)
    pred = np.array([11.0, 18.0, 33.0])
    y = np.array([10.0, 20.0, 30.0])

    assert engine._score(pred, y) == pytest.approx(expected)


def test_score_metric_validation():
    with pytest.raises(ValueError, match="score_metric"):
        NSREngine(score_metric="huber")


def test_train_test_split_defaults_without_validation():
    X, y = _make_data(n=10)

    X_train, X_test, X_validation, y_train, y_test, y_validation = (
        train_test_validation_split(
            X,
            y,
            train_frac=0.8,
            test_frac=0.2,
            validation_frac=None,
            seed=1,
        )
    )

    assert len(X_train) == 8
    assert len(y_train) == 8
    assert len(X_test) == 2
    assert len(y_test) == 2
    assert X_train.index.tolist() == list(range(8))
    assert X_train["a"].tolist() == X["a"].iloc[:8].tolist()
    assert X_test["a"].tolist() == X["a"].iloc[8:].tolist()
    assert X_validation is None
    assert y_validation is None


def test_train_test_split_can_shuffle():
    X, y = _make_data(n=10)

    X_train, X_test, _, y_train, y_test, _ = train_test_validation_split(
        X,
        y,
        train_frac=0.8,
        test_frac=0.2,
        validation_frac=None,
        seed=1,
        shuffle=True,
    )

    assert len(X_train) == 8
    assert len(y_train) == 8
    assert len(X_test) == 2
    assert len(y_test) == 2
    assert X_train["a"].tolist() != X["a"].iloc[:8].tolist()


def test_train_test_validation_split():
    X, y = _make_data(n=10)

    X_train, X_test, X_validation, y_train, y_test, y_validation = (
        train_test_validation_split(
            X,
            y,
            train_frac=0.7,
            test_frac=0.2,
            validation_frac=0.1,
            seed=1,
        )
    )

    assert len(X_train) == 7
    assert len(y_train) == 7
    assert len(X_test) == 2
    assert len(y_test) == 2
    assert X_validation is not None
    assert y_validation is not None
    assert len(X_validation) == 1
    assert len(y_validation) == 1
    assert X_train["a"].tolist() == X["a"].iloc[:7].tolist()
    assert X_validation["a"].tolist() == X["a"].iloc[7:8].tolist()
    assert X_test["a"].tolist() == X["a"].iloc[8:].tolist()


def test_k_fold_splits_preserve_order_by_default():
    X, y = _make_data(n=10)

    folds = list(k_fold_splits(X, y, n_splits=5, seed=1))

    assert len(folds) == 5
    assert folds[0].name == "k_fold_1"
    assert folds[0].X_eval["a"].tolist() == X["a"].iloc[:2].tolist()
    assert folds[-1].X_eval["a"].tolist() == X["a"].iloc[8:].tolist()
    assert len(folds[0].X_train) == 8
    assert len(folds[0].y_train) == 8
    assert len(folds[0].y_eval) == 2


def test_k_fold_splits_can_shuffle():
    X, y = _make_data(n=10)

    folds = list(k_fold_splits(X, y, n_splits=5, seed=1, shuffle=True))

    assert len(folds) == 5
    assert folds[0].X_eval["a"].tolist() != X["a"].iloc[:2].tolist()


def test_walk_forward_splits_expand_training_window():
    X, y = _make_data(n=10)

    folds = list(walk_forward_splits(X, y, n_splits=4))

    assert len(folds) == 4
    assert folds[0].name == "walk_forward_1"
    assert folds[0].X_train["a"].tolist() == X["a"].iloc[:2].tolist()
    assert folds[0].X_eval["a"].tolist() == X["a"].iloc[2:4].tolist()
    assert folds[1].X_train["a"].tolist() == X["a"].iloc[:4].tolist()
    assert folds[-1].X_eval["a"].tolist() == X["a"].iloc[8:].tolist()


def test_expanding_window_splits_use_explicit_name():
    X, y = _make_data(n=10)

    folds = list(expanding_window_splits(X, y, n_splits=4))

    assert len(folds) == 4
    assert folds[0].name == "expanding_window_1"
    assert folds[1].X_train["a"].tolist() == X["a"].iloc[:4].tolist()


def test_blocked_time_series_splits_use_adjacent_blocks():
    X, y = _make_data(n=10)

    folds = list(blocked_time_series_splits(X, y, n_splits=4))

    assert len(folds) == 4
    assert folds[0].name == "blocked_time_series_1"
    assert folds[0].X_train["a"].tolist() == X["a"].iloc[:2].tolist()
    assert folds[0].X_eval["a"].tolist() == X["a"].iloc[2:4].tolist()
    assert folds[1].X_train["a"].tolist() == X["a"].iloc[2:4].tolist()
    assert folds[1].X_eval["a"].tolist() == X["a"].iloc[4:6].tolist()
    assert folds[-1].X_eval["a"].tolist() == X["a"].iloc[8:].tolist()


def test_cli_validation_mode_defaults_to_none(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nsr-engine"])

    args = parse_args()

    assert args.validation_mode == "none"


def test_cli_boosts_residuals_by_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nsr-engine"])

    args = parse_args()

    assert args.boosting is True


def test_cli_no_boosting_opts_out_of_boosting(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nsr-engine", "--no-boosting"])

    args = parse_args()

    assert args.boosting is False


def test_cli_joint_refit_needs_the_default_boosting(monkeypatch):
    """Layer 3 consumes the boosted terms, so opting out of layer 1 is an error."""
    monkeypatch.setattr("sys.argv", ["nsr-engine", "--joint-refit"])
    assert parse_args().joint_refit is True

    monkeypatch.setattr("sys.argv", ["nsr-engine", "--joint-refit", "--no-boosting"])
    with pytest.raises(SystemExit):
        parse_args()


def test_cli_accepts_kfold_alias(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nsr-engine", "--validation-mode", "kfold"])

    args = parse_args()

    assert args.validation_mode == "k-fold"


@pytest.mark.parametrize(
    ("raw_mode", "expected"),
    [
        ("sequential-train-test", "sequential"),
        ("expanding", "expanding-window"),
        ("blocked", "blocked-time-series"),
    ],
)
def test_cli_accepts_time_series_validation_aliases(
    monkeypatch,
    raw_mode,
    expected,
):
    monkeypatch.setattr("sys.argv", ["nsr-engine", "--validation-mode", raw_mode])

    args = parse_args()

    assert args.validation_mode == expected


def test_split_fraction_range_validation():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        validate_split_fractions(1.2, -0.2, None)


def test_split_fraction_sum_validation():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_split_fractions(0.7, 0.2, None)


def test_pareto_front_dominance_filter():
    from nsr_engine.pareto import ParetoPoint

    pts = [
        ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=0.5),
        ParetoPoint(equation="b", sympy_expr=None, complexity=2, mse=0.3),
        ParetoPoint(equation="c", sympy_expr=None, complexity=2, mse=0.6),  # dominated
    ]
    front = ParetoFront(pts).dominance_filter()
    equations = {p.equation for p in front.points}
    assert "c" not in equations
    assert "a" in equations
    assert "b" in equations


def test_pareto_front_elbow():
    from nsr_engine.pareto import ParetoPoint

    pts = [
        ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=1.0),
        ParetoPoint(equation="b", sympy_expr=None, complexity=3, mse=0.2),
        ParetoPoint(equation="c", sympy_expr=None, complexity=5, mse=0.18),
    ]
    elbow = ParetoFront(pts).elbow()
    assert elbow.equation == "b"


# Token sequence that stalled a real `scale_mode="log"` fit (seed 7) for 6h40m:
# ten nested `tanh`, whose *construction* cost sympy 54 s before `simplify` --
# the phase the old `_SIMPLIFY_TIMEOUT_S` never covered.
_NESTED_TANH_TOKENS = [
    "tanh", "tanh", "tanh", "tanh", "tanh", "tanh",
    "-", "tanh", "tanh", "tanh", "tanh", "l",
    "-", "delta", "p",
]
_LOG_MEAN = {"p": -0.90, "l": -0.30, "delta": 2.60}
_LOG_STD = {"p": 0.75, "l": 1.10, "delta": 1.90}


def test_sympy_conversion_bounds_nested_unary_build(monkeypatch):
    """A candidate whose sympy *build* runs away is dropped, not waited on."""
    import time

    from nsr_engine import engine

    monkeypatch.setattr(engine, "_CONVERT_TIMEOUT_S", 0.5)
    start = time.monotonic()
    converted = engine._to_sympy_affine(
        _NESTED_TANH_TOKENS, 0.5, 1.2, _LOG_MEAN, _LOG_STD, feat_mode="log"
    )
    elapsed = time.monotonic() - start

    assert converted is None, "runaway build must be skipped, not returned"
    # Unbounded this takes ~54s; the budget is 0.5s. A generous ceiling keeps
    # the test from flaking on a loaded machine while still failing loudly if
    # the bound is removed.
    assert elapsed < 10.0, f"build was not bounded: took {elapsed:.1f}s"


def test_sympy_conversion_still_converts_benign_candidate(monkeypatch):
    """The build bound must not cost ordinary candidates their conversion."""
    from nsr_engine import engine

    monkeypatch.setattr(engine, "_CONVERT_TIMEOUT_S", 0.5)
    converted = engine._to_sympy_affine(
        ["+", "p", "l"], 0.0, 1.0, _LOG_MEAN, _LOG_STD, feat_mode="log"
    )

    assert converted is not None
    eq_str, expr = converted
    assert "log" in eq_str


def test_time_budget_runs_unbounded_when_disabled():
    """A non-positive limit means "no bound", and says so via the yielded flag."""
    from nsr_engine.engine import _time_budget

    with _time_budget(0.0) as bounded:
        assert bounded is False


def test_pareto_front_elbow_empty_raises():
    with pytest.raises(ValueError, match="empty Pareto front"):
        ParetoFront([]).elbow()


def test_pareto_front_elbow_empty_quotes_reason():
    front = ParetoFront([], empty_reason="boosting round 1 rejected: no finite rows")
    with pytest.raises(ValueError, match="boosting round 1 rejected: no finite rows"):
        front.elbow()


def test_boosted_rejecting_every_round_returns_empty_front_with_reason():
    """A booster whose every round is rejected must explain itself.

    Round 1 is unconditionally kept *unless* the front is empty, so a weak
    learner that discovers nothing is the path that leaves `points` untouched.
    The empty front must then say which round failed and why, rather than
    leaving `elbow()` to raise a bare IndexError from the library's insides.
    """
    from nsr_engine.boosting import ResidualBoostedNSR

    X, y = _make_data(n=50)
    booster = ResidualBoostedNSR(
        lambda k: _EmptyFrontEngine(), max_rounds=3, min_gain=0.01
    )
    front = booster.fit(X, y)

    assert len(front) == 0
    assert booster.rounds_[-1]["reason"] == "empty front"
    with pytest.raises(ValueError, match="boosting round 1 rejected: empty front"):
        front.elbow()


class _EmptyFrontEngine:
    """Weak learner that discovers nothing, as a failed NSR round does."""

    def fit(self, X, y):
        return ParetoFront([])


def test_residual_metric_accepts_every_engine_score_metric_but_mbd():
    """`residual_metric` must be able to follow `score_metric`."""
    from nsr_engine.boosting import _RESIDUAL_METRICS, ResidualBoostedNSR
    from nsr_engine.engine import _SCORE_METRICS

    assert set(_RESIDUAL_METRICS) == set(_SCORE_METRICS) - {"mbd"}
    for metric in _RESIDUAL_METRICS:
        booster = ResidualBoostedNSR(lambda k: None, residual_metric=metric)
        assert booster.residual_metric == metric


def test_residual_metric_rejects_mbd_with_a_reason():
    """The one metric that cannot drive the gain rule fails loudly."""
    from nsr_engine.boosting import ResidualBoostedNSR

    with pytest.raises(ValueError, match="cannot drive the round acceptance rule"):
        ResidualBoostedNSR(lambda k: None, residual_metric="mbd")


def test_boosted_front_is_scored_in_the_engines_metric():
    """`boosting=True` under a non-MSE metric reports that metric, not MSE.

    Before, anything outside mse/rmse fell back to MSE rounds and the round-1
    front was withheld from the merge, so the caller got a one-point front
    scored in a metric they never asked for.
    """
    X, y = _make_data(n=200)
    engine = NSREngine(
        n_lambda=2,
        n_iters=8,
        batch_size=16,
        max_len=7,
        random_state=0,
        score_metric="mape",
        boosting=True,
        boosting_max_rounds=3,
        device="cpu",
    )
    front = engine.fit(X, y)

    assert len(front) >= 1
    assert {p.score_metric for p in front.points} == {"mape"}


def test_r2_and_mse_gain_rules_agree():
    """`1 - r2` is proportional to MSE, so the two must accept the same rounds.

    This is the property that lets `r2` drive a *relative* gain threshold at
    all, so it is worth pinning rather than assuming.
    """
    from nsr_engine.boosting import ResidualBoostedNSR

    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    resid_before = y - 0.3 * y
    resid_after = y - 0.9 * y

    gains = {}
    for metric in ("mse", "r2"):
        booster = ResidualBoostedNSR(lambda k: None, residual_metric=metric)
        before = booster._loss_of(booster._score_of(resid_before, y, 1))
        after = booster._loss_of(booster._score_of(resid_after, y, 1))
        gains[metric] = (before - after) / before

    assert gains["mse"] == pytest.approx(gains["r2"], rel=1e-9)


def test_pareto_front_to_frame_uses_metric_column():
    from nsr_engine.pareto import ParetoPoint

    front = ParetoFront(
        [
            ParetoPoint(
                equation="a",
                sympy_expr=None,
                complexity=1,
                mse=0.2,
                score_metric="mae",
            )
        ]
    )

    assert list(front.to_frame().columns) == ["equation", "complexity", "mae"]


def test_pareto_front_maximizes_r2():
    from nsr_engine.pareto import ParetoPoint

    pts = [
        ParetoPoint(equation="a", sympy_expr=None, complexity=1, mse=0.7, score_metric="r2"),
        ParetoPoint(equation="b", sympy_expr=None, complexity=1, mse=0.8, score_metric="r2"),
    ]
    front = ParetoFront(pts).dominance_filter()

    assert [p.equation for p in front.points] == ["b"]
    assert front.to_frame()["r2"].iloc[0] == pytest.approx(0.8)


# ----------------------------------------------------------------------
# Feature scaling modes
# ----------------------------------------------------------------------


def _positive_frame(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.lognormal(2.0, 1.5, n).astype(np.float32),  # many decades
            "b": rng.uniform(0.5, 9.0, n).astype(np.float32),
        }
    )


def _scaled(engine: NSREngine, X: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays = {col: X[col].to_numpy(dtype=np.float32, copy=True) for col in X.columns}
    engine._set_stats_from_arrays(arrays)
    return engine._standardize_arrays(arrays, inplace=True)


@pytest.mark.parametrize("mode", ["scale", "minmax", "geometric"])
def test_scale_modes_keep_a_positive_column_positive(mode):
    """The point of the non-zscore modes: no centering out of the positive orthant."""
    X = _positive_frame()
    scaled = _scaled(NSREngine(scale_mode=mode), X)

    for col, arr in scaled.items():
        assert arr.min() > 0.0, f"{mode} sent column {col} non-positive"


@pytest.mark.parametrize("mode", ["zscore", "log"])
def test_centering_modes_do_not_keep_the_domain(mode):
    """The contrast the other three exist for — asserted so it cannot drift silently."""
    scaled = _scaled(NSREngine(scale_mode=mode), _positive_frame())

    assert min(arr.min() for arr in scaled.values()) < 0.0


def test_minmax_lands_on_the_requested_range():
    X = _positive_frame()
    scaled = _scaled(NSREngine(scale_mode="minmax", minmax_range=(0.25, 4.0)), X)

    for arr in scaled.values():
        assert arr.min() == pytest.approx(0.25, rel=1e-5)
        assert arr.max() == pytest.approx(4.0, rel=1e-5)


def test_scale_mode_scale_preserves_ratios():
    """Scale-only is multiplicative, so row-to-row ratios come through untouched."""
    X = _positive_frame()
    scaled = _scaled(NSREngine(scale_mode="scale"), X)

    raw = X["a"].to_numpy(dtype=np.float64)
    got = scaled["a"].astype(np.float64)
    assert np.allclose(got / got[0], raw / raw[0], rtol=1e-4)


def test_geometric_mode_centers_on_the_geometric_mean():
    scaled = _scaled(NSREngine(scale_mode="geometric"), _positive_frame())

    for arr in scaled.values():
        logs = np.log(arr.astype(np.float64))
        assert logs.mean() == pytest.approx(0.0, abs=1e-5)  # geometric mean 1
        assert logs.std() == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("mode", ["geometric", "log"])
def test_log_space_modes_reject_non_positive_columns(mode):
    engine = NSREngine(scale_mode=mode)
    arrays = {"a": np.array([1.0, 2.0, -3.0], dtype=np.float32)}

    with pytest.raises(ValueError, match="strictly positive"):
        engine._set_stats_from_arrays(arrays)


def test_scale_mode_validation():
    with pytest.raises(ValueError, match="scale_mode must be one of"):
        NSREngine(scale_mode="bogus")
    with pytest.raises(ValueError, match="lo < hi"):
        NSREngine(minmax_range=(2.0, 1.0))


@pytest.mark.parametrize("mode", ["zscore", "scale", "minmax", "geometric", "log"])
def test_scaled_expressions_convert_back_to_raw_features(mode):
    """The scaling must not leak into the reported equation, in any mode."""
    sp = pytest.importorskip("sympy")
    from nsr_engine.engine import _eval_prefix_numpy, _to_sympy_affine

    X = _positive_frame(n=200)
    tokens = ["+", "*", "a", "b", "log", "a"]
    b0, b1 = 0.37, 2.1

    engine = NSREngine(scale_mode=mode)
    scaled = _scaled(engine, X)
    on_scaled = b0 + b1 * _eval_prefix_numpy(tokens, scaled, len(X)).astype(np.float64)

    converted = _to_sympy_affine(
        tokens, b0, b1, engine._feat_mean, engine._feat_std, feat_mode=mode
    )
    assert converted is not None
    _, expr = converted
    fn = sp.lambdify((sp.Symbol("a"), sp.Symbol("b")), expr, "numpy")
    on_raw = np.asarray(
        fn(X["a"].to_numpy(np.float64), X["b"].to_numpy(np.float64)), dtype=np.float64
    )

    # float32 scaling vs float64 substitution, so relative rather than exact.
    assert np.allclose(on_raw, on_scaled, rtol=1e-3)


@pytest.mark.parametrize("mode", ["zscore", "scale", "minmax", "geometric", "log"])
def test_streaming_stats_match_in_memory_stats(mode):
    """The out-of-core accumulators duplicate the math; keep the two in step."""
    X = _positive_frame(n=997)

    class _ChunkStore:
        feature_cols = list(X.columns)

        def gather(self, sl):
            arrays = {c: X[c].to_numpy(np.float32)[sl].copy() for c in self.feature_cols}
            return arrays, np.zeros(1)

    in_memory = NSREngine(scale_mode=mode)
    _scaled(in_memory, X)
    streaming = NSREngine(scale_mode=mode)
    streaming._set_stats_streaming(_ChunkStore(), 0, len(X), 100)

    for col in X.columns:
        assert streaming._feat_mean[col] == pytest.approx(in_memory._feat_mean[col])
        assert streaming._feat_std[col] == pytest.approx(in_memory._feat_std[col])


def test_cli_scale_mode_defaults_to_zscore(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nsr-engine"])

    args = parse_args()

    assert args.scale_mode == "zscore"
    assert args.minmax_range == (1e-3, 1.0)


def test_cli_scale_mode_reaches_the_engine(monkeypatch):
    from nsr_engine.main import _build_engine

    monkeypatch.setattr(
        "sys.argv",
        ["nsr-engine", "--scale-mode", "minmax", "--minmax-range", "1,2"],
    )
    engine = _build_engine(parse_args())

    assert engine.standardize is True
    assert engine.scale_mode == "minmax"
    assert engine.minmax_range == (1.0, 2.0)


def test_cli_scale_mode_none_disables_scaling(monkeypatch):
    from nsr_engine.main import _build_engine

    monkeypatch.setattr("sys.argv", ["nsr-engine", "--scale-mode", "none"])

    assert _build_engine(parse_args()).standardize is False


def test_cli_rejects_an_inverted_minmax_range(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nsr-engine", "--minmax-range", "5,1"])

    with pytest.raises(SystemExit):
        parse_args()
