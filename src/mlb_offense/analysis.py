"""Leakage-aware exploratory analysis and grouped regression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "wrc_plus"
CONTEXT_NUMERIC = ("age",)
CONTEXT_CATEGORICAL = ("season", "bats")
DISCIPLINE_FEATURES = (
    "walk_rate",
    "strikeout_rate",
    "swing_rate",
    "chase_rate",
    "contact_rate",
    "zone_contact_rate",
    "swinging_strike_rate",
)
CONTACT_FEATURES = (
    "barrel_rate",
    "hard_hit_rate",
    "avg_exit_velocity",
    "avg_launch_angle",
)
BAT_TRACKING_FEATURES = (
    "avg_bat_speed",
    "fast_swing_rate",
    "swing_length",
    "squared_up_contact_rate",
    "squared_up_swing_rate",
    "blast_contact_rate",
    "blast_swing_rate",
    "whiff_rate",
    "bbe_per_swing",
)
STANCE_FEATURES = (
    "swing_tilt",
    "attack_angle",
    "attack_direction",
    "ideal_attack_angle_rate",
    "distance_off_plate",
    "depth_in_box",
    "intercept_y_vs_plate",
    "intercept_y_vs_batter",
)

# Do not allow production metrics or components that directly recreate wRC+.
LEAKAGE_COLUMNS = frozenset(
    {
        "ops",
        "ops_plus",
        "woba",
        "xwoba",
        "fangraphs_woba",
        "fangraphs_xwoba",
        "wrc_plus",
        "batter_run_value",
    }
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Settings for same-season, player-grouped model evaluation."""

    target: str = TARGET
    min_pa: int = 100
    min_competitive_swings: int = 50
    outer_splits: int = 5
    inner_splits: int = 3
    random_state: int = 42
    bootstrap_iterations: int = 500


@dataclass(frozen=True)
class FeatureSet:
    """A labelled predictor set used for incremental model comparisons."""

    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = CONTEXT_CATEGORICAL

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


def _available(columns: Iterable[str], data: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in columns if column in data.columns)


def feature_sets(data: pd.DataFrame) -> dict[str, FeatureSet]:
    """Return reproducible predictor blocks that exclude target leakage."""

    core_numeric = _available(
        CONTEXT_NUMERIC + DISCIPLINE_FEATURES + CONTACT_FEATURES, data
    )
    trait_numeric = _available(BAT_TRACKING_FEATURES + STANCE_FEATURES, data)
    categorical = _available(CONTEXT_CATEGORICAL, data)
    return {
        "core": FeatureSet("core", core_numeric, categorical),
        "core_plus_traits": FeatureSet(
            "core_plus_traits", core_numeric + trait_numeric, categorical
        ),
    }


def load_analysis_data(
    path: Path | str,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Load a processed dataset and apply the local analytic eligibility rule."""

    data = pd.read_parquet(path)
    required = {
        "player_id",
        "season",
        "pa",
        "competitive_swings",
        config.target,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required analysis columns: {missing}")

    eligible = data.loc[
        data["pa"].ge(config.min_pa)
        & data["competitive_swings"].ge(config.min_competitive_swings)
        & data[config.target].notna()
    ].copy()
    eligible = eligible.dropna(subset=["player_id", "season"])
    eligible["player_id"] = eligible["player_id"].astype(int)
    eligible["season"] = eligible["season"].astype(int)
    eligible["observation_id"] = np.arange(len(eligible))

    if eligible["player_id"].nunique() < 2:
        raise ValueError("At least two players are required for grouped evaluation.")
    return eligible.reset_index(drop=True)


def coverage_by_season(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize row counts and coverage for core outcome and trait fields."""

    fields = (
        "wrc_plus",
        "ops_plus",
        "woba",
        "xwoba",
        "avg_bat_speed",
        "attack_angle",
        "distance_off_plate",
    )
    available_fields = [field for field in fields if field in data.columns]
    coverage = (
        data.groupby("season", dropna=False)
        .agg(
            player_seasons=("player_id", "size"),
            unique_players=("player_id", "nunique"),
            median_pa=("pa", "median"),
            median_competitive_swings=("competitive_swings", "median"),
        )
        .reset_index()
    )
    for field in available_fields:
        available = data.groupby("season")[field].apply(lambda value: value.notna().mean())
        coverage[f"{field}_coverage"] = coverage["season"].map(available)
    return coverage


def missingness_by_season(data: pd.DataFrame) -> pd.DataFrame:
    """Return a long table suitable for visualising structural missingness."""

    metadata = {"player_id", "player_name", "season", "retrieved_on", "observation_id"}
    columns = [
        column
        for column in data.columns
        if column not in metadata and data[column].isna().any()
    ]
    missing = (
        data.groupby("season")[columns]
        .apply(lambda frame: frame.isna().mean())
        .stack()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"level_1": "metric"})
    )
    return missing


def correlation_matrix(data: pd.DataFrame, features: FeatureSet) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for numeric predictors and outcome."""

    columns = list(features.numeric) + [TARGET]
    columns = [column for column in columns if column in data]
    return data[columns].corr(numeric_only=True)


def _preprocessor(features: FeatureSet) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(features.numeric)),
            ("categorical", categorical_pipeline, list(features.categorical)),
        ],
        sparse_threshold=0,
    )


def _model_specifications(random_state: int) -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "mean_baseline": (DummyRegressor(strategy="mean"), {}),
        "ols": (LinearRegression(), {}),
        "ridge": (Ridge(), {"model__alpha": [1.0, 10.0, 100.0]}),
        "elastic_net": (
            ElasticNet(max_iter=25_000, random_state=random_state),
            {
                "model__alpha": [0.01, 0.1, 1.0],
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=150,
                max_features=0.7,
                random_state=random_state,
                n_jobs=1,
            ),
            {
                "model__max_depth": [3, 6],
                "model__min_samples_leaf": [5, 15],
            },
        ),
    }


def _n_splits(requested: int, groups: pd.Series) -> int:
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("Grouped evaluation needs at least two unique players.")
    return min(requested, unique_groups)


def _fit_estimator(
    pipeline: Pipeline,
    parameters: dict[str, list[object]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    config: AnalysisConfig,
) -> tuple[Pipeline, dict[str, object]]:
    if not parameters:
        pipeline.fit(x_train, y_train)
        return pipeline, {}

    inner_cv = GroupKFold(
        n_splits=_n_splits(config.inner_splits, groups_train)
    )
    search = GridSearchCV(
        pipeline,
        parameters,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(x_train, y_train, groups=groups_train)
    return search.best_estimator_, search.best_params_


def _metric_row(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    if np.isclose(np.ptp(predicted), 0):
        calibration_intercept = float("nan")
        calibration_slope = float("nan")
    else:
        calibration = LinearRegression().fit(predicted.reshape(-1, 1), actual)
        calibration_intercept = float(calibration.intercept_)
        calibration_slope = float(calibration.coef_[0])
    return {
        "mae": float(mean_absolute_error(actual, predicted, sample_weight=weights)),
        "rmse": float(
            root_mean_squared_error(actual, predicted, sample_weight=weights)
        ),
        "r2": float(r2_score(actual, predicted, sample_weight=weights)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def evaluate_models(
    data: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate model families with player-grouped nested cross-validation.

    Returns a performance table, out-of-fold prediction table, and selected
    hyperparameters. The results describe same-season association, not a
    next-season forecast.
    """

    if config.target not in data:
        raise ValueError(f"Target column {config.target!r} is absent.")
    if any(column in LEAKAGE_COLUMNS for column in feature_sets(data)["core"].columns):
        raise AssertionError("A production metric entered the core feature set.")

    models = _model_specifications(config.random_state)
    feature_blocks = feature_sets(data)
    outer_cv = GroupKFold(n_splits=_n_splits(config.outer_splits, data["player_id"]))
    performance_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []

    # The baseline has no meaningful feature block and is evaluated once.
    evaluations = [
        ("baseline", FeatureSet("baseline", (), ()), ("mean_baseline", models["mean_baseline"]))
    ]
    for block_name, features in feature_blocks.items():
        for model_name, specification in models.items():
            if model_name != "mean_baseline":
                evaluations.append((block_name, features, (model_name, specification)))

    for block_name, features, (model_name, (estimator, parameters)) in evaluations:
        all_predictions: list[pd.DataFrame] = []
        for fold, (train_index, test_index) in enumerate(
            outer_cv.split(data, groups=data["player_id"]), start=1
        ):
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            x_train = train.loc[:, list(features.columns)]
            x_test = test.loc[:, list(features.columns)]
            y_train = train[config.target]
            y_test = test[config.target]
            if model_name == "mean_baseline":
                pipeline = Pipeline([("model", estimator)])
            else:
                pipeline = Pipeline(
                    [
                        ("preprocess", _preprocessor(features)),
                        ("model", estimator),
                    ]
                )
            fitted, best_params = _fit_estimator(
                pipeline,
                parameters,
                x_train,
                y_train,
                train["player_id"],
                config,
            )
            predicted = fitted.predict(x_test)
            model_key = f"{model_name}:{block_name}"
            fold_predictions = pd.DataFrame(
                {
                    "observation_id": test["observation_id"].to_numpy(),
                    "player_id": test["player_id"].to_numpy(),
                    "season": test["season"].to_numpy(),
                    "pa": test["pa"].to_numpy(),
                    "actual": y_test.to_numpy(),
                    "predicted": predicted,
                    "model": model_key,
                    "fold": fold,
                }
            )
            all_predictions.append(fold_predictions)
            parameter_rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "best_parameters": best_params,
                }
            )

        predictions = pd.concat(all_predictions, ignore_index=True)
        prediction_rows.append(predictions)
        unweighted = _metric_row(
            predictions["actual"].to_numpy(), predictions["predicted"].to_numpy()
        )
        weighted = _metric_row(
            predictions["actual"].to_numpy(),
            predictions["predicted"].to_numpy(),
            predictions["pa"].to_numpy(),
        )
        performance_rows.append(
            {
                "model": model_key,
                "feature_block": block_name,
                "model_family": model_name,
                **unweighted,
                **{f"pa_weighted_{key}": value for key, value in weighted.items()},
                "n_observations": len(predictions),
                "n_players": predictions["player_id"].nunique(),
            }
        )

    performance = pd.DataFrame(performance_rows).sort_values(
        "mae", ignore_index=True
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    return performance, predictions, parameters


def bootstrap_model_difference(
    predictions: pd.DataFrame,
    candidate_model: str,
    reference_model: str,
    metric: str = "mae",
    iterations: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate player-clustered uncertainty for a model-performance difference."""

    if metric not in {"mae", "rmse", "r2"}:
        raise ValueError("metric must be one of: mae, rmse, r2")
    selected = predictions[
        predictions["model"].isin([candidate_model, reference_model])
    ]
    paired = selected.pivot(
        index=["observation_id", "player_id"], columns="model", values=["actual", "predicted", "pa"]
    )
    if candidate_model not in paired["predicted"] or reference_model not in paired["predicted"]:
        raise ValueError("Both models need predictions for every evaluated observation.")

    actual = paired["actual"].iloc[:, 0]
    pa = paired["pa"].iloc[:, 0]
    candidates = paired["predicted"][candidate_model]
    references = paired["predicted"][reference_model]
    paired_frame = pd.DataFrame(
        {
            "player_id": paired.index.get_level_values("player_id"),
            "actual": actual.to_numpy(),
            "candidate": candidates.to_numpy(),
            "reference": references.to_numpy(),
            "pa": pa.to_numpy(),
        }
    )
    groups = paired_frame["player_id"].unique()
    rng = np.random.default_rng(random_state)

    def calculate(frame: pd.DataFrame, prediction_column: str) -> float:
        if metric == "mae":
            return float(mean_absolute_error(frame["actual"], frame[prediction_column]))
        if metric == "rmse":
            return float(root_mean_squared_error(frame["actual"], frame[prediction_column]))
        return float(r2_score(frame["actual"], frame[prediction_column]))

    differences = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat(
            [
                paired_frame.loc[paired_frame["player_id"].eq(group)]
                for group in sampled_groups
            ],
            ignore_index=True,
        )
        differences.append(calculate(sample, "candidate") - calculate(sample, "reference"))

    return pd.DataFrame(
        {
            "metric": [metric],
            "candidate_model": [candidate_model],
            "reference_model": [reference_model],
            "difference": [calculate(paired_frame, "candidate") - calculate(paired_frame, "reference")],
            "ci_lower": [float(np.quantile(differences, 0.025))],
            "ci_upper": [float(np.quantile(differences, 0.975))],
            "iterations": [iterations],
        }
    )


def clustered_ols_coefficients(
    data: pd.DataFrame,
    features: FeatureSet,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Fit standardized OLS with player-clustered standard errors."""

    numeric = list(features.numeric)
    categorical = list(features.categorical)
    design = data[numeric + categorical].copy()
    for column in numeric:
        design[column] = pd.to_numeric(design[column], errors="coerce")
        design[column] = design[column].fillna(design[column].median())
        standard_deviation = design[column].std(ddof=0)
        if standard_deviation:
            design[column] = (design[column] - design[column].mean()) / standard_deviation
    design = pd.get_dummies(design, columns=categorical, drop_first=True, dtype=float)
    design = sm.add_constant(design, has_constant="add")
    fitted = sm.OLS(data[config.target], design).fit(
        cov_type="cluster", cov_kwds={"groups": data["player_id"]}
    )
    intervals = fitted.conf_int()
    return pd.DataFrame(
        {
            "feature": fitted.params.index,
            "coefficient": fitted.params.to_numpy(),
            "std_error": fitted.bse.to_numpy(),
            "ci_lower": intervals.iloc[:, 0].to_numpy(),
            "ci_upper": intervals.iloc[:, 1].to_numpy(),
            "p_value": fitted.pvalues.to_numpy(),
        }
    ).sort_values("coefficient", key=np.abs, ascending=False, ignore_index=True)


def grouped_permutation_importance(
    data: pd.DataFrame,
    features: FeatureSet,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Fit a constrained forest and return within-feature permutation importance.

    Importance is descriptive only: correlated traits can substitute for one
    another and should not be interpreted as isolated causal effects.
    """

    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(features)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=6,
                    min_samples_leaf=5,
                    max_features=0.7,
                    n_jobs=1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    x_data = data.loc[:, list(features.columns)]
    pipeline.fit(x_data, data[config.target])
    result = permutation_importance(
        pipeline,
        x_data,
        data[config.target],
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        random_state=config.random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)


def plot_outcome_distributions(data: pd.DataFrame, target: str = TARGET) -> plt.Figure:
    """Plot observed target distributions by season."""

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=data, x="season", y=target, inner="quartile", ax=axis)
    axis.set(
        title="Observed wRC+ distribution by season",
        xlabel="Season",
        ylabel="wRC+ (FanGraphs, league-adjusted rate index)",
    )
    return figure


def plot_missingness(data: pd.DataFrame) -> plt.Figure:
    """Plot non-missing coverage by season for high-level analysis fields."""

    fields = [
        "wrc_plus",
        "ops_plus",
        "woba",
        "xwoba",
        "avg_bat_speed",
        "attack_angle",
        "distance_off_plate",
    ]
    fields = [field for field in fields if field in data]
    coverage = (
        data.groupby("season")[fields].apply(lambda frame: frame.notna().mean())
        .T.reset_index(names="metric")
        .melt(id_vars="metric", var_name="season", value_name="coverage")
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=coverage, x="metric", y="coverage", hue="season", ax=axis)
    axis.set(
        title="Metric coverage among model-eligible player-seasons",
        xlabel="Metric",
        ylabel="Non-missing share",
        ylim=(0, 1.05),
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend(title="Season")
    return figure


def plot_correlation(data: pd.DataFrame, features: FeatureSet) -> plt.Figure:
    """Plot the correlation structure of the selected numerical predictor block."""

    matrix = correlation_matrix(data, features)
    figure, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis)
    axis.set(title="Pearson correlation among model predictors and wRC+")
    return figure


def plot_predictions(
    predictions: pd.DataFrame,
    model: str,
) -> plt.Figure:
    """Plot observed versus out-of-fold predicted production."""

    frame = predictions.loc[predictions["model"].eq(model)].copy()
    figure, axis = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(data=frame, x="predicted", y="actual", hue="season", alpha=0.65, ax=axis)
    low = min(frame["predicted"].min(), frame["actual"].min())
    high = max(frame["predicted"].max(), frame["actual"].max())
    axis.plot([low, high], [low, high], color="black", linestyle="--", label="Perfect calibration")
    axis.set(
        title=f"Out-of-fold wRC+ predictions: {model}",
        xlabel="Predicted wRC+",
        ylabel="Observed wRC+",
    )
    axis.legend(title="Season")
    return figure


def plot_residual_diagnostics(
    predictions: pd.DataFrame,
    model: str,
    data: pd.DataFrame,
) -> plt.Figure:
    """Plot residuals against fitted values, PA, age, and season."""

    prediction_frame = predictions.loc[predictions["model"].eq(model)].copy()
    merged = prediction_frame.merge(
        data[["observation_id", "age"]],
        on="observation_id",
        how="left",
        validate="one_to_one",
    )
    merged["residual"] = merged["actual"] - merged["predicted"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    plots: list[tuple[str, str, str]] = [
        ("predicted", "Predicted wRC+", "Residual versus fitted wRC+"),
        ("pa", "Plate appearances", "Residual versus plate appearances"),
        ("age", "Age (years)", "Residual versus age"),
        ("season", "Season", "Residual versus season"),
    ]
    for axis, (column, label, title) in zip(axes.ravel(), plots, strict=True):
        sns.scatterplot(data=merged, x=column, y="residual", alpha=0.55, ax=axis)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel=label, ylabel="Observed minus predicted wRC+")
    figure.tight_layout()
    return figure
