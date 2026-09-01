"""Leakage-aware exploratory analysis and grouped regression evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch
from scipy import stats
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
TRAIT_FEATURES = BAT_TRACKING_FEATURES + STANCE_FEATURES
RANKING_OUTCOMES = ("wrc_plus", "xwoba")

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

# Not swing/stance traits. They are either the production outcome itself or
# batted-ball quality sitting downstream of the swing. Ranking them against
# wRC+ would be circular (OPS, wOBA) or dominated by near-outcome contact
# results (barrels, exit velocity).
RANKING_EXCLUSIONS = LEAKAGE_COLUMNS | frozenset(CONTACT_FEATURES)

TRAIT_LABELS = {
    "avg_bat_speed": "Average bat speed",
    "fast_swing_rate": "Fast-swing rate",
    "swing_length": "Swing length",
    "squared_up_contact_rate": "Squared-up rate (contact)",
    "squared_up_swing_rate": "Squared-up rate (swings)",
    "blast_contact_rate": "Blast rate (contact)",
    "blast_swing_rate": "Blast rate (swings)",
    "whiff_rate": "Whiff rate",
    "bbe_per_swing": "Batted balls per swing",
    "swing_tilt": "Swing-path tilt",
    "attack_angle": "Attack angle",
    "attack_direction": "Attack direction",
    "ideal_attack_angle_rate": "Ideal attack-angle rate",
    "distance_off_plate": "Distance off the plate",
    "depth_in_box": "Depth in the box",
    "intercept_y_vs_plate": "Intercept vs plate",
    "intercept_y_vs_batter": "Intercept vs batter",
}
OUTCOME_LABELS = {
    "wrc_plus": "wRC+",
    "xwoba": "xwOBA",
}
CORE_OVERLAP_FEATURES = (
    "barrel_rate",
    "hard_hit_rate",
    "avg_exit_velocity",
    "strikeout_rate",
    "walk_rate",
)
CORE_LABELS = {
    "barrel_rate": "Barrel rate",
    "hard_hit_rate": "Hard-hit rate",
    "avg_exit_velocity": "Average exit velocity",
    "avg_launch_angle": "Average launch angle",
    "walk_rate": "Walk rate",
    "strikeout_rate": "Strikeout rate",
    "swing_rate": "Swing rate",
    "chase_rate": "Chase rate",
    "contact_rate": "Contact rate",
    "zone_contact_rate": "Zone-contact rate",
    "swinging_strike_rate": "Swinging-strike rate",
    "age": "Age",
    "season": "Season",
    "bats": "Handedness",
    "const": "Intercept",
    "season_2025": "Season 2025",
    "season_2026": "Season 2026",
    "bats_L": "Bats left",
    "bats_R": "Bats right",
}
BLOCK_LABELS = {
    "baseline": "Mean baseline",
    "core": "Core stats",
    "core_plus_traits": "Core + traits",
}
FAMILY_LABELS = {
    "mean_baseline": "Mean",
    "ols": "OLS",
    "ridge": "Ridge",
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
}


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
    trait_numeric = hitter_trait_columns(data)
    categorical = _available(CONTEXT_CATEGORICAL, data)
    return {
        "core": FeatureSet("core", core_numeric, categorical),
        "core_plus_traits": FeatureSet(
            "core_plus_traits", core_numeric + trait_numeric, categorical
        ),
    }


def hitter_trait_columns(data: pd.DataFrame) -> tuple[str, ...]:
    """Return swing and stance fields eligible for the trait ranking.

    Production metrics (OPS, wOBA, wRC+) and contact-quality results
    (barrel rate, exit velocity, hard-hit rate, launch angle) are excluded
    because they are outcomes or near-outcomes, not hitter traits.
    """

    columns = _available(TRAIT_FEATURES, data)
    leaked = sorted(set(columns) & RANKING_EXCLUSIONS)
    if leaked:
        raise ValueError(
            "Trait ranking cannot include production or contact-quality "
            f"fields: {leaked}"
        )
    return columns


def _pairwise_association(
    left: pd.Series, right: pd.Series, method: str
) -> tuple[float, float, int]:
    frame = pd.concat([left, right], axis=1).dropna()
    count = int(len(frame))
    if (
        count < 3
        or frame.iloc[:, 0].nunique() < 2
        or frame.iloc[:, 1].nunique() < 2
    ):
        return float("nan"), float("nan"), count
    if method == "pearson":
        result = stats.pearsonr(frame.iloc[:, 0], frame.iloc[:, 1])
    elif method == "spearman":
        result = stats.spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return float(result.statistic), float(result.pvalue), count


def rank_hitter_traits(
    data: pd.DataFrame,
    outcomes: tuple[str, ...] = RANKING_OUTCOMES,
) -> pd.DataFrame:
    """Rank swing/stance traits by association with offensive production.

    Pearson r captures linear association; Spearman rho captures monotonic
    association. Each row is a trait-outcome pair using pairwise-complete
    player-seasons. This is a descriptive ranking, not a causal effect.
    """

    traits = hitter_trait_columns(data)
    available_outcomes = _available(outcomes, data)
    if not available_outcomes:
        raise ValueError("None of the ranking outcomes are present in the data.")

    rows: list[dict[str, object]] = []
    for outcome in available_outcomes:
        for trait in traits:
            pearson_r, pearson_p, pearson_n = _pairwise_association(
                data[trait], data[outcome], "pearson"
            )
            spearman_rho, spearman_p, _spearman_n = _pairwise_association(
                data[trait], data[outcome], "spearman"
            )
            rows.append(
                {
                    "trait": trait,
                    "trait_label": TRAIT_LABELS.get(trait, trait),
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                    "pearson_r": pearson_r,
                    "pearson_p_value": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p_value": spearman_p,
                    "n": pearson_n,
                    "abs_pearson_r": abs(pearson_r) if pd.notna(pearson_r) else float("nan"),
                }
            )

    ranking = pd.DataFrame(rows)
    primary = available_outcomes[0]
    primary_order = (
        ranking.loc[ranking["outcome"].eq(primary)]
        .sort_values(
            ["abs_pearson_r", "trait"],
            ascending=[False, True],
            na_position="last",
        )["trait"]
        .tolist()
    )
    ranking["rank"] = ranking["trait"].map(
        {trait: index for index, trait in enumerate(primary_order)}
    )
    return ranking.sort_values(["rank", "outcome"], ignore_index=True).drop(
        columns="rank"
    )


def format_trait_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long ranking into one row per trait, ordered by |r| with wRC+."""

    if ranking.empty:
        return ranking
    value_columns = ["pearson_r", "spearman_rho", "n"]
    pieces = []
    for column in value_columns:
        wide = ranking.pivot(index=["trait", "trait_label"], columns="outcome", values=column)
        wide.columns = [f"{column}_{outcome}" for outcome in wide.columns]
        pieces.append(wide)
    table = pd.concat(pieces, axis=1).reset_index()
    sort_column = (
        "pearson_r_wrc_plus" if "pearson_r_wrc_plus" in table.columns else table.columns[2]
    )
    table["_abs"] = table[sort_column].abs()
    return table.sort_values(
        ["_abs", "trait"], ascending=[False, True], ignore_index=True
    ).drop(columns="_abs")


def _field_label(column: str) -> str:
    return TRAIT_LABELS.get(column, CORE_LABELS.get(column, column))


def trait_core_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations between swing/stance traits and core contact stats.

    Rows are traits from the Part 1 ranking; columns are barrel rate, hard-hit
    rate, exit velocity, strikeout rate, and walk rate. Production metrics are
    not included.
    """

    traits = hitter_trait_columns(data)
    core = _available(CORE_OVERLAP_FEATURES, data)
    if not traits or not core:
        raise ValueError("Trait-core correlations need both trait and core fields.")
    matrix = data[list(traits) + list(core)].corr(numeric_only=True)
    overlap = matrix.loc[list(traits), list(core)].copy()
    ranking = rank_hitter_traits(data)
    trait_order = ranking.loc[ranking["outcome"].eq("wrc_plus"), "trait"].tolist()
    ordered_traits = [trait for trait in trait_order if trait in overlap.index]
    return overlap.loc[ordered_traits]


def _ols_residuals(target: pd.Series, controls: pd.DataFrame) -> pd.Series:
    frame = pd.concat(
        [pd.to_numeric(target, errors="coerce").rename("_y"), controls],
        axis=1,
    ).dropna(subset=["_y"])
    design = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    for column in design.columns:
        design[column] = design[column].fillna(design[column].median())
    if frame["_y"].nunique() < 2 or design.shape[1] == 0:
        return pd.Series(np.nan, index=target.index)
    fitted = sm.OLS(
        frame["_y"].to_numpy(),
        sm.add_constant(design.to_numpy(), has_constant="add"),
    ).fit()
    return pd.Series(fitted.resid, index=frame.index).reindex(target.index)


def residualized_trait_associations(
    data: pd.DataFrame,
    outcome: str = TARGET,
    controls: tuple[str, ...] = CORE_OVERLAP_FEATURES,
    traits: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compare raw trait–outcome Pearson r with r after residualizing on core stats.

    Both the trait and the outcome are residualized on the same control block
    (barrels, hard-hit, EV, K%, BB% by default). The leftover correlation is
    the association that is not already sitting in those core stats.
    """

    selected_traits = traits if traits is not None else hitter_trait_columns(data)
    control_columns = _available(controls, data)
    if outcome not in data:
        raise ValueError(f"Outcome {outcome!r} is absent.")
    if not selected_traits:
        raise ValueError("No traits were provided for residualization.")
    if not control_columns:
        raise ValueError("No core control fields are available.")

    rows: list[dict[str, object]] = []
    for trait in selected_traits:
        raw_r, raw_p, raw_n = _pairwise_association(data[trait], data[outcome], "pearson")
        frame = data[[trait, outcome, *control_columns]].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=[trait, outcome])
        if len(frame) < 3 or frame[trait].nunique() < 2 or frame[outcome].nunique() < 2:
            residual_r, residual_p, residual_n = float("nan"), float("nan"), int(len(frame))
        else:
            trait_resid = _ols_residuals(frame[trait], frame[list(control_columns)])
            outcome_resid = _ols_residuals(frame[outcome], frame[list(control_columns)])
            residual_r, residual_p, residual_n = _pairwise_association(
                trait_resid, outcome_resid, "pearson"
            )
        rows.append(
            {
                "trait": trait,
                "trait_label": _field_label(trait),
                "outcome": outcome,
                "raw_r": raw_r,
                "raw_p_value": raw_p,
                "residualized_r": residual_r,
                "residualized_p_value": residual_p,
                "n": residual_n if pd.notna(residual_r) else raw_n,
                "abs_raw_r": abs(raw_r) if pd.notna(raw_r) else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["abs_raw_r", "trait"],
        ascending=[False, True],
        na_position="last",
        ignore_index=True,
    )


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


def plot_trait_ranking(ranking: pd.DataFrame) -> plt.Figure:
    """Plot Pearson correlations of swing/stance traits with wRC+ and xwOBA."""

    if ranking.empty:
        raise ValueError("Trait ranking is empty.")
    wrc = ranking.loc[ranking["outcome"].eq("wrc_plus")].sort_values(
        "abs_pearson_r", ascending=True, na_position="first"
    )
    trait_order = wrc["trait"].tolist()
    labels = wrc["trait_label"].tolist()
    outcomes = [
        outcome
        for outcome in ("wrc_plus", "xwoba")
        if outcome in set(ranking["outcome"])
    ]
    figure, axes = plt.subplots(
        1,
        len(outcomes),
        figsize=(6.2 * len(outcomes), max(5.5, 0.38 * len(trait_order) + 1.8)),
        sharey=True,
        squeeze=False,
    )
    palette = sns.color_palette("colorblind")
    positive = palette[0]
    negative = palette[3]
    for axis, outcome in zip(axes[0], outcomes, strict=True):
        frame = ranking.loc[ranking["outcome"].eq(outcome)].set_index("trait").loc[trait_order]
        values = frame["pearson_r"].to_numpy()
        colors = [positive if pd.notna(value) and value >= 0 else negative for value in values]
        axis.barh(range(len(trait_order)), values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        for index, value in enumerate(values):
            if pd.isna(value):
                continue
            ha = "left" if value >= 0 else "right"
            axis.text(
                value + (0.018 if value >= 0 else -0.018),
                index,
                f"{value:.2f}",
                va="center",
                ha=ha,
                fontsize=8,
            )
        axis.set_yticks(range(len(trait_order)), labels)
        axis.set_xlabel("Pearson correlation")
        axis.set_xlim(-1.05, 1.05)
        axis.set_title(f"vs {OUTCOME_LABELS.get(outcome, outcome)}")
    figure.suptitle("Swing and stance traits vs offensive production", fontsize=13)
    figure.tight_layout()
    return figure


def plot_trait_core_heatmap(overlap: pd.DataFrame) -> plt.Figure:
    """Plot Pearson correlations of swing/stance traits with core contact stats."""

    if overlap.empty:
        raise ValueError("Trait-core correlation matrix is empty.")
    labeled = overlap.rename(
        index=_field_label,
        columns=_field_label,
    )
    figure, axis = plt.subplots(
        figsize=(8.5, max(6.0, 0.38 * len(labeled.index) + 1.6))
    )
    sns.heatmap(
        labeled,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set(
        title="Swing traits vs barrels, exit velocity, and plate discipline",
        xlabel="Core contact and discipline stats",
        ylabel="Swing and stance traits",
    )
    figure.tight_layout()
    return figure


def plot_raw_vs_residualized(associations: pd.DataFrame) -> plt.Figure:
    """Plot raw vs residualized trait–wRC+ correlations as a dumbbell chart."""

    if associations.empty:
        raise ValueError("Residualized associations are empty.")
    frame = (
        associations.dropna(subset=["raw_r", "residualized_r"])
        .sort_values(["abs_raw_r", "trait"], ascending=[True, False], na_position="first")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("Residualized associations have no finite correlations.")
    y_positions = np.arange(len(frame))
    palette = sns.color_palette("colorblind")
    figure, axis = plt.subplots(figsize=(8.2, max(5.5, 0.38 * len(frame) + 1.8)))
    axis.hlines(
        y_positions,
        frame["raw_r"],
        frame["residualized_r"],
        color="0.75",
        linewidth=1.4,
        zorder=1,
    )
    axis.scatter(
        frame["raw_r"],
        y_positions,
        color=palette[0],
        s=42,
        zorder=3,
        label="Raw Pearson r with wRC+",
    )
    axis.scatter(
        frame["residualized_r"],
        y_positions,
        color=palette[1],
        s=42,
        zorder=3,
        label="After controlling for barrels, EV, hard-hit, K%, BB%",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y_positions, frame["trait_label"])
    axis.set_xlabel("Pearson correlation with wRC+")
    axis.set_xlim(-1.05, 1.05)
    axis.set_title("How much of the trait ranking remains after core stats")
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    return figure


def plot_reconstruction_mae(performance: pd.DataFrame) -> plt.Figure:
    """Plot grouped-CV MAE and R² for core vs core-plus-traits models."""

    if performance.empty:
        raise ValueError("Performance table is empty.")
    frame = performance.copy()
    frame["block_label"] = frame["feature_block"].map(
        lambda value: BLOCK_LABELS.get(value, value)
    )
    frame["family_label"] = frame["model_family"].map(
        lambda value: FAMILY_LABELS.get(value, value)
    )
    family_order = [
        label
        for key, label in FAMILY_LABELS.items()
        if label in set(frame["family_label"])
    ]
    block_order = [
        label
        for key, label in BLOCK_LABELS.items()
        if label in set(frame["block_label"])
    ]
    palette = sns.color_palette("colorblind")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    sns.barplot(
        data=frame,
        x="family_label",
        y="mae",
        hue="block_label",
        order=family_order,
        hue_order=block_order,
        palette=palette[: len(block_order)],
        ax=axes[0],
    )
    axes[0].set(
        title="Out-of-fold MAE",
        xlabel="Model",
        ylabel="Mean absolute error (wRC+ points)",
    )
    axes[0].legend(title="", frameon=False)
    sns.barplot(
        data=frame,
        x="family_label",
        y="r2",
        hue="block_label",
        order=family_order,
        hue_order=block_order,
        palette=palette[: len(block_order)],
        ax=axes[1],
    )
    axes[1].set(title="Out-of-fold R²", xlabel="Model", ylabel="R²")
    axes[1].legend(title="", frameon=False)
    figure.suptitle("Reconstructing same-season wRC+: core stats vs core plus traits")
    figure.tight_layout()
    return figure


def plot_trait_block_increment(comparison: pd.DataFrame) -> plt.Figure:
    """Plot the bootstrapped MAE difference when traits are added to core stats."""

    if comparison.empty:
        raise ValueError("Increment comparison is empty.")
    row = comparison.iloc[0]
    difference = float(row["difference"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    figure, axis = plt.subplots(figsize=(8.2, 2.4))
    axis.errorbar(
        difference,
        0,
        xerr=[[difference - lower], [upper - difference]],
        fmt="o",
        color=sns.color_palette("colorblind")[0],
        capsize=5,
        markersize=8,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks([])
    axis.set_xlabel("MAE difference in wRC+ points (core + traits minus core)")
    axis.set_xlim(min(-0.8, lower - 0.1), max(0.2, upper + 0.1))
    axis.set_title(
        "Adding swing/stance traits changes MAE by "
        f"{difference:.2f} (95% player-cluster interval {lower:.2f} to {upper:.2f})"
    )
    figure.tight_layout()
    return figure


def plot_permutation_importance(
    importance: pd.DataFrame, top_n: int = 12
) -> plt.Figure:
    """Plot grouped permutation importance, highlighting swing/stance traits."""

    if importance.empty:
        raise ValueError("Permutation importance table is empty.")
    frame = importance.sort_values("importance_mean", ascending=True).tail(top_n)
    palette = sns.color_palette("colorblind")
    colors = [
        palette[1] if feature in TRAIT_FEATURES else palette[0]
        for feature in frame["feature"]
    ]
    figure, axis = plt.subplots(figsize=(8.2, max(4.5, 0.38 * len(frame) + 1.4)))
    axis.barh(
        range(len(frame)),
        frame["importance_mean"],
        xerr=frame["importance_std"] if "importance_std" in frame else None,
        color=colors,
        capsize=3,
    )
    axis.set_yticks(range(len(frame)), [_field_label(feature) for feature in frame["feature"]])
    axis.set_xlabel("MAE increase when the feature is shuffled (wRC+ points)")
    axis.set_title("What the reconstruction model uses (permutation importance)")
    axis.legend(
        handles=[
            Patch(color=palette[0], label="Core stats"),
            Patch(color=palette[1], label="Swing/stance traits"),
        ],
        frameon=False,
        loc="lower right",
    )
    figure.tight_layout()
    return figure


def plot_clustered_coefficients(
    coefficients: pd.DataFrame, top_n: int = 12
) -> plt.Figure:
    """Plot standardized OLS coefficients with clustered confidence intervals."""

    if coefficients.empty:
        raise ValueError("Coefficient table is empty.")
    frame = coefficients.loc[~coefficients["feature"].eq("const")].copy()
    frame["abs_coefficient"] = frame["coefficient"].abs()
    frame = frame.sort_values("abs_coefficient", ascending=True).tail(top_n)
    y_positions = np.arange(len(frame))
    palette = sns.color_palette("colorblind")
    colors = [
        palette[1] if feature in TRAIT_FEATURES else palette[0]
        for feature in frame["feature"]
    ]
    figure, axis = plt.subplots(figsize=(8.2, max(4.5, 0.38 * len(frame) + 1.4)))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.errorbar(
        frame["coefficient"],
        y_positions,
        xerr=np.vstack(
            [
                (frame["coefficient"] - frame["ci_lower"]).to_numpy(),
                (frame["ci_upper"] - frame["coefficient"]).to_numpy(),
            ]
        ),
        fmt="none",
        ecolor="0.65",
        capsize=3,
        zorder=1,
    )
    axis.scatter(frame["coefficient"], y_positions, color=colors, zorder=3)
    axis.set_yticks(y_positions, [_field_label(feature) for feature in frame["feature"]])
    axis.set_xlabel("Standardized OLS coefficient for wRC+")
    axis.set_title("Joint associations after controlling for everything in the model")
    axis.legend(
        handles=[
            Patch(color=palette[0], label="Core stats"),
            Patch(color=palette[1], label="Swing/stance traits"),
        ],
        frameon=False,
        loc="lower right",
    )
    figure.tight_layout()
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
