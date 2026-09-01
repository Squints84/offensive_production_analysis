from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlb_offense import analysis
from mlb_offense.analysis import (
    AnalysisConfig,
    bootstrap_model_difference,
    feature_sets,
    format_trait_ranking,
    load_analysis_data,
    plot_trait_ranking,
    rank_hitter_traits,
)


def _sample_data() -> pd.DataFrame:
    rows = []
    for player_id in range(1, 9):
        for season in (2024, 2025):
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "season": season,
                    "pa": 250,
                    "competitive_swings": 100,
                    "wrc_plus": 80 + (3 * player_id) + (season - 2024),
                    "age": 24 + player_id,
                    "bats": "R" if player_id % 2 else "L",
                    "walk_rate": 0.08,
                    "strikeout_rate": 0.20,
                    "swing_rate": 0.45,
                    "chase_rate": 0.28,
                    "contact_rate": 0.76,
                    "zone_contact_rate": 0.84,
                    "swinging_strike_rate": 0.11,
                    "barrel_rate": 0.08 + player_id / 1000,
                    "hard_hit_rate": 0.38,
                    "avg_exit_velocity": 88 + player_id / 10,
                    "avg_launch_angle": 12.0,
                    "avg_bat_speed": 68.0 + player_id,
                    "fast_swing_rate": 0.25,
                    "swing_length": 7.2,
                    "squared_up_contact_rate": 0.28,
                    "squared_up_swing_rate": 0.18,
                    "blast_contact_rate": 0.08,
                    "blast_swing_rate": 0.05,
                    "whiff_rate": 0.24,
                    "bbe_per_swing": 0.31,
                    "swing_tilt": 34.0,
                    "attack_angle": 11.0,
                    "attack_direction": 3.0,
                    "ideal_attack_angle_rate": 0.48,
                    "distance_off_plate": 32.0 - player_id,
                    "depth_in_box": 20.0,
                    "intercept_y_vs_plate": 5.0,
                    "intercept_y_vs_batter": 14.0,
                    "ops": 0.800,
                    "woba": 0.340,
                    "xwoba": 0.300 + player_id * 0.01,
                    "ops_plus": 115,
                }
            )
    return pd.DataFrame(rows)


def test_load_analysis_data_applies_local_thresholds(tmp_path: Path) -> None:
    data = _sample_data()
    data.loc[0, "pa"] = 50
    path = tmp_path / "dataset.parquet"
    data.to_parquet(path)

    result = load_analysis_data(path, AnalysisConfig(min_pa=100))

    assert len(result) == len(data) - 1
    assert result["observation_id"].is_unique


def test_feature_sets_exclude_outcome_and_production_metrics() -> None:
    sets = feature_sets(_sample_data())
    selected = set(sets["core_plus_traits"].columns)

    assert not selected.intersection(analysis.LEAKAGE_COLUMNS)
    assert "avg_bat_speed" in selected
    assert "attack_angle" in selected


def test_grouped_bootstrap_returns_interval() -> None:
    predictions = pd.DataFrame(
        {
            "observation_id": [0, 1, 2, 3] * 2,
            "player_id": [1, 1, 2, 2] * 2,
            "actual": [100.0, 105.0, 110.0, 115.0] * 2,
            "predicted": [
                99.0,
                104.0,
                109.0,
                114.0,
                90.0,
                95.0,
                100.0,
                105.0,
            ],
            "pa": [300, 300, 300, 300] * 2,
            "model": ["candidate"] * 4 + ["reference"] * 4,
        }
    )

    result = bootstrap_model_difference(
        predictions,
        "candidate",
        "reference",
        iterations=30,
        random_state=1,
    )

    assert result.loc[0, "difference"] < 0
    assert result.loc[0, "ci_upper"] < 0


def test_constant_baseline_has_undefined_calibration() -> None:
    result = analysis._metric_row(
        np.array([90.0, 100.0, 110.0]), np.array([100.0, 100.0, 100.0])
    )

    assert np.isnan(result["calibration_intercept"])
    assert np.isnan(result["calibration_slope"])


def test_evaluate_models_keeps_players_within_grouped_folds(monkeypatch) -> None:
    monkeypatch.setattr(
        analysis,
        "_model_specifications",
        lambda random_state: {
            "mean_baseline": (analysis.DummyRegressor(strategy="mean"), {}),
            "ols": (analysis.LinearRegression(), {}),
        },
    )
    data = _sample_data()
    data["observation_id"] = np.arange(len(data))

    performance, predictions, _ = analysis.evaluate_models(
        data,
        AnalysisConfig(outer_splits=2, inner_splits=2),
    )

    assert set(performance["model"]) == {
        "mean_baseline:baseline",
        "ols:core",
        "ols:core_plus_traits",
    }
    assert predictions.groupby("model")["observation_id"].nunique().eq(len(data)).all()


def test_trait_ranking_excludes_production_and_contact_quality() -> None:
    assert not set(analysis.TRAIT_FEATURES) & analysis.RANKING_EXCLUSIONS

    ranking = rank_hitter_traits(_sample_data())

    assert not set(ranking["trait"]) & analysis.RANKING_EXCLUSIONS
    assert {"avg_bat_speed", "attack_angle", "distance_off_plate"} <= set(ranking["trait"])
    assert set(ranking["outcome"]) == {"wrc_plus", "xwoba"}


def test_trait_ranking_orders_by_absolute_association_with_wrc() -> None:
    table = format_trait_ranking(rank_hitter_traits(_sample_data()))

    assert table.iloc[0]["trait"] == "avg_bat_speed"
    assert table.iloc[0]["pearson_r_wrc_plus"] > 0.99
    assert table.iloc[1]["trait"] == "distance_off_plate"
    assert table.iloc[1]["pearson_r_wrc_plus"] < 0


def test_plot_trait_ranking_returns_figure() -> None:
    figure = plot_trait_ranking(rank_hitter_traits(_sample_data()))
    try:
        assert len(figure.axes) == 2
    finally:
        plt.close(figure)
