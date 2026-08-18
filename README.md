# MLB Offensive Production Data

This project builds a player-season dataset and grouped regression analysis for
studying how swing behavior and hitter traits relate to offensive production.
It uses 2024, 2025, and 2026 YTD by default because 2024 is the first full
season of public Statcast bat tracking.

## Sources

- **Baseball Savant:** wOBA, xwOBA, xBA, xSLG, bat speed, swing length,
  squared-up/blast/whiff rates, attack angle, swing-path tilt, stance position,
  and intercept position.
- **FanGraphs:** OPS, wRC+, plate-discipline rates, and contact-quality context.
- **Baseball-Reference:** exact combined player-season OPS+, including a single
  combined row for traded players.

Savant's CSV exports and FanGraphs' JSON endpoint are public but not versioned
APIs. Raw responses are therefore cached. FanGraphs says automated access is
unsupported, and Baseball-Reference rate limits automated requests; do not
delete caches merely to rerun an analysis.

No `pybaseball` code or dependency is used.

## Setup

With the existing virtual environment:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

## Collect data

```powershell
.venv\Scripts\python.exe data_collection.py
```

Or, after the editable install:

```powershell
collect-mlb-offense
```

Useful options:

```powershell
collect-mlb-offense --seasons 2024 2025 2026 --min-pa 100 --min-competitive-swings 50
collect-mlb-offense --refresh
```

`--refresh` replaces the current date's raw caches. Historical caches are not
overwritten because their retrieval date is part of the filename.

Outputs:

- `data/raw/`: source CSV, JSON, and HTML responses
- `data/processed/mlb_offense_2024_2026.parquet`: typed analysis dataset
- `data/processed/mlb_offense_2024_2026.csv`: portable copy
- `data/processed/collection_report.json`: row counts, coverage, and missingness

The full joined dataset is retained. `analysis_eligible` marks rows meeting the
configured PA and competitive-swing thresholds with all core production
metrics available.

## Important interpretation notes

- 2026 is year-to-date and is frozen at `retrieved_on`.
- `distance_off_plate` and `depth_in_box` are Statcast stance-position fields,
  measured from the hitter's center of mass. They are not pitch-location data.
- wRC+ and OPS+ come from different providers and should remain source-labeled.
- Same-season regressions describe associations, not causal effects or
  out-of-sample forecasts.

## Run the analysis

The notebook applies the same local eligibility thresholds as the collector
(`PA >= 100`, `competitive swings >= 50`) and evaluates:

- a season-adjusted mean baseline;
- OLS, ridge, and elastic-net regressions;
- a constrained random forest for nonlinearities;
- `core` predictors (context, discipline, contact quality) versus
  `core_plus_traits` (bat tracking and stance metrics).

It uses nested, player-grouped cross-validation and reports MAE, RMSE, R²,
calibration, PA-weighted sensitivity metrics, a player-cluster bootstrap
interval for the trait-block MAE difference, cluster-robust OLS intervals, and
descriptive permutation importance.

```powershell
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute offensive_analysis.ipynb --output offensive_analysis.executed.ipynb
```

The executed notebook writes reproducible analysis tables and figures to
`data/analysis/`. In particular, do not use OPS, OPS+, wOBA, xwOBA, or
`batter_run_value` as predictors of wRC+: they are excluded by the analysis
module because they would create formulaic target leakage.

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

Tests use local fixtures and do not consume provider request budgets.
