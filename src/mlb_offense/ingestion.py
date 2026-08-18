"""Collect player-season offensive production and hitter-trait data.

Baseball Savant provides the Statcast metrics, FanGraphs provides wRC+, and
Baseball-Reference provides OPS+. Raw responses are cached before they are
normalized and joined on MLBAM player ID plus season.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from fungo import fangraphs, lookup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SAVANT_BASE_URL = "https://baseballsavant.mlb.com"
BBREF_LEAGUE_URL = (
    "https://www.baseball-reference.com/leagues/majors/"
    "{season}-standard-batting.shtml"
)


class DataSourceError(RuntimeError):
    """Raised when a remote source returns unusable data."""


@dataclass(frozen=True)
class CollectionConfig:
    """Runtime settings for a reproducible collection run."""

    seasons: tuple[int, ...] = (2024, 2025, 2026)
    raw_dir: Path = Path("data/raw")
    output_path: Path = Path("data/processed/mlb_offense_2024_2026.parquet")
    min_pa: int = 100
    min_competitive_swings: int = 50
    refresh: bool = False
    retrieved_on: str = date.today().isoformat()


def _http_session() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "mlb-offense-analysis/0.1 "
                "(research use; sequential cached requests)"
            )
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], source: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataSourceError(f"{source} response is missing columns: {missing}")
    if frame.empty:
        raise DataSourceError(f"{source} response contains no rows")


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _read_or_fetch_savant(
    endpoint: str,
    params: dict[str, Any],
    cache_path: Path,
    refresh: bool,
    session: requests.Session,
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        content = cache_path.read_bytes()
    else:
        response = session.get(
            f"{SAVANT_BASE_URL}{endpoint}", params=params, timeout=60
        )
        response.raise_for_status()
        content = response.content
        preview = content.lstrip()[:100].lower()
        if not content or preview.startswith(b"<html") or preview.startswith(
            b"<!doctype"
        ):
            raise DataSourceError(
                f"Baseball Savant returned non-CSV content for {endpoint}"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)

    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise DataSourceError(
            f"Could not parse cached Savant response {cache_path}"
        ) from exc


def fetch_savant_expected(
    season: int,
    raw_dir: Path,
    retrieved_on: str,
    refresh: bool,
    session: requests.Session,
) -> pd.DataFrame:
    """Fetch broad expected-stat results (the local PA filter is applied later)."""

    frame = _read_or_fetch_savant(
        "/leaderboard/expected_statistics",
        {
            "type": "batter",
            "year": season,
            "position": "",
            "team": "",
            "filterType": "pa",
            "min": 1,
            "csv": "true",
        },
        raw_dir / f"savant_expected_{season}_{retrieved_on}.csv",
        refresh,
        session,
    )
    _require_columns(
        frame,
        {"player_id", "year", "pa", "woba", "est_woba"},
        f"Savant expected stats {season}",
    )
    frame = frame.rename(
        columns={
            "last_name, first_name": "savant_player_name",
            "year": "season",
            "pa": "savant_pa",
            "bip": "savant_bip",
            "ba": "batting_avg",
            "est_ba": "xba",
            "slg": "slugging",
            "est_slg": "xslg",
            "woba": "woba",
            "est_woba": "xwoba",
        }
    )
    keep = [
        "player_id",
        "season",
        "savant_player_name",
        "savant_pa",
        "savant_bip",
        "batting_avg",
        "xba",
        "slugging",
        "xslg",
        "woba",
        "xwoba",
    ]
    frame = frame[[column for column in keep if column in frame]]
    return _to_numeric(frame, [column for column in keep if column != "savant_player_name"])


def fetch_savant_bat_tracking(
    season: int,
    raw_dir: Path,
    retrieved_on: str,
    refresh: bool,
    session: requests.Session,
) -> pd.DataFrame:
    frame = _read_or_fetch_savant(
        "/leaderboard/bat-tracking",
        {
            "seasonStart": season,
            "seasonEnd": season,
            "type": "batter",
            "team": "",
            "gameType": "Regular",
            "minSwings": 1,
            "minGroupSwings": 1,
            "csv": "true",
        },
        raw_dir / f"savant_bat_tracking_{season}_{retrieved_on}.csv",
        refresh,
        session,
    )
    _require_columns(
        frame,
        {"id", "avg_bat_speed", "swings_competitive"},
        f"Savant bat tracking {season}",
    )
    frame["season"] = season
    frame = frame.rename(
        columns={
            "id": "player_id",
            "name": "bat_tracking_player_name",
            "swings_competitive": "competitive_swings",
            "percent_swings_competitive": "competitive_swing_rate",
            "hard_swing_rate": "fast_swing_rate",
            "squared_up_per_bat_contact": "squared_up_contact_rate",
            "squared_up_per_swing": "squared_up_swing_rate",
            "blast_per_bat_contact": "blast_contact_rate",
            "blast_per_swing": "blast_swing_rate",
            "whiff_per_swing": "whiff_rate",
            "batted_ball_event_per_swing": "bbe_per_swing",
        }
    )
    keep = [
        "player_id",
        "season",
        "bat_tracking_player_name",
        "competitive_swings",
        "competitive_swing_rate",
        "avg_bat_speed",
        "fast_swing_rate",
        "swing_length",
        "squared_up_contact_rate",
        "squared_up_swing_rate",
        "blast_contact_rate",
        "blast_swing_rate",
        "whiff_rate",
        "bbe_per_swing",
        "batter_run_value",
        "contact",
        "whiffs",
        "batted_ball_events",
        "swords",
    ]
    frame = frame[[column for column in keep if column in frame]]
    return _to_numeric(
        frame,
        [
            column
            for column in keep
            if column not in {"bat_tracking_player_name"}
        ],
    )


def fetch_savant_swing_path(
    season: int,
    raw_dir: Path,
    retrieved_on: str,
    refresh: bool,
    session: requests.Session,
) -> pd.DataFrame:
    frame = _read_or_fetch_savant(
        "/leaderboard/bat-tracking/swing-path-attack-angle",
        {
            "seasonStart": season,
            "seasonEnd": season,
            "type": "batter",
            "team": "",
            "gameType": "Regular",
            "minSwings": 1,
            "minGroupSwings": 1,
            "csv": "true",
        },
        raw_dir / f"savant_swing_path_{season}_{retrieved_on}.csv",
        refresh,
        session,
    )
    _require_columns(
        frame,
        {"id", "attack_angle", "avg_batter_x_position"},
        f"Savant swing path {season}",
    )
    frame["season"] = season
    frame = frame.rename(
        columns={
            "id": "player_id",
            "name": "swing_path_player_name",
            "competitive_swings": "path_competitive_swings",
            "avg_batter_x_position": "distance_off_plate",
            "avg_batter_y_position": "depth_in_box",
            "avg_intercept_y_vs_plate": "intercept_y_vs_plate",
            "avg_intercept_y_vs_batter": "intercept_y_vs_batter",
        }
    )
    keep = [
        "player_id",
        "season",
        "swing_path_player_name",
        "side",
        "path_competitive_swings",
        "swing_tilt",
        "attack_angle",
        "attack_direction",
        "ideal_attack_angle_rate",
        "distance_off_plate",
        "depth_in_box",
        "intercept_y_vs_plate",
        "intercept_y_vs_batter",
    ]
    frame = frame[[column for column in keep if column in frame]]
    return _to_numeric(
        frame,
        [
            column
            for column in keep
            if column not in {"swing_path_player_name", "side"}
        ],
    )


def fetch_fangraphs(
    season: int,
    raw_dir: Path,
    retrieved_on: str,
    refresh: bool,
) -> pd.DataFrame:
    cache_path = raw_dir / f"fangraphs_batting_{season}_{retrieved_on}.json"
    if cache_path.exists() and not refresh:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        try:
            rows = fangraphs.get_leaders("bat", season, qual=1)
        except Exception as exc:
            raise DataSourceError(
                "FanGraphs access failed. Its public JSON endpoint is unofficial; "
                "leave the raw cache in place between runs."
            ) from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )

    frame = pd.DataFrame(rows)
    _require_columns(
        frame,
        {"xMLBAMID", "Season", "PA", "OPS", "wRC+"},
        f"FanGraphs batting {season}",
    )
    frame = frame.rename(
        columns={
            "xMLBAMID": "player_id",
            "Season": "season",
            "PlayerName": "fangraphs_player_name",
            "PA": "fangraphs_pa",
            "OPS": "ops",
            "wOBA": "fangraphs_woba",
            "xwOBA": "fangraphs_xwoba",
            "wRC+": "wrc_plus",
            "BB%": "walk_rate",
            "K%": "strikeout_rate",
            "Swing%": "swing_rate",
            "O-Swing%": "chase_rate",
            "Contact%": "contact_rate",
            "Z-Contact%": "zone_contact_rate",
            "SwStr%": "swinging_strike_rate",
            "Barrel%": "barrel_rate",
            "HardHit%": "hard_hit_rate",
            "EV": "avg_exit_velocity",
            "LA": "avg_launch_angle",
            "Age": "age",
            "Bats": "bats",
        }
    )
    keep = [
        "player_id",
        "season",
        "fangraphs_player_name",
        "fangraphs_pa",
        "ops",
        "fangraphs_woba",
        "fangraphs_xwoba",
        "wrc_plus",
        "walk_rate",
        "strikeout_rate",
        "swing_rate",
        "chase_rate",
        "contact_rate",
        "zone_contact_rate",
        "swinging_strike_rate",
        "barrel_rate",
        "hard_hit_rate",
        "avg_exit_velocity",
        "avg_launch_angle",
        "age",
        "bats",
    ]
    frame = frame[[column for column in keep if column in frame]]
    return _to_numeric(
        frame,
        [
            column
            for column in keep
            if column not in {"fangraphs_player_name", "bats"}
        ],
    )


def _parse_bbref_table(html: str, season: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="players_standard_batting")
    if table is None:
        raise DataSourceError(
            f"Baseball-Reference player batting table was not found for {season}"
        )

    rows: list[dict[str, Any]] = []
    for row in table.select("tbody tr:not(.thead)"):
        name_cell = row.find("td", attrs={"data-stat": "name_display"})
        if name_cell is None or name_cell.find("a") is None:
            continue
        link = name_cell.find("a")
        match = re.search(r"/players/[a-z]/([^/.]+)\.shtml", link.get("href", ""))
        if not match:
            continue
        values = {
            cell.get("data-stat"): cell.get_text(strip=True)
            for cell in row.find_all(["th", "td"])
            if cell.get("data-stat")
        }
        rows.append(
            {
                "bbref_id": match.group(1),
                "bbref_player_name": link.get_text(strip=True),
                "season": season,
                "bbref_team": values.get("team_name_abbr"),
                "bbref_pa": values.get("b_pa"),
                "ops_plus": values.get("b_onbase_plus_slugging_plus"),
            }
        )

    frame = pd.DataFrame(rows)
    _require_columns(
        frame,
        {"bbref_id", "season", "bbref_pa", "ops_plus"},
        f"Baseball-Reference batting {season}",
    )

    # Traded players have a full combined row plus team-stint rows. A player
    # who changed leagues more than once can also have a partial "2TM" row
    # alongside the full "3TM" row, so the row with the greatest PA is the
    # complete player-season record.
    frame = _to_numeric(frame, ["season", "bbref_pa", "ops_plus"])
    complete_rows = frame.groupby("bbref_id")["bbref_pa"].idxmax()
    return frame.loc[complete_rows].reset_index(drop=True)


def fetch_baseball_reference(
    season: int,
    raw_dir: Path,
    retrieved_on: str,
    refresh: bool,
) -> pd.DataFrame:
    cache_path = raw_dir / f"baseball_reference_batting_{season}_{retrieved_on}.html"
    if cache_path.exists() and not refresh:
        html = cache_path.read_text(encoding="utf-8")
    else:
        response = curl_requests.get(
            BBREF_LEAGUE_URL.format(season=season),
            impersonate="chrome",
            timeout=60,
        )
        if response.status_code in {403, 429}:
            raise DataSourceError(
                "Baseball-Reference rate-limited this request. Stop and reuse "
                "existing caches; repeated retries can extend the block."
            )
        if response.status_code != 200:
            raise DataSourceError(
                f"Baseball-Reference returned HTTP {response.status_code}"
            )
        html = response.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
        time.sleep(7)
    return _parse_bbref_table(html, season)


def _normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", ascii_text)


def attach_bbref_player_ids(
    bbref_frame: pd.DataFrame, fangraphs_frame: pd.DataFrame
) -> pd.DataFrame:
    """Map B-R IDs through Chadwick, with unique normalized names as fallback."""

    register = lookup.lookup(mlb_only=False)
    id_map = {
        row["key_bbref"]: row["key_mlbam"]
        for row in register
        if row.get("key_bbref") and row.get("key_mlbam")
    }
    frame = bbref_frame.copy()
    frame["player_id"] = frame["bbref_id"].map(id_map)
    frame["bbref_join_method"] = frame["player_id"].notna().map(
        {True: "chadwick", False: pd.NA}
    )

    fg_names = fangraphs_frame[
        ["season", "player_id", "fangraphs_player_name"]
    ].dropna()
    fg_names["_normalized_name"] = fg_names["fangraphs_player_name"].map(
        _normalize_name
    )
    unique_names = fg_names.drop_duplicates(
        ["season", "_normalized_name"], keep=False
    )
    fallback_map = {
        (int(season), normalized_name): player_id
        for season, normalized_name, player_id in unique_names[
            ["season", "_normalized_name", "player_id"]
        ].itertuples(index=False, name=None)
    }
    unresolved = frame["player_id"].isna()
    normalized = frame.loc[unresolved, "bbref_player_name"].map(_normalize_name)
    fallback = [
        (
            str(fallback_map[(int(season), name)])
            if (int(season), name) in fallback_map
            else None
        )
        for season, name in zip(
            frame.loc[unresolved, "season"], normalized, strict=True
        )
    ]
    frame.loc[unresolved, "player_id"] = fallback
    used_fallback = unresolved & frame["player_id"].notna()
    frame.loc[used_fallback, "bbref_join_method"] = "unique_name"
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce").astype(
        "Int64"
    )
    return frame


def _assert_unique(frame: pd.DataFrame, source: str) -> None:
    duplicates = frame.duplicated(["player_id", "season"], keep=False)
    duplicates &= frame["player_id"].notna()
    if duplicates.any():
        sample = frame.loc[duplicates, ["player_id", "season"]].head().to_dict(
            "records"
        )
        raise DataSourceError(
            f"{source} has duplicate player-season keys; sample: {sample}"
        )


def combine_sources(
    expected: pd.DataFrame,
    bat_tracking: pd.DataFrame,
    swing_path: pd.DataFrame,
    fangraphs_frame: pd.DataFrame,
    bbref_frame: pd.DataFrame,
    config: CollectionConfig,
) -> pd.DataFrame:
    bbref_frame = bbref_frame[bbref_frame["player_id"].notna()].copy()
    sources = {
        "Savant expected": expected,
        "Savant bat tracking": bat_tracking,
        "Savant swing path": swing_path,
        "FanGraphs": fangraphs_frame,
        "Baseball-Reference": bbref_frame,
    }
    for source, frame in sources.items():
        _assert_unique(frame, source)

    keys = ["player_id", "season"]
    dataset = expected.merge(
        fangraphs_frame, on=keys, how="outer", validate="one_to_one"
    )
    dataset = dataset.merge(
        bat_tracking, on=keys, how="outer", validate="one_to_one"
    )
    dataset = dataset.merge(
        swing_path, on=keys, how="outer", validate="one_to_one"
    )
    dataset = dataset.merge(
        bbref_frame, on=keys, how="outer", validate="one_to_one"
    )

    name_columns = [
        "fangraphs_player_name",
        "bat_tracking_player_name",
        "swing_path_player_name",
        "bbref_player_name",
        "savant_player_name",
    ]
    dataset["player_name"] = dataset[
        [column for column in name_columns if column in dataset]
    ].bfill(axis=1).iloc[:, 0]
    dataset["pa"] = dataset[
        [
            column
            for column in ["fangraphs_pa", "savant_pa", "bbref_pa"]
            if column in dataset
        ]
    ].bfill(axis=1).iloc[:, 0]

    dataset["has_savant_expected"] = dataset["woba"].notna()
    dataset["has_bat_tracking"] = dataset["avg_bat_speed"].notna()
    dataset["has_swing_path"] = dataset["attack_angle"].notna()
    dataset["has_fangraphs"] = dataset["wrc_plus"].notna()
    dataset["has_baseball_reference"] = dataset["ops_plus"].notna()
    dataset["meets_pa_threshold"] = dataset["pa"].ge(config.min_pa)
    dataset["meets_swing_threshold"] = dataset["competitive_swings"].ge(
        config.min_competitive_swings
    )
    dataset["analysis_eligible"] = (
        dataset["meets_pa_threshold"]
        & dataset["meets_swing_threshold"]
        & dataset["wrc_plus"].notna()
        & dataset["ops_plus"].notna()
        & dataset["xwoba"].notna()
    )
    dataset["retrieved_on"] = config.retrieved_on

    front = [
        "player_id",
        "player_name",
        "season",
        "retrieved_on",
        "pa",
        "analysis_eligible",
    ]
    remaining = [column for column in dataset if column not in front]
    dataset = dataset[front + remaining]
    dataset["player_id"] = pd.to_numeric(
        dataset["player_id"], errors="coerce"
    ).astype("Int64")
    dataset["season"] = pd.to_numeric(dataset["season"], errors="coerce").astype(
        "Int64"
    )
    return dataset.sort_values(
        ["season", "pa", "player_name"], ascending=[True, False, True]
    ).reset_index(drop=True)


def _collection_report(
    dataset: pd.DataFrame,
    source_counts: dict[str, int],
    config: CollectionConfig,
) -> dict[str, Any]:
    eligible = dataset[dataset["analysis_eligible"]]
    important = [
        "ops",
        "woba",
        "xwoba",
        "ops_plus",
        "wrc_plus",
        "avg_bat_speed",
        "attack_angle",
        "distance_off_plate",
    ]
    return {
        "retrieved_on": config.retrieved_on,
        "seasons": list(config.seasons),
        "minimum_pa": config.min_pa,
        "minimum_competitive_swings": config.min_competitive_swings,
        "source_rows": source_counts,
        "dataset_rows": len(dataset),
        "eligible_rows": len(eligible),
        "eligible_by_season": {
            str(key): int(value)
            for key, value in eligible.groupby("season").size().items()
        },
        "missing_among_eligible": {
            column: int(eligible[column].isna().sum())
            for column in important
            if column in eligible
        },
        "unresolved_bbref_rows": source_counts["baseball_reference_unresolved"],
    }


def collect_dataset(config: CollectionConfig) -> pd.DataFrame:
    """Fetch, validate, join, persist, and return the complete dataset."""

    session = _http_session()
    expected_frames: list[pd.DataFrame] = []
    bat_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    fg_frames: list[pd.DataFrame] = []
    bbref_frames: list[pd.DataFrame] = []

    for season in config.seasons:
        expected_frames.append(
            fetch_savant_expected(
                season,
                config.raw_dir,
                config.retrieved_on,
                config.refresh,
                session,
            )
        )
        bat_frames.append(
            fetch_savant_bat_tracking(
                season,
                config.raw_dir,
                config.retrieved_on,
                config.refresh,
                session,
            )
        )
        path_frames.append(
            fetch_savant_swing_path(
                season,
                config.raw_dir,
                config.retrieved_on,
                config.refresh,
                session,
            )
        )
        fg_frames.append(
            fetch_fangraphs(
                season, config.raw_dir, config.retrieved_on, config.refresh
            )
        )
        bbref_frames.append(
            fetch_baseball_reference(
                season, config.raw_dir, config.retrieved_on, config.refresh
            )
        )

    expected = pd.concat(expected_frames, ignore_index=True)
    bat_tracking = pd.concat(bat_frames, ignore_index=True)
    swing_path = pd.concat(path_frames, ignore_index=True)
    fg_frame = pd.concat(fg_frames, ignore_index=True)
    bbref_frame = attach_bbref_player_ids(
        pd.concat(bbref_frames, ignore_index=True), fg_frame
    )
    unresolved_bbref = int(bbref_frame["player_id"].isna().sum())

    dataset = combine_sources(
        expected,
        bat_tracking,
        swing_path,
        fg_frame,
        bbref_frame,
        config,
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(config.output_path, index=False)
    dataset.to_csv(config.output_path.with_suffix(".csv"), index=False)

    source_counts = {
        "savant_expected": len(expected),
        "savant_bat_tracking": len(bat_tracking),
        "savant_swing_path": len(swing_path),
        "fangraphs": len(fg_frame),
        "baseball_reference": len(bbref_frame),
        "baseball_reference_unresolved": unresolved_bbref,
    }
    report = _collection_report(dataset, source_counts, config)
    report_path = config.output_path.with_name("collection_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return dataset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MLB offensive production and hitter-trait data."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2024, 2025, 2026],
        help="Seasons to collect (default: 2024 2025 2026).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mlb_offense_2024_2026.parquet"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--min-pa", type=int, default=100)
    parser.add_argument("--min-competitive-swings", type=int, default=50)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore dated raw caches and fetch each source again.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = CollectionConfig(
        seasons=tuple(args.seasons),
        raw_dir=args.raw_dir,
        output_path=args.output,
        min_pa=args.min_pa,
        min_competitive_swings=args.min_competitive_swings,
        refresh=args.refresh,
    )
    dataset = collect_dataset(config)
    print(
        f"Wrote {len(dataset):,} player-seasons to {config.output_path} "
        f"({int(dataset['analysis_eligible'].sum()):,} analysis-eligible)."
    )


if __name__ == "__main__":
    main()
