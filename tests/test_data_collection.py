from pathlib import Path

import pandas as pd
import pytest

from mlb_offense.ingestion import (
    CollectionConfig,
    DataSourceError,
    _parse_bbref_table,
    _read_or_fetch_savant,
    attach_bbref_player_ids,
    combine_sources,
)


def test_bbref_parser_prefers_combined_traded_player_row() -> None:
    html = """
    <table id="players_standard_batting"><tbody>
      <tr>
        <td data-stat="name_display"><a href="/players/a/arraelu01.shtml">Luis Arraez</a></td>
        <td data-stat="team_name_abbr">2TM</td><td data-stat="b_pa">672</td>
        <td data-stat="b_onbase_plus_slugging_plus">107</td>
      </tr>
      <tr>
        <td data-stat="name_display"><a href="/players/a/arraelu01.shtml">Luis Arraez</a></td>
        <td data-stat="team_name_abbr">MIA</td><td data-stat="b_pa">148</td>
        <td data-stat="b_onbase_plus_slugging_plus">100</td>
      </tr>
      <tr>
        <td data-stat="name_display"><a href="/players/a/arraelu01.shtml">Luis Arraez</a></td>
        <td data-stat="team_name_abbr">SDP</td><td data-stat="b_pa">524</td>
        <td data-stat="b_onbase_plus_slugging_plus">108</td>
      </tr>
    </tbody></table>
    """

    result = _parse_bbref_table(html, 2024)

    assert len(result) == 1
    assert result.iloc[0]["bbref_team"] == "2TM"
    assert result.iloc[0]["ops_plus"] == 107


def test_bbref_id_mapping_uses_unique_name_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_offense.ingestion.lookup.lookup",
        lambda mlb_only=False: [
            {"key_bbref": "known01", "key_mlbam": "100"},
        ],
    )
    bbref = pd.DataFrame(
        {
            "bbref_id": ["known01", "rookie01"],
            "bbref_player_name": ["Known Player", "José Rookie"],
            "season": [2026, 2026],
            "bbref_pa": [200, 150],
            "ops_plus": [120, 105],
        }
    )
    fangraphs = pd.DataFrame(
        {
            "player_id": [100, 200],
            "season": [2026, 2026],
            "fangraphs_player_name": ["Known Player", "Jose Rookie"],
        }
    )

    result = attach_bbref_player_ids(bbref, fangraphs)

    assert result["player_id"].tolist() == [100, 200]
    assert result["bbref_join_method"].tolist() == ["chadwick", "unique_name"]


def test_combination_flags_analysis_eligibility() -> None:
    keys = {"player_id": [1], "season": [2025]}
    expected = pd.DataFrame(
        {**keys, "savant_player_name": ["Hitter"], "woba": [.360], "xwoba": [.370]}
    )
    bat = pd.DataFrame(
        {**keys, "avg_bat_speed": [74.0], "competitive_swings": [120]}
    )
    path = pd.DataFrame(
        {**keys, "attack_angle": [12.0], "distance_off_plate": [28.0]}
    )
    fangraphs = pd.DataFrame(
        {
            **keys,
            "fangraphs_player_name": ["Hitter"],
            "fangraphs_pa": [400],
            "ops": [.850],
            "wrc_plus": [130],
        }
    )
    bbref = pd.DataFrame(
        {
            **keys,
            "bbref_id": ["hitter01"],
            "bbref_player_name": ["Hitter"],
            "ops_plus": [125],
        }
    )
    config = CollectionConfig(min_pa=100, min_competitive_swings=50)

    result = combine_sources(expected, bat, path, fangraphs, bbref, config)

    assert len(result) == 1
    assert bool(result.iloc[0]["analysis_eligible"])
    assert result.iloc[0]["player_name"] == "Hitter"


def test_savant_html_response_is_rejected(tmp_path: Path) -> None:
    class FakeResponse:
        content = b"<html>blocked</html>"

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    with pytest.raises(DataSourceError, match="non-CSV"):
        _read_or_fetch_savant(
            "/leaderboard/test",
            {},
            tmp_path / "response.csv",
            False,
            FakeSession(),
        )
