from __future__ import annotations

import pandas as pd


def _require_columns(
    df: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Check that all required columns are present."""
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(f"Missing columns: {sorted(missing_columns)}")


def _add_place_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add placement-based indicator columns."""
    _require_columns(df, {"place"})

    result = df.copy()

    result["wins"] = result["place"].eq(1).astype("int64")
    result["podiums"] = result["place"].between(1, 3).astype("int64")
    result["top_5"] = result["place"].between(1, 5).astype("int64")
    result["top_10"] = result["place"].between(1, 10).astype("int64")

    return result


def compute_global_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute global statistics for the filtered results.

    Results with id_rider == 0 are excluded from rider-specific
    statistics.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "score",
            "place",
        },
    )

    result = df[df["id_rider"] != 0].copy()
    result = _add_place_metrics(result)

    return pd.DataFrame(
        {
            "results_count": [len(result)],
            "riders_count": [result["id_rider"].nunique()],
            "score": [result["score"].sum()],
            "wins": [result["wins"].sum()],
            "podiums": [result["podiums"].sum()],
            "top_5": [result["top_5"].sum()],
            "top_10": [result["top_10"].sum()],
        }
    )


def compute_rider_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics for each rider.

    Returns one row per rider.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "score",
            "place",
        },
    )

    result = df[df["id_rider"] != 0].copy()
    result = _add_place_metrics(result)

    return (
        result.groupby("id_rider", as_index=False)
        .agg(
            results_count=("id_rider", "size"),
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
        )
        .sort_values(
            ["score", "id_rider"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def compute_rider_identity(
    df: pd.DataFrame,
    rider_id: int,
) -> pd.DataFrame:
    """
    Return the identity information of a rider.

    The dataframe is expected to be enriched and therefore to contain
    the rider's nationality information.

    Returns one row for the selected rider.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "first_name",
            "surname",
            "nat_p",
            "nat_s",
            "country_n3_p",
            "country_n3_s",
        },
    )

    result = df[df["id_rider"] == rider_id].copy()

    if result.empty:
        return pd.DataFrame(
            columns=[
                "id_rider",
                "first_name",
                "surname",
                "nat_p",
                "nat_s",
                "country_n3_p",
                "country_n3_s",
            ]
        )

    return (
        result[
            [
                "id_rider",
                "first_name",
                "surname",
                "nat_p",
                "nat_s",
                "country_n3_p",
                "country_n3_s",
            ]
        ]
        .drop_duplicates(subset=["id_rider"])
        .reset_index(drop=True)
    )


def compute_rider_career_span(
    df: pd.DataFrame,
    rider_id: int,
) -> pd.DataFrame:
    """
    Return the first and last year in which a rider appears.

    The career span is calculated from the complete dataframe and is
    therefore independent of the page filters.

    Returns one row with:
    - id_rider
    - start_year
    - end_year
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
        },
    )

    result = df[df["id_rider"] == rider_id].copy()

    if result.empty:
        return pd.DataFrame(
            columns=[
                "id_rider",
                "start_year",
                "end_year",
            ]
        )

    return pd.DataFrame(
        {
            "id_rider": [rider_id],
            "start_year": [result["year"].min()],
            "end_year": [result["year"].max()],
        }
    )


def compute_rider_year_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics for each rider and year.

    Returns one row per rider/year.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "score",
            "place",
        },
    )

    result = df[df["id_rider"] != 0].copy()
    result = _add_place_metrics(result)

    return (
        result.groupby(
            ["id_rider", "year"],
            as_index=False,
        )
        .agg(
            results_count=("id_rider", "size"),
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
        )
        .sort_values(
            ["year", "score", "id_rider"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def compute_rider_category_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics for each rider, year and edition category.

    Returns one row per rider/year/category.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "edition_cat",
            "score",
            "place",
        },
    )

    result = df[df["id_rider"] != 0].copy()
    result = _add_place_metrics(result)

    return (
        result.groupby(
            ["id_rider", "year", "edition_cat"],
            as_index=False,
        )
        .agg(
            results_count=("id_rider", "size"),
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
        )
        .sort_values(
            [
                "id_rider",
                "year",
                "edition_cat",
            ],
        )
        .reset_index(drop=True)
    )


def compute_nation_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics for each primary nationality.

    nat_p is used as the rider's nationality.
    nat_s is deliberately ignored.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "nat_p",
            "score",
            "place",
        },
    )

    result = df[(df["id_rider"] != 0) & df["nat_p"].notna()].copy()

    result = _add_place_metrics(result)

    return (
        result.groupby("nat_p", as_index=False)
        .agg(
            results_count=("id_rider", "size"),
            riders_count=("id_rider", "nunique"),
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
        )
        .sort_values(
            ["score", "nat_p"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def compute_year_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute global statistics for each year.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "score",
            "place",
        },
    )

    result = df[df["id_rider"] != 0].copy()
    result = _add_place_metrics(result)

    return (
        result.groupby("year", as_index=False)
        .agg(
            results_count=("id_rider", "size"),
            riders_count=("id_rider", "nunique"),
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )


def compute_race_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics for each logical race.

    Stage results are grouped with the corresponding general
    classification according to the normalization performed in
    aggregations.normalize_race_for_aggregation().
    """
    _require_columns(
        df,
        {
            "race_id_aggregate",
            "id_rider",
            "score",
            "place",
            "edition_cat",
        },
    )

    result = df[df["id_rider"] != 0].copy()
    result = _add_place_metrics(result)

    result["stage_wins"] = (
        result["edition_cat"].eq("S") & result["place"].eq(1)
    ).astype("int64")

    return (
        result.groupby(
            "race_id_aggregate",
            as_index=False,
        )
        .agg(
            results_count=("id_rider", "size"),
            riders_count=("id_rider", "nunique"),
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
            stage_wins=("stage_wins", "sum"),
        )
        .sort_values(
            ["score", "race_id_aggregate"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
