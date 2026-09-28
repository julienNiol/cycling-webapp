from __future__ import annotations

import pandas as pd

F1_POINTS = {
    1: 25,
    2: 18,
    3: 15,
    4: 12,
    5: 10,
    6: 8,
    7: 6,
    8: 4,
    9: 2,
    10: 1,
}


def _require_columns(
    df: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Check that all required columns are present."""
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(f"Missing columns: {sorted(missing_columns)}")


def rank_by_score(
    df: pd.DataFrame,
    *,
    score_column: str = "score",
) -> pd.DataFrame:
    """
    Rank rows by score in descending order.

    Equal scores receive the same rank.
    The following rank skips the occupied positions.

    Example:
        scores 100, 90, 90, 80
        ranks  1,   2,  2,  4
    """
    _require_columns(df, {score_column})

    result = df.copy()

    result["rank"] = (
        result[score_column]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("int64")
    )

    return result.sort_values(
        [score_column],
        ascending=[False],
    ).reset_index(drop=True)


def rank_riders_yearly_and_cumulative(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate annual and cumulative rankings for every rider and year.

    Annual ranking:
    - based on the rider's score for the year;
    - only exists for years in which the rider has results.

    Cumulative ranking:
    - based on the score accumulated from the first year of the
      database through the current year;
    - continues after a rider's last participation;
    - a rider only enters the cumulative ranking from the first year
      in which they appear.

    Missing years after a rider's first appearance are therefore kept
    so that a retired rider can continue to have a cumulative rank.

    Returns:
        id_rider
        year
        score
        rank_year
        score_cumulative
        rank_cumulative
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "score",
        },
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "id_rider",
                "year",
                "score",
                "rank_year",
                "score_cumulative",
                "rank_cumulative",
            ]
        )

    source = df[df["id_rider"] != 0].copy()

    if source.empty:
        return pd.DataFrame(
            columns=[
                "id_rider",
                "year",
                "score",
                "rank_year",
                "score_cumulative",
                "rank_cumulative",
            ]
        )

    # Annual score for every rider/year actually present in the data.
    yearly = source.groupby(
        ["id_rider", "year"],
        as_index=False,
    ).agg(score=("score", "sum"))

    # Keep track of years in which the rider actually has results.
    yearly["has_results"] = True

    # First appearance of each rider.
    first_year = (
        yearly.groupby("id_rider", as_index=False)["year"]
        .min()
        .rename(columns={"year": "first_year"})
    )

    # Last year available in the complete database.
    last_year = yearly["year"].max()

    # Create one row per rider/year from their first appearance
    # until the last year in the database.
    riders = first_year["id_rider"].tolist()

    panels = []

    for rider_id in riders:
        rider_first_year = first_year.loc[
            first_year["id_rider"] == rider_id,
            "first_year",
        ].iloc[0]

        panels.append(
            pd.DataFrame(
                {
                    "id_rider": rider_id,
                    "year": range(
                        int(rider_first_year),
                        int(last_year) + 1,
                    ),
                }
            )
        )

    panel = pd.concat(
        panels,
        ignore_index=True,
    )

    # Add annual scores. Missing years receive 0 for the cumulative
    # calculation, but remain identifiable as years without results.
    panel = panel.merge(
        yearly,
        on=["id_rider", "year"],
        how="left",
    )

    panel["has_results"] = panel["has_results"].fillna(False)
    panel["score"] = panel["score"].fillna(0.0)

    # Cumulative score is calculated independently for every rider.
    panel = panel.sort_values(
        ["id_rider", "year"],
    ).reset_index(drop=True)

    panel["score_cumulative"] = panel.groupby("id_rider")["score"].cumsum()

    # Annual rank only exists when the rider has results that year.
    panel["rank_year"] = pd.NA

    has_results = panel["has_results"]

    panel.loc[has_results, "rank_year"] = (
        panel.loc[has_results]
        .groupby("year")["score"]
        .rank(
            method="min",
            ascending=False,
        )
    )

    panel["rank_year"] = panel["rank_year"].astype("Int64")

    # Cumulative rank is calculated among all riders who have already
    # appeared in the database by the given year.
    panel["rank_cumulative"] = (
        panel.groupby("year")["score_cumulative"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("int64")
    )

    return (
        panel[
            [
                "id_rider",
                "year",
                "score",
                "rank_year",
                "score_cumulative",
                "rank_cumulative",
            ]
        ]
        .sort_values(
            ["year", "rank_cumulative", "id_rider"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )


def add_relative_score(
    df: pd.DataFrame,
    *,
    score_column: str = "score",
    relative_column: str = "score_relative",
) -> pd.DataFrame:
    """
    Add a relative score expressed on a 0-100 scale.

    The highest score in the provided dataframe is 100.
    The reference score is therefore calculated after all
    active filters have been applied.
    """
    _require_columns(df, {score_column})

    result = df.copy()

    if result.empty:
        result[relative_column] = pd.Series(
            dtype="float64",
        )
        return result

    first_score = result[score_column].max()

    if first_score == 0:
        result[relative_column] = 0.0
    else:
        result[relative_column] = result[score_column] / first_score * 100

    return result


def rank_with_relative_score(
    df: pd.DataFrame,
    *,
    score_column: str = "score",
    relative_column: str = "score_relative",
) -> pd.DataFrame:
    """
    Rank rows by score and add their relative score on 100.
    """
    result = rank_by_score(
        df,
        score_column=score_column,
    )

    return add_relative_score(
        result,
        score_column=score_column,
        relative_column=relative_column,
    )


def _add_f1_points(
    df: pd.DataFrame,
    *,
    rank_column: str = "rank",
    points_column: str = "f1_points",
) -> pd.DataFrame:
    """Assign F1 points according to the rank."""
    _require_columns(df, {rank_column})

    result = df.copy()

    result[points_column] = result[rank_column].map(F1_POINTS).fillna(0).astype("int64")

    return result


def rank_f1(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the cumulative F1 ranking.

    For each year:
    1. riders are ranked by their annual score;
    2. equal scores receive the same rank;
    3. F1 points are assigned according to the rank;
    4. all yearly F1 points are summed.

    The input dataframe must already contain the filtered results
    and must have one row per rider/year, typically produced by
    aggregate_rider_year().

    Ties are preserved. There is no tie-breaker.

    A tie around the 10th place can therefore result in more than
    10 riders receiving F1 points.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "score",
        },
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "id_rider",
                "score",
                "f1_points",
            ]
        )

    yearly = df.copy()

    yearly["rank"] = (
        yearly.groupby("year")["score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("int64")
    )

    yearly["f1_points"] = yearly["rank"].map(F1_POINTS).fillna(0).astype("int64")

    yearly = yearly[yearly["f1_points"] > 0]

    if yearly.empty:
        return pd.DataFrame(
            columns=[
                "id_rider",
                "score",
                "f1_points",
            ]
        )

    result = (
        yearly.groupby("id_rider", as_index=False)
        .agg(
            score=("score", "sum"),
            f1_points=("f1_points", "sum"),
        )
        .sort_values(
            ["f1_points", "score"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    result["rank"] = (
        result["f1_points"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("int64")
    )

    return result
