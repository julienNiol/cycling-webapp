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


def _add_place_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add placement-based indicator columns.

    The source data guarantees that place is between 1 and 20.
    """
    _require_columns(df, {"place"})

    result = df.copy()

    result["wins"] = result["place"].eq(1).astype("int64")
    result["podiums"] = result["place"].between(1, 3).astype("int64")
    result["top_5"] = result["place"].between(1, 5).astype("int64")
    result["top_10"] = result["place"].between(1, 10).astype("int64")

    return result


def aggregate_rider_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate scores by rider.

    Results with id_rider == 0 are excluded because they cannot
    be attributed to a rider.
    """
    _require_columns(df, {"id_rider", "score"})

    result = df[df["id_rider"] != 0]

    return result.groupby("id_rider", as_index=False).agg(score=("score", "sum"))


def aggregate_rider_season_category(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate a rider's results by year and edition category.

    Metrics:
    - score
    - wins
    - podiums
    - top 5
    - top 10
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "edition_cat",
            "place",
            "score",
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
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
        )
        .sort_values(["id_rider", "year", "edition_cat"])
        .reset_index(drop=True)
    )


def aggregate_rider_year(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate the score of each rider for each year.

    nat_p is retained because this aggregation is also used
    for nation rankings.
    """
    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "nat_p",
            "score",
        },
    )

    result = df[df["id_rider"] != 0]

    return result.groupby(
        ["id_rider", "year", "nat_p"],
        as_index=False,
    ).agg(score=("score", "sum"))


def select_top_n_riders_by_nationality(
    df: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    Keep the N highest-scoring riders for each year and nationality.

    The top N is therefore calculated independently for every
    (year, nationality) pair.
    """
    if n < 1:
        raise ValueError("n must be greater than or equal to 1.")

    _require_columns(
        df,
        {
            "id_rider",
            "year",
            "nat_p",
            "score",
        },
    )

    result = (
        df.sort_values(
            ["year", "nat_p", "score", "id_rider"],
            ascending=[True, True, False, True],
        )
        .groupby(
            ["year", "nat_p"],
            sort=False,
            group_keys=False,
        )
        .head(n)
        .reset_index(drop=True)
    )

    return result


def aggregate_nation_ranking(
    df: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    Aggregate the nation ranking over all selected years.

    For each year:
    1. calculate each rider's annual score;
    2. keep the N best riders for each nationality;
    3. sum their scores for that year.

    Then sum the yearly scores over all years present in df.

    The top N is therefore determined separately for each year
    before the multi-year cumulative score is calculated.
    """
    rider_scores = aggregate_rider_year(df)

    top_riders = select_top_n_riders_by_nationality(
        rider_scores,
        n=n,
    )

    nation_year_scores = top_riders.groupby(
        ["year", "nat_p"],
        as_index=False,
    ).agg(
        score=("score", "sum"),
    )

    return (
        nation_year_scores.groupby("nat_p", as_index=False)
        .agg(score=("score", "sum"))
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def normalize_race_for_aggregation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize race identifiers so that a stage race's general
    classification and stages are attached to the same logical race.

    Business rules:
    - edition_cat == "T": general classification
    - edition_cat == "S": stage
    - stage id_race = general id_race + 1
    - stage race names end with "(S)"
    """
    _require_columns(
        df,
        {
            "id_race",
            "race_name",
            "edition_cat",
        },
    )

    result = df.copy()

    is_stage = result["edition_cat"].eq("S")

    result["race_id_aggregate"] = result["id_race"]

    result.loc[is_stage, "race_id_aggregate"] -= 1

    result["race_name_aggregate"] = result["race_name"].str.replace(
        r"\s*\(S\)\s*$",
        "",
        regex=True,
    )

    return result


def aggregate_race_riders(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate results by logical race and rider.

    For stage races, the general classification and stages are
    aggregated into the same logical race.

    Metrics:
    - score: general classifications + stages
    - wins: general classifications only
    - podiums: general classifications only
    - top 5: general classifications only
    - top 10: general classifications only
    - stage wins: stages only
    """
    _require_columns(
        df,
        {
            "id_race",
            "id_rider",
            "edition_cat",
            "place",
            "score",
            "race_name",
        },
    )

    result = df[df["id_rider"] != 0].copy()

    result = normalize_race_for_aggregation(result)

    # Les statistiques de classement général ne doivent pas
    # comptabiliser les étapes.
    general_results = result[result["edition_cat"] != "S"].copy()

    general_results = _add_place_metrics(general_results)

    # On conserve les résultats d'étapes uniquement pour
    # calculer les victoires d'étape et les points.
    stage_results = result[result["edition_cat"] == "S"].copy()

    stage_results["stage_wins"] = (stage_results["place"].eq(1)).astype("int64")

    # Pour les résultats généraux, aucune victoire d'étape.
    general_results["stage_wins"] = 0

    result = pd.concat(
        [
            general_results,
            stage_results,
        ],
        ignore_index=True,
    )

    return (
        result.groupby(
            [
                "race_id_aggregate",
                "id_rider",
            ],
            as_index=False,
        )
        .agg(
            score=("score", "sum"),
            wins=("wins", "sum"),
            podiums=("podiums", "sum"),
            top_5=("top_5", "sum"),
            top_10=("top_10", "sum"),
            stage_wins=("stage_wins", "sum"),
        )
        .sort_values(
            [
                "race_id_aggregate",
                "score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def aggregate_year_calendar(
    df: pd.DataFrame,
    excluded_categories: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build the calendar for a selected year.

    The dataframe is expected to have already been filtered to the
    requested year.

    Only editions whose category is not excluded are retained.

    The winner is the rider with place == 1. If several rows exist
    for the same edition with place == 1, the rider with the lowest
    id_rider is retained.

    The aggregation keeps all information required to display the
    calendar:
    - edition identifier
    - race identifier
    - year
    - edition category
    - edition sub-category
    - race name
    - race nationality
    - winner identity

    The calendar is sorted by id_edition.
    """
    _require_columns(
        df,
        {
            "id_edition",
            "id_race",
            "year",
            "edition_cat",
            "s_cat",
            "race_name",
            "id_nat",
            "id_rider",
            "place",
            "first_name",
            "surname",
        },
    )

    result = df.copy()

    if excluded_categories:
        result = result[~result["edition_cat"].isin(excluded_categories)]

    winners = (
        result[result["place"].eq(1)]
        .sort_values(
            ["id_edition", "id_rider"],
        )
        .drop_duplicates(
            subset=["id_edition"],
            keep="first",
        )
    )

    return (
        winners[
            [
                "id_edition",
                "id_race",
                "year",
                "edition_cat",
                "s_cat",
                "race_name",
                "id_nat",
                "id_rider",
                "first_name",
                "surname",
            ]
        ]
        .sort_values("id_edition")
        .reset_index(drop=True)
    )


def aggregate_race_palmares(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the palmares of a race.

    One row is returned per edition, using only the main race result.
    Stage results are excluded.

    The winner is the rider with place == 1.
    """

    _require_columns(
        df,
        {
            "id_edition",
            "year",
            "edition_cat",
            "id_rider",
            "place",
            "first_name",
            "surname",
            "nat_p",
        },
    )

    result = df[~df["edition_cat"].eq("S")].copy()

    winners = (
        result[result["place"].eq(1)]
        .sort_values(
            ["year", "id_edition", "id_rider"],
        )
        .drop_duplicates(
            subset=["id_edition"],
            keep="first",
        )
    )

    return (
        winners[
            [
                "id_edition",
                "year",
                "id_rider",
                "first_name",
                "surname",
                "nat_p",
            ]
        ]
        .sort_values(
            ["year", "id_edition"],
        )
        .reset_index(drop=True)
    )
