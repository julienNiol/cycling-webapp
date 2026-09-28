import pandas as pd

# ---------------------------------------------------------------------------
# Generic validation helpers
# ---------------------------------------------------------------------------


def _validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """
    Validate that a DataFrame contains all required columns.
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"{table_name}: missing columns: {sorted(missing)}")


def _validate_unique_id(
    df: pd.DataFrame,
    column: str,
    table_name: str,
) -> None:
    """
    Validate that an identifier column contains no duplicates.
    """
    if df[column].isna().any():
        raise ValueError(f"{table_name}: column '{column}' contains missing values.")

    duplicates = df.loc[
        df[column].duplicated(keep=False),
        column,
    ]

    if not duplicates.empty:
        values = duplicates.unique().tolist()

        raise ValueError(
            f"{table_name}: column '{column}' contains duplicate IDs: {values[:10]}"
        )


def _validate_not_null(
    df: pd.DataFrame,
    column: str,
    table_name: str,
) -> None:
    """
    Validate that a column contains no missing values.
    """
    if df[column].isna().any():
        count = int(df[column].isna().sum())

        raise ValueError(
            f"{table_name}: column '{column}' contains {count} missing value(s)."
        )


def _validate_foreign_key(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    child_table: str,
    parent_table: str,
    *,
    allow_null: bool = False,
    missing_values: set = None,
) -> None:
    """Validate that values in a column exist in a parent table."""

    if missing_values is None:
        missing_values = set()

    values = child[child_column]

    # Valeurs considérées comme absentes
    missing_mask = values.isna() | values.isin(missing_values)

    # Une FK obligatoire ne peut pas contenir de valeur absente
    if not allow_null and missing_mask.any():
        invalid_rows = child.loc[missing_mask, [child_column]]

        raise ValueError(
            f"{child_table}.{child_column} contains missing values "
            f"but the foreign key is mandatory:\n{invalid_rows}"
        )

    # On retire les valeurs absentes avant de vérifier la FK
    values_to_check = values.loc[~missing_mask]

    parent_values = set(parent[parent_column])

    invalid_mask = ~values_to_check.isin(parent_values)

    if invalid_mask.any():
        invalid_values = values_to_check.loc[invalid_mask].unique()

        raise ValueError(
            f"Invalid foreign key in {child_table}.{child_column}: "
            f"{invalid_values.tolist()} do not exist in "
            f"{parent_table}.{parent_column}"
        )


def _validate_unique_combination(
    df: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    """
    Validate that a combination of columns is unique.
    """
    duplicates = df.loc[
        df.duplicated(columns, keep=False),
        columns,
    ]

    if not duplicates.empty:
        examples = duplicates.drop_duplicates().head(10).to_dict("records")

        raise ValueError(
            f"{table_name}: combination {columns} is not unique. Examples: {examples}"
        )


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------


def validate_countries(
    countries: pd.DataFrame,
) -> None:
    """
    Validate the countries table.
    """
    table = "countries"

    _validate_columns(
        countries,
        [
            "id_country",
            "name",
            "n_2",
            "n_3",
        ],
        table,
    )

    _validate_unique_id(
        countries,
        "id_country",
        table,
    )

    _validate_not_null(
        countries,
        "name",
        table,
    )

    _validate_not_null(
        countries,
        "n_2",
        table,
    )

    _validate_not_null(
        countries,
        "n_3",
        table,
    )

    if countries["n_2"].str.len().ne(2).any():
        raise ValueError("countries.n_2 must contain 2-character codes.")

    if countries["n_3"].str.len().ne(3).any():
        raise ValueError("countries.n_3 must contain 3-character codes.")

    _validate_unique_combination(
        countries,
        ["n_2"],
        table,
    )

    _validate_unique_combination(
        countries,
        ["n_3"],
        table,
    )


# ---------------------------------------------------------------------------
# Riders
# ---------------------------------------------------------------------------


def validate_riders(
    riders: pd.DataFrame,
    countries: pd.DataFrame,
) -> None:
    """
    Validate the riders table and its country references.
    """
    table = "riders"

    _validate_columns(
        riders,
        [
            "id_rider",
            "first_name",
            "surname",
            "nat_p",
            "nat_s",
        ],
        table,
    )

    _validate_unique_id(
        riders,
        "id_rider",
        table,
    )

    _validate_not_null(
        riders,
        "first_name",
        table,
    )

    _validate_not_null(
        riders,
        "surname",
        table,
    )

    _validate_foreign_key(
        riders,
        "nat_p",
        countries,
        "id_country",
        "riders",
        "countries",
    )

    _validate_foreign_key(
        riders,
        "nat_s",
        countries,
        "id_country",
        "riders",
        "countries",
        allow_null=True,
        missing_values={0},
    )


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------


def validate_races(
    races: pd.DataFrame,
    countries: pd.DataFrame,
) -> None:
    """
    Validate the races table and its country references.
    """
    table = "races"

    _validate_columns(
        races,
        [
            "id_race",
            "name",
            "name_fr",
            "cat",
            "id_nat",
        ],
        table,
    )

    _validate_unique_id(
        races,
        "id_race",
        table,
    )

    _validate_not_null(
        races,
        "name",
        table,
    )

    _validate_not_null(
        races,
        "cat",
        table,
    )

    _validate_foreign_key(
        races,
        "id_nat",
        countries,
        "id_country",
        "races",
        "countries",
        allow_null=True,
        missing_values={0},
    )


# ---------------------------------------------------------------------------
# Editions
# ---------------------------------------------------------------------------


def validate_editions(
    editions: pd.DataFrame,
    races: pd.DataFrame,
) -> None:
    """
    Validate the editions table and its race references.
    """
    table = "editions"

    _validate_columns(
        editions,
        [
            "id_edition",
            "id_race",
            "year",
            "cat",
            "s_cat",
        ],
        table,
    )

    _validate_unique_id(
        editions,
        "id_edition",
        table,
    )

    _validate_not_null(
        editions,
        "id_race",
        table,
    )

    _validate_not_null(
        editions,
        "year",
        table,
    )

    _validate_not_null(
        editions,
        "cat",
        table,
    )

    _validate_foreign_key(
        editions,
        "id_race",
        races,
        "id_race",
        "editions",
        "races",
    )

    if (editions["year"] < 1900).any():
        raise ValueError("editions.year contains an invalid year (< 1900).")

    if (editions["year"] > 2100).any():
        raise ValueError("editions.year contains an invalid year (> 2100).")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def validate_results(
    results: pd.DataFrame,
    riders: pd.DataFrame,
    editions: pd.DataFrame,
) -> None:
    """
    Validate the results table and its references.
    """
    table = "results"

    _validate_columns(
        results,
        [
            "id_result",
            "id_edition",
            "place",
            "id_rider",
            "score",
        ],
        table,
    )

    _validate_unique_id(
        results,
        "id_result",
        table,
    )

    _validate_not_null(
        results,
        "id_edition",
        table,
    )

    _validate_not_null(
        results,
        "id_rider",
        table,
    )

    _validate_not_null(
        results,
        "score",
        table,
    )

    _validate_foreign_key(
        results,
        "id_edition",
        editions,
        "id_edition",
        "results",
        "editions",
    )

    _validate_foreign_key(
        results,
        "id_rider",
        riders,
        "id_rider",
        "results",
        "riders",
        allow_null=True,
        missing_values={0},
    )

    # A final ranking position, when present, must be positive.
    valid_places = results["place"].dropna()

    if (valid_places <= 0).any():
        raise ValueError(
            "results.place must be strictly positive when a place is provided."
        )

    # Scores must not be negative.
    if (results["score"] < 0).any():
        raise ValueError("results.score cannot contain negative values.")


# ---------------------------------------------------------------------------
# Global validation
# ---------------------------------------------------------------------------


def validate_data(
    countries: pd.DataFrame,
    riders: pd.DataFrame,
    races: pd.DataFrame,
    editions: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """
    Validate the complete dataset.

    Raises
    ------
    ValueError
        If one or more integrity constraints are violated.
    """

    validate_countries(countries)

    validate_riders(
        riders,
        countries,
    )

    validate_races(
        races,
        countries,
    )

    validate_editions(
        editions,
        races,
    )

    validate_results(
        results,
        riders,
        editions,
    )
