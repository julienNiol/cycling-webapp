from pathlib import Path

import pandas as pd

from dataloader.validate import validate_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data/raw"


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------


def _load_csv(
    filename: str,
    *,
    dtype: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load a CSV file from the project's data directory.

    Parameters
    ----------
    filename:
        Name of the CSV file.
    dtype:
        Optional pandas dtype mapping.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(
        path,
        header=0,
        sep=";",
        dtype=dtype,
        keep_default_na=False,
    )


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def load_countries() -> pd.DataFrame:
    """
    Load countries data.
    """
    return _load_csv(
        "countries.csv",
        dtype={
            "id_country": "int64",
            "name": "string",
            "n_2": "string",
            "n_3": "string",
        },
    )


def load_riders() -> pd.DataFrame:
    """
    Load riders data.
    """
    return _load_csv(
        "riders.csv",
        dtype={
            "id_rider": "int64",
            "first_name": "string",
            "surname": "string",
            "nat_p": "Int64",
            "nat_s": "Int64",
        },
    )


def load_races() -> pd.DataFrame:
    """
    Load races data.
    """
    return _load_csv(
        "races.csv",
        dtype={
            "id_race": "int64",
            "name": "string",
            "name_fr": "string",
            "cat": "string",
            "id_nat": "Int64",
        },
    )


def load_editions() -> pd.DataFrame:
    """
    Load race editions data.
    """
    return _load_csv(
        "editions.csv",
        dtype={
            "id_edition": "int64",
            "id_race": "int64",
            "year": "int64",
            "cat": "string",
            "s_cat": "string",
        },
    )


def load_results() -> pd.DataFrame:
    """
    Load results data.
    """
    return _load_csv(
        "results.csv",
        dtype={
            "id_result": "int64",
            "id_edition": "int64",
            "place": "Int64",
            "id_rider": "int64",
            "score": "float64",
        },
    )


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrich_results(
    countries: pd.DataFrame,
    riders: pd.DataFrame,
    races: pd.DataFrame,
    editions: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical enriched results DataFrame.

    The resulting DataFrame keeps one row per result and adds:
    - edition information,
    - race information,
    - rider information,
    - primary nationality information.

    IDs are intentionally preserved because they remain useful for
    calculations and joins.
    """

    edition_columns = [
        "id_edition",
        "id_race",
        "year",
        "cat",
        "s_cat",
    ]

    race_columns = [
        "id_race",
        "name",
        "name_fr",
        "cat",
        "id_nat",
    ]

    rider_columns = [
        "id_rider",
        "first_name",
        "surname",
        "nat_p",
        "nat_s",
    ]

    country_columns = [
        "id_country",
        "n_2",
        "n_3",
    ]

    enriched = (
        results.merge(
            editions[edition_columns],
            on="id_edition",
            how="left",
            validate="many_to_one",
        )
        .rename(
            columns={
                "cat": "edition_cat",
            }
        )
        .merge(
            races[race_columns],
            on="id_race",
            how="left",
            validate="many_to_one",
        )
        .rename(
            columns={
                "name": "race_name",
                "name_fr": "race_name_fr",
                "cat": "race_cat",
            }
        )
        .merge(
            riders[rider_columns],
            on="id_rider",
            how="left",
            validate="many_to_one",
        )
        .merge(
            countries[country_columns],
            left_on="nat_p",
            right_on="id_country",
            how="left",
            validate="many_to_one",
        )
        .rename(
            columns={
                "n_2": "country_n2_p",
                "n_3": "country_n3_p",
            }
        )
        .drop(columns=["id_country"])
        .merge(
            countries[country_columns],
            left_on="nat_s",
            right_on="id_country",
            how="left",
            validate="many_to_one",
        )
        .rename(
            columns={
                "n_2": "country_n2_s",
                "n_3": "country_n3_s",
            }
        )
        .drop(columns=["id_country"])
    )

    return enriched


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_data() -> dict[str, pd.DataFrame]:
    """
    Load all application data.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing both raw and enriched DataFrames.

    Keys
    ----
    countries
        Countries reference table.

    riders
        Riders reference table.

    races
        Races reference table.

    editions
        Race editions table.

    results
        Raw results fact table.

    results_enriched
        Results enriched with edition, race, rider and nationality data.
    """

    countries = load_countries()
    riders = load_riders()
    races = load_races()
    editions = load_editions()
    results = load_results()

    validate_data(
        countries,
        riders,
        races,
        editions,
        results,
    )

    results_enriched = enrich_results(
        countries=countries,
        riders=riders,
        races=races,
        editions=editions,
        results=results,
    )

    return {
        "countries": countries,
        "riders": riders,
        "races": races,
        "editions": editions,
        "results": results,
        "results_enriched": results_enriched,
    }
