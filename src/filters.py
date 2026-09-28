from __future__ import annotations

import pandas as pd


def filter_by_year(
    df: pd.DataFrame,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Filter results by an inclusive year range.

    If only start_year is provided, years greater than or equal to it
    are kept.

    If only end_year is provided, years less than or equal to it
    are kept.

    If start_year and end_year are equal, only that year is kept.

    Parameters
    ----------
    df:
        DataFrame containing a ``year`` column.
    start_year:
        First year of the period, inclusive.
    end_year:
        Last year of the period, inclusive.
    """
    if start_year is None and end_year is None:
        return df.copy()

    if "year" not in df.columns:
        raise KeyError("Column 'year' not found in DataFrame.")

    if start_year is not None and end_year is not None:
        if start_year > end_year:
            raise ValueError("start_year must be less than or equal to end_year.")

    mask = pd.Series(True, index=df.index)

    if start_year is not None:
        mask &= df["year"] >= start_year

    if end_year is not None:
        mask &= df["year"] <= end_year

    return df.loc[mask].copy()


def filter_by_rider(
    df: pd.DataFrame,
    rider_ids: int | list[int] | None = None,
) -> pd.DataFrame:
    """Filter results by rider ID.

    ``rider_ids`` can be a single rider ID or a list of IDs.
    ``None`` means that no rider filter is applied.

    Rider ID 0 is treated as a valid value here, representing
    an unidentified rider in the source data.
    """
    if rider_ids is None:
        return df.copy()

    if "id_rider" not in df.columns:
        raise KeyError("Column 'id_rider' not found in DataFrame.")

    if isinstance(rider_ids, int):
        rider_ids = [rider_ids]

    rider_ids = list(rider_ids)

    if not rider_ids:
        return df.copy()

    return df[df["id_rider"].isin(rider_ids)].copy()


def filter_by_country(
    df: pd.DataFrame,
    country_id: int | None = None,
) -> pd.DataFrame:
    """Filter results by the selected country.

    This filter is intended for the 'Nation' page and uses the
    rider's primary nationality (``nat_p``).
    """
    if country_id is None:
        return df.copy()

    if "nat_p" not in df.columns:
        raise KeyError("Column 'nat_p' not found in DataFrame.")

    return df[df["nat_p"] == country_id].copy()


def filter_by_rider_nationality(
    df: pd.DataFrame,
    country_ids: int | list[int] | None = None,
) -> pd.DataFrame:
    """Filter results by the rider's primary nationality.

    Only ``nat_p`` is considered. ``nat_s`` is deliberately ignored.
    """
    if country_ids is None:
        return df.copy()

    if "nat_p" not in df.columns:
        raise KeyError("Column 'nat_p' not found in DataFrame.")

    if isinstance(country_ids, int):
        country_ids = [country_ids]

    country_ids = list(country_ids)

    if not country_ids:
        return df.copy()

    return df[df["nat_p"].isin(country_ids)].copy()


def filter_by_race_nationality(
    df: pd.DataFrame,
    country_ids: int | list[int] | None = None,
) -> pd.DataFrame:
    """Filter results by the nationality of the race.

    The race nationality is represented by ``id_nat`` in ``races``.
    """
    if country_ids is None:
        return df.copy()

    if "id_nat" not in df.columns:
        raise KeyError("Column 'id_nat' not found in DataFrame.")

    if isinstance(country_ids, int):
        country_ids = [country_ids]

    country_ids = list(country_ids)

    if not country_ids:
        return df.copy()

    return df[df["id_nat"].isin(country_ids)].copy()


def filter_by_edition_category(
    df: pd.DataFrame,
    categories: str | list[str] | None = None,
) -> pd.DataFrame:
    """Filter results by edition category.

    Uses ``edition_cat`` from ``editions.cat``.
    """
    if categories is None:
        return df.copy()

    if "edition_cat" not in df.columns:
        raise KeyError("Column 'edition_cat' not found in DataFrame.")

    if isinstance(categories, str):
        categories = [categories]

    categories = list(categories)

    if not categories:
        return df.copy()

    return df[df["edition_cat"].isin(categories)].copy()


def filter_by_edition_sub_category(
    df: pd.DataFrame,
    sub_categories: str | list[str] | None = None,
) -> pd.DataFrame:
    """Filter results by edition sub-category.

    Uses ``s_cat`` from ``editions.s_cat``.
    """
    if sub_categories is None:
        return df.copy()

    if "s_cat" not in df.columns:
        raise KeyError("Column 's_cat' not found in DataFrame.")

    if isinstance(sub_categories, str):
        sub_categories = [sub_categories]

    sub_categories = list(sub_categories)

    if not sub_categories:
        return df.copy()

    return df[df["s_cat"].isin(sub_categories)].copy()


def filter_by_race(
    df: pd.DataFrame,
    race_ids: int | list[int] | None = None,
) -> pd.DataFrame:
    """Filter results by race ID."""
    if race_ids is None:
        return df.copy()

    if "id_race" not in df.columns:
        raise KeyError("Column 'id_race' not found in DataFrame.")

    if isinstance(race_ids, int):
        race_ids = [race_ids]

    race_ids = list(race_ids)

    if not race_ids:
        return df.copy()

    return df[df["id_race"].isin(race_ids)].copy()


def filter_by_max_place(
    df: pd.DataFrame,
    max_place: int | None = None,
) -> pd.DataFrame:
    """Filter results up to a maximum finishing place.

    A ``max_place`` of ``None`` means that the filter is inactive.

    The source data uses 0 to represent a missing/unidentified place.
    Such rows are therefore excluded when this filter is active.

    ``max_place`` must be between 1 and 20.
    """
    if max_place is None:
        return df.copy()

    if "place" not in df.columns:
        raise KeyError("Column 'place' not found in DataFrame.")

    if not 1 <= max_place <= 20:
        raise ValueError("max_place must be between 1 and 20.")

    return df[(df["place"] > 0) & (df["place"] <= max_place)].copy()
