from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agregations import aggregate_rider_scores, aggregate_rider_year
from src.filters import (
    filter_by_edition_category,
    filter_by_edition_sub_category,
    filter_by_max_place,
    filter_by_race,
    filter_by_race_nationality,
    filter_by_rider,
    filter_by_rider_nationality,
    filter_by_year,
)
from src.ranking import rank_by_score, rank_f1, rank_with_relative_score

data = st.session_state["data"]

countries = data["countries"]
riders = data["riders"]
results_enriched = data["results_enriched"]


def build_rider_mapping(riders: pd.DataFrame) -> dict[str, int]:
    """Construit un mapping nom affiché -> identifiant coureur."""
    display_names = (
        riders["first_name"].fillna("") + " " + riders["surname"].fillna("")
    ).str.strip()

    return dict(zip(display_names, riders["id_rider"]))


def build_country_mapping(
    countries: pd.DataFrame,
    country_ids: pd.Series,
) -> dict[str, int]:
    """Construit un mapping nom de pays -> identifiant pays."""
    ids = pd.Series(country_ids).dropna().unique()

    available = countries[countries["id_country"].isin(ids)].copy()

    return dict(
        zip(
            available["name"],
            available["id_country"],
        )
    )


def prepare_absolute_table(
    ranked_df: pd.DataFrame,
    riders: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau d'affichage du classement absolu."""

    result = ranked_df.merge(
        riders[["id_rider", "first_name", "surname", "nat_p"]],
        on="id_rider",
        how="left",
    )

    result = result.merge(
        countries[["id_country", "n_3"]],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    result["Nom"] = (
        result["first_name"].fillna("") + " " + result["surname"].fillna("")
    ).str.strip()

    result = result[["rank", "Nom", "n_3", "score"]].copy()

    return result.rename(
        columns={
            "rank": "Rang",
            "n_3": "Nat",
            "score": "Points",
        }
    )


def prepare_relative_table(
    ranked_df: pd.DataFrame,
    riders: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau d'affichage du classement relatif."""

    result = ranked_df.merge(
        riders[["id_rider", "first_name", "surname", "nat_p"]],
        on="id_rider",
        how="left",
    )

    result = result.merge(
        countries[["id_country", "n_3"]],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    result["Nom"] = (
        result["first_name"].fillna("") + " " + result["surname"].fillna("")
    ).str.strip()

    result = result[["rank", "Nom", "n_3", "score_relative"]].copy()

    return result.rename(
        columns={
            "rank": "Rang",
            "n_3": "Nat",
            "score_relative": "Ratio",
        }
    )


def prepare_f1_table(
    ranked_df: pd.DataFrame,
    riders: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau d'affichage du classement F1."""

    result = ranked_df.merge(
        riders[["id_rider", "first_name", "surname", "nat_p"]],
        on="id_rider",
        how="left",
    )

    result = result.merge(
        countries[["id_country", "n_3"]],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    result["Nom"] = (
        result["first_name"].fillna("") + " " + result["surname"].fillna("")
    ).str.strip()

    result = result[["rank", "Nom", "n_3", "f1_points"]].copy()

    return result.rename(
        columns={
            "rank": "Rang",
            "n_3": "Nat",
            "f1_points": "Points",
        }
    )


def prepare_year_winners_table(
    rider_year_scores: pd.DataFrame,
    riders: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau des vainqueurs par année."""

    winners = (
        rider_year_scores.sort_values(
            ["year", "score"],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset="year",
            keep="first",
        )
        .copy()
    )

    winners = winners.merge(
        riders[["id_rider", "first_name", "surname"]],
        on="id_rider",
        how="left",
    )

    winners["Vainqueur"] = (
        winners["first_name"].fillna("") + " " + winners["surname"].fillna("")
    ).str.strip()

    winners = winners[["year", "Vainqueur"]]

    return winners.rename(
        columns={
            "year": "Année",
        }
    )


def get_table_height(
    df: pd.DataFrame,
    row_height: int = 35,
    header_height: int = 38,
    max_height: int = 560,
) -> int:
    """Calcule une hauteur adaptée au nombre de lignes."""
    return min(
        header_height + len(df) * row_height,
        max_height,
    )


st.title("🏆 Classement")


# ---------------------------------------------------------------------------
# Type de classement
# ---------------------------------------------------------------------------

ranking_type = st.radio(
    "Type de classement",
    options=[
        "Absolu",
        "Relatif",
        "F1",
    ],
    index=0,
    horizontal=True,
)

# Coureurs
rider_mapping = build_rider_mapping(riders)

selected_riders = st.multiselect(
    "Cyclistes",
    options=sorted(rider_mapping),
)

selected_rider_ids = [rider_mapping[name] for name in selected_riders]


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------

st.sidebar.header("Filtres")


# Années
min_year = int(results_enriched["year"].min())
max_year = int(results_enriched["year"].max())

year_range = st.sidebar.slider(
    "Années",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

start_year, end_year = year_range


# Nationalité des coureurs
rider_country_mapping = build_country_mapping(
    countries,
    results_enriched["nat_p"],
)

selected_rider_countries = st.sidebar.multiselect(
    "Nationalité des coureurs",
    options=sorted(rider_country_mapping),
)

selected_rider_country_ids = [
    rider_country_mapping[name] for name in selected_rider_countries
]


# Nationalité des courses
race_country_mapping = build_country_mapping(
    countries,
    results_enriched["id_nat"],
)

selected_race_countries = st.sidebar.multiselect(
    "Nationalité des courses",
    options=sorted(race_country_mapping),
)

selected_race_country_ids = [
    race_country_mapping[name] for name in selected_race_countries
]


# Catégorie d'édition
edition_categories = sorted(results_enriched["edition_cat"].dropna().unique())

selected_edition_categories = st.sidebar.multiselect(
    "Catégorie",
    options=edition_categories,
)


# Sous-catégorie
edition_sub_categories = sorted(results_enriched["s_cat"].dropna().unique())

selected_edition_sub_categories = st.sidebar.multiselect(
    "Sous-catégorie",
    options=edition_sub_categories,
)


# Courses
race_mapping = dict(
    zip(
        results_enriched["race_name"],
        results_enriched["id_race"],
    )
)

selected_races = st.sidebar.multiselect(
    "Courses",
    options=sorted(race_mapping),
)

selected_race_ids = [race_mapping[name] for name in selected_races]


# Place maximale
max_place = st.sidebar.slider(
    "Place maximale",
    min_value=1,
    max_value=20,
    value=20,
)


# ---------------------------------------------------------------------------
# Application des filtres
# ---------------------------------------------------------------------------

filtered_df = results_enriched.copy()

filtered_df = filter_by_year(
    filtered_df,
    start_year=start_year,
    end_year=end_year,
)

if selected_rider_ids:
    filtered_df = filter_by_rider(
        filtered_df,
        rider_ids=selected_rider_ids,
    )

if selected_rider_country_ids:
    filtered_df = filter_by_rider_nationality(
        filtered_df,
        country_ids=selected_rider_country_ids,
    )

if selected_race_country_ids:
    filtered_df = filter_by_race_nationality(
        filtered_df,
        country_ids=selected_race_country_ids,
    )

if selected_edition_categories:
    filtered_df = filter_by_edition_category(
        filtered_df,
        categories=selected_edition_categories,
    )

if selected_edition_sub_categories:
    filtered_df = filter_by_edition_sub_category(
        filtered_df,
        sub_categories=selected_edition_sub_categories,
    )

if selected_race_ids:
    filtered_df = filter_by_race(
        filtered_df,
        race_ids=selected_race_ids,
    )

filtered_df = filter_by_max_place(
    filtered_df,
    max_place=max_place,
)


# ---------------------------------------------------------------------------
# Calcul du classement
# ---------------------------------------------------------------------------

if filtered_df.empty:
    st.info("Aucun résultat ne correspond aux filtres sélectionnés.")
    st.stop()

rider_year_scores = aggregate_rider_year(
    filtered_df,
)

year_winners = prepare_year_winners_table(
    rider_year_scores,
    riders,
)


if ranking_type == "Absolu":
    scores = aggregate_rider_scores(filtered_df)

    ranked_df = rank_by_score(
        scores,
        score_column="score",
    )

    ranking_table = prepare_absolute_table(
        ranked_df,
        riders,
        countries,
    )


elif ranking_type == "Relatif":
    scores = aggregate_rider_scores(filtered_df)

    ranked_df = rank_with_relative_score(
        scores,
        score_column="score",
    )

    ranking_table = prepare_relative_table(
        ranked_df,
        riders,
        countries,
    )


else:  # F1
    rider_year_scores = aggregate_rider_year(
        filtered_df,
    )

    ranked_df = rank_f1(
        rider_year_scores,
    )

    ranking_table = prepare_f1_table(
        ranked_df,
        riders,
        countries,
    )


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

col_ranking, col_winners = st.columns([2, 1])

with col_ranking:
    st.subheader("Classement")

    st.dataframe(
        ranking_table,
        width="stretch",
        hide_index=True,
        height=get_table_height(ranking_table),
        column_config={
            "Rang": st.column_config.NumberColumn(
                "Rang",
                width="small",
            ),
            "Nom": st.column_config.TextColumn(
                "Nom",
                width="large",
            ),
            "Nat": st.column_config.TextColumn(
                "Nat",
                width="small",
            ),
            "Points": st.column_config.NumberColumn(
                "Points",
                width="small",
            ),
            "Ratio": st.column_config.NumberColumn(
                "Ratio",
                format="%.2f",
                width="small",
            ),
        },
    )

with col_winners:
    st.subheader("Vainqueur annuel")

    st.dataframe(
        year_winners,
        width="stretch",
        hide_index=True,
        height=get_table_height(year_winners),
        column_config={
            "Année": st.column_config.NumberColumn(
                "Année",
                width="small",
            ),
            "Vainqueur": st.column_config.TextColumn(
                "Vainqueur",
                width="large",
            ),
        },
    )
