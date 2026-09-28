from __future__ import annotations

import pandas as pd
import streamlit as st

from src.filters import (
    filter_by_edition_category,
    filter_by_edition_sub_category,
    filter_by_max_place,
    filter_by_race,
    filter_by_race_nationality,
    filter_by_rider,
    filter_by_year,
)

data = st.session_state["data"]

countries = data["countries"]
riders = data["riders"]
results_enriched = data["results_enriched"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    return dict(zip(available["name"], available["id_country"]))


def build_race_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Construit le mapping nom de course -> identifiant.

    Les courses d'étapes sont exclues de la sélection principale, mais les
    autres catégories d'édition sont conservées.
    """
    races = (
        df.loc[df["edition_cat"] != "S", ["race_name", "id_race"]]
        .dropna(subset=["race_name", "id_race"])
        .drop_duplicates()
        .sort_values("race_name")
    )

    return dict(zip(races["race_name"], races["id_race"]))


def get_table_height(
    df: pd.DataFrame,
    row_height: int = 35,
    header_height: int = 38,
    max_height: int = 650,
) -> int:
    """Calcule une hauteur adaptée au nombre de lignes."""
    return min(header_height + len(df) * row_height, max_height)


def prepare_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les résultats d'une course pour l'affichage."""
    result = df.copy()

    result["Nom"] = (
        result["first_name"].fillna("") + " " + result["surname"].fillna("")
    ).str.strip()

    # Les étapes sont affichées avec un identifiant Ex. Comme toutes les
    # lignes d'une même étape partagent le même id_edition, le numéro est
    # simplement incrémenté à chaque ligne d'étape dans l'ordre d'affichage.
    result["Place"] = result["place"].astype("object")
    stage_mask = result["edition_cat"].eq("S")
    result.loc[stage_mask, "Place"] = [f"E{i}" for i in range(1, stage_mask.sum() + 1)]

    return result[["Place", "Nom", "country_n3_p", "score"]].rename(
        columns={
            "country_n3_p": "Nat",
            "score": "Points",
        }
    )


def prepare_rider_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare le palmarès annuel d'un coureur pour l'affichage."""
    result = df.copy()

    result["Type"] = (
        result["edition_cat"].fillna("").astype(str)
        + result["s_cat"].fillna("").astype(str)
    ).str.strip()

    result["Place"] = result["place"]
    result["Points"] = result["score"]

    return result[["Place", "race_name", "Type", "Points"]].rename(
        columns={
            "race_name": "Course",
        }
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


st.title("📋 Résultats")

selection_type = st.radio(
    "Choisissez les résultats à afficher :",
    options=["Course", "Cycliste"],
    index=0,
    horizontal=True,
)


# ---------------------------------------------------------------------------
# Filtres secondaires
# ---------------------------------------------------------------------------

st.sidebar.header("Filtres")

max_place = st.sidebar.slider(
    "Place maximale",
    min_value=1,
    max_value=20,
    value=20,
)


# ---------------------------------------------------------------------------
# Sélection principale : course
# ---------------------------------------------------------------------------

if selection_type == "Course":
    race_mapping = build_race_mapping(results_enriched)

    col_race, col_year = st.columns(2)

    with col_race:
        race_options = list(race_mapping)
        default_race_name = next(
            (name for name, race_id in race_mapping.items() if race_id == 2040),
            None,
        )
        race_index = (
            race_options.index(default_race_name)
            if default_race_name in race_options
            else 0
        )

        selected_race_name = st.selectbox(
            "Course",
            options=race_options,
            index=race_index,
        )
        selected_race_id = race_mapping[selected_race_name]

    # Les années sont prises parmi toutes les éditions non-étapes de la
    # course sélectionnée, quelle que soit leur catégorie.
    race_years = (
        results_enriched.loc[
            (results_enriched["id_race"] == selected_race_id)
            & (results_enriched["edition_cat"] != "S"),
            "year",
        ]
        .dropna()
        .unique()
    )
    race_years = sorted(race_years, reverse=True)

    with col_year:
        selected_year = st.selectbox(
            "Année",
            options=race_years,
        )

    # La course sélectionnée peut être accompagnée d'étapes. On conserve
    # toutes ses catégories principales et les étapes associées (+1).
    selected_edition_races = (
        results_enriched.loc[
            (
                (results_enriched["id_race"] == selected_race_id)
                & (results_enriched["edition_cat"] != "S")
            )
            | (
                (results_enriched["id_race"] == selected_race_id + 1)
                & (results_enriched["edition_cat"] == "S")
            ),
            "id_race",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    filtered_df = filter_by_race(results_enriched, selected_edition_races)
    filtered_df = filter_by_year(
        filtered_df,
        start_year=int(selected_year),
        end_year=int(selected_year),
    )
    filtered_df = filter_by_max_place(filtered_df, max_place=max_place)

    # Certaines lignes peuvent exister sans coureur identifié (0 ou valeur
    # manquante). Elles ne sont pas utiles à l'affichage des résultats.
    filtered_df = filtered_df.loc[
        filtered_df["id_rider"].notna() & filtered_df["id_rider"].ne(0)
    ].copy()

    if filtered_df.empty:
        st.info("Aucun résultat pour cette course et cette année.")
        st.stop()

    filtered_df = filtered_df.assign(
        _edition_cat_order=filtered_df["edition_cat"].map({"T": 0, "S": 1}).fillna(0)
    ).sort_values(
        ["id_edition", "_edition_cat_order", "place"],
        ascending=[True, True, True],
    )

    race_category = results_enriched.loc[
        results_enriched["id_race"] == selected_race_id,
        "race_cat",
    ].dropna()
    race_category = str(race_category.iloc[0]) if not race_category.empty else ""

    sub_category = (
        filtered_df.loc[filtered_df["edition_cat"] != "S", "s_cat"].dropna().astype(str)
    )
    sub_category = sub_category.iloc[0] if not sub_category.empty else ""

    category_code = f"{race_category}{sub_category}"
    title = f"{selected_race_name} — {selected_year}"
    if category_code:
        title += f" — {category_code}"

    st.subheader(title)

    # Séparation du classement général et des étapes.
    general_df = filtered_df.loc[filtered_df["edition_cat"] != "S"].copy()
    stages_df = filtered_df.loc[filtered_df["edition_cat"] == "S"].copy()

    general_table = prepare_results_table(general_df)
    stages_table = prepare_results_table(stages_df)

else:
    # -----------------------------------------------------------------------
    # Sélection principale : cycliste
    # -----------------------------------------------------------------------

    rider_mapping = build_rider_mapping(riders)

    col_rider, col_year = st.columns(2)

    with col_rider:
        rider_options = sorted(rider_mapping)
        default_rider_name = next(
            (name for name, rider_id in rider_mapping.items() if rider_id == 2004),
            None,
        )
        rider_index = (
            rider_options.index(default_rider_name)
            if default_rider_name in rider_options
            else 0
        )

        selected_rider_name = st.selectbox(
            "Cycliste",
            options=rider_options,
            index=rider_index,
        )
        selected_rider_id = rider_mapping[selected_rider_name]

    rider_years = (
        results_enriched.loc[
            results_enriched["id_rider"] == selected_rider_id,
            "year",
        ]
        .dropna()
        .unique()
    )
    rider_years = sorted(rider_years, reverse=True)

    with col_year:
        selected_year = st.selectbox(
            "Année",
            options=rider_years,
        )

    # Filtres spécifiques au mode Cycliste.
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

    edition_categories = sorted(results_enriched["edition_cat"].dropna().unique())
    selected_edition_categories = st.sidebar.multiselect(
        "Catégorie",
        options=edition_categories,
    )

    edition_sub_categories = sorted(results_enriched["s_cat"].dropna().unique())
    selected_edition_sub_categories = st.sidebar.multiselect(
        "Sous-catégorie",
        options=edition_sub_categories,
    )

    filtered_df = filter_by_rider(
        results_enriched,
        rider_ids=selected_rider_id,
    )
    filtered_df = filter_by_year(
        filtered_df,
        start_year=int(selected_year),
        end_year=int(selected_year),
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

    filtered_df = filter_by_max_place(filtered_df, max_place=max_place)

    if filtered_df.empty:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés.")
        st.stop()

    # Le palmarès est d'abord ordonné par le résultat obtenu, puis par les
    # points pour départager les résultats de même place.
    filtered_df = filtered_df.sort_values(
        ["place", "score"],
        ascending=[True, False],
        na_position="last",
    )

    total_score = filtered_df["score"].sum()
    st.subheader(
        f"Résultats de {selected_rider_name} — {selected_year} — {total_score:g} pts"
    )

    # Séparation des classements finaux et des étapes.
    final_results_df = filtered_df.loc[filtered_df["edition_cat"] != "S"].copy()
    stages_results_df = filtered_df.loc[filtered_df["edition_cat"] == "S"].copy()

    result_table = prepare_rider_results_table(final_results_df)
    stages_result_table = prepare_rider_results_table(stages_results_df)


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

if selection_type == "Course":
    course_column_config = {
        "Place": st.column_config.TextColumn("Place", width="small"),
        "Nom": st.column_config.TextColumn("Nom", width="large"),
        "Nat": st.column_config.TextColumn("Nat", width="small"),
        "Points": st.column_config.NumberColumn("Points", width="small"),
    }

    st.dataframe(
        general_table,
        use_container_width=True,
        hide_index=True,
        height=get_table_height(general_table),
        column_config=course_column_config,
    )

    if not stages_table.empty:
        # Les étapes restent un vrai dataframe Streamlit pour conserver
        # exactement le même rendu et une hauteur adaptée au nombre de lignes.
        # Les libellés de colonnes sont volontairement vides pour ne pas
        # afficher un second en-tête.
        stages_column_config = {
            "Place": st.column_config.TextColumn("", width="small"),
            "Nom": st.column_config.TextColumn("", width="large"),
            "Nat": st.column_config.TextColumn("", width="small"),
            "Points": st.column_config.NumberColumn("", width="small"),
        }

        st.dataframe(
            stages_table,
            use_container_width=True,
            hide_index=True,
            height=get_table_height(stages_table),
            column_config=stages_column_config,
        )

else:
    rider_column_config = {
        "Place": st.column_config.NumberColumn("Place", width="small"),
        "Course": st.column_config.TextColumn("Course", width="large"),
        "Type": st.column_config.TextColumn("Type", width="small"),
        "Points": st.column_config.NumberColumn("Points", width="small"),
    }

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True,
        height=get_table_height(result_table),
        column_config=rider_column_config,
    )

    if not stages_result_table.empty:
        stages_rider_column_config = {
            "Place": st.column_config.NumberColumn("", width="small"),
            "Course": st.column_config.TextColumn("", width="large"),
            "Type": st.column_config.TextColumn("", width="small"),
            "Points": st.column_config.NumberColumn("", width="small"),
        }

        st.dataframe(
            stages_result_table,
            use_container_width=True,
            hide_index=True,
            height=get_table_height(stages_result_table),
            column_config=stages_rider_column_config,
        )
