from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agregations import (
    aggregate_race_riders,
    normalize_race_for_aggregation,
)
from src.filters import (
    filter_by_max_place,
    filter_by_rider_nationality,
    filter_by_year,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data = st.session_state["data"]

countries = data["countries"]
results_enriched = data["results_enriched"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def build_race_selector(
    results: pd.DataFrame,
    categories: list[str],
) -> pd.DataFrame:
    """
    Construit la liste des courses disponibles pour le sélecteur.

    Les courses sont dédupliquées par id_race et triées
    alphabétiquement par nom.
    """

    races = (
        results[results["edition_cat"].isin(categories)][
            [
                "id_race",
                "race_name",
            ]
        ]
        .drop_duplicates("id_race")
        .sort_values(
            "race_name",
            key=lambda column: column.str.lower(),
        )
        .reset_index(drop=True)
    )

    return races


def build_rider_names(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Construit le nom affiché de chaque coureur."""

    riders = (
        results[results["id_rider"] != 0][
            [
                "id_rider",
                "first_name",
                "surname",
                "nat_p",
                "country_n3_p",
            ]
        ]
        .drop_duplicates("id_rider")
        .copy()
    )

    riders["Nom"] = (
        riders["first_name"].fillna("").str.strip()
        + " "
        + riders["surname"].fillna("").str.strip()
    ).str.strip()

    return riders


def build_leaders_table(
    race_stats: pd.DataFrame,
    rider_names: pd.DataFrame,
    metrics: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Construit le tableau des leaders statistiques de la course.
    """

    if race_stats.empty:
        return pd.DataFrame(
            columns=[
                "Statistique",
                "Coureur",
                "Nat",
                "Valeur",
            ]
        )

    result = race_stats.merge(
        rider_names[
            [
                "id_rider",
                "Nom",
                "country_n3_p",
            ]
        ],
        on="id_rider",
        how="left",
    )

    if metrics is None:
        metrics = [
            ("wins", "Victoires"),
            ("podiums", "Podiums"),
            ("top_5", "Top 5"),
            ("top_10", "Top 10"),
            ("score", "Points"),
            ("stage_wins", "Victoires d'étape"),
        ]

    rows = []

    for metric, label in metrics:
        if metric not in result.columns:
            continue

        total = result[metric].sum()

        if total <= 0:
            continue

        best = result.sort_values(
            by=[
                metric,
                "score",
                "id_rider",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

        rows.append(
            {
                "Statistique": label,
                "Coureur": best["Nom"],
                "Nat": best["country_n3_p"],
                "Valeur": best[metric],
            }
        )

    return pd.DataFrame(rows)


def build_palmares_table(
    race_results: pd.DataFrame,
    selected_race_id: int,
) -> pd.DataFrame:
    """
    Construit le palmarès de la course.

    Les étapes sont exclues explicitement.
    Seule la classification générale de chaque édition
    est utilisée pour déterminer le vainqueur.
    """

    required_columns = {
        "id_edition",
        "id_race",
        "year",
        "edition_cat",
        "id_rider",
        "place",
        "first_name",
        "surname",
        "country_n3_p",
    }

    missing_columns = required_columns - set(race_results.columns)

    if missing_columns:
        raise KeyError(
            f"Colonnes manquantes pour le palmarès : {sorted(missing_columns)}"
        )

    winners = race_results[
        (race_results["id_race"] == selected_race_id)
        & (race_results["edition_cat"] != "S")
        & (race_results["place"] == 1)
        & (race_results["id_rider"] != 0)
    ].copy()

    winners = winners.sort_values(
        [
            "year",
            "id_edition",
            "id_rider",
        ]
    ).drop_duplicates(
        subset=["id_edition"],
        keep="first",
    )

    winners["Vainqueur"] = (
        winners["first_name"].fillna("").str.strip()
        + " "
        + winners["surname"].fillna("").str.strip()
    ).str.strip()

    return (
        winners[
            [
                "year",
                "Vainqueur",
                "country_n3_p",
            ]
        ]
        .rename(
            columns={
                "year": "Année",
                "country_n3_p": "Nat",
            }
        )
        .sort_values(
            "Année",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_race_french_name(
    race_results: pd.DataFrame,
) -> str:
    """
    Retourne le nom français de la course si une colonne dédiée existe.

    Le dataset actuellement disponible ne documente pas de colonne
    race_name_fr, d'où le fallback explicite.
    """

    column = "race_name_fr"

    values = race_results[column].dropna().astype(str).str.strip()

    values = values[values != ""]

    if not values.empty:
        return values.iloc[0]

    return "Non disponible"


def build_category_leaders_table(
    race_results: pd.DataFrame,
    categories: list[str],
    metric: str,
    limit: int,
) -> pd.DataFrame:
    """Construit les leaders d'une statistique pour chaque catégorie."""

    columns = [
        "Rang",
        "Nom",
        "Valeur",
    ]

    if race_results.empty:
        return pd.DataFrame(columns=columns)

    rider_stats = aggregate_race_riders(
        race_results,
    )

    rider_names = build_rider_names(
        race_results,
    )

    rider_stats = rider_stats.merge(
        rider_names[
            [
                "id_rider",
                "Nom",
            ]
        ],
        on="id_rider",
        how="left",
    )

    rows = []

    for category in categories:
        category_results = race_results[race_results["edition_cat"] == category]

        category_rider_ids = category_results.loc[
            category_results["id_rider"] != 0,
            "id_rider",
        ].unique()

        category_stats = rider_stats[
            rider_stats["id_rider"].isin(category_rider_ids)
        ].copy()

        if category_stats.empty:
            continue

        if metric not in category_stats.columns:
            continue

        if category_stats[metric].max() <= 0:
            continue

        ranked = (
            category_stats[category_stats[metric] > 0][
                [
                    "id_rider",
                    "Nom",
                    metric,
                ]
            ]
            .rename(
                columns={
                    metric: "Valeur",
                }
            )
            .sort_values(
                by=[
                    "Valeur",
                    "id_rider",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        ranked["Rang"] = (
            ranked["Valeur"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype("int64")
        )

        ranked = ranked.head(limit)[["Rang", "Nom", "Valeur"]]

        rows.append(
            ranked[
                [
                    "Rang",
                    "Nom",
                    "Valeur",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🏁 Course")


# ---------------------------------------------------------------------------
# Sélection du type de course
# ---------------------------------------------------------------------------

selection_mode = st.radio(
    "Type",
    options=[
        "Courses",
        "Championnats nationaux",
    ],
    index=0,
    horizontal=True,
)


# ---------------------------------------------------------------------------
# Sélection de la course
# ---------------------------------------------------------------------------

if selection_mode == "Courses":
    available_races = build_race_selector(
        results_enriched,
        categories=["C", "T", "I"],
    )

    if available_races.empty:
        st.info("Aucune course disponible.")
        st.stop()

    race_name_mapping = dict(
        zip(
            available_races["id_race"],
            available_races["race_name"],
        )
    )

    race_ids = available_races["id_race"].tolist()

    default_index = race_ids.index(2040) if 2040 in race_ids else 0

    selected_race_id = st.selectbox(
        "Course",
        options=race_ids,
        index=default_index,
        format_func=lambda race_id: race_name_mapping[race_id],
    )

else:
    available_races = build_race_selector(
        results_enriched,
        categories=["N"],
    )

    if available_races.empty:
        st.info("Aucun championnat national disponible.")
        st.stop()

    race_name_mapping = dict(
        zip(
            available_races["id_race"],
            available_races["race_name"],
        )
    )

    race_ids = available_races["id_race"].tolist()

    selected_race_id = st.selectbox(
        "Championnat national",
        options=race_ids,
        format_func=lambda race_id: race_name_mapping[race_id],
    )


# ---------------------------------------------------------------------------
# Données de la course sélectionnée
# ---------------------------------------------------------------------------

normalized_results = normalize_race_for_aggregation(
    results_enriched,
)

race_results = normalized_results[
    normalized_results["race_id_aggregate"] == selected_race_id
].copy()


if race_results.empty:
    st.info("Aucun résultat disponible pour cette course.")
    st.stop()


# ---------------------------------------------------------------------------
# Identité de la course
# ---------------------------------------------------------------------------

race_name = race_name_mapping[selected_race_id]

french_name = get_race_french_name(
    race_results,
)

number_of_editions = race_results["year"].nunique()

first_year = int(race_results["year"].min())
last_year = int(race_results["year"].max())


# ---------------------------------------------------------------------------
# Filtres statistiques
# ---------------------------------------------------------------------------

st.sidebar.header("Filtres")


# Années

min_year = int(race_results["year"].min())
max_year = int(race_results["year"].max())

year_range = st.sidebar.slider(
    "Années",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

selected_start_year, selected_end_year = year_range


# Nationalité des coureurs

rider_country_mapping = build_country_mapping(
    countries,
    race_results["nat_p"],
)

selected_rider_countries = st.sidebar.multiselect(
    "Nationalité des coureurs",
    options=sorted(rider_country_mapping),
)

selected_rider_country_ids = [
    rider_country_mapping[name] for name in selected_rider_countries
]


# Place maximale

max_place = st.sidebar.slider(
    "Place maximale",
    min_value=1,
    max_value=20,
    value=20,
)

leaders_limit = st.sidebar.slider(
    "Nombre de leaders",
    min_value=3,
    max_value=20,
    value=10,
)

# ---------------------------------------------------------------------------
# Application des filtres
#
# IMPORTANT :
# Ces filtres ne s'appliquent qu'aux statistiques.
# L'identité et le palmarès utilisent toujours race_results complet.
# ---------------------------------------------------------------------------

leader_results = race_results.copy()

leader_results = filter_by_year(
    leader_results,
    start_year=selected_start_year,
    end_year=selected_end_year,
)

if selected_rider_country_ids:
    leader_results = filter_by_rider_nationality(
        leader_results,
        country_ids=selected_rider_country_ids,
    )

leader_results = filter_by_max_place(
    leader_results,
    max_place=max_place,
)

# ---------------------------------------------------------------------------
# Agrégation des statistiques
# ---------------------------------------------------------------------------

race_stats = aggregate_race_riders(
    leader_results,
)

rider_names = build_rider_names(
    results_enriched,
)

if selection_mode == "Championnats nationaux":
    leader_metrics = [
        ("score", "Points"),
        ("wins", "Victoires"),
    ]
else:
    leader_metrics = [
        ("score", "Points"),
        ("wins", "Victoires"),
        ("podiums", "Podiums"),
        ("top_5", "Top 5"),
        ("top_10", "Top 10"),
        ("stage_wins", "Victoires d'étape"),
    ]


leaders_table = build_leaders_table(
    race_stats,
    rider_names,
    metrics=leader_metrics,
)

# ---------------------------------------------------------------------------
# Section 1 : identité + statistiques
# ---------------------------------------------------------------------------

identity_column, stats_column = st.columns(
    [1, 1],
)


with identity_column:
    st.subheader("Informations")

    st.markdown(f"**Nom :** {race_name}")

    st.markdown(f"**Nom français :** {french_name}")

    st.markdown(f"**Nombre d'éditions :** {number_of_editions}")

    st.markdown(f"**Première édition :** {int(first_year)}")

    st.markdown(f"**Dernière édition :** {int(last_year)}")


with stats_column:
    st.subheader("Leaders")

    if leaders_table.empty:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés.")

    else:
        st.dataframe(
            leaders_table,
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Section 2 : Palmarès
#
# Aucun filtre ne s'applique ici.
# Les étapes sont ignorées pour les courses par étapes.
# ---------------------------------------------------------------------------

palmares_column, category_leaders_column = st.columns(
    [1, 1],
)

with palmares_column:
    st.subheader("Palmarès")
    palmares_table = build_palmares_table(
        race_results,
        selected_race_id,
    )

    if palmares_table.empty:
        st.info("Aucun vainqueur disponible pour cette course.")

    else:
        st.dataframe(
            palmares_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Année": st.column_config.NumberColumn(
                    "Année",
                    width="small",
                ),
                "Vainqueur": st.column_config.TextColumn(
                    "Vainqueur",
                    width="large",
                ),
                "Nat": st.column_config.TextColumn(
                    "Nat",
                    width="small",
                ),
            },
        )

with category_leaders_column:
    selected_metric_label = st.selectbox(
        "Statistique",
        options=[label for _, label in leader_metrics],
    )

    selected_metric = next(
        metric for metric, label in leader_metrics if label == selected_metric_label
    )

    if selection_mode == "Championnats nationaux":
        leader_categories = ["N"]
    else:
        leader_categories = [
            category
            for category in ["C", "T", "I"]
            if category in leader_results["edition_cat"].unique()
        ]

    category_leaders = build_category_leaders_table(
        leader_results,
        categories=leader_categories,
        metric=selected_metric,
        limit=leaders_limit,
    )

    if category_leaders.empty:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés.")
    else:
        st.dataframe(
            category_leaders,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rang": st.column_config.NumberColumn(
                    "Rang",
                    width="small",
                ),
                "Nom": st.column_config.TextColumn(
                    "Nom",
                    width="large",
                ),
                "Valeur": st.column_config.NumberColumn(
                    "Valeur",
                    width="small",
                ),
            },
        )
