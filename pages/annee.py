from __future__ import annotations

import pandas as pd
import streamlit as st

from src.agregations import (
    aggregate_rider_year,
    aggregate_year_calendar,
    select_top_n_riders_by_nationality,
)
from src.filters import (
    filter_by_edition_category,
    filter_by_edition_sub_category,
    filter_by_race_nationality,
)
from src.ranking import (
    rank_by_score,
    rank_riders_yearly_and_cumulative,
)
from src.stats import compute_rider_year_stats

# =============================================================================
# Données
# =============================================================================

data = st.session_state["data"]

countries = data["countries"]
riders = data["riders"]
results_enriched = data["results_enriched"]


# =============================================================================
# Fonctions utilitaires
# =============================================================================


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


def get_table_height(
    df: pd.DataFrame,
    row_height: int = 35,
    header_height: int = 38,
    max_height: int = 388,
) -> int:
    """Calcule une hauteur adaptée au nombre de lignes."""

    return min(
        header_height + len(df) * row_height,
        max_height,
    )


def prepare_rider_ranking_table(
    ranking: pd.DataFrame,
    riders: pd.DataFrame,
    countries: pd.DataFrame,
    *,
    rank_column: str,
    score_column: str,
) -> pd.DataFrame:
    """Prépare un tableau de classement coureurs."""

    result = ranking.merge(
        riders[
            [
                "id_rider",
                "first_name",
                "surname",
                "nat_p",
            ]
        ],
        on="id_rider",
        how="left",
    )

    result = result.merge(
        countries[
            [
                "id_country",
                "n_3",
            ]
        ],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    result["Nom"] = (
        result["first_name"].fillna("") + " " + result["surname"].fillna("")
    ).str.strip()

    result = result[
        [
            rank_column,
            "Nom",
            "n_3",
            score_column,
        ]
    ].copy()

    return result.rename(
        columns={
            rank_column: "Rang",
            "n_3": "Nat.",
            score_column: "Points",
        }
    )


def prepare_nation_ranking_table(
    ranking: pd.DataFrame,
    countries: pd.DataFrame,
    *,
    rank_column: str,
    score_column: str,
) -> pd.DataFrame:
    """Prépare un tableau de classement nations."""

    result = ranking.merge(
        countries[
            [
                "id_country",
                "name",
            ]
        ],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    result = result[
        [
            rank_column,
            "name",
            score_column,
        ]
    ].copy()

    return result.rename(
        columns={
            rank_column: "Rang",
            "name": "Nation",
            score_column: "Points",
        }
    )


def prepare_calendar_table(
    calendar: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau d'affichage du calendrier."""

    if calendar.empty:
        return pd.DataFrame(
            columns=[
                "Type",
                "Course",
                "Nat.",
                "Vainqueur",
            ]
        )

    result = calendar.copy()

    result["Type"] = result["edition_cat"].fillna("") + result["s_cat"].fillna("")

    result = result.merge(
        countries[
            [
                "id_country",
                "n_3",
            ]
        ],
        left_on="id_nat",
        right_on="id_country",
        how="left",
    )

    result["Vainqueur"] = (
        result["first_name"].fillna("") + " " + result["surname"].fillna("")
    ).str.strip()

    return result[
        [
            "Type",
            "race_name",
            "n_3",
            "Vainqueur",
        ]
    ].rename(
        columns={
            "race_name": "Course",
            "n_3": "Nat.",
        }
    )


def prepare_stat_leaders_table(
    df: pd.DataFrame,
    riders: pd.DataFrame,
    countries: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Prépare le classement des leaders d'une statistique."""

    stats = compute_rider_year_stats(df)

    if stats.empty or metric not in stats.columns:
        return pd.DataFrame(
            columns=[
                "Rang",
                "Nom",
                "Nat.",
                "Valeur",
            ]
        )

    stats = stats[stats[metric] > 0].copy()

    if stats.empty:
        return pd.DataFrame(
            columns=[
                "Rang",
                "Nom",
                "Nat.",
                "Valeur",
            ]
        )

    stats = stats.sort_values(
        [
            metric,
            "id_rider",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    stats["Rang"] = (
        stats[metric]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("int64")
    )

    stats = stats.merge(
        riders[
            [
                "id_rider",
                "first_name",
                "surname",
                "nat_p",
            ]
        ],
        on="id_rider",
        how="left",
    )

    stats = stats.merge(
        countries[
            [
                "id_country",
                "n_3",
            ]
        ],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    stats["Nom"] = (
        stats["first_name"].fillna("") + " " + stats["surname"].fillna("")
    ).str.strip()

    return stats[
        [
            "Rang",
            "Nom",
            "n_3",
            metric,
        ]
    ].rename(
        columns={
            "n_3": "Nat.",
            metric: "Valeur",
        }
    )


# =============================================================================
# Sélection de l'année
# =============================================================================

st.title("📅 Année")

min_year = int(results_enriched["year"].min())
max_year = int(results_enriched["year"].max())

years = list(
    range(
        max_year,
        min_year - 1,
        -1,
    )
)

if "selected_year" not in st.session_state:
    st.session_state.selected_year = max_year

selected_index = years.index(st.session_state.selected_year)

st.subheader("Sélectionner une année")

col_prev, col_year, col_next = st.columns([1, 4, 1])

with col_prev:
    if st.button(
        "←",
        disabled=selected_index == len(years) - 1,
        help="Année précédente",
        width="stretch",
    ):
        st.session_state.selected_year = years[selected_index + 1]
        st.rerun()

with col_year:
    selected_year = st.selectbox(
        "Année",
        options=years,
        index=selected_index,
        label_visibility="collapsed",
    )

    if selected_year != st.session_state.selected_year:
        st.session_state.selected_year = selected_year
        st.rerun()

with col_next:
    if st.button(
        "→",
        disabled=selected_index == 0,
        help="Année suivante",
        width="stretch",
    ):
        st.session_state.selected_year = years[selected_index - 1]
        st.rerun()

selected_year = st.session_state.selected_year


# =============================================================================
# SECTION 1 — Classements coureurs
# =============================================================================

st.header("Classement des coureurs")


rider_year_scores = aggregate_rider_year(
    results_enriched,
)

rider_rankings = rank_riders_yearly_and_cumulative(
    rider_year_scores,
)

selected_rider_rankings = rider_rankings[rider_rankings["year"] == selected_year].copy()


annual_riders = selected_rider_rankings[
    selected_rider_rankings["rank_year"].notna()
].copy()

annual_riders = annual_riders.sort_values(
    by=["rank_year", "id_rider"],
    ascending=[True, True],
).rename(
    columns={
        "rank_year": "Rang",
        "score": "Points",
    }
)

cumulative_riders = selected_rider_rankings.rename(
    columns={
        "rank_cumulative": "Rang",
        "score_cumulative": "Points",
    }
)


annual_riders_table = prepare_rider_ranking_table(
    annual_riders,
    riders,
    countries,
    rank_column="Rang",
    score_column="Points",
)

cumulative_riders_table = prepare_rider_ranking_table(
    cumulative_riders,
    riders,
    countries,
    rank_column="Rang",
    score_column="Points",
)


col_annual, col_cumulative = st.columns(2)

with col_annual:
    st.subheader(f"Classement annuel — {selected_year}")

    st.dataframe(
        annual_riders_table,
        width="stretch",
        hide_index=True,
        height=get_table_height(
            annual_riders_table,
        ),
        column_config={
            "Rang": st.column_config.NumberColumn(
                "Rang",
                width="small",
            ),
            "Nom": st.column_config.TextColumn(
                "Nom",
                width="large",
            ),
            "Nat.": st.column_config.TextColumn(
                "Nat.",
                width="small",
            ),
            "Points": st.column_config.NumberColumn(
                "Points",
                width="small",
            ),
        },
    )

with col_cumulative:
    st.subheader(f"Classement cumulé — {selected_year}")

    st.dataframe(
        cumulative_riders_table,
        width="stretch",
        hide_index=True,
        height=get_table_height(
            cumulative_riders_table,
        ),
        column_config={
            "Rang": st.column_config.NumberColumn(
                "Rang",
                width="small",
            ),
            "Nom": st.column_config.TextColumn(
                "Nom",
                width="large",
            ),
            "Nat.": st.column_config.TextColumn(
                "Nat.",
                width="small",
            ),
            "Points": st.column_config.NumberColumn(
                "Points",
                width="small",
            ),
        },
    )


# =============================================================================
# SECTION 2 — Classements nations
# =============================================================================

st.header("Classement des nations")


top_n_riders = st.slider(
    "Nombre de coureurs considérés par nation",
    min_value=1,
    max_value=50,
    value=10,
)


nation_rider_scores = aggregate_rider_year(
    results_enriched,
)

nation_top_riders = select_top_n_riders_by_nationality(
    nation_rider_scores,
    n=top_n_riders,
)

nation_year_scores = nation_top_riders.groupby(
    [
        "year",
        "nat_p",
    ],
    as_index=False,
).agg(
    score=("score", "sum"),
)


# Classement annuel.
nation_annual = nation_year_scores[nation_year_scores["year"] == selected_year].copy()

nation_annual = rank_by_score(
    nation_annual[
        [
            "nat_p",
            "score",
        ]
    ],
    score_column="score",
)


# Classement cumulé.
nation_cumulative = (
    nation_year_scores[nation_year_scores["year"] <= selected_year]
    .groupby(
        "nat_p",
        as_index=False,
    )
    .agg(
        score=("score", "sum"),
    )
)

nation_cumulative = rank_by_score(
    nation_cumulative,
    score_column="score",
)


nation_annual_table = prepare_nation_ranking_table(
    nation_annual,
    countries,
    rank_column="rank",
    score_column="score",
)

nation_cumulative_table = prepare_nation_ranking_table(
    nation_cumulative,
    countries,
    rank_column="rank",
    score_column="score",
)


col_annual_nation, col_cumulative_nation = st.columns(2)

with col_annual_nation:
    st.subheader(f"Classement annuel — {selected_year}")

    st.dataframe(
        nation_annual_table,
        width="stretch",
        hide_index=True,
        height=get_table_height(
            nation_annual_table,
        ),
        column_config={
            "Rang": st.column_config.NumberColumn(
                "Rang",
                width="small",
            ),
            "Nation": st.column_config.TextColumn(
                "Nation",
                width="large",
            ),
            "Points": st.column_config.NumberColumn(
                "Points",
                width="small",
            ),
        },
    )

with col_cumulative_nation:
    st.subheader(f"Classement cumulé — {selected_year}")

    st.dataframe(
        nation_cumulative_table,
        width="stretch",
        hide_index=True,
        height=get_table_height(
            nation_cumulative_table,
        ),
        column_config={
            "Rang": st.column_config.NumberColumn(
                "Rang",
                width="small",
            ),
            "Nation": st.column_config.TextColumn(
                "Nation",
                width="large",
            ),
            "Points": st.column_config.NumberColumn(
                "Points",
                width="small",
            ),
        },
    )


# =============================================================================
# SECTION 3 — Calendrier et leaders statistiques
# =============================================================================

st.header("Calendrier et statistiques")


# -----------------------------------------------------------------------------
# Filtres — uniquement pour la section 3
# -----------------------------------------------------------------------------

st.sidebar.header("Filtres")


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


# -----------------------------------------------------------------------------
# Filtrage de l'année
# -----------------------------------------------------------------------------

year_df = results_enriched[results_enriched["year"] == selected_year].copy()

filtered_year_df = year_df.copy()


if selected_race_country_ids:
    filtered_year_df = filter_by_race_nationality(
        filtered_year_df,
        country_ids=selected_race_country_ids,
    )


if selected_edition_categories:
    filtered_year_df = filter_by_edition_category(
        filtered_year_df,
        categories=selected_edition_categories,
    )


if selected_edition_sub_categories:
    filtered_year_df = filter_by_edition_sub_category(
        filtered_year_df,
        sub_categories=selected_edition_sub_categories,
    )


# =============================================================================
# Calendrier
# =============================================================================

col_calendar, col_stats = st.columns(2)


with col_calendar:
    st.subheader("Calendrier")

    calendar = aggregate_year_calendar(
        filtered_year_df,
        excluded_categories={
            "S",
            "N",
        },
    )

    calendar_table = prepare_calendar_table(
        calendar,
        countries,
    )

    if calendar_table.empty:
        st.info("Aucune édition ne correspond aux filtres sélectionnés.")
    else:
        st.dataframe(
            calendar_table,
            width="stretch",
            hide_index=True,
            height=get_table_height(
                calendar_table,
            ),
            column_config={
                "Type": st.column_config.TextColumn(
                    "Type",
                    width="small",
                ),
                "Course": st.column_config.TextColumn(
                    "Course",
                    width="medium",
                ),
                "Nat.": st.column_config.TextColumn(
                    "Nat.",
                    width="small",
                ),
                "Vainqueur": st.column_config.TextColumn(
                    "Vainqueur",
                    width="medium",
                ),
            },
        )


# =============================================================================
# Leaders statistiques
# =============================================================================

with col_stats:
    stat_options = {
        "Victoires": "wins",
        "Podiums": "podiums",
        "Top 5": "top_5",
        "Top 10": "top_10",
    }

    selected_stat_label = st.selectbox(
        "Statistique",
        options=list(stat_options),
    )

    selected_stat = stat_options[selected_stat_label]

    stat_leaders = prepare_stat_leaders_table(
        filtered_year_df,
        riders,
        countries,
        selected_stat,
    )

    if stat_leaders.empty:
        st.info(
            "Aucun coureur ne possède cette statistique dans les données sélectionnées."
        )
    else:
        st.dataframe(
            stat_leaders,
            width="stretch",
            hide_index=True,
            height=get_table_height(
                stat_leaders,
            ),
            column_config={
                "Rang": st.column_config.NumberColumn(
                    "Rang",
                    width="small",
                ),
                "Nom": st.column_config.TextColumn(
                    "Nom",
                    width="medium",
                ),
                "Nat.": st.column_config.TextColumn(
                    "Nat.",
                    width="small",
                ),
                "Valeur": st.column_config.NumberColumn(
                    "Valeur",
                    width="small",
                ),
            },
        )
