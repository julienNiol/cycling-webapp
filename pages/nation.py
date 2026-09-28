from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agregations import (
    aggregate_nation_ranking,
    aggregate_rider_season_category,
    aggregate_rider_year,
    select_top_n_riders_by_nationality,
)
from src.filters import (
    filter_by_edition_category,
    filter_by_edition_sub_category,
    filter_by_max_place,
    filter_by_race,
    filter_by_race_nationality,
    filter_by_year,
)
from src.ranking import rank_by_score, rank_f1, rank_with_relative_score
from src.stats import compute_nation_stats, compute_rider_stats

data = st.session_state["data"]

countries = data["countries"]
results_enriched = data["results_enriched"]


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


def prepare_nation_table(
    ranked_df: pd.DataFrame,
    countries: pd.DataFrame,
    *,
    score_column: str = "score",
    display_column: str = "Points",
) -> pd.DataFrame:
    """Prépare le tableau d'affichage d'un classement par nation."""

    result = ranked_df.merge(
        countries[["id_country", "name"]],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    result = result[["rank", "name", score_column]].copy()

    return result.rename(
        columns={
            "rank": "Rang",
            "name": "Nation",
            score_column: display_column,
        }
    )


def prepare_f1_nation_table(
    ranked_df: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau d'affichage du classement F1 par nation."""

    result = ranked_df.merge(
        countries[["id_country", "name"]],
        left_on="id_rider",
        right_on="id_country",
        how="left",
    )

    result = result[["rank", "name", "f1_points"]].copy()

    return result.rename(
        columns={
            "rank": "Rang",
            "name": "Nation",
            "f1_points": "Points",
        }
    )


def prepare_year_leaders_table(
    nation_year_scores: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    """Prépare le tableau de la nation en tête pour chaque année."""

    leaders = (
        nation_year_scores.sort_values(
            ["year", "score", "nat_p"],
            ascending=[True, False, True],
        )
        .drop_duplicates(
            subset="year",
            keep="first",
        )
        .copy()
    )

    leaders = leaders.merge(
        countries[["id_country", "name"]],
        left_on="nat_p",
        right_on="id_country",
        how="left",
    )

    leaders = leaders[["year", "name"]]

    return leaders.rename(
        columns={
            "year": "Année",
            "name": "Nation",
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


st.title("🌍 Nation")


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


# Nombre de coureurs retenus par nation
top_n_riders = st.sidebar.slider(
    "Coureurs retenus par nation",
    min_value=1,
    max_value=50,
    value=10,
)


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
# Calcul des scores annuels par nation
# ---------------------------------------------------------------------------

if filtered_df.empty:
    st.info("Aucun résultat ne correspond aux filtres sélectionnés.")
    st.stop()

rider_year_scores = aggregate_rider_year(
    filtered_df,
)

top_riders_year = select_top_n_riders_by_nationality(
    rider_year_scores,
    n=top_n_riders,
)

nation_year_scores = top_riders_year.groupby(
    ["year", "nat_p"],
    as_index=False,
).agg(score=("score", "sum"))


# ---------------------------------------------------------------------------
# Calcul du classement
# ---------------------------------------------------------------------------

if ranking_type == "Absolu":
    scores = aggregate_nation_ranking(
        filtered_df,
        n=top_n_riders,
    )

    # La fonction d'agrégation trie déjà par score, mais le ranking est
    # volontairement recalculé ici pour conserver la même chaîne que
    # la page Classement.
    scores = rank_by_score(
        scores,
        score_column="score",
    )

    ranking_table = prepare_nation_table(
        scores,
        countries,
    )


elif ranking_type == "Relatif":
    scores = aggregate_nation_ranking(
        filtered_df,
        n=top_n_riders,
    )

    ranked_df = rank_with_relative_score(
        scores,
        score_column="score",
    )

    ranking_table = prepare_nation_table(
        ranked_df,
        countries,
        score_column="score_relative",
        display_column="Ratio",
    )


else:  # F1
    # rank_f1 attend les colonnes id_rider / year / score.
    # Ici, id_rider représente temporairement l'identifiant de nation.
    nation_year_for_f1 = nation_year_scores.rename(
        columns={"nat_p": "id_rider"},
    )

    ranked_df = rank_f1(
        nation_year_for_f1,
    )

    ranking_table = prepare_f1_nation_table(
        ranked_df,
        countries,
    )


# ---------------------------------------------------------------------------
# Nation en tête par année
# ---------------------------------------------------------------------------

year_leaders = prepare_year_leaders_table(
    nation_year_scores,
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
        use_container_width=True,
        hide_index=True,
        height=get_table_height(ranking_table),
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
            "Ratio": st.column_config.NumberColumn(
                "Ratio",
                format="%.2f",
                width="small",
            ),
        },
    )

with col_winners:
    st.subheader("Vainqueurs annuels")

    st.dataframe(
        year_leaders,
        use_container_width=True,
        hide_index=True,
        height=get_table_height(year_leaders),
        column_config={
            "Année": st.column_config.NumberColumn(
                "Année",
                width="small",
            ),
            "Nation": st.column_config.TextColumn(
                "Nation",
                width="large",
            ),
        },
    )


# ---------------------------------------------------------------------------
# Statistiques et graphiques de la nation
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    "T": "Tours",
    "S": "Etapes",
    "C": "Classiques",
    "I": "Chronos",
    "N": "Championnats nationaux",
}

STAT_OPTIONS = {
    "Points": "score",
    "Victoires": "wins",
    "Podiums": "podiums",
    "Top 5": "top_5",
    "Top 10": "top_10",
    "Classement": "ranking",
}


def _build_country_mapping(
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


def _build_category_chart(
    stats: pd.DataFrame,
    metric: str,
) -> go.Figure:
    """Build the yearly stacked bar chart."""

    chart_data = stats.copy()

    chart_data["category"] = (
        chart_data["edition_cat"].map(CATEGORY_LABELS).fillna(chart_data["edition_cat"])
    )

    # Plotly must treat years as discrete categories:
    # one year = one bar = one tick.
    chart_data["year"] = chart_data["year"].astype(str)
    years = sorted(chart_data["year"].unique())

    figure = go.Figure()

    category_order = [
        category
        for category in CATEGORY_LABELS.values()
        if category in chart_data["category"].unique()
    ]

    for category in category_order:
        category_data = chart_data[chart_data["category"] == category]

        figure.add_trace(
            go.Bar(
                x=category_data["year"],
                y=category_data[metric],
                name=category,
                hovertemplate=(
                    f"<b>{category}</b><br>"
                    "Année : %{x}<br>"
                    "Valeur : %{y}<extra></extra>"
                ),
            )
        )

    # Total of the complete stacked bar.
    totals = chart_data.groupby("year", as_index=False)[metric].sum()

    figure.add_trace(
        go.Scatter(
            x=totals["year"],
            y=totals[metric],
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    for i, (y, label) in enumerate(
        zip(totals[metric], totals[metric].round(0).astype(int).astype(str))
    ):
        figure.add_annotation(
            x=i,
            y=y,
            text=label,
            textangle=-90,  # Orientez le texte selon l'angle souhaité (-45°, 90°, etc.)
            showarrow=False,  # Supprime la flèche pointant vers la coordonnée
            xanchor="center",  # Aligne strictement le milieu horizontal du texte sur la barre
            yanchor="bottom",  # Pose la "base" du texte sur le sommet de la barre (utilisez "top" si le texte pointe vers le bas)
            yshift=5,  # Crée un petit espace vertical pour ne pas coller à la barre
        )

    figure.update_layout(
        barmode="stack",
        xaxis=dict(
            title="Année",
            type="category",
            categoryorder="array",
            categoryarray=sorted(
                chart_data["year"].unique(),
                key=int,
            ),
        ),
        yaxis=dict(
            title=None,
        ),
        legend_title="Catégorie",
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
            t=45,
            b=30,
        ),
    )

    figure.update_xaxes(
        tickmode="array",
        tickvals=years,
        ticktext=[str(year) for year in years],
        tickangle=-90,
    )

    return figure


def _build_annual_ranking_chart(
    ranking: pd.DataFrame,
) -> go.Figure:
    """Build the annual ranking chart for the selected nation."""

    figure = go.Figure()

    ranking = ranking.dropna(subset=["rank_year"]).copy()

    if ranking.empty:
        figure.update_layout(
            yaxis=dict(visible=False),
        )
        return figure

    max_rank = ranking["rank_year"].max()
    y_step = max(1, round(max_rank / 10))

    figure.add_trace(
        go.Scatter(
            x=ranking["year"],
            y=ranking["rank_year"],
            mode="lines+markers+text",
            text=ranking["rank_year"].astype(int).astype(str),
            textposition="top center",
            name="Annuel",
            hovertemplate="Annuel : %{y}<extra></extra>",
            connectgaps=False,
        )
    )

    figure.update_layout(
        yaxis=dict(
            title="Classement annuel",
            autorange="reversed",
            dtick=y_step,
            showgrid=True,
        ),
        yaxis2=dict(
            visible=False,
        ),
        xaxis=dict(
            title="Année",
            tickmode="array",
            tickvals=ranking["year"],
            ticktext=[str(year) for year in ranking["year"]],
            tickangle=-90,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=30,
            r=30,
            t=45,
            b=80,
        ),
    )

    return figure


def _build_nation_leaders(
    filtered_results: pd.DataFrame,
    nation_id: int,
) -> pd.DataFrame:
    """Retourne le meilleur coureur de la nation pour chaque statistique."""

    nation_results = filtered_results[
        (filtered_results["nat_p"] == nation_id) & (filtered_results["id_rider"] != 0)
    ].copy()

    if nation_results.empty:
        return pd.DataFrame(columns=["Statistique", "Coureur", "Valeur"])

    rider_stats = compute_rider_stats(nation_results)

    # Récupération des noms des coureurs
    rider_names = (
        nation_results[["id_rider", "first_name", "surname"]]
        .drop_duplicates("id_rider")
        .copy()
    )

    rider_names["Coureur"] = (
        rider_names["first_name"].fillna("").str.strip()
        + " "
        + rider_names["surname"].fillna("").str.strip()
    ).str.strip()

    rider_stats = rider_stats.merge(
        rider_names[["id_rider", "Coureur"]],
        on="id_rider",
        how="left",
    )

    stat_labels = {
        "score": "Points",
        "wins": "Victoires",
        "podiums": "Podiums",
        "top_5": "Top 5",
        "top_10": "Top 10",
    }

    rows = []

    for metric, label in stat_labels.items():
        if metric not in rider_stats.columns:
            continue

        total = rider_stats[metric].sum()

        # On n'affiche la statistique que si la nation possède
        # au moins une occurrence de celle-ci.
        if total <= 0:
            continue

        best = rider_stats.sort_values(
            by=[metric, "score", "id_rider"],
            ascending=[False, False, True],
        ).iloc[0]

        rows.append(
            {
                "Statistique": label,
                "Coureur": best["Coureur"],
                "Valeur": best[metric],
            }
        )

    return pd.DataFrame(rows)


def _get_nation_identity(
    results: pd.DataFrame,
    countries: pd.DataFrame,
    country_id: int,
) -> pd.DataFrame:
    """
    Build the identity information of a nation.

    The number of riders and scoring years are calculated from the
    complete database and are independent of page filters.
    """

    country = countries[countries["id_country"] == country_id]

    if country.empty:
        return pd.DataFrame()

    country_name = country.iloc[0]["name"]

    nation_results = results[
        (results["nat_p"] == country_id) & (results["id_rider"] != 0)
    ].copy()

    if nation_results.empty:
        return pd.DataFrame(
            {
                "country_id": [country_id],
                "country_name": [country_name],
                "riders_count": [0],
                "start_year": [pd.NA],
                "end_year": [pd.NA],
            }
        )

    scoring_results = nation_results[nation_results["score"].fillna(0) > 0]

    if scoring_results.empty:
        start_year = pd.NA
        end_year = pd.NA
    else:
        start_year = int(scoring_results["year"].min())
        end_year = int(scoring_results["year"].max())

    return pd.DataFrame(
        {
            "country_id": [country_id],
            "country_name": [country_name],
            "riders_count": [nation_results["id_rider"].nunique()],
            "start_year": [start_year],
            "end_year": [end_year],
        }
    )


# ---------------------------------------------------------------------------
# Nation selector
# ---------------------------------------------------------------------------

country_mapping = _build_country_mapping(
    countries,
    results_enriched["nat_p"],
)

country_options = sorted(
    country_mapping,
    key=country_mapping.get,
)

if not country_options:
    st.info("Aucune nation disponible.")
    st.stop()

st.subheader("Statistiques de la nation")

selected_country = st.selectbox(
    "Sélectionner une nation",
    options=country_options,
)

selected_country_id = country_mapping[selected_country]

identity = _get_nation_identity(
    results_enriched,
    countries,
    selected_country_id,
)

if identity.empty:
    st.warning("Impossible de récupérer les informations de cette nation.")
    st.stop()

nation = identity.iloc[0]

# ---------------------------------------------------------------------------
# Nation stats data
# ---------------------------------------------------------------------------

filtered_results = filtered_df[filtered_df["nat_p"] == selected_country_id].copy()

# ---------------------------------------------------------------------------
# Nation header
# ---------------------------------------------------------------------------

info_column, stats_column, leaders_column = st.columns(
    [1, 1, 1.2],
)

with info_column:
    st.subheader("Informations")

    st.markdown(f"**Pays :** {nation['country_name']}")

    st.markdown(f"**Coureurs :** {int(nation['riders_count'])}")

    if pd.isna(nation["start_year"]):
        st.markdown("**Première année :** -")
        st.markdown("**Dernière année :** -")
    else:
        st.markdown(f"**Première année :** {int(nation['start_year'])}")
        st.markdown(f"**Dernière année :** {int(nation['end_year'])}")

with stats_column:
    st.subheader("Bilan")

    nation_stats = compute_nation_stats(
        filtered_results,
    )

    if nation_stats.empty:
        stats = None
    else:
        stats = nation_stats.iloc[0]

    metric_1, metric_2 = st.columns(2)
    metric_3, metric_4 = st.columns(2)
    metric_5, _ = st.columns(2)

    with metric_1:
        st.metric(
            "Victoires",
            0 if stats is None else int(stats["wins"]),
        )

    with metric_2:
        st.metric(
            "Podiums",
            0 if stats is None else int(stats["podiums"]),
        )

    with metric_3:
        st.metric(
            "Top 5",
            0 if stats is None else int(stats["top_5"]),
        )

    with metric_4:
        st.metric(
            "Top 10",
            0 if stats is None else int(stats["top_10"]),
        )

    with metric_5:
        st.metric(
            "Points",
            "0" if stats is None else f"{stats['score']:.0f}",
        )

with leaders_column:
    st.subheader("Leaders")

    leaders = _build_nation_leaders(
        filtered_results,
        selected_country_id,
    )

    if leaders.empty:
        st.info("Aucune statistique positive pour la période sélectionnée.")
    else:
        st.dataframe(
            leaders,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Statistique": st.column_config.TextColumn(
                    "Statistique",
                    width="small",
                ),
                "Coureur": st.column_config.TextColumn(
                    "Coureur",
                    width="medium",
                ),
                "Valeur": st.column_config.NumberColumn(
                    "Valeur",
                    width="small",
                ),
            },
        )

# ---------------------------------------------------------------------------
# Bottom statistics
# ---------------------------------------------------------------------------

st.divider()

statistic = st.selectbox(
    "Statistique",
    options=list(STAT_OPTIONS.keys()),
)

# ---------------------------------------------------------------------------
# Annual ranking
# ---------------------------------------------------------------------------

if statistic == "Classement":
    ranking_rows = []

    for year, yearly in nation_year_scores.groupby("year"):
        ranked = rank_by_score(
            yearly[["nat_p", "score"]],
            score_column="score",
        )

        ranked["year"] = year
        ranking_rows.append(ranked)

    if not ranking_rows:
        st.info("Aucune donnée de classement disponible.")
    else:
        ranking_history = pd.concat(
            ranking_rows,
            ignore_index=True,
        )

        nation_ranking = ranking_history[
            ranking_history["nat_p"] == selected_country_id
        ].copy()

        nation_ranking = nation_ranking.rename(
            columns={"rank": "rank_year"},
        )

        if nation_ranking.empty:
            st.info(
                "Aucune donnée de classement disponible pour la période sélectionnée."
            )
        else:
            figure = _build_annual_ranking_chart(
                nation_ranking,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Annual statistics
# ---------------------------------------------------------------------------

else:
    metric = STAT_OPTIONS[statistic]

    if filtered_results.empty:
        st.info("Aucun résultat pour la période sélectionnée.")
    else:
        category_stats = aggregate_rider_season_category(
            filtered_results,
        )

        # Re-aggregate the rider/category statistics at nation level.
        nation_category_stats = (
            category_stats.groupby(
                ["year", "edition_cat"],
                as_index=False,
            )
            .agg(
                score=("score", "sum"),
                wins=("wins", "sum"),
                podiums=("podiums", "sum"),
                top_5=("top_5", "sum"),
                top_10=("top_10", "sum"),
            )
            .sort_values(
                ["year", "edition_cat"],
            )
            .reset_index(drop=True)
        )

        if nation_category_stats.empty:
            st.info("Aucune statistique pour la période sélectionnée.")
        else:
            figure = _build_category_chart(
                nation_category_stats,
                metric,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )
