from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agregations import aggregate_rider_scores, aggregate_rider_season_category
from src.filters import (
    filter_by_edition_category,
    filter_by_edition_sub_category,
    filter_by_max_place,
    filter_by_race,
    filter_by_race_nationality,
)
from src.ranking import rank_by_score, rank_riders_yearly_and_cumulative
from src.stats import (
    compute_rider_career_span,
    compute_rider_category_stats,
    compute_rider_identity,
    compute_rider_stats,
)

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


def _rider_label(row: pd.Series) -> str:
    """Build the label displayed in the rider selector."""
    return f"{row['first_name']} {row['surname']}"


def _format_nationality(
    country_code: object,
    country_name: object,
) -> str:
    """Build a readable nationality label."""
    if pd.isna(country_code) or country_code in (0, "", None):
        return "-"

    if pd.isna(country_name) or country_name in ("", None):
        return str(country_code)

    return f"{country_name}"


def _build_rider_selector(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the list of riders available in the selector.

    Riders with id_rider == 0 are excluded.
    """
    riders = (
        results.loc[
            results["id_rider"] != 0,
            [
                "id_rider",
                "first_name",
                "surname",
            ],
        ]
        .drop_duplicates("id_rider")
        .sort_values(
            ["surname", "first_name"],
            key=lambda column: column.str.lower(),
        )
        .reset_index(drop=True)
    )

    riders["label"] = riders.apply(
        _rider_label,
        axis=1,
    )

    return riders


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


def _compute_category_max_scores(
    filtered_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compute the best total score reached by a rider in each category."""
    filtered_period = filtered_df[
        filtered_df["year"].between(start_year, end_year)
        & (filtered_df["id_rider"] != 0)
    ]

    return (
        filtered_period.groupby(
            ["id_rider", "edition_cat"],
            as_index=False,
        )
        .agg(score=("score", "sum"))
        .groupby(
            "edition_cat",
            as_index=False,
        )
        .agg(category_max_score=("score", "max"))
    )


def _build_radar(
    category_stats: pd.DataFrame,
    category_max_scores: pd.DataFrame,
) -> go.Figure:
    """Build the rider's relative score radar."""
    radar = category_stats.groupby(
        "edition_cat",
        as_index=False,
    ).agg(score=("score", "sum"))

    # Compare each category score with the best total reached in that
    # category among the currently filtered data.
    radar = radar.merge(
        category_max_scores,
        on="edition_cat",
        how="left",
    )

    radar["score_relative"] = 0.0
    has_reference = radar["category_max_score"] > 0

    radar.loc[has_reference, "score_relative"] = (
        radar.loc[has_reference, "score"]
        / radar.loc[has_reference, "category_max_score"]
        * 100
    )

    # Keep the canonical category order.
    categories = list(CATEGORY_LABELS.keys())

    radar = (
        radar.set_index("edition_cat").reindex(categories, fill_value=0).reset_index()
    )

    codes = radar["edition_cat"].tolist()
    labels = [CATEGORY_LABELS.get(code, code) for code in codes]
    relative_values = radar["score_relative"].tolist()
    real_scores = radar["score"].tolist()

    # Close the radar.
    codes_closed = codes + [codes[0]]
    labels_closed = labels + [labels[0]]
    relative_values_closed = relative_values + [relative_values[0]]
    real_scores_closed = real_scores + [real_scores[0]]

    customdata = list(zip(labels_closed, real_scores_closed))

    figure = go.Figure()

    figure.add_trace(
        go.Scatterpolar(
            r=relative_values_closed,
            theta=codes_closed,
            fill="toself",
            name="Score relatif",
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Code : %{theta}<br>"
                "Relatif : %{r:.1f}%<br>"
                "Points : %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20,
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(128, 128, 128, 0.35)",
                linecolor="rgba(128, 128, 128, 0.5)",
            ),
            angularaxis=dict(
                gridcolor="rgba(128, 128, 128, 0.35)",
                linecolor="rgba(128, 128, 128, 0.5)",
            ),
        ),
    )

    return figure


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
            mode="text",
            text=totals[metric].round(0).astype(int).astype(str),
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
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

    return figure


def _build_ranking_chart(
    ranking: pd.DataFrame,
    show_annual: bool = True,
    show_cumulative: bool = True,
) -> go.Figure:
    """Build the annual/cumulative ranking chart."""

    figure = go.Figure()

    annual = ranking.dropna(subset=["rank_year"]).copy()
    cumulative = ranking.dropna(subset=["rank_cumulative"]).copy()

    max_rank_annual = annual["rank_year"].max()
    max_rank_cumulative = cumulative["rank_cumulative"].max()

    y_step_annual = max(1, round(max_rank_annual / 10))
    y_step_cumulative = max(1, round(max_rank_cumulative / 10))

    # ------------------------------------------------------------------
    # Cas 1 : annuel + cumulé
    # ------------------------------------------------------------------
    if show_annual and show_cumulative:
        figure.add_trace(
            go.Scatter(
                x=annual["year"],
                y=annual["rank_year"],
                mode="lines+markers+text",
                text=annual["rank_year"].astype(int).astype(str),
                textposition="top center",
                name="Annuel",
                hovertemplate="Annuel : %{y}<extra></extra>",
                connectgaps=False,
            )
        )

        figure.add_trace(
            go.Scatter(
                x=cumulative["year"],
                y=cumulative["rank_cumulative"],
                mode="lines+markers+text",
                text=cumulative["rank_cumulative"].astype(int).astype(str),
                textposition="bottom center",
                name="Cumulé",
                hovertemplate="Cumulé : %{y}<extra></extra>",
                connectgaps=False,
                yaxis="y2",
            )
        )

        figure.update_layout(
            yaxis=dict(
                title="Classement annuel",
                autorange="reversed",
                dtick=y_step_annual,
                showgrid=True,
            ),
            yaxis2=dict(
                title="Classement cumulé",
                overlaying="y",
                side="right",
                autorange="reversed",
                dtick=y_step_cumulative,
                showgrid=False,
            ),
        )

    # ------------------------------------------------------------------
    # Cas 2 : annuel seul
    # ------------------------------------------------------------------
    elif show_annual:
        figure.add_trace(
            go.Scatter(
                x=annual["year"],
                y=annual["rank_year"],
                mode="lines+markers+text",
                text=annual["rank_year"].astype(int).astype(str),
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
                dtick=y_step_annual,
                showgrid=True,
            ),
            yaxis2=dict(
                visible=False,
            ),
        )

    # ------------------------------------------------------------------
    # Cas 3 : cumulé seul
    # ------------------------------------------------------------------
    elif show_cumulative:
        figure.add_trace(
            go.Scatter(
                x=cumulative["year"],
                y=cumulative["rank_cumulative"],
                mode="lines+markers+text",
                text=cumulative["rank_cumulative"].astype(int).astype(str),
                textposition="top center",
                name="Cumulé",
                hovertemplate="Cumulé : %{y}<extra></extra>",
                connectgaps=False,
            )
        )

        # Le cumulé devient l'axe principal.
        figure.update_layout(
            yaxis=dict(
                title="Classement cumulé",
                autorange="reversed",
                dtick=y_step_cumulative,
                showgrid=True,
            ),
            yaxis2=dict(
                visible=False,
            ),
        )

    # ------------------------------------------------------------------
    # Cas 4 : aucune courbe
    # ------------------------------------------------------------------
    else:
        figure.update_layout(
            yaxis=dict(
                visible=False,
            ),
            yaxis2=dict(
                visible=False,
            ),
        )

    years = ranking["year"].tolist()

    figure.update_layout(
        xaxis=dict(
            title="Année",
            tickmode="array",
            tickvals=years,
            ticktext=[str(year) for year in years],
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data = st.session_state["data"]
countries = data["countries"]
results = data["results_enriched"]

# ---------------------------------------------------------------------------
# Rider selector
# ---------------------------------------------------------------------------

riders = _build_rider_selector(results)

general_ranking = rank_by_score(aggregate_rider_scores(results))
default_rider_id = general_ranking.iloc[0]["id_rider"]

st.title("🚴 Cycliste")

selected_rider_id = st.selectbox(
    "Sélectionner une cycliste",
    options=riders["id_rider"].tolist(),
    index=riders["id_rider"].tolist().index(default_rider_id),
    format_func=lambda rider_id: riders.loc[
        riders["id_rider"] == rider_id,
        "label",
    ].iloc[0],
)

if selected_rider_id is None:
    st.info("Sélectionnez un cycliste.")
    st.stop()

# ---------------------------------------------------------------------------
# Complete rider information
# ---------------------------------------------------------------------------

identity = compute_rider_identity(
    results,
    selected_rider_id,
)

career = compute_rider_career_span(
    results,
    selected_rider_id,
)

career_stats = compute_rider_stats(results[results["id_rider"] == selected_rider_id])

if identity.empty or career.empty:
    st.warning("Impossible de récupérer les informations de ce cycliste.")
    st.stop()

rider = identity.iloc[0]
career_info = career.iloc[0]

start_year = int(career_info["start_year"])
end_year = int(career_info["end_year"])

database_first_year = int(results["year"].min())
database_last_year = int(results["year"].max())

display_end_year = "" if end_year == database_last_year else str(end_year)

career_label = f"{start_year} - {display_end_year}"

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filtres")

filter_start_year = (
    start_year if start_year < database_last_year else database_last_year - 1
)

selected_years = st.sidebar.slider(
    "Intervalle d'années",
    min_value=filter_start_year,
    max_value=database_last_year,
    value=(start_year, end_year),
)

selected_start_year, selected_end_year = selected_years

# Nationalité des courses
race_country_mapping = _build_country_mapping(
    countries,
    results["id_nat"],
)

selected_race_countries = st.sidebar.multiselect(
    "Nationalité des courses",
    options=sorted(race_country_mapping),
)

selected_race_country_ids = [
    race_country_mapping[name] for name in selected_race_countries
]


# Catégorie d'édition
edition_categories = sorted(results["edition_cat"].dropna().unique())

selected_edition_categories = st.sidebar.multiselect(
    "Catégorie",
    options=edition_categories,
)


# Sous-catégorie
edition_sub_categories = sorted(results["s_cat"].dropna().unique())

selected_edition_sub_categories = st.sidebar.multiselect(
    "Sous-catégorie",
    options=edition_sub_categories,
)


# Courses
race_mapping = dict(
    zip(
        results["race_name"],
        results["id_race"],
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

filtered_df = results.copy()

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
# Rider header
# ---------------------------------------------------------------------------

info_column, stats_column, radar_column = st.columns(
    [1, 1, 1.2],
)

with info_column:
    st.subheader("Informations")

    st.markdown(f"**Nom :** {rider['surname']}")

    st.markdown(f"**Prénom :** {rider['first_name']}")

    st.markdown(
        "**Nationalité principale :** "
        + _format_nationality(
            rider["nat_p"],
            rider["country_n3_p"],
        )
    )

    st.markdown(
        "**Nationalité secondaire :** "
        + _format_nationality(
            rider["nat_s"],
            rider["country_n3_s"],
        )
    )

    st.markdown(f"**Carrière :** {career_label}")

with stats_column:
    st.subheader("Bilan de carrière")

    stats = career_stats.iloc[0]

    metric_1, metric_2 = st.columns(2)
    metric_3, metric_4 = st.columns(2)
    metric_5, _ = st.columns(2)

    with metric_1:
        st.metric("Victoires", int(stats["wins"]))

    with metric_2:
        st.metric("Podiums", int(stats["podiums"]))

    with metric_3:
        st.metric("Top 5", int(stats["top_5"]))

    with metric_4:
        st.metric("Top 10", int(stats["top_10"]))

    with metric_5:
        st.metric("Points", f"{stats['score']:.0f}")

with radar_column:
    st.subheader("Répartition des points")

    filtered_results = filtered_df[
        (filtered_df["id_rider"] == selected_rider_id)
        & (
            filtered_df["year"].between(
                selected_start_year,
                selected_end_year,
            )
        )
    ].copy()

    category_stats = compute_rider_category_stats(
        filtered_results,
    )

    category_max_scores = _compute_category_max_scores(
        filtered_df,
        database_first_year,
        database_last_year,
    )

    radar_figure = _build_radar(
        category_stats,
        category_max_scores,
    )

    st.plotly_chart(
        radar_figure,
        use_container_width=True,
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
# Cumulative ranking
# ---------------------------------------------------------------------------

if statistic == "Classement":
    # IMPORTANT:
    # This calculation uses the complete database before applying
    # the display year interval.
    ranking_history = rank_riders_yearly_and_cumulative(
        results,
    )

    rider_ranking = ranking_history[
        (ranking_history["id_rider"] == selected_rider_id)
        & (
            ranking_history["year"].between(
                selected_start_year,
                selected_end_year,
            )
        )
    ].copy()

    if rider_ranking.empty:
        st.info("Aucune donnée de classement disponible pour la période sélectionnée.")
    else:
        show_annual = st.checkbox(
            "Annuel",
            value=True,
            key="show_annual_ranking",
        )

        show_cumulative = st.checkbox(
            "Cumulé",
            value=True,
            key="show_cumulative_ranking",
        )

        figure = _build_ranking_chart(
            rider_ranking,
            show_annual,
            show_cumulative,
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

    filtered_results = filtered_df[
        (filtered_df["id_rider"] == selected_rider_id)
        & (
            filtered_df["year"].between(
                selected_start_year,
                selected_end_year,
            )
        )
    ].copy()

    if filtered_results.empty:
        st.info("Aucun résultat pour la période sélectionnée.")
    else:
        category_stats = aggregate_rider_season_category(
            filtered_results,
        )

        figure = _build_category_chart(
            category_stats,
            metric,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )
