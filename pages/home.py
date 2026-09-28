from __future__ import annotations

import streamlit as st

data = st.session_state["data"]

countries = data["countries"]
riders = data["riders"]
races = data["races"]
editions = data["editions"]
results = data["results"]
results_enriched = data["results_enriched"]

with st.sidebar:
    st.header("Données")
    st.metric(
        label="Cyclistes",
        value=f"{len(riders):,}",
    )
    st.metric(
        label="Courses",
        value=f"{len(races):,}",
    )
    st.metric(
        label="Éditions",
        value=f"{len(editions):,}",
    )
    st.metric(
        label="Pays",
        value=f"{len(countries):,}",
    )
    st.metric(
        label="Résultats",
        value=f"{len(results):,}",
    )
    st.divider()
    st.subheader("État")
    st.success("Données chargées en mémoire")
    st.caption("Actualisation : 06/12/2025")

st.title("🚴 Women Cycling Database")

st.markdown(
    """
    ### Bienvenue

    Cette application permet d'explorer les résultats cyclistes féminins,
    construire un classement historique et consulter quelques statistiques.
    """
)

st.subheader("Quelques statistiques")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Première année",
        editions["year"].min(),
    )

with col2:
    st.metric(
        "Dernière année",
        editions["year"].max(),
    )

with col3:
    st.metric(
        "Résultats",
        f"{len(results):,}",
    )

st.subheader("Contenu")

st.markdown("#### 📋 Résultats")
st.write("Consulter le palmarès d'une course ou d'une cycliste.")

st.markdown("#### 🏆 Classement")
st.write("Consulter le classement all-time et le filtrer suivant différents critères.")

st.markdown("#### 🚴 Cycliste")
st.write("Consulter la fiche individuelle de chaque cycliste et ses statistiques.")

st.markdown("#### 🌍 Nation")
st.write(
    "Consulter le classement all-time par nation ainsi que les statistiques nationales."
)

st.markdown("#### 🏁 Course")
st.write("Consulter la fiche de chaque course et ses statistiques.")

st.markdown("#### 📅 Année")
st.write(
    "Consulter le bilan d'une saison avec son classement, son calendrier et ses statistiques."
)
