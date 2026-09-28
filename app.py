from __future__ import annotations

import streamlit as st

from dataloader.load import load_data

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Women Cycling DB",
    page_icon="🚴",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def get_data():
    """
    Load and validate all application data.

    Streamlit caches the returned data so that the CSV files
    are not reloaded on every interaction.
    """
    return load_data()


data = get_data()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = {
    "Women Cycling DB": [
        st.Page(
            "pages/home.py",
            title="Accueil",
            icon="🏠",
            default=True,
        ),
        st.Page(
            "pages/resultats.py",
            title="Résultats",
            icon="📋",
        ),
        st.Page(
            "pages/classement.py",
            title="Classement",
            icon="🏆",
        ),
        st.Page(
            "pages/cycliste.py",
            title="Cycliste",
            icon="🚴",
        ),
        st.Page(
            "pages/nation.py",
            title="Nation",
            icon="🌍",
        ),
        st.Page(
            "pages/course.py",
            title="Course",
            icon="🏁",
        ),
        st.Page(
            "pages/annee.py",
            title="Année",
            icon="📅",
        ),
    ],
}

pg = st.navigation(pages)


# ---------------------------------------------------------------------------
# Shared application data
# ---------------------------------------------------------------------------

# The data is made available to the pages through session state.
st.session_state["data"] = data


# ---------------------------------------------------------------------------
# Run selected page
# ---------------------------------------------------------------------------

pg.run()
