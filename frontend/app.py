"""Interface Streamlit interactive de demonstration et d'explicabilite."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
QUARTERS = {
    "Dakar": ["Almadies", "Mermoz", "Plateau", "Yoff", "Parcelles Assainies"],
    "Thiès": ["Grand Standing", "Randoulène", "Médina Fall"],
    "Saint-Louis": ["Île", "Sor", "Hydrobase"],
    "Mbour": ["Zone résidentielle", "Grand Mbour", "Mbour centre"],
    "Saly": ["Saly Portudal", "Saly centre", "Niakh Niakhal"],
}
PROPERTY_TYPES = ["Appartement", "Maison", "Studio", "Villa"]
EQUIPMENT = ["climatisation", "parking", "gardiennage", "piscine", "groupe_electrogene"]


def api_get(path: str) -> dict[str, Any] | None:
    """Interroge une route GET et transforme toute panne en etat lisible."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_predict(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Demande une estimation a l'API sans exposer d'exception technique."""
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def fcfa(value: float) -> str:
    """Formate une valeur monetaire sans fausse precision."""
    return f"{value:,.0f}".replace(",", " ") + " FCFA"


def scenario_payload(
    city: str,
    quarter: str,
    property_type: str,
    surface: float,
    rooms: int,
    bedrooms: int,
    furnished: bool,
    equipment: list[str],
) -> dict[str, Any]:
    """Construit le contrat JSON partage par l'estimation et les simulations."""
    return {
        "ville": city,
        "quartier": quarter,
        "type_bien": property_type,
        "surface_m2": surface,
        "nb_pieces": rooms,
        "nb_chambres": bedrooms,
        "meuble": furnished,
        "equipements": equipment,
    }


st.set_page_config(
    page_title="Teranga Loyer | Estimation pédagogique",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    :root {
      --ink: #17211b; --muted: #5d6b62; --paper: #f5f2e9; --panel: #fffdf7;
      --green: #1f5a3d; --green-soft: #dfeade; --gold: #c79a3b; --line: #d9ddd5;
    }
    html { scroll-behavior: smooth; }
    html, body, [class*="css"] {
      font-family: "Aptos", "Segoe UI", sans-serif;
      color: var(--ink);
    }
    .stApp {
      background:
        radial-gradient(circle at 86% 4%, rgba(199,154,59,.12), transparent 24rem),
        radial-gradient(circle at 4% 28%, rgba(31,90,61,.09), transparent 28rem),
        var(--paper);
      overflow-x:hidden;
    }
    [data-testid="stHeader"] { background: rgba(245,242,233,.8); backdrop-filter: blur(10px); }
    [data-testid="stAppViewContainer"] > .main .block-container {
      max-width: 1240px; padding-top: 1.2rem; padding-bottom: 5rem;
    }
    #MainMenu, footer { visibility: hidden; }
    .masthead {
      display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
      padding:.65rem 0 1.2rem;
      border-bottom:1px solid var(--line); margin-bottom:2.4rem;
    }
    .brand {font-weight:750; letter-spacing:-.03em; font-size:1.15rem;}
    .brand-mark {color:var(--green); margin-right:.35rem;}
    .mode {font-size:.78rem; color:var(--muted); letter-spacing:.06em; overflow-wrap:anywhere;}
    .eyebrow {color:var(--green); font-size:.75rem; font-weight:750; letter-spacing:.13em;
      text-transform:uppercase; margin-bottom:.65rem;}
    .hero-title {font-size:clamp(2.6rem,6vw,5.5rem); line-height:.94; letter-spacing:-.065em;
      max-width:880px; font-weight:760; text-wrap:balance; margin:0 0 1.2rem;}
    .hero-copy {font-size:1.08rem; line-height:1.65; color:var(--muted); max-width:65ch;}
    .trust-strip {display:flex; flex-wrap:wrap; gap:.55rem; margin:1.4rem 0 2.8rem;}
    .trust-item {background:rgba(255,253,247,.76); border:1px solid var(--line);
      padding:.55rem .8rem; border-radius:.45rem; font-size:.82rem;}
    .section-label {font-size:.75rem; font-weight:750; letter-spacing:.12em; color:var(--green);
      text-transform:uppercase; margin-bottom:.35rem;}
    .result-shell {background:var(--green); color:white; border-radius:1.2rem .35rem 1.2rem .35rem;
      padding:1.65rem 1.7rem 1.9rem; box-shadow:0 22px 55px rgba(31,90,61,.18);
      animation:rise .5s cubic-bezier(.2,.7,.2,1) both;}
    .result-shell .value {font-size:clamp(1.75rem,3vw,2.8rem); font-weight:760;
      letter-spacing:-.055em; font-variant-numeric:tabular-nums; line-height:1.05;
      overflow-wrap:anywhere;}
    .result-shell .range {margin-top:.6rem; color:#dce9df; font-size:.92rem;}
    .result-shell .caption {margin-top:1.25rem; padding-top:1rem;
      border-top:1px solid rgba(255,255,255,.22); font-size:.8rem; color:#dce9df;}
    .empty-result {border:1px dashed #aab5ad; border-radius:1rem; padding:2rem;
      min-height:14rem; display:grid; align-content:center; color:var(--muted);
      background:rgba(255,253,247,.52);}
    .why-card {background:var(--panel); padding:1rem 1.1rem; border-left:3px solid var(--gold);
      margin:.55rem 0; border-radius:0 .55rem .55rem 0; animation:rise .35s ease both;}
    .honesty {background:#edf2e9; border:1px solid #ced9cd; padding:1rem 1.15rem;
      border-radius:.65rem; color:#35483a; font-size:.9rem; line-height:1.55;}
    div[data-baseweb="select"] > div, div[data-testid="stNumberInputContainer"],
    div[data-testid="stMultiSelect"] > div {background:var(--panel); border-color:var(--line);}
    .stButton > button {background:var(--green); color:white; border:0; border-radius:.55rem;
      min-height:3.1rem; font-weight:700; transition:transform .2s ease, box-shadow .2s ease;}
    .stButton > button[kind="primary"] {background:var(--green)!important; color:white!important;}
    .stButton > button:hover {background:#17472f; color:white; transform:translateY(-2px);
      box-shadow:0 10px 24px rgba(31,90,61,.18);}
    .stButton > button:active {transform:translateY(0) scale(.985);}
    .stButton > button:focus-visible {outline:3px solid rgba(199,154,59,.55);}
    [data-testid="stMetric"] {background:rgba(255,253,247,.7); padding:.9rem 1rem;
      border-top:2px solid var(--green);}
    [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums; letter-spacing:-.04em;}
    .stTabs [data-baseweb="tab-list"] {gap:.35rem;}
    .stTabs [data-baseweb="tab"] {border-radius:.4rem; padding:.55rem 1rem;}
    @keyframes rise {from {opacity:0; transform:translateY(12px)} to {opacity:1; transform:none}}
    @media (prefers-reduced-motion: reduce) { * {animation:none!important; transition:none!important;} }
    @media (max-width: 1000px) {
      [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
      [data-testid="column"] {min-width:100%!important; flex:1 1 100%!important;}
      .hero-title {font-size:clamp(2.45rem,10vw,4.3rem);}
      .result-shell .value {font-size:clamp(1.75rem,6vw,2.35rem);}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="masthead">
      <div class="brand"><span class="brand-mark">◆</span> Teranga Loyer</div>
      <div class="mode">ESTIMATEUR LOCATIF · SÉNÉGAL</div>
    </div>
    <div class="eyebrow">Décider avec un ordre de grandeur explicable</div>
    <div class="hero-title">Un loyer estimé.<br>Un raisonnement visible.</div>
    <p class="hero-copy">Configurez un bien, obtenez une estimation mensuelle et voyez immédiatement
    la fourchette d'erreur, les facteurs utilisés et la sensibilité à la surface.</p>
    <div class="trust-strip">
      <div class="trust-item">✓ Estimation instantanée</div>
      <div class="trust-item">✓ Fourchette indicative</div>
      <div class="trust-item">✓ Facteurs visibles</div>
      <div class="trust-item">⚑ Référentiel pédagogique</div>
    </div>
    """,
    unsafe_allow_html=True,
)

health = api_get("/health")
model_info = api_get("/model/info")
if health and health.get("status") == "ok":
    st.caption("● Service d'estimation disponible")
else:
    st.error("Le service d'estimation est momentanément indisponible.")

estimate_tab, reference_tab = st.tabs(["Estimer un bien", "Repères comparatifs"])

with estimate_tab:
    st.markdown('<div class="section-label">01 · Décrire le bien</div>', unsafe_allow_html=True)
    form_column, result_column = st.columns([1.08, 0.92], gap="large")
    with form_column:
        city, property_type = st.columns(2)
        selected_city = city.selectbox("Ville", list(QUARTERS))
        selected_type = property_type.selectbox("Type de bien", PROPERTY_TYPES)
        selected_quarter = st.selectbox("Quartier", QUARTERS[selected_city])
        surface = st.slider("Surface habitable (m²)", 18, 500, 85, 1)
        rooms_column, bedrooms_column = st.columns(2)
        rooms = rooms_column.number_input("Nombre de pièces", 1, 20, 3)
        bedrooms = bedrooms_column.number_input("Nombre de chambres", 0, min(15, int(rooms)), 2)
        furnished = st.toggle("Le bien est meublé")
        selected_equipment = st.multiselect("Équipements", EQUIPMENT)
        estimate = st.button("Estimer le loyer", use_container_width=True, type="primary")

        if estimate:
            payload = scenario_payload(
                selected_city,
                selected_quarter,
                selected_type,
                float(surface),
                int(rooms),
                int(bedrooms),
                furnished,
                selected_equipment,
            )
            with st.spinner("Comparaison du scénario avec le référentiel…"):
                st.session_state["last_payload"] = payload
                st.session_state["last_prediction"] = api_predict(payload)

    with result_column:
        prediction = st.session_state.get("last_prediction")
        if not prediction:
            st.markdown(
                """
                <div class="empty-result"><div><strong>Le résultat apparaîtra ici.</strong><br>
                Commencez par choisir la ville, le quartier et la surface.</div></div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-shell">
                  <div class="section-label" style="color:#dce9df">Estimation mensuelle</div>
                  <div class="value">{fcfa(float(prediction["prix_loyer_mensuel_estime"]))}</div>
                  <div class="range">Fourchette indicative :
                    {fcfa(float(prediction["fourchette_basse"]))} – {fcfa(float(prediction["fourchette_haute"]))}
                  </div>
                  <div class="caption">Niveau de confiance indicatif : {prediction["fiabilite"]}.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("#### Pourquoi ce résultat ?")
            for factor in prediction["facteurs_principaux"]:
                st.markdown(f'<div class="why-card">{factor}</div>', unsafe_allow_html=True)
            st.caption("La fourchette rappelle qu'une estimation reste un ordre de grandeur.")

    if prediction and st.session_state.get("last_payload"):
        st.markdown("---")
        st.markdown(
            '<div class="section-label">02 · Tester la sensibilité</div>', unsafe_allow_html=True
        )
        base_payload = st.session_state["last_payload"]
        surfaces = sorted(
            {
                max(18, int(base_payload["surface_m2"] * factor))
                for factor in (0.70, 0.85, 1.0, 1.15, 1.30)
            }
        )
        sensitivity: list[dict[str, float]] = []
        for candidate_surface in surfaces:
            candidate = {**base_payload, "surface_m2": candidate_surface}
            candidate_prediction = api_predict(candidate)
            if candidate_prediction:
                sensitivity.append(
                    {
                        "Surface (m²)": candidate_surface,
                        "Loyer estimé (FCFA)": candidate_prediction["prix_loyer_mensuel_estime"],
                    }
                )
        if sensitivity:
            chart_data = pd.DataFrame(sensitivity).set_index("Surface (m²)")
            st.line_chart(chart_data, color="#1f5a3d", use_container_width=True)
            st.caption("Simulation du même bien en faisant varier uniquement sa surface.")

with reference_tab:
    st.markdown(
        '<div class="section-label">03 · Situer l’ordre de grandeur</div>',
        unsafe_allow_html=True,
    )
    if not model_info:
        st.warning("Les repères comparatifs ne sont pas disponibles.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### Loyer médian par ville")
            city_medians = model_info.get("reference", {}).get("city_medians", {})
            if city_medians:
                medians = pd.DataFrame.from_dict(
                    city_medians, orient="index", columns=["Loyer médian (FCFA)"]
                ).sort_values("Loyer médian (FCFA)")
                st.bar_chart(medians, color="#1f5a3d", horizontal=True)
        with right:
            st.markdown("#### Loyer médian par type de bien")
            type_medians = model_info.get("reference", {}).get("type_medians", {})
            if type_medians:
                medians_by_type = pd.DataFrame.from_dict(
                    type_medians, orient="index", columns=["Loyer médian (FCFA)"]
                ).sort_values("Loyer médian (FCFA)")
                st.bar_chart(medians_by_type, color="#c79a3b", horizontal=True)

        st.markdown(
            """
            <div class="honesty"><strong>À retenir :</strong> ces repères servent à comparer des
            scénarios. Ils proviennent d'un référentiel pédagogique et ne remplacent pas une
            expertise immobilière réalisée sur place.</div>
            """,
            unsafe_allow_html=True,
        )
