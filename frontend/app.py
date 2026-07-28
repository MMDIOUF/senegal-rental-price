"""Interface Streamlit de demonstration."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Estimation loyer Senegal", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: #f5f3ee; color: #20231f; }
    [data-testid="stHeader"] { background: transparent; }
    .eyebrow { letter-spacing: .14em; text-transform: uppercase; color: #49705a;
      font-size: .76rem; font-weight: 700; }
    h1 { letter-spacing: -.045em !important; max-width: 720px; }
    .result { border-top: 1px solid #b9c3bb; padding-top: 1.2rem; margin-top: 1rem; }
    .result strong { font-size: clamp(2.1rem, 5vw, 4.4rem); letter-spacing: -.05em; }
    .note { color: #59615b; max-width: 62ch; line-height: 1.6; }
    .stButton button { background: #315d45; color: white; border: 0; border-radius: .5rem;
      min-height: 3rem; font-weight: 600; transition: transform .18s ease; }
    .stButton button:active { transform: translateY(1px) scale(.99); }
    </style>
    """,
    unsafe_allow_html=True,
)

intro, status = st.columns([3, 1])
with intro:
    st.markdown('<div class="eyebrow">Projet M2 DSIA · Démonstrateur</div>', unsafe_allow_html=True)
    st.title("Estimer un loyer au Sénégal, sans jargon.")
    st.markdown(
        '<p class="note">Cette estimation est produite par un modèle pédagogique entraîné '
        "sur des données synthétiques. Elle aide à comparer des scénarios et ne remplace pas "
        "une expertise immobilière.</p>",
        unsafe_allow_html=True,
    )
with status:
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        st.success("API et modèle disponibles" if health.ok else "API indisponible")
    except requests.RequestException:
        st.error("API indisponible")

st.divider()
form_column, result_column = st.columns([1.15, 0.85], gap="large")
with form_column:
    with st.form("rental-form"):
        city = st.selectbox("Ville", ["Dakar", "Thiès", "Saint-Louis", "Mbour", "Saly"])
        property_type = st.selectbox("Type de bien", ["Appartement", "Maison", "Studio", "Villa"])
        quarter = st.text_input("Quartier", value="Mermoz" if city == "Dakar" else "Centre")
        surface = st.number_input("Surface (m²)", min_value=10.0, max_value=2000.0, value=85.0)
        rooms, bedrooms = st.columns(2)
        room_count = rooms.number_input("Pièces", min_value=1, max_value=20, value=3)
        bedroom_count = bedrooms.number_input("Chambres", min_value=0, max_value=15, value=2)
        furnished = st.toggle("Bien meublé")
        equipment = st.multiselect(
            "Équipements",
            ["climatisation", "parking", "gardiennage", "piscine", "groupe_electrogene"],
        )
        submitted = st.form_submit_button("Calculer l'estimation", use_container_width=True)

with result_column:
    st.subheader("Résultat")
    if not submitted:
        st.info("Renseignez le bien puis lancez l'estimation.")
    elif bedroom_count > room_count:
        st.error("Le nombre de chambres ne peut pas dépasser le nombre de pièces.")
    else:
        payload = {
            "ville": city,
            "quartier": quarter,
            "type_bien": property_type,
            "surface_m2": surface,
            "nb_pieces": room_count,
            "nb_chambres": bedroom_count,
            "meuble": furnished,
            "equipements": equipment,
        }
        try:
            with st.spinner("Calcul en cours..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
            if response.ok:
                prediction = response.json()
                formatted = f"{prediction['prix_loyer_mensuel_estime']:,.0f}".replace(",", " ")
                st.markdown(
                    f'<div class="result"><strong>{formatted} FCFA</strong>'
                    f"<p>Estimation mensuelle · modèle {prediction['model_version']}</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                detail = response.json().get("detail", "Saisie refusée par l'API.")
                st.error(f"Impossible de calculer : {detail}")
        except requests.RequestException:
            st.error("Le service de prédiction ne répond pas. Vérifiez docker compose.")
