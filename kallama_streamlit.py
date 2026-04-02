import streamlit as st
import requests
import json
import time

# --- CONFIGURATION ---
API_URL = "http://50.7.159.181:8020/chat"  # Changez l'URL si nécessaire

# Configuration de la page
st.set_page_config(
    page_title="Kallama LLM - Assistant Wolof",
    page_icon="🤖",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .wolof-text {
        font-size: 1.1rem;
        color: #2e86ab;
    }
    .french-text {
        font-size: 1rem;
        color: #a23b72;
        font-style: italic;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# --- FONCTION D'APPEL API ---
def send_to_kallama(text: str, system_prompt: str) -> dict:
    """Envoie la requête à l'API Kallama"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": text,
        "system_prompt": system_prompt
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        return {"error": "Délai d'attente dépassé. Le traitement prend trop de temps."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur de connexion: {e}"}


# --- INTERFACE UTILISATEUR ---
st.markdown('<div class="main-header">🤖 Kallama LLM - Assistant en Wolof</div>', unsafe_allow_html=True)

# Description
st.markdown("""
<div class="info-box">
    <b>Comment ça marche :</b><br>
    1. Vous écrivez votre question en Wolof<br>
    2. Le système traduit en Français et la soumet à l'IA<br>
    3. L'IA répond en Français, puis la réponse est retraduite en Wolof<br>
    4. Vous recevez la réponse finale en Wolof !
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Paramètres système
with st.sidebar:
    st.header("⚙️ Paramètres")
    system_prompt = st.text_area(
        "Prompt système (personnalité de l'IA):",
        value="Vous êtes un assistant IA utile, précis et concis qui répond aux questions de manière claire et informative.",
        height=100
    )

    st.markdown("---")
    st.info("""
    **Instructions:**
    - Écrivez votre message en Wolof
    - Le système gère automatiquement les traductions
    - Les réponses peuvent prendre quelques secondes
    """)

# Zone de chat principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💬 Posez votre question en Wolof")
    user_input = st.text_area(
        "Votre message:",
        placeholder="Tapez votre question ou message en wolof ici...",
        height=150,
        key="user_input"
    )

with col2:
    st.subheader("🔍 Réponse en Wolof")
    response_placeholder = st.empty()

    # Zone pour afficher la réponse française (optionnelle)
    with st.expander("Voir la réponse en Français (débogage)"):
        french_placeholder = st.empty()

# Bouton d'envoi
if st.button("🚀 Envoyer à l'IA", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Veuillez entrer un message en Wolof.")
    else:
        with st.spinner("🔄 Traitement en cours... (Traduction → IA → Traduction)"):

            # Barre de progression simulée
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("📤 Envoi du message...")
            progress_bar.progress(20)
            time.sleep(5)

            status_text.text("🔄 Traduction Wolof → Français...")
            progress_bar.progress(40)
            time.sleep(5)

            status_text.text("🤖 Traitement par l'IA Llama...")
            progress_bar.progress(70)

            # Appel API
            result = send_to_kallama(user_input, system_prompt)

            status_text.text("🔄 Traduction Français → Wolof...")
            progress_bar.progress(90)
            time.sleep(5)

            progress_bar.progress(100)
            status_text.text("✅ Traitement terminé !")

            # Affichage des résultats
            if "error" in result:
                st.error(f"❌ Erreur: {result['error']}")
            else:
                # Réponse en Wolof
                response_placeholder.markdown(
                    f'<div class="wolof-text"><b>Réponse:</b><br>{result["translated_response"]}</div>',
                    unsafe_allow_html=True
                )

                # Réponse en Français (pour débogage)
                french_placeholder.markdown(
                    f'<div class="french-text"><b>Réponse Française:</b><br>{result["french_response"]}</div>',
                    unsafe_allow_html=True
                )

            time.sleep(5)
            progress_bar.empty()
            status_text.empty()

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Système Kallama LLM avec Traduction Wolof-Français • Powered by Llama 3.1 & NLLB"
    "</div>",
    unsafe_allow_html=True
)