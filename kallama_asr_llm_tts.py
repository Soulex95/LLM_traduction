import streamlit as st
import requests
import json
import time
import io
from audiorecorder import audiorecorder

# --- CONFIGURATION DES URLS ---
KALLAMA_API_URL = "http://3.212.177.54:8020/chat"
ASR_API_URL = "http://3.212.177.54:8003/transcribe/"
TTS_API_URL = "http://3.212.177.54:8010/synthesize/"

# Configuration de la page
st.set_page_config(
    page_title="Kallama Vocal 🗣️ - Assistant Wolof (ASR-LLM-TTS)",
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
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALISATION SESSION STATE ---
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_format' not in st.session_state:
    st.session_state.audio_format = None
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "Vous êtes un assistant IA utile, précis et concis qui répond aux questions de manière claire et informative avec une longueure de 500 maximum donc soit bref et concis."


# --- FONCTIONS D'APPEL API ---

def send_to_kallama(text: str, system_prompt: str) -> dict:
    """Envoie la requête de chat/traduction à l'API Kallama (texte → texte)"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": text,
        "system_prompt": system_prompt
    }

    try:
        response = requests.post(KALLAMA_API_URL, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Délai d'attente (LLM) dépassé."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur de connexion à l'API Kallama: {e}"}


def call_asr(audio_data: bytes, audio_format: str = "wav") -> dict:
    """Appelle le service ASR pour transcrire l'audio en texte Wolof."""
    try:
        # Préparation du fichier
        filename = f"audio.{audio_format}"
        files = {
            'file': (filename, audio_data, f'audio/{audio_format}')
        }

        # Debug info
        debug_info = {
            "url": ASR_API_URL,
            "file_size": len(audio_data),
            "file_type": audio_format
        }

        # Envoi à l'ASR
        response = requests.post(ASR_API_URL, files=files, timeout=60)

        if response.status_code != 200:
            # Essayer avec l'endpoint sans slash
            ASR_API_URL_NO_SLASH = ASR_API_URL.rstrip('/')
            response = requests.post(ASR_API_URL_NO_SLASH, files=files, timeout=60)

        response.raise_for_status()

        result = response.json()
        return result

    except requests.exceptions.Timeout:
        return {"error": "Délai d'attente (ASR) dépassé."}
    except requests.exceptions.RequestException as e:
        error_detail = f"Status: {e.response.status_code if e.response else 'N/A'}"
        return {"error": f"Erreur de connexion à l'API ASR: {e} | {error_detail}"}
    except Exception as e:
        return {"error": f"Erreur inattendue ASR: {e}"}


def call_tts(text: str) -> dict:
    """Appelle le service TTS pour synthétiser le texte Wolof en audio."""
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}

    try:
        response = requests.post(TTS_API_URL, headers=headers, data=json.dumps(payload), timeout=180)
        response.raise_for_status()
        return {"audio_data": response.content, "success": True}
    except requests.exceptions.Timeout:
        return {"error": "Délai d'attente (TTS) dépassé."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Erreur de connexion à l'API TTS: {e}"}


# --- INTERFACE UTILISATEUR ---
st.markdown('<div class="main-header">🗣️ Kallama Vocal - Assistant en Wolof (ASR-LLM-TTS)</div>',
            unsafe_allow_html=True)

# Description du pipeline
st.markdown("""
<div class="info-box">
    <b>Workflow Vocal :</b><br>
    1. 🎤 Vous parlez en Wolof (enregistrement direct)<br>
    2. 🔊 <b>ASR</b>: Transcription audio → texte Wolof<br>
    3. 🤖 <b>LLM</b>: Traitement intelligent de la question<br>
    4. 🗣️ <b>TTS</b>: Synthèse vocale de la réponse<br>
    5. ✅ Vous écoutez la réponse vocale !
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Paramètres système
with st.sidebar:
    st.header("⚙️ Paramètres LLM")

    system_prompt = st.text_area(
        "Prompt système:",
        value=st.session_state.system_prompt,
        height=100,
        key="system_prompt_input"
    )

    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt

    st.markdown("---")
    st.info("""
    **Instructions:**
    - Parlez clairement en Wolof
    - Durée: 1-30 secondes
    - Évitez le bruit de fond
    - Attendez la réponse vocale complète
    """)

# Zone de chat principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎤 Enregistrement Vocal")

    # Enregistreur audio
    audio = audiorecorder("🎤 Démarrer", "⏹️ Arrêter", key="audio_recorder")

    if audio and len(audio) > 0:
        # Afficher l'audio enregistré
        audio_bytes = audio.export().read()
        st.audio(audio_bytes, format="audio/wav")
        st.success(f"✅ Audio enregistré ({len(audio)} échantillons)")

        # Sauvegarder dans session state
        st.session_state.audio_data = audio_bytes
        st.session_state.audio_format = "wav"

        st.info("💡 Cliquez sur 'Lancer le pipeline' pour traiter.")

    st.markdown("---")

    # Alternative avec téléchargement de fichier
    uploaded_file = st.file_uploader(
        "Ou téléchargez un fichier audio",
        type=['wav', 'mp3', 'ogg'],
        key="file_uploader"
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        st.session_state.audio_data = uploaded_file.read()
        st.session_state.audio_format = file_extension
        st.audio(st.session_state.audio_data, format=f'audio/{file_extension}')
        st.success(f"✅ Fichier '{uploaded_file.name}' chargé")

with col2:
    st.subheader("🔍 Réponse Vocalisée")

    # Indicateur d'état
    if st.session_state.audio_data:
        st.success("✅ Audio prêt pour traitement")
    else:
        st.info("📝 En attente d'un enregistrement audio...")

    transcription_placeholder = st.empty()
    audio_response_placeholder = st.empty()
    wolof_text_response_placeholder = st.empty()

    with st.expander("📋 Détails du traitement"):
        french_placeholder = st.empty()
        debug_placeholder = st.empty()

# Bouton d'envoi principal
st.markdown("---")
if st.button("🚀 Lancer le pipeline Vocal", use_container_width=True, type="primary"):
    if st.session_state.audio_data is None:
        st.error("❌ Veuillez d'abord enregistrer ou télécharger un audio.")
        st.stop()

    # Initialisation
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Étape 1: ASR - Transcription audio
        status_text.text("🎯 Étape 1/4 : Transcription audio en cours...")
        progress_bar.progress(25)

        # Appel ASR
        asr_result = call_asr(st.session_state.audio_data, st.session_state.audio_format)

        # Vérification erreur ASR
        if "error" in asr_result:
            st.error(f"❌ Erreur ASR: {asr_result['error']}")
            raise Exception(f"ASR failed: {asr_result['error']}")

        # EXTRACTION CORRECTE DE LA TRANSCRIPTION
        # L'ASR retourne {'transcription': 'texte'} selon vos logs
        wolof_input_text = asr_result.get("transcription", "").strip()

        # Fallback pour d'autres clés possibles
        if not wolof_input_text:
            wolof_input_text = asr_result.get("transcribed_text", "").strip()
        if not wolof_input_text:
            wolof_input_text = asr_result.get("text", "").strip()

        # Debug info
        debug_placeholder.text(f"""
🔍 Debug ASR:
- Clés disponibles: {list(asr_result.keys())}
- Transcription: '{wolof_input_text}'
- Longueur: {len(wolof_input_text)}
        """)

        if not wolof_input_text:
            st.warning("⚠️ L'ASR a retourné un texte vide. Vérifiez l'audio.")
            raise Exception("ASR returned empty text")

        # Affichage transcription
        transcription_placeholder.markdown(
            f'<div class="success-box"><b>✅ Transcription Wolof :</b> {wolof_input_text}</div>',
            unsafe_allow_html=True
        )

        # Étape 2: LLM - Traitement par IA
        status_text.text("🎯 Étape 2/4 : Traitement par l'IA...")
        progress_bar.progress(50)

        kallama_result = send_to_kallama(wolof_input_text, st.session_state.system_prompt)

        # Vérification erreur LLM
        if "error" in kallama_result:
            st.error(f"❌ Erreur LLM: {kallama_result['error']}")
            raise Exception(f"LLM failed: {kallama_result['error']}")

        wolof_response_text = kallama_result.get("translated_response", "")
        french_response_text = kallama_result.get("french_response", "")

        # Affichage des réponses intermédiaires
        french_placeholder.markdown(
            f'<div class="french-text"><b>🇫🇷 Texte Français :</b> {french_response_text}</div>',
            unsafe_allow_html=True
        )
        wolof_text_response_placeholder.markdown(
            f'<div class="wolof-text"><b>🇸🇳 Réponse Wolof :</b> {wolof_response_text}</div>',
            unsafe_allow_html=True
        )

        # Étape 3: TTS - Synthèse vocale
        status_text.text("🎯 Étape 3/4 : Synthèse vocale...")
        progress_bar.progress(75)

        tts_result = call_tts(wolof_response_text)

        # Vérification erreur TTS
        if "error" in tts_result:
            st.error(f"❌ Erreur TTS: {tts_result['error']}")
            raise Exception(f"TTS failed: {tts_result['error']}")

        # Étape 4: Affichage résultat final
        status_text.text("🎯 Étape 4/4 : Finalisation...")
        progress_bar.progress(90)

        # Affichage de l'audio de réponse
        audio_response_placeholder.markdown("### 🔊 Réponse vocale :")
        audio_response_placeholder.audio(tts_result["audio_data"], format='audio/wav')

        # Finalisation
        progress_bar.progress(100)
        status_text.text("✅ Pipeline vocal terminé avec succès !")

        # Message de succès
        st.balloons()
        st.success("🎉 Votre assistant vocal a répondu avec succès !")

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {str(e)}")

    finally:
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

# Bouton de nettoyage
st.markdown("---")
if st.button("🧹 Effacer l'audio et recommencer", use_container_width=True):
    st.session_state.audio_data = None
    st.session_state.audio_format = None
    st.rerun()

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Système Kallama Vocal • ASR, Llama 3.1, NLLB & TTS Wolof"
    "</div>",
    unsafe_allow_html=True
)