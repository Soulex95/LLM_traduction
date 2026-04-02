from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import logging
from dotenv import load_dotenv
from typing import Optional

# --- CONFIGURATION DU LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("KallamaAPI")

load_dotenv()

# --- CONFIGURATION DU MODÈLE ---
# Utilisation du chemin vers votre modèle final réentraîné
CHECKPOINT_PATH = "./NLLB-fr-wolof-bidirectional-finetuned/final_model/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128

app = FastAPI(title="Kallama API - Traduction Robuste Bidirectionnelle")


# --- CLASSES DE DONNÉES ---
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "fra_Latn"
    target_lang: str = "wol_Latn"
    temperature: Optional[float] = 1.0
    num_beams: Optional[int] = 5


class ChatResponse(BaseModel):
    original_text: str
    translated_response: str
    detection_info: Optional[dict] = None


# --- CHARGEMENT DU MODÈLE ---
logger.info(f"Chargement du modèle depuis {CHECKPOINT_PATH} sur {DEVICE}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
    translation_model = None


# --- DÉTECTION DE COPIE ---
def detect_copy(source: str, translation: str, threshold: float = 0.8) -> bool:
    """Détecte si la traduction est une simple copie du texte source"""
    source_words = set(source.lower().split())
    translation_words = set(translation.lower().split())

    if len(source_words) == 0:
        return False

    common_words = source_words.intersection(translation_words)
    copy_ratio = len(common_words) / len(source_words)
    return copy_ratio > threshold


def _calculate_copy_ratio(source: str, translation: str) -> float:
    source_words = set(source.lower().split())
    translation_words = set(translation.lower().split())
    if len(source_words) == 0: return 0.0
    common_words = source_words.intersection(translation_words)
    return round(len(common_words) / len(source_words), 2)


# --- LOGIQUE DE TRADUCTION MULTI-STRATÉGIE ---
def translate_text(text: str, src_lang: str, tgt_lang: str,
                   temperature: float = 1.0, num_beams: int = 5) -> tuple[str, dict]:
    """Traduction avec repli automatique sur plusieurs stratégies"""
    if not text.strip():
        return "", {}

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    # Liste des stratégies par ordre de priorité
    strategies = [
        {"name": "greedy", "params": {"num_beams": 1, "do_sample": False, "repetition_penalty": 2.0}},
        {"name": "standard", "params": {"num_beams": num_beams, "repetition_penalty": 2.5, "no_repeat_ngram_size": 3}},
        {"name": "aggressive", "params": {"num_beams": 8, "repetition_penalty": 3.0, "no_repeat_ngram_size": 2}},
        {"name": "sampling", "params": {"do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 1.2}}
    ]

    best_translation = text
    best_strategy = "none"

    for strategy in strategies:
        try:
            with torch.no_grad():
                generated_tokens = translation_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=MAX_LENGTH,
                    **strategy["params"]
                )

            translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            if not detect_copy(text, translation):
                return translation, {
                    "is_copy_detected": False,
                    "strategy_used": strategy["name"],
                    "copy_ratio": _calculate_copy_ratio(text, translation)
                }

            best_translation = translation
            best_strategy = strategy["name"]

        except Exception as e:
            logger.warning(f"Échec stratégie {strategy['name']}: {e}")
            continue

    return best_translation, {
        "is_copy_detected": True,
        "strategy_used": best_strategy,
        "copy_ratio": _calculate_copy_ratio(text, best_translation)
    }


# --- ENDPOINTS ---
@app.post("/translate", response_model=ChatResponse)
async def handle_translation(request_obj: TranslationRequest):
    if translation_model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    translated_text, detection_info = translate_text(
        request_obj.text, request_obj.source_lang, request_obj.target_lang,
        request_obj.temperature, request_obj.num_beams
    )

    return ChatResponse(
        original_text=request_obj.text,
        translated_response=translated_text,
        detection_info=detection_info
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": DEVICE, "checkpoint": CHECKPOINT_PATH}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)