from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import json
import re
import logging
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# --- CONFIGURATION DU LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("KallamaAPI")

load_dotenv()

# --- CONFIGURATION DU MODÈLE ---
CHECKPOINT_PATH = "checkpoint-71000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
MODEL_ID = "facebook/nllb-200-distilled-600M"

app = FastAPI(title="Kallama API - Mode Protection Amélioré v2")

# --- MOTS COMMUNS FRANÇAIS À NE PAS PROTÉGER ---
# Ces mots commencent souvent par une majuscule mais ne sont pas des noms propres
COMMON_FRENCH_WORDS = {
    # Articles et déterminants
    "le", "la", "les", "un", "une", "des", "du", "de", "l",
    # Pronoms
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "leur", "y", "en",
    "ce", "cet", "cette", "ces", "celui", "celle", "ceux", "celles",
    "qui", "que", "quoi", "dont", "où",
    # Verbes auxiliaires courants
    "est", "sont", "était", "être", "avoir", "fait", "faut",
    # Prépositions et conjonctions
    "et", "ou", "mais", "donc", "car", "ni", "or",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "entre",
    "vers", "chez", "après", "avant", "depuis", "pendant",
    # Adverbes courants
    "ne", "pas", "plus", "moins", "très", "bien", "mal", "aussi",
    "encore", "toujours", "jamais", "déjà", "ici", "là",
    # Autres mots courants
    "tout", "tous", "toute", "toutes", "autre", "autres",
    "même", "si", "comme", "quand", "comment", "pourquoi",
    "son", "sa", "ses", "mon", "ma", "mes", "ton", "ta", "tes",
    "notre", "votre", "nos", "vos", "leur", "leurs",
}

# Convertir en set avec majuscules pour comparaison rapide
COMMON_WORDS_CAPITALIZED = {word.capitalize() for word in COMMON_FRENCH_WORDS}


# --- CHARGEMENT DU GLOSSAIRE FIXE ---
def load_glossaries() -> Tuple[List[str], Dict[str, str]]:
    """
    Charge les glossaires et retourne:
    - Une liste triée des termes (du plus long au plus court)
    - Un dictionnaire terme_source -> terme_wolof (si disponible)
    """
    glossary_terms = set()
    term_translations = {}
    
    files = ["mecanicien.json", "legal_admin.json", "agriculture.json", 
             "it_ia.json", "finance.json", "medical.json"]
    
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    source = item["source_term"].lower()
                    glossary_terms.add(source)
                    # Si le glossaire contient une traduction Wolof, on la garde
                    if "target_term" in item:
                        term_translations[source] = item["target_term"]
        except FileNotFoundError:
            logger.warning(f"Fichier glossaire non trouvé : {file}")
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON dans {file}: {e}")
    
    sorted_terms = sorted(list(glossary_terms), key=len, reverse=True)
    logger.info(f"Glossaire chargé: {len(sorted_terms)} termes techniques")
    
    return sorted_terms, term_translations


TECHNICAL_GLOSSARY, GLOSSARY_TRANSLATIONS = load_glossaries()

# --- INITIALISATION DU MODÈLE ---
try:
    logger.info(f"Chargement du modèle sur {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
    translation_model.eval()
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur fatale au démarrage : {e}")
    raise e


class ChatResponse(BaseModel):
    original_text: str
    translated_response: str
    debug_info: dict = None


def is_proper_noun(word: str, position_in_sentence: int, text: str) -> bool:
    """
    Détermine si un mot est un vrai nom propre à protéger.
    
    Critères:
    - N'est pas un mot français commun
    - Contient des majuscules internes (LinkedIn, WhatsApp)
    - Est un nom propre connu (personnes, lieux, marques)
    - N'est pas le premier mot d'une phrase (sauf si clairement un nom propre)
    """
    word_lower = word.lower()
    
    # 1. Exclure les mots français communs
    if word in COMMON_WORDS_CAPITALIZED or word_lower in COMMON_FRENCH_WORDS:
        return False
    
    # 2. Mots avec majuscules internes = marques (LinkedIn, WhatsApp, iPhone)
    if re.search(r'[a-z][A-Z]', word):
        return True
    
    # 3. Mots tout en majuscules de plus de 2 lettres (acronymes)
    if word.isupper() and len(word) > 2:
        return True
    
    # 4. Premier mot de phrase - vérifier si c'est vraiment un nom propre
    if position_in_sentence == 0:
        # Si le mot suivant commence aussi par majuscule, c'est probablement un nom
        # Ex: "Jean Pierre" vs "Le chat"
        words_after = text[len(word):].strip().split()
        if words_after and words_after[0][0].isupper():
            next_word = words_after[0]
            if next_word not in COMMON_WORDS_CAPITALIZED:
                return True
        return False
    
    # 5. Mot avec majuscule pas en début de phrase = nom propre probable
    return True


def protect_entities(text: str, placeholders: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """
    Protège les entités nommées (noms propres, marques, lieux) de manière intelligente.
    """
    protected_text = text
    
    # Diviser en phrases pour gérer la position
    sentences = re.split(r'([.!?]+\s*)', text)
    
    # Reconstruire avec protection
    result_parts = []
    placeholder_idx = len(placeholders)
    
    for sentence in sentences:
        if not sentence.strip():
            result_parts.append(sentence)
            continue
        
        # Trouver tous les mots avec majuscule
        words_with_positions = []
        for match in re.finditer(r'\b([A-Z][a-zÀ-ÿ]*(?:[A-Z][a-zÀ-ÿ0-9]*)*)\b', sentence):
            words_with_positions.append((match.group(1), match.start()))
        
        # Vérifier chaque mot
        for word, pos in words_with_positions:
            # Ignorer si déjà un placeholder
            if word.startswith("TERM") or word.startswith("NAME"):
                continue
            
            # Vérifier si c'est un nom propre
            is_start = pos == 0 or sentence[pos-1] in '.!?\n'
            position_idx = 0 if is_start else 1
            
            if is_proper_noun(word, position_idx, sentence[pos:]):
                # Éviter les doublons
                if word not in placeholders.values():
                    placeholder = f"NAME{placeholder_idx}"
                    placeholders[placeholder] = word
                    sentence = re.sub(rf'\b{re.escape(word)}\b', placeholder, sentence)
                    placeholder_idx += 1
        
        result_parts.append(sentence)
    
    protected_text = ''.join(result_parts)
    return protected_text, placeholders


def protect_technical_terms(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Protège les termes techniques du glossaire.
    Stratégie: Protection minimale pour garder le contexte.
    """
    placeholders = {}
    protected_text = text
    
    # Compter combien de mots au total
    word_count = len(text.split())
    protected_count = 0
    max_protection_ratio = 0.4  # Ne pas protéger plus de 40% des mots
    
    for i, term in enumerate(TECHNICAL_GLOSSARY):
        # Vérifier si on a atteint la limite de protection
        if protected_count / max(word_count, 1) >= max_protection_ratio:
            logger.warning(f"Limite de protection atteinte ({protected_count}/{word_count} mots)")
            break
        
        pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
        matches = pattern.findall(protected_text)
        
        if matches:
            placeholder = f"TERM{i}"
            placeholders[placeholder] = matches[0]
            protected_text = pattern.sub(placeholder, protected_text)
            protected_count += len(term.split())
    
    return protected_text, placeholders


def translate_with_model(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Effectue la traduction avec le modèle NLLB.
    """
    lang_map = {"fr": "fra_Latn", "wol": "wol_Latn"}
    tokenizer.src_lang = lang_map[src_lang]
    tgt_lang_token = lang_map[tgt_lang]
    
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH
    ).to(DEVICE)
    
    with torch.no_grad():
        generated_tokens = translation_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=MAX_LENGTH,
            num_beams=5,
            repetition_penalty=1.2,  # Réduit de 1.5 pour moins de rigidité
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.0,  # Ajouté pour meilleur équilibre
        )
    
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def restore_placeholders(text: str, placeholders: Dict[str, str]) -> str:
    """
    Restaure les placeholders dans le texte traduit.
    Gère les cas où le placeholder a été modifié (espaces ajoutés, etc.)
    """
    result = text
    
    # Trier par longueur décroissante pour éviter les remplacements partiels
    sorted_keys = sorted(placeholders.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        original_word = placeholders[key]
        
        # Patterns à chercher (le modèle peut modifier les placeholders)
        patterns = [
            rf'\b{key}\b',           # Exact
            rf'{key}',               # Sans frontières
            rf'{key[:-1]}\s*{key[-1]}',  # Avec espace inséré (TERM 480)
        ]
        
        for pattern in patterns:
            result = re.sub(pattern, original_word, result, flags=re.IGNORECASE)
    
    return result


def detect_translation_failure(original: str, translated: str, src_lang: str, tgt_lang: str) -> bool:
    """
    Détecte si la traduction a échoué (texte identique ou presque).
    """
    # Normaliser pour comparaison
    orig_normalized = re.sub(r'\s+', ' ', original.lower().strip())
    trans_normalized = re.sub(r'\s+', ' ', translated.lower().strip())
    
    # Si la traduction est identique à l'original
    if orig_normalized == trans_normalized:
        return True
    
    # Si plus de 80% des mots sont identiques
    orig_words = set(orig_normalized.split())
    trans_words = set(trans_normalized.split())
    
    if orig_words and trans_words:
        overlap = len(orig_words & trans_words) / len(orig_words)
        if overlap > 0.8 and src_lang != tgt_lang:
            return True
    
    return False


def fallback_translation(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Traduction de secours sans protection - laisse le modèle faire son travail.
    """
    logger.info("Utilisation de la traduction de secours (sans protection)")
    return translate_with_model(text, src_lang, tgt_lang)


# --- ENDPOINT PRINCIPAL ---
@app.post("/chat", response_model=ChatResponse)
async def safe_translate(request_obj: Request):
    try:
        body = await request_obj.body()
        logger.info("--- NOUVELLE REQUÊTE REÇUE ---")
        
        try:
            decoded_body = body.decode("utf-8")
            raw_data = json.loads(decoded_body)
        except UnicodeDecodeError as ue:
            logger.error(f"ERREUR ENCODAGE : {ue}")
            raise HTTPException(status_code=400, detail="L'entrée n'est pas en UTF-8 valide")
        
        text = raw_data.get("text", "").strip()
        src_lang = raw_data.get("src_lang", "fr")
        tgt_lang = raw_data.get("tgt_lang", "wol")
        debug_mode = raw_data.get("debug", False)
        
        if not text:
            raise HTTPException(status_code=400, detail="Le texte est vide")
        
        logger.info(f"INPUT ({src_lang} -> {tgt_lang}): {text}")
        
        debug_info = {"original": text} if debug_mode else None
        placeholders = {}
        protected_text = text
        
        if src_lang == "fr":
            # 1. Protection des termes techniques (glossaire)
            protected_text, placeholders = protect_technical_terms(protected_text)
            
            # 2. Protection des entités nommées (noms propres, marques)
            protected_text, placeholders = protect_entities(protected_text, placeholders)
            
            if placeholders:
                logger.info(f"PROTECTED TEXT: {protected_text}")
                logger.info(f"PLACEHOLDERS: {placeholders}")
                if debug_info:
                    debug_info["protected_text"] = protected_text
                    debug_info["placeholders"] = placeholders
        
        # 3. Traduction
        translated_text = translate_with_model(protected_text, src_lang, tgt_lang)
        
        if debug_info:
            debug_info["raw_translation"] = translated_text
        
        # 4. Restauration des placeholders
        final_text = restore_placeholders(translated_text, placeholders)
        
        # 5. Vérification de la qualité de traduction
        if detect_translation_failure(text, final_text, src_lang, tgt_lang):
            logger.warning("Traduction potentiellement échouée, tentative sans protection...")
            final_text = fallback_translation(text, src_lang, tgt_lang)
            if debug_info:
                debug_info["fallback_used"] = True
                debug_info["fallback_result"] = final_text
        
        logger.info(f"OUTPUT: {final_text}")
        logger.info("--- FIN DU TRAITEMENT ---")
        
        response = ChatResponse(
            original_text=text,
            translated_response=final_text
        )
        
        if debug_info:
            response.debug_info = debug_info
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERREUR SERVEUR : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT DE HEALTH CHECK ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "glossary_terms_count": len(TECHNICAL_GLOSSARY),
        "model_loaded": translation_model is not None
    }


# --- ENDPOINT DE TEST ---
@app.post("/test")
async def test_translation(request_obj: Request):
    """
    Endpoint de test avec mode debug activé par défaut.
    """
    body = await request_obj.body()
    raw_data = json.loads(body.decode("utf-8"))
    raw_data["debug"] = True
    
    # Recréer la requête avec debug activé
    class MockRequest:
        async def body(self):
            return json.dumps(raw_data).encode("utf-8")
    
    return await safe_translate(MockRequest())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
