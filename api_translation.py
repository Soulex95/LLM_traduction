from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import re  # Ajouté pour la segmentation par phrase

# --- CONFIGURATION API ET MODÈLE ---
CHECKPOINT_PATH = "./NLLB-fr-wolof-bidirectional-finetuned/checkpoint-92000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512  # Longueur max pour l'encodage du modèle principal (pour la génération)
SENTENCE_MAX_LENGTH = 128  # Nouvelle constante pour la longueur max des segments
# Codes NLLB
FR_CODE = "fr_Latn"
WOL_CODE = "wol_Latn"
MODEL_ID = "facebook/nllb-200-distilled-600M"

# --- INITIALISATION DU MODÈLE ET DU TOKENIZER ---
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    app = FastAPI(title="NLLB Fr-Wolof Bidirectional API")


    # --- SCHÉMAS DE DONNÉES (Pydantic) ---
    class TranslationRequest(BaseModel):
        text: str
        source_lang: str
        target_lang: str


    class TranslationResponse(BaseModel):
        translated_text: str

except Exception as e:
    print(f"Erreur fatale au chargement du modèle: {e}")
    raise e


# --- FONCTION DE SEGMENTATION/TRADUCTION PAR LOTS ---
def split_and_translate_in_batches(text: str, src_lang: str, tgt_lang_token: str, forced_bos_token_id: int) -> str:
    """
    Divise le texte en phrases, les regroupe en lots pour l'encodage/traduction,
    puis concatène le résultat.
    """
    # Segmentation simple par phrases (garde les séparateurs)
    # Cherche . ? ! suivis d'un espace ou fin de chaîne, mais capture le séparateur.
    sentences = re.split(r'([.?!]\s*|\n)', text)
    # Filtre les chaînes vides résultant du split si elles n'ont pas de contenu significatif
    sentences = [s.strip() for s in sentences if s.strip() or re.match(r'^[.?!]\s*|\n$', s)]

    # 1. Grouper en Lots pour le traitement
    # On va traiter chaque segment individuellement pour garantir max_length=128
    # NOTE: Pour la traduction, il est préférable de grouper par phrase.

    segments_to_translate = []
    current_segment = ""

    for item in sentences:
        if re.match(r'^[.?!]\s*|\n$', item) and not current_segment:
            # Si le segment est vide mais que c'est un séparateur, on l'ajoute
            # pour maintenir la structure si le texte original était juste des séparateurs
            if segments_to_translate and re.match(r'^[.?!]\s*|\n$', segments_to_translate[-1]):
                # Évite les doubles séparateurs, on ajoute juste le séparateur au dernier segment
                segments_to_translate[-1] += item
            else:
                segments_to_translate.append(item)
            continue

        # On utilise le tokenizer pour vérifier la longueur, en s'assurant que le `src_lang` est bien défini
        tokenizer.src_lang = src_lang
        # Encodage temporaire de la phrase/partie pour obtenir la taille en tokens
        tokens = tokenizer.encode(item, add_special_tokens=True)
        token_count = len(tokens)

        if token_count > SENTENCE_MAX_LENGTH:
            # Si le segment est trop long, on le prend tel quel, le tokenizer le tronquera
            # ou vous pourriez implémenter une division par tokens ici.
            print(f"ATTENTION: Segment de {token_count} tokens > {SENTENCE_MAX_LENGTH}. Traité en un seul segment.")
            segments_to_translate.append(item)
        else:
            segments_to_translate.append(item)

    translated_segments = []
    BATCH_SIZE = 8  # Taille du lot de segments pour l'encodage par batch

    # Traitement par lots (ici, chaque segment est un lot de taille 1,
    # mais on peut regrouper plusieurs segments courts pour optimiser la VRAM si on augmente BATCH_SIZE)

    for i in range(0, len(segments_to_translate), BATCH_SIZE):
        batch = segments_to_translate[i:i + BATCH_SIZE]

        # Filtrer les séparateurs qui n'ont pas de traduction (e.g. les sauts de ligne purs)
        text_batch = [s for s in batch if not re.match(r'^[.?!]\s*|\n$', s)]
        separator_batch = [s for s in batch if re.match(r'^[.?!]\s*|\n$', s)]

        # Encodage du lot pour le modèle
        if text_batch:
            tokenizer.src_lang = src_lang  # Ré-assurer la langue source
            encoded_input = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=SENTENCE_MAX_LENGTH  # Utilise 128 tokens
            ).to(DEVICE)

            # Génération
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded_input,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=SENTENCE_MAX_LENGTH,  # Génère au max 128 tokens
                    num_beams=4,
                    early_stopping=True
                )

            # Décodage
            translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # Reconstruction pour inclure les séparateurs non traduits
            translated_index = 0
            for segment in batch:
                if re.match(r'^[.?!]\s*|\n$', segment):
                    translated_segments.append(segment)  # Garde le séparateur intact
                else:
                    if translated_index < len(translations):
                        translated_segments.append(translations[translated_index])
                        translated_index += 1

        else:
            # Si le lot ne contient que des séparateurs (ex: un bloc de \n)
            translated_segments.extend(separator_batch)

    # 2. Concaténation et Nettoyage
    # Le texte traduit final
    final_translation = "".join(translated_segments)

    # Nettoyage supplémentaire des espaces autour des signes de ponctuation si nécessaire
    # (Dépend de la langue cible, ici on suppose que la traduction maintient un format correct)

    return final_translation


# --- FONCTION D'INFÉRENCE PRINCIPALE (Version corrigée) ---
def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Traduit le texte en utilisant le modèle NLLB chargé et la gestion par lots pour les longs textes."""

    # 1. Vérification des codes (pour l'erreur 400)
    if src_lang not in [FR_CODE, WOL_CODE] or tgt_lang not in [FR_CODE, WOL_CODE]:
        raise ValueError("Codes de langue non valides. Utilisez 'fr_Latn' ou 'wol_Latn'.")

    try:
        # 2. Définition de la langue source
        tokenizer.src_lang = src_lang

        # 3. CORRECTION: Mapping des codes de langue vers les tokens NLLB
        # Les tokens NLLB utilisent le format avec chevrons
        lang_to_token = {
            "fr_Latn": "fra_Latn",  # NLLB utilise "fra" au lieu de "fr"
            "wol_Latn": "wol_Latn"
        }

        tgt_lang_token = lang_to_token.get(tgt_lang)
        if not tgt_lang_token:
            raise ValueError(f"Code de langue cible '{tgt_lang}' non supporté.")

        # 4. Récupération de l'ID du token cible
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)

        # Vérification que l'ID a été trouvé (logique de secours)
        if forced_bos_token_id == tokenizer.unk_token_id:
            # Tentative de récupération en cas d'échec du premier convert
            if tgt_lang == "fr_Latn":
                forced_bos_token_id = tokenizer.convert_tokens_to_ids("fra_Latn")
            elif tgt_lang == "wol_Latn":
                forced_bos_token_id = tokenizer.convert_tokens_to_ids("wol_Latn")

            if forced_bos_token_id == tokenizer.unk_token_id:
                raise ValueError(f"Token de langue cible '{tgt_lang_token}' non trouvé dans le vocabulaire.")

        print(f"Token cible: {tgt_lang_token}, ID: {forced_bos_token_id}")

        # 5. Appel de la nouvelle fonction de traitement par lots
        translation = split_and_translate_in_batches(text, src_lang, tgt_lang_token, forced_bos_token_id)

        return translation

    except Exception as e:
        # Attrape toute autre erreur d'exécution Python et la transforme en 500
        print(f"Erreur interne lors de la traduction: {e}")
        # On relance l'exception comme HTTPException si elle n'a pas été gérée
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Erreur de traitement du modèle: {type(e).__name__} - {str(e)}")


# --- POINT DE TERMINAISON API ---
@app.post("/translate", response_model=TranslationResponse)
def api_translate(request: TranslationRequest):
    # Log de débogage pour voir la requête reçue
    print(f"DEBUG: Reçu src={request.source_lang}, tgt={request.target_lang}, text={request.text[:30]}...")

    try:
        translated_text = translate_text(
            request.text,
            request.source_lang,
            request.target_lang
        )
        # 200 OK si la traduction réussit
        return TranslationResponse(translated_text=translated_text)

    except ValueError as e:
        # 400 Bad Request si les codes de langue sont incorrects
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Already raised by translate_text
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"Erreur inattendue: {str(e)}")