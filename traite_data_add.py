import json
import os

# 1. Définition des fichiers
source_file = "dev_wo_Latn_fr_Latn_1.json"
output_file = "dev_bidirectional_new.json"

# 2. Codes linguistiques NLLB
WOL_CODE = "wol_Latn"
FR_CODE = "fr_Latn"

# Initialisation
lines_processed = 0
entries_generated = 0
all_new_entries = []

print(f"Démarrage de la conversion du fichier : {source_file}...")

# 3. Traitement du fichier ligne par ligne
try:
    with open(source_file, 'r', encoding='utf-8') as infile:

        for line in infile:
            lines_processed += 1
            try:
                # Charger l'objet JSON de la ligne
                data = json.loads(line.strip())

                wolof_text = data["translation"]["wol"]
                french_text = data["translation"]["fr"]

                # --- GÉNÉRATION DE LA DIRECTION 1 : Wolof -> Français ---
                entry_wo_to_fr = {
                    "translation": {
                        "src": wolof_text,
                        "tgt": french_text
                    },
                    "codes": {
                        "src": WOL_CODE,
                        "tgt": FR_CODE
                    }
                }
                all_new_entries.append(entry_wo_to_fr)
                entries_generated += 1

                # --- GÉNÉRATION DE LA DIRECTION 2 : Français -> Wolof ---
                entry_fr_to_wo = {
                    "translation": {
                        "src": french_text,
                        "tgt": wolof_text
                    },
                    "codes": {
                        "src": FR_CODE,
                        "tgt": WOL_CODE
                    }
                }
                all_new_entries.append(entry_fr_to_wo)
                entries_generated += 1

            except json.JSONDecodeError:
                print(f"Erreur: Impossible de décoder la ligne {lines_processed}. Ligne ignorée.")
            except KeyError as e:
                print(f"Erreur: Clé manquante {e} dans la ligne {lines_processed}. Ligne ignorée.")

    # 4. Écriture des nouvelles entrées dans le fichier de sortie
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in all_new_entries:
            # ensure_ascii=False est crucial pour conserver les caractères spéciaux du Wolof/Français
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("-" * 60)
    print(f"✅ Conversion terminée.")
    print(f"Lignes originales traitées (paires): {lines_processed}")
    print(f"Entrées bidirectionnelles générées: {entries_generated}")
    print(f"Le fichier prêt pour NLLB est : {output_file}")

except FileNotFoundError:
    print(f"Erreur: Le fichier source '{source_file}' est introuvable.")

except Exception as e:
    print(f"Une erreur inattendue s'est produite: {e}")