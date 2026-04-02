import json
import os
from typing import List

# Liste des fichiers d'entrée à fusionner
INPUT_FILES: List[str] = [
    "test_final.json",
    "techniques.json"
]

# Nom du fichier de sortie
OUTPUT_FILE: str = "test_final.json"

total_entries_merged: int = 0
entries_in_file: List[dict] = []

print(f"Démarrage de la fusion des fichiers dans {OUTPUT_FILE}...")

try:
    # 1. Parcourir tous les fichiers d'entrée
    for filename in INPUT_FILES:
        file_entries: int = 0
        print(f"-> Traitement du fichier : {filename}")

        with open(filename, 'r', encoding='utf-8') as infile:
            for line_number, line in enumerate(infile):
                try:
                    # Charger l'objet JSON de la ligne
                    entry = json.loads(line.strip())

                    # Vérification simple de la structure (optionnel mais recommandé)
                    if not all(key in entry for key in ["translation", "codes"]):
                        print(
                            f"Avertissement : Ligne {line_number + 1} dans {filename} ne contient pas les clés 'translation' ou 'codes'. Ignorée.")
                        continue

                    # Ajouter l'entrée à la liste
                    entries_in_file.append(entry)
                    file_entries += 1

                except json.JSONDecodeError:
                    print(
                        f"Erreur : Impossible de décoder le JSON à la ligne {line_number + 1} dans {filename}. Ligne ignorée.")

        print(f"   {file_entries} entrées lues depuis {filename}.")
        total_entries_merged += file_entries

    # 2. Écrire toutes les entrées dans le fichier de sortie
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for entry in entries_in_file:
            # ensure_ascii=False est crucial pour conserver les caractères spéciaux du Wolof/Français
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("-" * 60)
    print(f"✅ Fusion terminée avec succès.")
    print(f"Total des entrées uniques dans {OUTPUT_FILE} : {total_entries_merged}")
    print(f"Ce fichier est votre jeu de données d'entraînement complet pour NLLB.")

except FileNotFoundError as e:
    print(
        f"Erreur: Le fichier d'entrée '{e.filename}' est introuvable. Veuillez vérifier le nom du fichier et son emplacement.")

except Exception as e:
    print(f"Une erreur inattendue s'est produite lors de la fusion : {e}")