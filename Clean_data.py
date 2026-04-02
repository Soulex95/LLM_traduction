import json
import os

# 1. Définition des fichiers
source_file = "dev.json"
output_file = "dev_nllb.json"

# 2. Définition de la carte de conversion des codes NLLB
# 'fr' -> 'fr_Latn' et 'wo' -> 'wol_Latn'
code_map = {
    "fr": "fr_Latn",
    "wo": "wol_Latn"
}

# Initialisation du compteur de lignes traitées
lines_processed = 0

print(f"Démarrage de la conversion des codes linguistiques...")

# 3. Traitement du fichier ligne par ligne
try:
    with open(source_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            lines_processed += 1
            try:
                # Charger l'objet JSON de la ligne
                data = json.loads(line.strip())

                # Récupérer et modifier les codes source et cible
                codes = data.get('codes', {})

                if 'src' in codes and codes['src'] in code_map:
                    codes['src'] = code_map[codes['src']]

                if 'tgt' in codes and codes['tgt'] in code_map:
                    codes['tgt'] = code_map[codes['tgt']]

                # S'assurer que les codes mis à jour sont bien dans l'objet principal
                data['codes'] = codes

                # Écrire l'objet modifié dans le nouveau fichier (format JSON Lines)
                # ensure_ascii=False permet de conserver les caractères spéciaux comme 'ñ'
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"Erreur: Impossible de décoder la ligne {lines_processed} du JSON. Ligne ignorée.")

    print("-" * 40)
    print(f"✅ Conversion terminée.")
    print(f"Total des lignes traitées: {lines_processed}")
    print(f"Le nouveau fichier prêt pour NLLB est : {output_file}")
    print(f"Utilisez maintenant '{output_file}' pour le fine-tuning.")

except FileNotFoundError:
    print(f"Erreur: Le fichier source '{source_file}' est introuvable.")

except Exception as e:
    print(f"Une erreur inattendue s'est produite: {e}")