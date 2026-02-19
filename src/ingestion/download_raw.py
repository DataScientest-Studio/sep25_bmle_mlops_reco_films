import subprocess
import os
import sys

def download_from_dagshub():
    """
    Récupère les données brutes (raw) via DVC.
    Cible spécifiquement le fichier data/raw.dvc.
    """
    # On se place à la racine du projet pour que les chemins DVC soient corrects
    
    print("🔍 Vérification du pointeur DVC...")
    if not os.path.exists("data/raw.dvc"):
        print("❌ Erreur : Le fichier 'data/raw.dvc' est introuvable.")
        print("Avez-vous fait 'dvc add data/raw' ?")
        return False

    try:
        print("📡 Lancement du pull DVC (Synchronisation avec DagsHub)...")
        # On utilise subprocess pour appeler DVC
        result = subprocess.run(
            ["dvc", "pull", "data/raw.dvc"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Rapport DVC :")
        print(result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du pull DVC : {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Erreur : La commande 'dvc' n'est pas accessible. Est-il installé ?")
        return False

if __name__ == "__main__":
    # Permet de tester le téléchargement en lançant : 
    # python src/ingestion/download_raw.py
    if download_from_dagshub():
        print("🚀 Prêt pour l'ingestion SQL.")
    else:
        sys.exit(1)