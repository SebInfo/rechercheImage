import os
import time
import random
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import hashlib
import cv2
import numpy as np
import imagehash
import sys
import platform
import gradio as gr

print("Python executable :", sys.executable)
print("Python version    :", sys.version)
print("Platform          :", platform.platform())

# --- Paramètres de recherche  ---
api_key = "votre identifiant API"
search_engine_id = "Identifiant navigateur"
output_dir = "images/"
taille_min = 800
max_images = 5
seuil_phash = 5  # seuil de tolérance de similarité

# --- Préparation du dossier ---
os.makedirs(output_dir, exist_ok=True)

# --- En-tête HTTP ---
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# --- Initialisation du détecteur de visages ---
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascades", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)



# --- Fonctions utilitaires ---
def contient_visage(image_pil):
    """Retourne True si un visage est détecté dans l’image PIL"""
    try:
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return len(faces) > 0
    except Exception as e:
        print(f"[ERREUR] Détection visage : {e}")
        return False

def contient_un_seul_visage(image_pil):
    try:
        img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return len(faces) == 1
    except Exception as e:
        print(f"[ERREUR] Détection visage : {e}")
        return False

def get_md5_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def est_similaire(phash_new, phashes_connus, seuil=5):
    for old_hash in phashes_connus:
        if phash_new - old_hash <= seuil:
            return True
    return False

def creer_repertoire_unique(base_dir, nom):
    """
    Crée un dossier basé sur 'nom'.
    Si le dossier existe déjà, ajoute un suffixe (_1, _2, ...).
    """
    # Normaliser le nom
    dossier_nom = nom.replace(" ", "_").lower()
    chemin = os.path.join(base_dir, dossier_nom)

    # Ajouter un indice si nécessaire
    indice = 1
    chemin_unique = chemin
    while os.path.exists(chemin_unique):
        chemin_unique = f"{chemin}_{indice}"
        indice += 1

    # Créer le dossier
    os.makedirs(chemin_unique, exist_ok=True)
    return chemin_unique

def chercher_images(nom):
    # ... même code qu’avant (boucle API Google) ...
    # et à la fin :

    # --- Variables de contrôle ---
    downloaded = 0
    start_index = 1
    tried_urls = set()
    md5_hashes_seen = set()
    phashes_seen = []

    # --- Boucle principale ---
    urls_gardees = []
    repertoire_nom = creer_repertoire_unique(output_dir, nom)
    while downloaded < max_images:
        print(f"🌀 Requête à partir de l’index {start_index}")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": nom,
            "searchType": "image",
            "num": 10,
            "start": start_index,
            "imgSize": "xlarge",
            "imgType": "photo",
            "imgColorType": "color",
            "fileType": "jpg",
            "safe": "off"  # désactive SafeSearch
        }

        response = requests.get(url, params=params)
        data = response.json()

        items = data.get("items", [])
        if not items:
            print("❌ Plus de résultats disponibles.")
            break

        for item in items:
            img_url = item.get("link")
            if img_url in tried_urls:
                continue
            tried_urls.add(img_url)

            try:
                img_response = requests.get(img_url, headers=headers, timeout=10)
                content_type = img_response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    print(f"[IGNORÉ] Non image : {img_url} ({content_type})")
                    continue

                img_bytes = img_response.content
                md5 = get_md5_hash(img_bytes)
                if md5 in md5_hashes_seen:
                    print(f"[IGNORÉ] Doublon exact détecté : {img_url}")
                    continue
                md5_hashes_seen.add(md5)

                try:
                    img = Image.open(BytesIO(img_bytes))
                    img.verify()
                    img = Image.open(BytesIO(img_bytes))

                    if max(img.width, img.height) >= taille_min:

                        if not contient_un_seul_visage(img):
                            print(f"[IGNORÉ] Pas un seul visage détecté : {img_url}")
                            continue

                        phash = imagehash.phash(img)
                        if est_similaire(phash, phashes_seen, seuil_phash):
                            print(f"[IGNORÉ] Image visuellement similaire : {img_url}")
                            continue
                        phashes_seen.append(phash)


                        ext = img.format.lower() or "jpg"
                        filename = f"{nom.replace(' ', '_').lower()}_{downloaded+1}.{ext}"
                        filepath = os.path.join(repertoire_nom, filename)
                        urls_gardees.append(img_url)
                        img.save(filepath)
                        downloaded += 1
                        print(f"[{downloaded:03}] ✅ {filepath}")

                        if downloaded >= max_images:
                            break
                    else:
                        print(f"[IGNORÉ] Trop petite ({img.width}x{img.height}): {img_url}")

                except UnidentifiedImageError:
                    print(f"[ERREUR] Format inconnu : {img_url}")

            except Exception as e:
                print(f"[ERREUR] Téléchargement : {img_url} ({e})")

        start_index += 10

        print(f"\n✅ Téléchargement terminé : {downloaded} images enregistrées dans {output_dir}")
    return urls_gardees

# Interface Gradio
demo = gr.Interface(
    fn=chercher_images,
    inputs=gr.Textbox(label="Entrez un nom"),
    outputs=gr.Gallery(label="Résultats", columns=3, height="auto"),
    title="Recherche d'images par nom",
    description="Tapez un nom et regardez les images affichées ci-dessous."
)

demo.launch()