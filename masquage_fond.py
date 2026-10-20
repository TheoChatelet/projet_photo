"""
Masquage du fond des photos (optionnel).

Les points d'intérêt trouvés sur le fond (feuille, table) ajoutent du bruit
autour de l'objet dans le nuage 3D. En masquant le fond, COLMAP ne cherche
des points que sur l'objet.

Contient aussi lister_images, lire_image et ecrire_image, réutilisées par
main.py pour que les deux fichiers voient exactement les mêmes images.
"""

import os

import cv2
import numpy as np

# Comparées en minuscules dans lister_images, donc .JPG et .jpg passent tous les deux
EXTENSIONS_IMAGES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def lister_images(dossier):
    """Renvoie la liste triée des images d'un dossier."""
    if not os.path.isdir(dossier):
        return []
    # Un seul os.listdir avec l'extension en minuscules, et pas plusieurs
    # glob('*.jpg') + glob('*.JPG') : sous Windows la casse est ignorée, donc
    # ces deux glob trouvent les mêmes fichiers et chaque photo était traitée
    # deux fois (GrabCut deux fois plus long)
    return sorted(
        os.path.join(dossier, nom)
        for nom in os.listdir(dossier)
        if os.path.splitext(nom)[1].lower() in EXTENSIONS_IMAGES
    )


def lire_image(chemin):
    """Lit une image, ou renvoie None si le fichier n'est pas lisible."""
    # On passe par numpy + imdecode au lieu de cv2.imread, parce que
    # cv2.imread renvoie None sous Windows dès que le chemin contient un
    # accent (ex : "C:/Users/Théo/Bureau/Photos dé")
    try:
        donnees = np.fromfile(chemin, dtype=np.uint8)
    except OSError:
        return None
    if donnees.size == 0:
        return None
    return cv2.imdecode(donnees, cv2.IMREAD_COLOR)


def ecrire_image(chemin, image):
    """Écrit une image. Renvoie True si tout s'est bien passé."""
    # Même problème d'accents que cv2.imread, donc même contournement
    extension = os.path.splitext(chemin)[1]
    ok, tampon = cv2.imencode(extension, image)
    if ok:
        tampon.tofile(chemin)
    return ok


def generer_masques(dossier_images, dossier_masques, marge_bord_pct=0.06, iterations=5):
    """
    Pour chaque image, crée un masque noir et blanc (blanc = objet,
    noir = fond) au format attendu par COLMAP : "12.jpg" -> "12.jpg.png".

    Paramètres :
        marge_bord_pct : marge (en % de la taille de l'image) considérée
                         comme du fond au départ
        iterations     : nombre de passes de GrabCut (plus = plus précis
                         mais plus lent)
    """
    os.makedirs(dossier_masques, exist_ok=True)

    # Même fonction que main.py : sinon une photo .tif ou .bmp acceptée par
    # main.py se retrouvait sans masque
    chemins = lister_images(dossier_images)
    if not chemins:
        raise FileNotFoundError(f"Aucune image trouvée dans {dossier_images}")

    print(f"Génération des masques pour {len(chemins)} images (GrabCut)...")
    for i, chemin in enumerate(chemins):
        img = lire_image(chemin)
        if img is None:
            print(f"  Ignoré (illisible) : {chemin}")
            continue
        h, w = img.shape[:2]

        # GrabCut a besoin d'un rectangle de départ autour de l'objet. On prend
        # toute l'image moins une marge, parce que le protocole de prise de vue
        # (voir l'étude) met l'objet au centre avec du fond visible sur les bords.
        marge_x = int(w * marge_bord_pct)
        marge_y = int(h * marge_bord_pct)
        rect = (marge_x, marge_y, w - 2 * marge_x, h - 2 * marge_y)

        # Tableaux de travail imposés par l'API de GrabCut (65 = taille fixée
        # par OpenCV, on ne choisit pas)
        masque = np.zeros((h, w), np.uint8)
        modele_fond = np.zeros((1, 65), np.float64)
        modele_objet = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, masque, rect, modele_fond, modele_objet,
                    iterations, cv2.GC_INIT_WITH_RECT)

        # GrabCut répond en 4 valeurs (0/2 = fond certain/probable, 1/3 = objet
        # certain/probable) mais COLMAP veut du tout ou rien : on garde aussi
        # le "probable" pour ne pas grignoter les bords de l'objet
        masque_binaire = np.where((masque == 2) | (masque == 0), 0, 255).astype("uint8")

        # COLMAP retrouve le masque d'une image uniquement par ce nom exact
        nom_fichier = os.path.basename(chemin)
        chemin_masque = os.path.join(dossier_masques, nom_fichier + ".png")
        ecrire_image(chemin_masque, masque_binaire)

        # GrabCut est lent : sans ces messages on croit que le programme est planté
        if (i + 1) % 10 == 0 or (i + 1) == len(chemins):
            print(f"  {i + 1}/{len(chemins)} masques générés")

    print(f"Masques enregistrés dans : {dossier_masques}")
    print("Vérifie visuellement quelques masques avant de lancer le pipeline complet "
          "(ouvre une image .png du dossier de masques à côté de la photo d'origine : "
          "le blanc doit correspondre à l'objet, le noir au fond).")
    return dossier_masques


if __name__ == "__main__":
    # Sur les images redimensionnées et pas sur Photos/ : COLMAP n'accepte
    # que des masques de la même taille que les images qu'il traite
    generer_masques(dossier_images="colmap_workspace/images_redimensionnees",
                    dossier_masques="colmap_workspace/masques")
