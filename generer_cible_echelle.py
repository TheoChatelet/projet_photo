"""
Génère la cible d'échelle à imprimer (marqueurs ArUco) et sa description.

Pourquoi : la photogrammétrie donne un modèle à la bonne forme mais dans une
unité arbitraire (celle de COLMAP). Pour mesurer en millimètres, il faut
poser dans la scène un objet dont la taille est connue. Des marqueurs
ArUco imprimés conviennent bien : l'ordinateur les reconnaît seul, leurs
coins sont repérés avec une précision inférieure au pixel, et ils sont
tous dans le même plan (celui du plateau), ce qui donne en plus la
position du plan sur lequel l'objet est posé.

Sorties (dans le dossier choisi) :
  - cible_echelle.svg  : la feuille à imprimer (échelle 100 %, pas "ajuster
                         à la page")
  - cible_echelle.json : la position exacte de chaque marqueur, relue par
                         echelle.py

Repère de la feuille (vue de dessus) : origine au centre, x vers la droite,
y vers le haut, z vers l'observateur. Les 4 coins d'un marqueur sont
donnés dans l'ordre de détection d'OpenCV : haut-gauche, haut-droit,
bas-droit, bas-gauche.

Lancement : python generer_cible_echelle.py [--dossier .] [--taille-mm 30]
                [--nb 8] [--rayon-mm 80] [--page A4]

Une fois imprimée, MESURER la feuille au pied à coulisse (voir la barre de
100 mm dessinée en bas) : une imprimante peut déformer de quelques dixièmes
de pour cent, et cette erreur se retrouverait dans toutes les mesures.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# Pages en mm (largeur, hauteur) en orientation portrait
FORMATS_PAGE_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}

# Dictionnaire de 50 marqueurs de 4x4 cases : peu de cases = chaque case est
# grosse, donc la détection reste fiable quand le marqueur est petit ou vu
# de biais sur la photo
DICTIONNAIRE_PAR_DEFAUT = "DICT_4X4_50"

# Nombre de cases d'un marqueur, bordure noire comprise (4 cases + 2 de bordure)
def _cases_par_cote(nom_dictionnaire):
    # "DICT_4X4_50" -> 4
    try:
        return int(nom_dictionnaire.split("_")[1].split("X")[0]) + 2
    except (IndexError, ValueError):
        raise ValueError(f"Nom de dictionnaire ArUco inattendu : {nom_dictionnaire}")


def obtenir_dictionnaire(nom_dictionnaire):
    """Renvoie le dictionnaire ArUco d'OpenCV (même appel quelle que soit la version)."""
    import cv2

    if not hasattr(cv2, "aruco"):
        raise ImportError(
            "Ce module OpenCV n'a pas cv2.aruco. Installer : pip install opencv-python>=4.7 "
            "(ou opencv-contrib-python), un seul des deux."
        )
    identifiant = getattr(cv2.aruco, nom_dictionnaire, None)
    if identifiant is None:
        raise ValueError(f"Dictionnaire ArUco inconnu : {nom_dictionnaire}")
    return cv2.aruco.getPredefinedDictionary(identifiant)


def cases_du_marqueur(nom_dictionnaire, identifiant):
    """Matrice (n, n) de 0/1 d'un marqueur : 1 = case noire, bordure comprise."""
    import cv2

    dictionnaire = obtenir_dictionnaire(nom_dictionnaire)
    n = _cases_par_cote(nom_dictionnaire)
    nb_max = dictionnaire.bytesList.shape[0]
    if not 0 <= identifiant < nb_max:
        raise ValueError(f"Identifiant {identifiant} hors du dictionnaire (0 à {nb_max - 1})")
    # 10 pixels par case, puis on relit le centre de chaque case : évite tout
    # souci de bord entre cases
    cote_px = 10 * n
    if hasattr(cv2.aruco, "generateImageMarker"):
        image = cv2.aruco.generateImageMarker(dictionnaire, identifiant, cote_px, borderBits=1)
    else:  # OpenCV < 4.7
        image = cv2.aruco.drawMarker(dictionnaire, identifiant, cote_px, borderBits=1)
    centres = np.arange(n) * 10 + 5
    return (image[np.ix_(centres, centres)] < 128).astype(np.uint8)


def positions_en_cercle(nb, rayon_mm):
    """Centres de `nb` marqueurs répartis sur un cercle, le premier en haut."""
    angles = [math.pi / 2 - 2 * math.pi * i / nb for i in range(nb)]
    return [(rayon_mm * math.cos(a), rayon_mm * math.sin(a)) for a in angles]


def construire_mise_en_page(taille_mm=30.0, nb=8, rayon_mm=70.0, page="A4",
                            nom_dictionnaire=DICTIONNAIRE_PAR_DEFAUT, premier_id=0):
    """Décrit la cible : renvoie un dictionnaire prêt à être écrit en JSON.

    Lève ValueError si la cible ne tient pas sur la page ou si les marqueurs
    se chevauchent : mieux vaut le savoir avant d'imprimer.
    """
    if page not in FORMATS_PAGE_MM:
        raise ValueError(f"Page inconnue : {page} (choix : {', '.join(FORMATS_PAGE_MM)})")
    if nb < 2:
        # Avec un seul marqueur l'échelle repose sur 4 coins très proches :
        # trop sensible au bruit pour un outil de mesure
        raise ValueError("Il faut au moins 2 marqueurs (4 recommandés).")
    if taille_mm <= 0 or rayon_mm <= 0:
        raise ValueError("La taille et le rayon doivent être positifs.")

    largeur_page, hauteur_page = FORMATS_PAGE_MM[page]
    # Orientation paysage : la plus grande dimension disponible compte pour le cercle
    petit_cote = min(largeur_page, hauteur_page)
    etendue = 2 * (rayon_mm + taille_mm * math.sqrt(2) / 2)
    marge_mm = 10.0
    if etendue > petit_cote - 2 * marge_mm:
        raise ValueError(
            f"La cible ({etendue:.0f} mm de large) ne tient pas sur une page {page} "
            f"(max {petit_cote - 2 * marge_mm:.0f} mm). Réduire --rayon-mm ou --taille-mm, "
            f"ou passer en --page A3."
        )

    # Distance entre deux centres voisins sur le cercle : corde = 2 R sin(pi/n)
    ecart_centres = 2 * rayon_mm * math.sin(math.pi / nb)
    if ecart_centres < taille_mm * 1.5:
        raise ValueError(
            f"Les marqueurs se touchent ({ecart_centres:.0f} mm entre centres pour "
            f"{taille_mm:.0f} mm de côté). Réduire --nb ou --taille-mm, ou augmenter --rayon-mm."
        )

    nb_dispo = obtenir_dictionnaire(nom_dictionnaire).bytesList.shape[0]
    if premier_id + nb > nb_dispo:
        raise ValueError(f"Le dictionnaire {nom_dictionnaire} n'a que {nb_dispo} marqueurs.")

    marqueurs = []
    for i, (cx, cy) in enumerate(positions_en_cercle(nb, rayon_mm)):
        marqueurs.append({"id": premier_id + i, "centre_mm": [round(cx, 4), round(cy, 4)]})

    return {
        "version": 1,
        "dictionnaire": nom_dictionnaire,
        "taille_marqueur_mm": float(taille_mm),
        "page": page,
        "marqueurs": marqueurs,
    }


def coins_marqueur_mm(mise_en_page, identifiant):
    """Coordonnées (4, 3) en mm des coins d'un marqueur, dans l'ordre OpenCV.

    Repère de la feuille : x droite, y haut, z = 0 (la feuille est plane).
    """
    for m in mise_en_page["marqueurs"]:
        if m["id"] == identifiant:
            cx, cy = m["centre_mm"]
            h = mise_en_page["taille_marqueur_mm"] / 2.0
            return np.array(
                [[cx - h, cy + h, 0.0],   # haut-gauche
                 [cx + h, cy + h, 0.0],   # haut-droit
                 [cx + h, cy - h, 0.0],   # bas-droit
                 [cx - h, cy - h, 0.0]]   # bas-gauche
            )
    raise KeyError(f"Marqueur {identifiant} absent de la mise en page")


def charger_mise_en_page(chemin):
    """Lit cible_echelle.json et vérifie qu'il est complet."""
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            mep = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{chemin} introuvable. Le générer avec : python generer_cible_echelle.py"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"{chemin} n'est pas un JSON valide ({e})")
    for cle in ("dictionnaire", "taille_marqueur_mm", "marqueurs"):
        if cle not in mep:
            raise ValueError(f"{chemin} : champ « {cle} » manquant")
    if not mep["marqueurs"]:
        raise ValueError(f"{chemin} : aucun marqueur décrit")
    ids = [m["id"] for m in mep["marqueurs"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{chemin} : identifiants de marqueurs en double")
    if mep["taille_marqueur_mm"] <= 0:
        raise ValueError(f"{chemin} : taille_marqueur_mm doit être positive")
    return mep


def mise_en_page_a_echelle(mise_en_page, taille_mesuree_mm):
    """Copie de la mise en page corrigée de l'erreur d'impression.

    On suppose que l'imprimante a agrandi ou réduit TOUTE la feuille du même
    facteur : si un marqueur de 30 mm mesure 29,9 mm au pied à coulisse, les
    distances entre marqueurs sont aussi 0,33 % trop courtes.
    """
    if taille_mesuree_mm <= 0:
        raise ValueError("La taille mesurée doit être positive.")
    facteur = taille_mesuree_mm / mise_en_page["taille_marqueur_mm"]
    copie = json.loads(json.dumps(mise_en_page))
    copie["taille_marqueur_mm"] = float(taille_mesuree_mm)
    for m in copie["marqueurs"]:
        m["centre_mm"] = [m["centre_mm"][0] * facteur, m["centre_mm"][1] * facteur]
    return copie


def ecrire_svg(mise_en_page, chemin_svg):
    """Dessine la feuille en SVG, aux dimensions réelles (1 unité = 1 mm)."""
    largeur_page, hauteur_page = FORMATS_PAGE_MM[mise_en_page["page"]]
    # Paysage : plus de place pour le cercle et la barre de contrôle
    largeur, hauteur = hauteur_page, largeur_page
    cx0, cy0 = largeur / 2, hauteur / 2
    n = _cases_par_cote(mise_en_page["dictionnaire"])
    taille = mise_en_page["taille_marqueur_mm"]
    case = taille / n

    morceaux = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largeur}mm" height="{hauteur}mm" '
        f'viewBox="0 0 {largeur} {hauteur}">',
        f'<rect x="0" y="0" width="{largeur}" height="{hauteur}" fill="white"/>',
        # crispEdges : bords nets, sans anti-crénelage qui grignoterait les cases
        '<g fill="black" shape-rendering="crispEdges">',
    ]
    for m in mise_en_page["marqueurs"]:
        cx, cy = m["centre_mm"]
        x0 = cx0 + cx - taille / 2
        y0 = cy0 - cy - taille / 2  # SVG : y vers le bas
        cases = cases_du_marqueur(mise_en_page["dictionnaire"], m["id"])
        for ligne in range(n):
            for colonne in range(n):
                if cases[ligne, colonne]:
                    morceaux.append(
                        f'<rect x="{x0 + colonne * case:.4f}" y="{y0 + ligne * case:.4f}" '
                        f'width="{case:.4f}" height="{case:.4f}"/>'
                    )
    morceaux.append("</g>")

    # Numéros en petit, hors du marqueur (la zone blanche autour doit rester vide)
    morceaux.append('<g font-family="sans-serif" font-size="3.5" fill="#444" text-anchor="middle">')
    for m in mise_en_page["marqueurs"]:
        cx, cy = m["centre_mm"]
        morceaux.append(
            f'<text x="{cx0 + cx:.3f}" y="{cy0 - cy + taille / 2 + 5:.3f}">id {m["id"]}</text>'
        )
    morceaux.append("</g>")

    # Barre de contrôle d'impression, doit faire exactement 100 mm. Verticale
    # dans la marge gauche : le cercle de marqueurs occupe presque toute la
    # hauteur de la page, mais il reste de la place sur les côtés
    x_barre = 22.0
    y_haut = cy0 - 50
    morceaux.append(
        f'<g stroke="black" stroke-width="0.4" fill="none">'
        f'<line x1="{x_barre}" y1="{y_haut}" x2="{x_barre}" y2="{y_haut + 100}"/>'
        f'<line x1="{x_barre - 2}" y1="{y_haut}" x2="{x_barre + 2}" y2="{y_haut}"/>'
        f'<line x1="{x_barre - 2}" y1="{y_haut + 100}" x2="{x_barre + 2}" y2="{y_haut + 100}"/>'
        f"</g>"
    )
    lignes_texte = [
        "Contrôle d'impression :",
        "la barre ci-contre doit",
        "mesurer 100 mm.",
        "Imprimer à 100 %,",
        "pas « ajuster à la page ».",
        f"Marqueurs : {taille:g} mm",
        mise_en_page["dictionnaire"],
    ]
    morceaux.append('<g font-family="sans-serif" font-size="3.2" fill="#222">')
    for i, ligne in enumerate(lignes_texte):
        morceaux.append(f'<text x="28" y="{y_haut + 6 + i * 5}">{ligne}</text>')
    morceaux.append("</g>")
    morceaux.append("</svg>")

    with open(chemin_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(morceaux))


def generer(dossier=".", **options):
    """Écrit cible_echelle.svg et cible_echelle.json. Renvoie leurs chemins."""
    mep = construire_mise_en_page(**options)
    os.makedirs(dossier, exist_ok=True)
    chemin_json = os.path.join(dossier, "cible_echelle.json")
    chemin_svg = os.path.join(dossier, "cible_echelle.svg")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(mep, f, indent=2, ensure_ascii=False)
    ecrire_svg(mep, chemin_svg)
    return chemin_svg, chemin_json


def main(argv=None):
    p = argparse.ArgumentParser(description="Génère la cible d'échelle ArUco à imprimer.")
    p.add_argument("--dossier", default=".", help="où écrire les fichiers (défaut : dossier courant)")
    p.add_argument("--taille-mm", type=float, default=30.0, help="côté d'un marqueur, bordure noire comprise")
    p.add_argument("--nb", type=int, default=8, help="nombre de marqueurs sur le cercle")
    p.add_argument("--rayon-mm", type=float, default=70.0, help="rayon du cercle (centre des marqueurs)")
    p.add_argument("--page", default="A4", choices=sorted(FORMATS_PAGE_MM))
    args = p.parse_args(argv)
    try:
        svg, js = generer(args.dossier, taille_mm=args.taille_mm, nb=args.nb,
                          rayon_mm=args.rayon_mm, page=args.page)
    except (ValueError, ImportError) as e:
        print(f"ERREUR : {e}")
        return 1
    print(f"Feuille à imprimer : {svg}")
    print(f"Description        : {js}")
    print("À faire : imprimer à 100 %, vérifier la barre de 100 mm au pied à coulisse, "
          "mesurer un marqueur et le donner à echelle.py (--taille-marqueur-mm).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
