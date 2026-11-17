"""
Étape 4a du pipeline : mise à l'échelle métrique du modèle.

Pourquoi : COLMAP ne connaît pas la taille réelle de ce qu'il reconstruit.
Son modèle est juste dans les proportions, mais exprimé dans une unité
arbitraire (le STL de colmap_workspace/ mesure 0,6 « unités » de large).
Sans échelle on ne peut ni mesurer un volume ni comparer à un pied à coulisse.

Principe :
  1. On pose la cible ArUco (generer_cible_echelle.py) à côté de l'objet,
     à plat sur le plateau.
  2. On détecte les 4 coins de chaque marqueur dans chaque photo.
  3. Avec les positions de caméras de COLMAP, on triangule ces coins : on
     obtient leurs positions 3D dans le repère de COLMAP.
  4. On connaît leurs vraies positions en mm (la cible est imprimée à
     une taille connue). On cherche la similitude (échelle + rotation +
     translation) qui superpose les deux : c'est l'échelle, et en plus le
     plan du plateau (z = 0, vers le haut), utilisé pour détourer l'objet.

Entrées : colmap_workspace/sparse_final, colmap_workspace/images_redimensionnees
          et cible_echelle.json (créé par generer_cible_echelle.py)
Sortie  : colmap_workspace/echelle.json

Sans cible, deux solutions de repli : --facteur (mm par unité, déjà connu) ou
--distance-unites / --distance-mm (une longueur connue entre deux points du
modèle). Attention : ne JAMAIS mettre à l'échelle avec la cote de la pièce
qu'on veut ensuite valider, la validation n'aurait plus aucun sens.

Lancement seul : python echelle.py [--travail colmap_workspace]
                    [--mise-en-page cible_echelle.json] [--taille-marqueur-mm 29.9]
                    [--debug]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

from generer_cible_echelle import (
    charger_mise_en_page,
    coins_marqueur_mm,
    mise_en_page_a_echelle,
    obtenir_dictionnaire,
)

# Seuils de qualité. Ils ne bloquent rien : ils décident du drapeau « fiable »
# et des avertissements affichés. À régler avec les essais de validation.
ANGLE_MIN_TRIANGULATION_DEG = 3.0   # en dessous, la profondeur d'un coin est trop imprécise
REPROJECTION_MAX_PX = 2.0           # un coin qui s'écarte plus de sa position triangulée est rejeté
COTE_MIN_PX = 14.0                  # un marqueur plus petit est trop mal localisé
NB_MARQUEURS_MIN = 2
RMS_RELATIF_MAX_PCT = 1.0           # résidu de la superposition / taille de la cible
ECART_METHODES_MAX_PCT = 1.5        # échelle par similitude vs échelle par distances


class ErreurEchelle(Exception):
    """Mise à l'échelle impossible. Le message dit quoi corriger."""


# ----------------------------------------------------------------------
# Détection des marqueurs
# ----------------------------------------------------------------------
def _creer_detecteur(nom_dictionnaire):
    """Renvoie une fonction image_gris -> (liste de coins, liste d'ids)."""
    import cv2

    dictionnaire = obtenir_dictionnaire(nom_dictionnaire)
    if hasattr(cv2.aruco, "ArucoDetector"):  # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        # Affinage des coins au sous-pixel : c'est ce qui fait la précision de l'échelle
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

        def detecter(image):
            coins, ids, _ = detecteur.detectMarkers(image)
            return coins, ids
    else:  # anciennes versions d'OpenCV
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        def detecter(image):
            coins, ids, _ = cv2.aruco.detectMarkers(image, dictionnaire, parameters=params)
            return coins, ids

    return detecter


def detecter_marqueurs(image, nom_dictionnaire, ids_autorises=None, _detecteur=None):
    """Détecte les marqueurs ArUco d'une image.

    Renvoie {id: tableau (4, 2) des coins en pixels}, convention OpenCV (le
    centre du premier pixel est en (0, 0)). On écarte :
      - les identifiants hors de la cible (faux positifs sur le décor) ;
      - un identifiant vu deux fois dans la même photo (on ne sait pas lequel
        est le bon) ;
      - les marqueurs trop petits ou déformés (quadrilatère non convexe).
    """
    import cv2

    if image is None or image.size == 0:
        return {}
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detecter = _detecteur or _creer_detecteur(nom_dictionnaire)
    coins, ids = detecter(image)
    if ids is None or len(ids) == 0:
        return {}

    trouves, doublons = {}, set()
    for c, i in zip(coins, ids.flatten()):
        i = int(i)
        if ids_autorises is not None and i not in ids_autorises:
            continue
        c = np.asarray(c, dtype=np.float64).reshape(4, 2)
        if i in trouves:
            doublons.add(i)
            continue
        if not _quadrilatere_valide(c):
            continue
        trouves[i] = c
    for i in doublons:
        trouves.pop(i, None)
    return trouves


def _quadrilatere_valide(c):
    """Convexe, et côtés d'au moins COTE_MIN_PX pixels."""
    cotes = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    if cotes.min() < COTE_MIN_PX:
        return False
    # Produits vectoriels des côtés consécutifs : tous de même signe si convexe
    aretes = np.roll(c, -1, axis=0) - c
    suivantes = np.roll(aretes, -1, axis=0)
    croix = aretes[:, 0] * suivantes[:, 1] - aretes[:, 1] * suivantes[:, 0]
    return bool(np.all(croix > 0) or np.all(croix < 0))


# ----------------------------------------------------------------------
# Vues (caméras positionnées) : indépendantes de pycolmap pour être testables
# ----------------------------------------------------------------------
class Vue:
    """Une photo dont la caméra est positionnée.

    nom           : nom du fichier image
    cam_from_world: matrice (3, 4) [R | t] qui passe du repère COLMAP à la caméra
    normaliser    : fonction (N, 2) pixels COLMAP -> (N, 2) coordonnées normalisées
                    sans distorsion (x/z, y/z) ; c'est elle qui gère l'objectif
    focale_px     : pour convertir une erreur normalisée en pixels
    marqueurs     : {id: (4, 2) pixels COLMAP}, rempli par la détection
    """

    def __init__(self, nom, cam_from_world, normaliser, focale_px):
        self.nom = nom
        self.cam_from_world = np.asarray(cam_from_world, dtype=np.float64).reshape(3, 4)
        self.normaliser = normaliser
        self.focale_px = float(focale_px)
        self.marqueurs = {}

    def centre(self):
        R, t = self.cam_from_world[:, :3], self.cam_from_world[:, 3]
        return -R.T @ t


def _valeur(x):
    """Selon la version de pycolmap, une même information est un attribut ou une méthode."""
    return x() if callable(x) else x


def vues_depuis_reconstruction(reconstruction):
    """Construit les Vue d'une reconstruction pycolmap (images positionnées seulement)."""
    vues = []
    for image in reconstruction.images.values():
        # has_pose n'existe pas dans les anciennes versions : toute image présente est alors positionnée
        if not _valeur(getattr(image, "has_pose", True)):
            continue
        camera = reconstruction.cameras[image.camera_id]

        def normaliser(points, camera=camera):
            points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            # cam_from_img gère toutes les distorsions de COLMAP (SIMPLE_RADIAL, OPENCV...)
            return np.array([camera.cam_from_img(p) for p in points], dtype=np.float64)

        pose = _valeur(image.cam_from_world)
        vues.append(Vue(image.name, _valeur(pose.matrix), normaliser, _valeur(camera.mean_focal_length)))
    return vues


def detecter_dans_vues(vues, dossier_images, nom_dictionnaire, ids_autorises=None,
                       dossier_debug=None):
    """Remplit vue.marqueurs pour chaque vue. Renvoie le nombre d'images lisibles."""
    import cv2
    from masquage_fond import ecrire_image, lire_image

    detecteur = _creer_detecteur(nom_dictionnaire)
    lues = 0
    for vue in vues:
        image = lire_image(os.path.join(dossier_images, vue.nom))
        if image is None:
            continue
        lues += 1
        trouves = detecter_marqueurs(image, nom_dictionnaire, ids_autorises, _detecteur=detecteur)
        # COLMAP place le centre du premier pixel en (0,5 ; 0,5), OpenCV en (0 ; 0)
        vue.marqueurs = {i: c + 0.5 for i, c in trouves.items()}
        if dossier_debug is not None:
            os.makedirs(dossier_debug, exist_ok=True)
            for i, c in trouves.items():
                cv2.polylines(image, [c.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                cv2.putText(image, str(i), tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            ecrire_image(os.path.join(dossier_debug, os.path.splitext(vue.nom)[0] + ".jpg"), image)
    return lues


# ----------------------------------------------------------------------
# Triangulation
# ----------------------------------------------------------------------
def trianguler_point(observations):
    """Position 3D d'un point vu par plusieurs caméras.

    observations : liste de (cam_from_world (3,4), xy_normalise (2,), focale_px, centre (3,))
    Renvoie (X, erreur_max_px, angle_max_deg, nb_vues_gardees) ou None si impossible.

    Méthode : triangulation linéaire (DLT), puis affinage par moindres carrés
    sur l'erreur de reprojection. Si une vue s'écarte trop, on la retire et on
    recommence : une détection ratée (marqueur flou) ne doit pas fausser le point.
    """
    from scipy.optimize import least_squares

    obs = list(observations)
    while len(obs) >= 2:
        X = _dlt(obs)
        if X is None:
            return None
        X = least_squares(_residus_reprojection, X, args=(obs,), loss="soft_l1", f_scale=1.0).x
        erreurs = np.abs(_residus_reprojection(X, obs)).reshape(-1, 2).max(axis=1)
        pire = int(np.argmax(erreurs))
        if erreurs[pire] <= REPROJECTION_MAX_PX:
            # Le point doit être devant toutes les caméras, sinon c'est une triangulation fausse
            if any((P @ np.append(X, 1.0))[2] <= 0 for P, _, _, _ in obs):
                return None
            return X, float(erreurs[pire]), _angle_max_deg(X, obs), len(obs)
        obs.pop(pire)
    return None


def _dlt(obs):
    lignes = []
    for P, (x, y), _, _ in obs:
        lignes.append(x * P[2] - P[0])
        lignes.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(lignes))
    h = Vt[-1]
    if abs(h[3]) < 1e-12:
        return None
    return h[:3] / h[3]


def _residus_reprojection(X, obs):
    """Écarts en pixels (x puis y pour chaque vue) entre observé et reprojeté."""
    res = []
    for P, xy, f, _ in obs:
        p = P @ np.append(X, 1.0)
        # Un point presque dans le plan de la caméra ferait diverger l'optimisation
        z = p[2] if abs(p[2]) > 1e-9 else 1e-9
        res.extend([(p[0] / z - xy[0]) * f, (p[1] / z - xy[1]) * f])
    return np.array(res)


def _angle_max_deg(X, obs):
    """Plus grand angle entre deux rayons caméra -> point : la « base » de la triangulation."""
    rayons = np.array([(c - X) / np.linalg.norm(c - X) for _, _, _, c in obs])
    cosinus = np.clip(rayons @ rayons.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosinus.min())))


def trianguler_coins(vues, mise_en_page):
    """Triangule chaque coin de chaque marqueur vu par au moins 2 caméras.

    Renvoie une liste de dict {id, coin, X, mm (3,), erreur_px, angle, nb_vues}
    et un dict de comptage pour le diagnostic.
    """
    par_coin = {}
    for vue in vues:
        for identifiant, coins in vue.marqueurs.items():
            normalises = vue.normaliser(coins)
            for k in range(4):
                par_coin.setdefault((identifiant, k), []).append(
                    (vue.cam_from_world, normalises[k], vue.focale_px, vue.centre()))

    points, diag = [], {"coins_vus": len(par_coin), "rejetes_peu_de_vues": 0,
                        "rejetes_angle": 0, "rejetes_reprojection": 0}
    for (identifiant, k), obs in sorted(par_coin.items()):
        if len(obs) < 2:
            diag["rejetes_peu_de_vues"] += 1
            continue
        res = trianguler_point(obs)
        if res is None:
            diag["rejetes_reprojection"] += 1
            continue
        X, erreur, angle, nb = res
        if angle < ANGLE_MIN_TRIANGULATION_DEG:
            diag["rejetes_angle"] += 1
            continue
        points.append({"id": identifiant, "coin": k, "X": X,
                       "mm": coins_marqueur_mm(mise_en_page, identifiant)[k],
                       "erreur_px": erreur, "angle": angle, "nb_vues": nb})
    return points, diag


# ----------------------------------------------------------------------
# Similitude (échelle + rotation + translation)
# ----------------------------------------------------------------------
def estimer_similitude(source, cible):
    """Similitude cible ≈ s · R · source + t au sens des moindres carrés (Umeyama).

    Renvoie (s, R, t). R est une vraie rotation (det = +1) : une cible
    plane ne doit jamais renvoyer un miroir.
    """
    source = np.asarray(source, dtype=np.float64)
    cible = np.asarray(cible, dtype=np.float64)
    n = len(source)
    if n < 3 or len(cible) != n:
        raise ErreurEchelle("Au moins 3 points sont nécessaires pour estimer l'échelle.")
    mu_s, mu_c = source.mean(axis=0), cible.mean(axis=0)
    xs, xc = source - mu_s, cible - mu_c
    variance = (xs ** 2).sum() / n
    if variance < 1e-18:
        raise ErreurEchelle("Les points triangulés sont tous confondus : échelle indéterminée.")
    U, D, Vt = np.linalg.svd(xc.T @ xs / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / variance)
    t = mu_c - s * R @ mu_s
    return s, R, t


def _residus_mm(points_colmap, points_mm, s, R, t):
    return np.linalg.norm((s * (R @ points_colmap.T)).T + t - points_mm, axis=1)


def _echelle_par_distances(points_colmap, points_mm):
    """Médiane des rapports (distance réelle / distance COLMAP) sur toutes les paires.

    Estimation indépendante de la rotation : sert de contrôle croisé de la
    similitude, et résiste aux points aberrants grâce à la médiane.
    """
    n = len(points_colmap)
    rapports = []
    for i in range(n):
        for j in range(i + 1, n):
            d_colmap = np.linalg.norm(points_colmap[i] - points_colmap[j])
            if d_colmap > 1e-12:
                rapports.append(np.linalg.norm(points_mm[i] - points_mm[j]) / d_colmap)
    return float(np.median(rapports)) if rapports else float("nan")


def calculer_echelle_cible(vues, mise_en_page):
    """Calcule l'échelle à partir de marqueurs déjà détectés dans les vues.

    Renvoie le dictionnaire écrit dans echelle.json (voir ecrire_echelle).
    Lève ErreurEchelle si les marqueurs ne suffisent pas.
    """
    ids_cible = {m["id"] for m in mise_en_page["marqueurs"]}
    nb_images_avec = sum(1 for v in vues if v.marqueurs)
    if nb_images_avec == 0:
        raise ErreurEchelle(
            "Aucun marqueur de la cible n'a été détecté dans les photos. Vérifier : cible "
            "bien dans le champ, nette, dictionnaire identique à celui imprimé, marqueurs "
            "assez gros dans l'image (au moins ~15 px de côté). Relancer avec --debug "
            "pour voir ce qui est détecté."
        )

    points, diag = trianguler_coins(vues, mise_en_page)
    if len(points) < 4:
        raise ErreurEchelle(
            f"Seulement {len(points)} coin(s) de marqueur exploitable(s) "
            f"(vus : {diag['coins_vus']}, rejetés : {diag['rejetes_peu_de_vues']} vus par une "
            f"seule photo, {diag['rejetes_angle']} avec un angle trop faible, "
            f"{diag['rejetes_reprojection']} incohérents). Il faut que chaque marqueur soit "
            f"visible depuis des directions variées (au moins {ANGLE_MIN_TRIANGULATION_DEG:.0f}° d'écart)."
        )

    avertissements = []   # ce qui rend l'échelle douteuse : fait passer « fiable » à False
    notes = []            # simple information (marqueur écarté, marqueur non vu)
    # Rejet au niveau du MARQUEUR : un marqueur mal détecté fausse ses 4 coins à la fois
    gardes = sorted({p["id"] for p in points})
    while True:
        pts = [p for p in points if p["id"] in gardes]
        colmap = np.array([p["X"] for p in pts])
        mm = np.array([p["mm"] for p in pts])
        if len(pts) < 4:
            raise ErreurEchelle("Trop peu de coins cohérents après rejet des marqueurs aberrants.")
        s, R, t = estimer_similitude(colmap, mm)
        residus = _residus_mm(colmap, mm, s, R, t)
        ids_pts = np.array([p["id"] for p in pts])
        par_marqueur = {i: float(np.sqrt(np.mean(residus[ids_pts == i] ** 2))) for i in gardes}
        if len(gardes) <= 2:
            break
        pire = max(par_marqueur, key=par_marqueur.get)
        autres = [v for k, v in par_marqueur.items() if k != pire]
        # Aberrant si bien plus mauvais que les autres (et plus d'un demi-mm : sous ce
        # seuil on est dans le bruit normal, inutile de jeter de la donnée)
        if par_marqueur[pire] > max(3.0 * np.median(autres), 0.5):
            notes.append(
                f"Marqueur {pire} écarté (résidu {par_marqueur[pire]:.2f} mm contre "
                f"{np.median(autres):.2f} mm en médiane pour les autres).")
            gardes.remove(pire)
        else:
            break

    nb_marqueurs = len(gardes)
    rms = float(np.sqrt(np.mean(residus ** 2)))
    etendue = float(np.max(np.linalg.norm(mm[:, None, :] - mm[None, :, :], axis=2)))
    rms_relatif = 100.0 * rms / etendue
    s_distances = _echelle_par_distances(colmap, mm)
    ecart_methodes = 100.0 * abs(s - s_distances) / s

    # Incertitude par « jackknife » : on retire chaque marqueur tour à tour et on
    # regarde de combien l'échelle bouge. Estimation de l'écart-type de l'échelle.
    incertitude_pct = None
    if nb_marqueurs >= 3:
        echelles = []
        for retire in gardes:
            m = np.array([p["id"] != retire for p in pts])
            echelles.append(estimer_similitude(colmap[m], mm[m])[0])
        echelles = np.array(echelles)
        n = len(echelles)
        ecart = math.sqrt((n - 1) / n * float(np.sum((echelles - echelles.mean()) ** 2)))
        incertitude_pct = 100.0 * ecart / s

    if nb_marqueurs < NB_MARQUEURS_MIN:
        avertissements.append(f"Un seul marqueur exploitable : échelle peu fiable (minimum {NB_MARQUEURS_MIN}).")
    elif nb_marqueurs < 4:
        avertissements.append("Moins de 4 marqueurs exploitables : mettre plus de marqueurs dans le champ.")
    if rms_relatif > RMS_RELATIF_MAX_PCT:
        avertissements.append(
            f"Superposition cible/modèle imprécise (résidu {rms:.2f} mm = {rms_relatif:.2f} % de la "
            f"cible, seuil {RMS_RELATIF_MAX_PCT} %). Feuille gondolée, photos floues ou reconstruction déformée ?")
    if ecart_methodes > ECART_METHODES_MAX_PCT:
        avertissements.append(
            f"Les deux estimations d'échelle diffèrent de {ecart_methodes:.2f} % "
            f"(seuil {ECART_METHODES_MAX_PCT} %) : un marqueur est probablement mal détecté.")
    reproj = float(np.mean([p["erreur_px"] for p in pts]))
    ignores = len(ids_cible) - nb_marqueurs
    if ignores > 0:
        notes.append(f"{ignores} marqueur(s) de la cible sur {len(ids_cible)} non exploités.")

    transformation = np.eye(4)
    transformation[:3, :3] = s * R
    transformation[:3, 3] = t
    return {
        "version": 1,
        "methode": "cible_aruco",
        "mm_par_unite": float(s),
        "transformation_4x4": transformation.tolist(),
        "repere": "mm, z vers le haut, plan z=0 = plan de la cible",
        "fiable": len(avertissements) == 0,
        "avertissements": avertissements,
        "notes": notes,
        "qualite": {
            "nb_marqueurs": nb_marqueurs,
            "nb_coins": len(pts),
            "nb_images_avec_marqueurs": nb_images_avec,
            "residu_rms_mm": rms,
            "residu_relatif_pct": rms_relatif,
            "echelle_par_distances": s_distances,
            "ecart_entre_methodes_pct": ecart_methodes,
            "incertitude_echelle_pct": incertitude_pct,
            "reprojection_moyenne_px": reproj,
            "diagnostic_triangulation": diag,
        },
        "cible": {
            "taille_marqueur_mm": mise_en_page["taille_marqueur_mm"],
            "dictionnaire": mise_en_page["dictionnaire"],
            "marqueurs_utilises": gardes,
        },
    }


# ----------------------------------------------------------------------
# Solutions de repli sans cible
# ----------------------------------------------------------------------
def echelle_depuis_facteur(mm_par_unite, methode="facteur_manuel"):
    """Échelle simple (pas de repère métrique : pas de plan de plateau)."""
    if not (isinstance(mm_par_unite, (int, float)) and math.isfinite(mm_par_unite) and mm_par_unite > 0):
        raise ErreurEchelle(f"Facteur d'échelle invalide : {mm_par_unite!r} (nombre > 0 attendu).")
    return {
        "version": 1,
        "methode": methode,
        "mm_par_unite": float(mm_par_unite),
        "transformation_4x4": None,
        "repere": "mm, origine et axes de COLMAP conservés (pas de plan de plateau connu)",
        "fiable": True,
        "avertissements": [],
        "notes": ["Échelle fournie à la main : sa précision dépend de la mesure de départ."],
        "qualite": {},
        "cible": None,
    }


def echelle_depuis_distance(distance_unites, distance_mm):
    """Échelle d'après une longueur connue entre deux points du modèle."""
    if not distance_unites > 0 or not distance_mm > 0:
        raise ErreurEchelle("Les deux distances doivent être strictement positives.")
    return echelle_depuis_facteur(distance_mm / distance_unites, methode="distance_connue")


# ----------------------------------------------------------------------
# Lecture / écriture de echelle.json
# ----------------------------------------------------------------------
def empreinte_reconstruction(dossier_sparse):
    """Empreinte du contenu d'une reconstruction COLMAP (poses des caméras), ou None si illisible.

    Elle est enregistrée dans echelle.json : si la reconstruction est refaite ensuite, l'échelle
    calculée pour l'ancienne ne vaut plus rien, et mesure.py peut le détecter.
    """
    h, trouve = hashlib.sha256(), False
    for nom in ("cameras.bin", "images.bin", "cameras.txt", "images.txt"):
        chemin = os.path.join(dossier_sparse, nom)
        if os.path.isfile(chemin):
            with open(chemin, "rb") as f:
                h.update(nom.encode() + b"\0" + f.read())
            trouve = True
    return h.hexdigest() if trouve else None


def chemin_echelle(dossier_travail):
    return os.path.join(dossier_travail, "echelle.json")


def ecrire_echelle(dossier_travail, resultat):
    os.makedirs(dossier_travail, exist_ok=True)
    chemin = chemin_echelle(dossier_travail)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(resultat, f, indent=2, ensure_ascii=False)
    return chemin


def charger_echelle(chemin):
    """Lit echelle.json et refuse tout fichier incohérent (un facteur faux ne se voit pas ensuite)."""
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            e = json.load(f)
    except FileNotFoundError:
        raise ErreurEchelle(
            f"{chemin} introuvable. Lancer d'abord la mise à l'échelle : python echelle.py")
    except json.JSONDecodeError as ex:
        raise ErreurEchelle(f"{chemin} illisible ({ex})")

    s = e.get("mm_par_unite")
    if not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0):
        raise ErreurEchelle(f"{chemin} : « mm_par_unite » invalide ({s!r}).")
    T = e.get("transformation_4x4")
    if T is not None:
        T = np.array(T, dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ErreurEchelle(f"{chemin} : transformation_4x4 doit être une matrice 4x4 finie.")
        A = T[:3, :3] / s
        # Une similitude = échelle × rotation : A doit être orthonormale et directe
        if not np.allclose(A @ A.T, np.eye(3), atol=1e-6) or np.linalg.det(A) < 0:
            raise ErreurEchelle(
                f"{chemin} : la transformation n'est pas une similitude directe (miroir ou déformation).")
        e["transformation_4x4"] = T
    return e


def resumer(resultat):
    """Texte lisible d'un résultat d'échelle."""
    q = resultat.get("qualite", {})
    lignes = [f"Échelle : {resultat['mm_par_unite']:.6g} mm par unité COLMAP "
              f"(méthode : {resultat['methode']})"]
    if q:
        lignes.append(f"  Marqueurs utilisés : {q['nb_marqueurs']} ({q['nb_coins']} coins, "
                      f"vus dans {q['nb_images_avec_marqueurs']} photos)")
        lignes.append(f"  Résidu de superposition : {q['residu_rms_mm']:.3f} mm "
                      f"({q['residu_relatif_pct']:.2f} % de la cible)")
        lignes.append(f"  Écart entre les deux estimations d'échelle : {q['ecart_entre_methodes_pct']:.2f} %")
        if q.get("incertitude_echelle_pct") is not None:
            lignes.append(f"  Incertitude d'échelle (1 écart-type, jackknife) : "
                          f"{q['incertitude_echelle_pct']:.3f} %")
    lignes.append("  Fiabilité : " + ("OK" if resultat["fiable"] else "À VÉRIFIER"))
    for a in resultat["avertissements"]:
        lignes.append(f"  ! {a}")
    for n in resultat.get("notes", []):
        lignes.append(f"  - {n}")
    return "\n".join(lignes)


# ----------------------------------------------------------------------
# Point d'entrée utilisé par l'interface et la ligne de commande
# ----------------------------------------------------------------------
def lancer_echelle(dossier_travail="colmap_workspace", chemin_mise_en_page="cible_echelle.json",
                   taille_marqueur_mm=None, dossier_images=None, debug=False):
    """Détecte la cible, calcule l'échelle, écrit echelle.json et renvoie le résultat."""
    import pycolmap

    chemin_sparse = os.path.join(dossier_travail, "sparse_final")
    if not os.path.isdir(chemin_sparse):
        raise ErreurEchelle(
            f"{chemin_sparse} introuvable. Lancer d'abord l'étape 1 (main.py).")
    dossier_images = dossier_images or os.path.join(dossier_travail, "images_redimensionnees")
    if not os.path.isdir(dossier_images):
        raise ErreurEchelle(f"Dossier d'images introuvable : {dossier_images}")

    mise_en_page = charger_mise_en_page(chemin_mise_en_page)
    if taille_marqueur_mm is not None:
        # Correction d'impression mesurée au pied à coulisse
        mise_en_page = mise_en_page_a_echelle(mise_en_page, taille_marqueur_mm)

    reconstruction = pycolmap.Reconstruction(chemin_sparse)
    vues = vues_depuis_reconstruction(reconstruction)
    if len(vues) < 2:
        raise ErreurEchelle("Moins de 2 caméras positionnées : impossible de trianguler.")

    ids = {m["id"] for m in mise_en_page["marqueurs"]}
    dossier_debug = os.path.join(dossier_travail, "echelle_debug") if debug else None
    lues = detecter_dans_vues(vues, dossier_images, mise_en_page["dictionnaire"], ids, dossier_debug)
    if lues == 0:
        raise ErreurEchelle(f"Aucune image lisible dans {dossier_images}.")
    print(f"{lues} images analysées, marqueurs trouvés dans "
          f"{sum(1 for v in vues if v.marqueurs)} d'entre elles.")

    resultat = calculer_echelle_cible(vues, mise_en_page)
    resultat["reconstruction"] = {"empreinte": empreinte_reconstruction(chemin_sparse)}
    chemin = ecrire_echelle(dossier_travail, resultat)
    print(resumer(resultat))
    print(f"Écrit : {chemin}")
    return resultat


def main(argv=None):
    p = argparse.ArgumentParser(description="Mise à l'échelle métrique du modèle COLMAP.")
    p.add_argument("--travail", default="colmap_workspace")
    p.add_argument("--mise-en-page", default="cible_echelle.json")
    p.add_argument("--taille-marqueur-mm", type=float, default=None,
                   help="côté d'un marqueur mesuré sur la feuille imprimée (corrige l'impression)")
    p.add_argument("--facteur", type=float, default=None, help="échelle connue, en mm par unité COLMAP")
    p.add_argument("--distance-unites", type=float, default=None)
    p.add_argument("--distance-mm", type=float, default=None)
    p.add_argument("--debug", action="store_true", help="enregistre les photos avec marqueurs détectés")
    args = p.parse_args(argv)
    try:
        if args.facteur is not None:
            resultat = echelle_depuis_facteur(args.facteur)
            print(resumer(resultat))
            print("Écrit :", ecrire_echelle(args.travail, resultat))
        elif args.distance_unites is not None or args.distance_mm is not None:
            if args.distance_unites is None or args.distance_mm is None:
                raise ErreurEchelle("--distance-unites et --distance-mm vont ensemble.")
            resultat = echelle_depuis_distance(args.distance_unites, args.distance_mm)
            print(resumer(resultat))
            print("Écrit :", ecrire_echelle(args.travail, resultat))
        else:
            lancer_echelle(args.travail, args.mise_en_page, args.taille_marqueur_mm, debug=args.debug)
    except (ErreurEchelle, FileNotFoundError, ValueError) as e:
        print(f"ERREUR : {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
