"""
Banc d'essai synthétique : on fabrique des photos et des caméras dont la
vérité est connue, pour vérifier que les calculs la retrouvent.

Convention : repère « mm » = repère de la cible (z vers le haut). Le repère
« COLMAP » de la reconstruction est un repère arbitraire, relié au repère mm
par une similitude connue (échelle S_VRAIE, rotation et translation au choix) ;
c'est ce que echelle.py doit retrouver.
"""
import numpy as np

from echelle import Vue
from generer_cible_echelle import cases_du_marqueur, coins_marqueur_mm

LARGEUR, HAUTEUR, FOCALE = 1600, 1066, 2600.0


def rotation(axe, angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axe]


def camera_regarde(centre, cible, haut=(0.0, 0.0, 1.0)):
    """(R, t) monde -> caméra, convention OpenCV/COLMAP : x droite, y bas, z devant."""
    centre, cible = np.asarray(centre, float), np.asarray(cible, float)
    z = cible - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, haut)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ centre


def positions_autour(nb=12, rayon=420.0, elevation_deg=38.0, jitter=0.0, graine=0):
    """Centres de caméras répartis sur un cercle autour de l'origine."""
    rng = np.random.default_rng(graine)
    e = np.radians(elevation_deg)
    centres = []
    for i in range(nb):
        a = 2 * np.pi * i / nb + rng.normal(0, jitter)
        centres.append([rayon * np.cos(e) * np.cos(a), rayon * np.cos(e) * np.sin(a), rayon * np.sin(e)])
    return centres


class SimilitudeVraie:
    """Passage mm -> COLMAP : x_colmap = (1/S) * Rc @ x_mm + tc (S en mm par unité COLMAP)."""

    def __init__(self, s=137.0, axe="x", angle=63.0, translation=(2.5, -1.0, 4.0)):
        self.s = s
        self.R = rotation(axe, angle) @ rotation("z", 31.0)
        self.t = np.array(translation, float)

    def colmap_depuis_mm(self, x):
        return (self.R @ np.asarray(x, float).T).T / self.s + self.t

    def cam_from_world_colmap(self, R_mm, t_mm):
        """Pose caméra exprimée dans le repère COLMAP à partir de la pose dans le repère mm."""
        # x_cam = R_mm x_mm + t_mm  avec  x_mm = s * Rc^T (x_colmap - tc)
        R = self.s * R_mm @ self.R.T
        t = t_mm - self.s * R_mm @ self.R.T @ self.t
        # L'échelle est dans la rotation : on la renvoie à la profondeur (x_cam est en mm,
        # on le veut en unités COLMAP, donc on divise tout par s)
        return np.hstack([R / self.s, (t / self.s).reshape(3, 1)])


def modele_distorsion_radiale(k1, f=FOCALE):
    """Fonctions (distordre, normaliser) pour un modèle SIMPLE_RADIAL, comme COLMAP."""
    cx, cy = LARGEUR / 2, HAUTEUR / 2

    def distordre(xy_norm):
        r2 = (xy_norm ** 2).sum(axis=1, keepdims=True)
        return xy_norm * (1 + k1 * r2)

    def normaliser(pixels):
        d = (np.asarray(pixels, float).reshape(-1, 2) - [cx, cy]) / f
        u = d.copy()
        for _ in range(30):  # inversion par point fixe
            r2 = (u ** 2).sum(axis=1, keepdims=True)
            u = d / (1 + k1 * r2)
        return u

    return distordre, normaliser


def vues_synthetiques(mise_en_page, vrai, centres, bruit_px=0.0, k1=0.0, graine=1,
                      ids_vus=None, cibles_regardees=(0.0, 0.0, 0.0)):
    """Vues dont les marqueurs sont obtenus en PROJETANT les coins réels (sans image).

    Permet de tester la partie calcul seule, avec bruit et distorsion maîtrisés.
    """
    rng = np.random.default_rng(graine)
    distordre, normaliser = modele_distorsion_radiale(k1)
    vues = []
    for n, centre in enumerate(centres):
        R, t = camera_regarde(centre, cibles_regardees)
        P_colmap = vrai.cam_from_world_colmap(R, t)
        vue = Vue(f"synth_{n:02d}.jpg", P_colmap, normaliser, FOCALE)
        for m in mise_en_page["marqueurs"]:
            if ids_vus is not None and m["id"] not in ids_vus:
                continue
            coins = coins_marqueur_mm(mise_en_page, m["id"])
            cam = (R @ coins.T).T + t
            if np.any(cam[:, 2] <= 0):
                continue
            norm = distordre(cam[:, :2] / cam[:, 2:3])
            px = norm * FOCALE + [LARGEUR / 2, HAUTEUR / 2]
            px = px + rng.normal(0, bruit_px, px.shape)
            if px[:, 0].min() < 0 or px[:, 0].max() > LARGEUR or px[:, 1].min() < 0 or px[:, 1].max() > HAUTEUR:
                continue
            vue.marqueurs[m["id"]] = px
        vues.append(vue)
    return vues


def rendre_image(mise_en_page, R, t, ppm=8.0, flou=0.7, bruit=2.0, graine=0):
    """Photo synthétique de la feuille vue par la caméra (R, t) dans le repère mm."""
    import cv2

    cote = 260.0  # mm : la feuille entière
    nb = int(cote * ppm)
    canevas = np.full((nb, nb), 255, np.uint8)
    n_cases = cases_du_marqueur(mise_en_page["dictionnaire"], mise_en_page["marqueurs"][0]["id"]).shape[0]
    taille = mise_en_page["taille_marqueur_mm"]
    for m in mise_en_page["marqueurs"]:
        cases = cases_du_marqueur(mise_en_page["dictionnaire"], m["id"])
        px = int(round(taille * ppm / n_cases)) * n_cases
        marq = np.kron(1 - cases, np.ones((px // n_cases, px // n_cases))).astype(np.uint8) * 255
        # coin haut-gauche du marqueur dans le canevas (y du canevas vers le bas)
        cx, cy = m["centre_mm"]
        u0 = int(round((cx - taille / 2 + cote / 2) * ppm))
        v0 = int(round((cote / 2 - cy - taille / 2) * ppm))
        canevas[v0:v0 + px, u0:u0 + px] = marq

    K = np.array([[FOCALE, 0, LARGEUR / 2], [0, FOCALE, HAUTEUR / 2], [0, 0, 1]])
    # pixel du canevas (centre = entier + 0,5) -> monde (x, y, 1)
    A = np.array([[1 / ppm, 0, 0.5 / ppm - cote / 2], [0, -1 / ppm, cote / 2 - 0.5 / ppm], [0, 0, 1]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A      # convention COLMAP (centre pixel = +0,5)
    decalage = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])  # -> convention OpenCV
    image = cv2.warpPerspective(canevas, decalage @ H, (LARGEUR, HAUTEUR), flags=cv2.INTER_LINEAR,
                                borderValue=200)
    if flou > 0:
        image = cv2.GaussianBlur(image, (0, 0), flou)
    rng = np.random.default_rng(graine)
    image = np.clip(image.astype(float) + rng.normal(0, bruit, image.shape), 0, 255).astype(np.uint8)
    return image


def echelle_vraie(vrai):
    """Contenu d'un echelle.json exact pour une SimilitudeVraie : COLMAP -> mm, z vers le haut."""
    A = vrai.s * vrai.R.T
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = -A @ vrai.t
    return {"version": 1, "methode": "cible_aruco", "mm_par_unite": float(vrai.s),
            "transformation_4x4": T.tolist(), "fiable": True, "avertissements": [], "notes": [],
            "qualite": {"incertitude_echelle_pct": 0.05}}
