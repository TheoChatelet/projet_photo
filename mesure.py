"""
Étape 4b du pipeline : mesures en millimètres (volume, cotes) et export du modèle à l'échelle.

Pourquoi : maillage.py produit un STL dans l'unité de COLMAP, avec le fond et le
dessous de l'objet comme Poisson les a "inventés". Pour mesurer, il faut :
  1. passer en millimètres (echelle.py) ;
  2. couper l'objet au plan du plateau : le dessous n'est pas visible sur les
     photos, Poisson le referme avec une surface arbitraire qui fausserait le volume ;
  3. refermer proprement la coupe (le volume d'un maillage n'a de sens que fermé) ;
  4. dire si le résultat est fiable ou non, plutôt que de sortir un chiffre faux.

Entrées : colmap_workspace/modele_pour_impression.stl (maillage.py)
          colmap_workspace/echelle.json (echelle.py)
          colmap_workspace/openmvs/scene_dense.ply (facultatif, sert à repérer les
          surfaces inventées loin des vrais points)
Sorties : colmap_workspace/mesures/<nom>/modele_mm.stl, rapport.json, rapport.txt
          colmap_workspace/mesures/historique.csv (une ligne par mesure, s'ouvre
          dans Excel)

Le volume est calculé par la formule de la divergence (somme des volumes signés
des tétraèdres origine-triangle), exacte pour une surface fermée. La coupe au
plateau est refermée virtuellement, et la tranche entre la coupe et le plateau
(z_coupe, 1 mm par défaut) est ajoutée comme un prisme : le bas d'un objet
posé monte à la verticale sur ce premier millimètre.

Lancement seul : python mesure.py [--travail colmap_workspace] [--nom essai1]
                    [--z-coupe-mm 1.0] [--formats stl obj ply]
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

from echelle import ErreurEchelle, charger_echelle, empreinte_reconstruction

# Tolérance pour dire qu'un sommet est dans le plan de coupe (mm)
EPS_PLAN = 1e-9
# Au-delà de cette part de surface « inventée » pour boucher des trous (hors coupe), le volume est douteux
PART_COMBLEE_MAX_PCT = 1.0
# Au-delà de cette part de surface écartée (fragments flottants), on prévient : l'objet est peut-être en deux morceaux
PART_ECARTEE_MAX_PCT = 5.0
# Sommets plus loin du nuage dense que ce multiple de son espacement : surface probablement inventée
FACTEUR_DISTANCE_NUAGE = 5.0
PART_LOIN_NUAGE_MAX_PCT = 2.0


class ErreurMesure(Exception):
    """Mesure impossible. Le message dit quoi corriger."""


# ----------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------
def charger_maillage(chemin):
    """Lit un maillage STL / PLY / OBJ. Renvoie (sommets, faces, infos de nettoyage).

    Les sommets identiques sont fusionnés : un STL stocke chaque triangle séparément
    (3 sommets par triangle, sans partage), il faut recoller les triangles voisins
    avant tout calcul de topologie.
    """
    import trimesh

    if not os.path.isfile(chemin):
        raise ErreurMesure(f"Maillage introuvable : {chemin}")
    try:
        objet = trimesh.load(chemin, process=False)
    except Exception as e:  # trimesh lève des types variés selon le format
        raise ErreurMesure(f"Impossible de lire {chemin} : {e}")
    if isinstance(objet, trimesh.Scene):
        geometries = [g for g in objet.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ErreurMesure(f"{chemin} ne contient aucun maillage.")
        objet = trimesh.util.concatenate(geometries)
    if not isinstance(objet, trimesh.Trimesh) or len(objet.faces) == 0:
        raise ErreurMesure(
            f"{chemin} n'est pas un maillage (nuage de points ou fichier vide). "
            f"Lancer d'abord l'étape 3 (maillage.py).")
    sommets_avant = len(objet.vertices)
    objet.merge_vertices()
    return _nettoyer(np.asarray(objet.vertices, np.float64), np.asarray(objet.faces, np.int64),
                     sommets_avant)


def _nettoyer(V, F, sommets_avant=None):
    """Retire faces non finies ou dégénérées et sommets inutilisés."""
    faces_avant = len(F)
    sommets_avant = len(V) if sommets_avant is None else sommets_avant
    if not np.all(np.isfinite(V)):
        raise ErreurMesure("Le maillage contient des coordonnées non finies (NaN ou infini).")
    garder = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    F = F[garder]
    aires = _aires_faces(V, F)
    F = F[aires > 1e-18]
    if len(F) == 0:
        raise ErreurMesure("Le maillage ne contient aucun triangle exploitable.")
    utilises, inverse = np.unique(F, return_inverse=True)
    V, F = V[utilises], inverse.reshape(-1, 3)
    return V, F, {
        "sommets_fichier": int(sommets_avant),
        "faces_fichier": int(faces_avant),
        "faces_degenerees_supprimees": int(faces_avant - len(F)),
    }


# ----------------------------------------------------------------------
# Géométrie de base
# ----------------------------------------------------------------------
def _aires_faces(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def _volume_signe(V, F, origine=None):
    """Somme des volumes signés des tétraèdres (origine, triangle) / 6."""
    o = V.mean(axis=0) if origine is None else origine
    a, b, c = V[F[:, 0]] - o, V[F[:, 1]] - o, V[F[:, 2]] - o
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def appliquer_echelle(V, echelle):
    """Passe les sommets en millimètres (et dans le repère du plateau si connu)."""
    T = echelle.get("transformation_4x4")
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        return V @ T[:3, :3].T + T[:3, 3]
    return V * float(echelle["mm_par_unite"])


# ----------------------------------------------------------------------
# Découpe au plan z = z0 (on garde z > z0)
# ----------------------------------------------------------------------
def decouper_au_plan(V, F, z0):
    """Garde la partie du maillage au-dessus du plan z = z0.

    Un triangle entièrement au-dessus est gardé, entièrement dessous il disparaît,
    et un triangle coupé par le plan est recoupé (Sutherland-Hodgman). Les points
    d'intersection sont partagés entre triangles voisins (même arête = même point) :
    la coupe est donc continue, sans trou ni doublon.
    """
    d = V[:, 2] - z0
    dessus, dessous = d[F] > EPS_PLAN, d[F] < -EPS_PLAN
    nb_dessus, nb_dessous = dessus.sum(axis=1), dessous.sum(axis=1)

    entieres = (nb_dessous == 0) & (nb_dessus > 0)        # au-dessus (peut toucher le plan)
    coupees = np.flatnonzero((nb_dessous > 0) & (nb_dessus > 0))

    nouveaux, aretes_coupees, nouvelles_faces = [], {}, []
    nb_V = len(V)

    def point_sur_plan(i, j):
        """Intersection de l'arête (i, j) avec le plan, partagée entre faces."""
        cle = (min(i, j), max(i, j))
        if cle not in aretes_coupees:
            a, b = (i, j) if d[i] > 0 else (j, i)    # a au-dessus, b dessous ou sur le plan
            t = d[a] / (d[a] - d[b])
            p = V[a] + t * (V[b] - V[a])
            p[2] = z0
            aretes_coupees[cle] = nb_V + len(nouveaux)
            nouveaux.append(p)
        return aretes_coupees[cle]

    for f in coupees:
        sommets = F[f]
        polygone = []
        for k in range(3):
            i, j = sommets[k], sommets[(k + 1) % 3]
            i_dedans, j_dedans = d[i] > EPS_PLAN, d[j] > EPS_PLAN
            i_dehors, j_dehors = d[i] < -EPS_PLAN, d[j] < -EPS_PLAN
            if i_dedans:
                polygone.append(i)
            elif not i_dehors:
                polygone.append(i)                   # déjà sur le plan : il reste tel quel
            if (i_dedans and j_dehors) or (i_dehors and j_dedans):
                polygone.append(point_sur_plan(i, j))
        # Un triangle recoupé donne un triangle ou un quadrilatère
        for k in range(1, len(polygone) - 1):
            nouvelles_faces.append([polygone[0], polygone[k], polygone[k + 1]])

    V2 = np.vstack([V] + ([np.array(nouveaux)] if nouveaux else []))
    F2 = np.vstack([F[entieres]] + ([np.array(nouvelles_faces, np.int64)] if nouvelles_faces else []))
    if len(F2) == 0:
        return V2[:0], F2.reshape(0, 3)
    return _nettoyer(V2, F2)[:2]


# ----------------------------------------------------------------------
# Topologie : arêtes, composantes, boucles
# ----------------------------------------------------------------------
def analyser_aretes(F, nV):
    """Bilan des arêtes : renvoie un dict.

    aretes_nettes : tableau (k, 2) des arêtes orientées dont l'orientation n'est pas équilibrée
                    (une face la parcourt a->b sans que l'autre la parcoure b->a) : c'est le
                    bord réel de la surface, avec multiplicité.
    libres        : arêtes portées par une seule face
    non_manifold  : arêtes portées par plus de 2 faces
    """
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    a, b = e[:, 0], e[:, 1]
    cle = np.minimum(a, b).astype(np.int64) * nV + np.maximum(a, b)
    signe = np.where(a < b, 1, -1)
    ordre = np.argsort(cle, kind="stable")
    cle, signe = cle[ordre], signe[ordre]
    debuts = np.r_[0, np.flatnonzero(np.diff(cle)) + 1]
    bilan = np.add.reduceat(signe, debuts)
    nb_faces = np.diff(np.r_[debuts, len(cle)])
    lo, hi = cle[debuts] // nV, cle[debuts] % nV

    positives, negatives = bilan > 0, bilan < 0
    nettes = np.vstack([
        np.repeat(np.column_stack([lo[positives], hi[positives]]), bilan[positives], axis=0),
        np.repeat(np.column_stack([hi[negatives], lo[negatives]]), -bilan[negatives], axis=0),
    ]) if (positives.any() or negatives.any()) else np.zeros((0, 2), np.int64)
    return {"aretes_nettes": nettes.astype(np.int64), "libres": int((nb_faces == 1).sum()),
            "non_manifold": int((nb_faces > 2).sum()), "nb_aretes": int(len(lo))}


def composantes(F, nV):
    """Étiquette de composante connexe de chaque face (connexité par sommets partagés)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ligne = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    colonne = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    graphe = coo_matrix((np.ones(len(ligne)), (ligne, colonne)), shape=(nV, nV))
    _, etiquettes = connected_components(graphe, directed=False)
    return etiquettes[F[:, 0]]


def chainer_boucles(aretes):
    """Assemble des arêtes orientées en boucles fermées. Renvoie (boucles, nb_chaines_ouvertes).

    Une chaîne ouverte (impasse) signifie que le bord n'est pas fermable : volume non fiable.
    """
    sortant = defaultdict(list)
    for u, v in aretes:
        sortant[int(u)].append(int(v))
    boucles, ouvertes = [], 0
    for depart in list(sortant):
        while sortant[depart]:
            chemin, courant = [depart], depart
            while True:
                if not sortant[courant]:
                    ouvertes += 1
                    break
                suivant = sortant[courant].pop()
                if suivant == depart:
                    boucles.append(chemin)
                    break
                chemin.append(suivant)
                courant = suivant
    return boucles, ouvertes


def _aire_signee_xy(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ----------------------------------------------------------------------
# Rectangle d'aire minimale (cotes de l'empreinte de l'objet vu de dessus)
# ----------------------------------------------------------------------
def rectangle_min(points_xy):
    """Plus petit rectangle (tourné) contenant les points. Renvoie (longueur, largeur, angle_deg)."""
    from scipy.spatial import ConvexHull

    pts = np.asarray(points_xy, np.float64)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:  # points alignés : pas d'enveloppe 2D
        ecart = pts.max(axis=0) - pts.min(axis=0)
        return float(max(ecart)), float(min(ecart)), 0.0
    meilleur = None
    for i in range(len(hull)):
        arete = hull[(i + 1) % len(hull)] - hull[i]
        norme = np.linalg.norm(arete)
        if norme < 1e-12:
            continue
        u = arete / norme
        v = np.array([-u[1], u[0]])
        a, b = hull @ u, hull @ v
        largeur, hauteur = a.max() - a.min(), b.max() - b.min()
        if meilleur is None or largeur * hauteur < meilleur[0]:
            meilleur = (largeur * hauteur, max(largeur, hauteur), min(largeur, hauteur),
                        math.degrees(math.atan2(u[1], u[0])))
    return float(meilleur[1]), float(meilleur[2]), float(meilleur[3])


# ----------------------------------------------------------------------
# Fermeture virtuelle (pour le volume)
# ----------------------------------------------------------------------
def volume_ferme(V, F, aretes_coupe, boucles_trous, z0):
    """Volume du maillage refermé, sans construire la fermeture : on ajoute des cônes.

    aretes_coupe  : arêtes (u, v) de la coupe, orientées comme la FERMETURE les parcourt
                    (donc à l'envers de celles du maillage).
    boucles_trous : listes de sommets des autres bords ouverts, même orientation.

    Un bord plan (la coupe) se ferme par un cône dont le sommet est n'importe quel point du
    plan : le volume obtenu est exact, quelle que soit la forme de la coupe (trous
    compris). Pour un trou non plan, le sommet est le centre du trou, l'approximation la
    plus neutre. Renvoie (volume, surface_comblée) : la surface des cônes de trous est
    la part « inventée », qui sert à juger de la fiabilité.
    """
    c0 = V.mean(axis=0)    # on travaille autour du centre : évite de perdre des chiffres significatifs
    volume = _volume_signe(V, F, c0)
    if len(aretes_coupe):
        sommet = np.array([0.0, 0.0, z0 - c0[2]])
        a, b = V[aretes_coupe[:, 0]] - c0, V[aretes_coupe[:, 1]] - c0
        volume += float(np.einsum("j,ij->i", sommet, np.cross(a, b)).sum() / 6.0)
    surface_comblee = 0.0
    for boucle in boucles_trous:
        pts = V[boucle] - c0
        suivants = np.roll(pts, -1, axis=0)
        centre = pts.mean(axis=0)
        volume += float(np.einsum("j,ij->i", centre, np.cross(pts, suivants)).sum() / 6.0)
        surface_comblee += float(0.5 * np.linalg.norm(np.cross(pts - centre, suivants - centre), axis=1).sum())
    return volume, surface_comblee


# ----------------------------------------------------------------------
# Fermeture réelle (pour l'export du modèle)
# ----------------------------------------------------------------------
def _dans_polygone(point, polygone):
    """Test du rayon : le point est-il dans le polygone (2D) ?"""
    x, y = point
    x1, y1 = polygone[:, 0], polygone[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    traverse = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-300, y2 - y1) + x1)
    return bool(traverse.sum() % 2)


def _trianguler_section(boucles_xy):
    """Triangule la région plane délimitée par des boucles (contours et trous).

    Renvoie un tableau (t, 3) d'indices dans la concaténation des boucles, ou None si
    mapbox_earcut n'est pas installé.
    """
    try:
        import mapbox_earcut as earcut
    except ImportError:
        return None
    aires = np.array([_aire_signee_xy(b) for b in boucles_xy])
    sens_contour = np.sign(aires[int(np.argmax(np.abs(aires)))])
    contours = [i for i, a in enumerate(aires) if np.sign(a) == sens_contour]
    trous = [i for i in range(len(boucles_xy)) if i not in contours]

    decalages = np.r_[0, np.cumsum([len(b) for b in boucles_xy])]
    triangles = []
    trous_par_contour = defaultdict(list)
    for t in trous:
        # Le trou appartient au plus petit contour qui le contient
        candidats = [c for c in contours if _dans_polygone(boucles_xy[t][0], boucles_xy[c])]
        if candidats:
            trous_par_contour[min(candidats, key=lambda c: abs(aires[c]))].append(t)
        else:
            contours.append(t)       # orientation inattendue : on le traite comme un contour
    for c in contours:
        anneaux = [c] + trous_par_contour[c]
        pts = np.vstack([boucles_xy[i] for i in anneaux])
        fins = np.cumsum([len(boucles_xy[i]) for i in anneaux]).astype(np.uint32)
        idx = np.asarray(earcut.triangulate_float64(pts, fins)).reshape(-1, 3)
        correspondance = np.concatenate([np.arange(decalages[i], decalages[i + 1]) for i in anneaux])
        triangles.append(correspondance[idx])
    return np.vstack(triangles) if triangles else np.zeros((0, 3), np.int64)


def construire_maillage_ferme(V, F, boucles_plan, boucles_trous, z0, prolonger):
    """Maillage fermé pour l'export : coupe prolongée jusqu'au plateau (z = 0) puis bouchée.

    Renvoie (sommets, faces, notes). Les trous hors coupe sont bouchés par un cône
    (voir volume_ferme). Sans mapbox_earcut, seule une coupe sans trou est refermée.
    """
    notes = []
    morceaux_V, morceaux_F, n = [V], [F], len(V)

    # Trous hors coupe : un cône de triangles autour du centre du trou
    for boucle in boucles_trous:
        centre = V[boucle].mean(axis=0)
        morceaux_V.append(centre.reshape(1, 3))
        for i in range(len(boucle)):
            morceaux_F.append(np.array([[n, boucle[i], boucle[(i + 1) % len(boucle)]]]))
        n += 1

    if boucles_plan:
        descendre = prolonger and z0 is not None and z0 > 1e-9
        bords = []
        for boucle in boucles_plan:
            if descendre:
                bas = V[boucle].copy()
                bas[:, 2] = 0.0
                indices_bas = np.arange(n, n + len(boucle))
                morceaux_V.append(bas)
                n += len(boucle)
                L = len(boucle)
                for i in range(L):
                    ci, cj = boucle[i], boucle[(i + 1) % L]
                    bi, bj = indices_bas[i], indices_bas[(i + 1) % L]
                    # Jupe verticale : les deux triangles contiennent l'arête du maillage à l'envers
                    morceaux_F.append(np.array([[ci, cj, bj], [ci, bj, bi]]))
                bords.append(indices_bas)
            else:
                bords.append(np.asarray(boucle))
        sommets_tous = np.vstack(morceaux_V)
        boucles_xy = [sommets_tous[b][:, :2] for b in bords]
        tri = _trianguler_section(boucles_xy)
        if tri is None and len(bords) == 1:
            # Repli sans earcut : éventail depuis le centre (correct si la coupe est convexe)
            b = bords[0]
            centre = sommets_tous[b].mean(axis=0)
            morceaux_V.append(centre.reshape(1, 3))
            tri_globaux = np.array([[n, b[i], b[(i + 1) % len(b)]] for i in range(len(b))])
            n += 1
            notes.append("mapbox_earcut absent : base de l'export fermée en éventail (suppose une coupe convexe).")
        elif tri is None:
            notes.append("mapbox_earcut absent et coupe en plusieurs morceaux : base de l'export non fermée "
                         "(pip install mapbox-earcut). Le volume mesuré n'est pas concerné.")
            tri_globaux = np.zeros((0, 3), np.int64)
        else:
            tous_les_indices = np.concatenate(bords)
            tri_globaux = tous_les_indices[tri]
        if len(tri_globaux):
            sommets_tous = np.vstack(morceaux_V)
            a, b, c = (sommets_tous[tri_globaux[:, k]] for k in range(3))
            # Face du dessous : normale vers -z (l'objet est au-dessus). On retourne TOUS les
            # triangles ensemble selon l'aire totale signée : earcut les sort dans un même
            # sens, et décider triangle par triangle serait faux pour les triangles d'aire
            # nulle qu'il crée pour relier un trou au contour.
            if np.cross(b - a, c - a)[:, 2].sum() > 0:
                tri_globaux = tri_globaux[:, ::-1]
            morceaux_F.append(tri_globaux)
    return np.vstack(morceaux_V), np.vstack(morceaux_F), notes


def hauteur_robuste(V, F, facteur_rayon=3.0):
    """Hauteur maximale après moyenne locale des z (rayon = 3 arêtes « courtes »).

    La hauteur « brute » (point le plus haut) est biaisée vers le haut par le bruit de surface :
    sur un dessus plat de ~100 sommets, le plus haut est à environ +2,4 écarts-types du vrai plan.
    On moyenne donc le z de chaque sommet voisin du sommet avec tous les sommets situés dans un
    rayon de quelques arêtes (la distance, pas les liens du maillage : le résultat ne dépend pas
    de la forme des triangles). Le rayon suit la densité du maillage : sur un maillage fin le sommet
    d'un dôme est abaissé de r²/(4R) (R = rayon de courbure), négligeable devant la hauteur.
    """
    from scipy.spatial import cKDTree

    pas = max(1, len(F) // 100000)
    p = V[F[::pas]]
    aretes = np.concatenate([np.linalg.norm(p[:, 0] - p[:, 1], axis=1), np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
                             np.linalg.norm(p[:, 2] - p[:, 0], axis=1)])
    # 20e centile et pas médiane : un maillage qui mêle arêtes courtes et très longues (pièce de CAO,
    # maillage décimé) a une médiane trop grande, le rayon couvrirait tout l'objet. Le petit quantile ne
    # sur-lisse jamais ; sur un maillage de Poisson (arêtes uniformes) il donne la même chose que la médiane.
    rayon = facteur_rayon * float(np.percentile(aretes, 20))
    z = V[:, 2]
    candidats = np.flatnonzero(z > z.max() - 5.0 * rayon)       # seuls les sommets proches du haut comptent
    voisins = cKDTree(V).query_ball_point(V[candidats], rayon)
    return float(max(z[idx].mean() for idx in voisins))


def part_loin_du_nuage(V, nuage, facteur=FACTEUR_DISTANCE_NUAGE):
    """Part (en %) des sommets du maillage trop loin de tout point du nuage dense.

    Poisson invente parfois des surfaces là où il n'y a pas de points (bulles) : ces
    sommets s'en éloignent. Renvoie (part_en_pct, espacement_médian_du_nuage).
    """
    from scipy.spatial import cKDTree

    arbre = cKDTree(nuage)
    echantillon = nuage[:: max(1, len(nuage) // 20000)]
    espacement = float(np.median(arbre.query(echantillon, k=2)[0][:, 1]))
    distances = arbre.query(V, workers=-1)[0]
    return 100.0 * float((distances > facteur * espacement).mean()), espacement


# ----------------------------------------------------------------------
# Mesure d'un maillage déjà en millimètres
# ----------------------------------------------------------------------
def _classer_bords(V, F, z0):
    """Chaîne tout le bord du maillage en boucles, puis classe chaque boucle ENTIÈRE.

    Une boucle est « de coupe » si tous ses sommets sont dans le plan z0 (c'est la coupe au
    plateau, refermée par un plan), sinon c'est un trou. On chaîne d'abord et on classe
    ensuite : un trou qui descend jusqu'au plan de coupe donne une boucle mixte, qui doit
    rester d'un seul tenant.
    """
    ar = analyser_aretes(F, len(V))
    fermeture = ar["aretes_nettes"][:, ::-1]       # la fermeture parcourt chaque arête à l'envers
    boucles, ouvertes = chainer_boucles(fermeture)
    boucles_plan, boucles_trous = [], []
    for b in boucles:
        if z0 is not None and np.all(np.abs(V[b][:, 2] - z0) < 1e-6):
            boucles_plan.append(b)
        else:
            boucles_trous.append(b)
    aretes_plan = np.array([(b[i], b[(i + 1) % len(b)]) for b in boucles_plan for i in range(len(b))],
                           dtype=np.int64).reshape(-1, 2)
    return ar, aretes_plan, boucles_plan, boucles_trous, ouvertes


def mesurer_maillage(V, F, z_coupe_mm=1.0, repere_plateau=True, prolonger=True, garder_toutes=False,
                     nuage_mm=None):
    """Mesure un maillage en millimètres. Renvoie (mesures, qualite, avertissements, notes, (Ve, Fe)).

    repere_plateau : True si z = 0 est le plan du plateau et z vers le haut (venant de echelle.py).
    z_coupe_mm     : hauteur de coupe au-dessus du plateau (None = pas de coupe).
    prolonger      : ajoute la tranche plateau -> coupe (prisme) au volume et à l'export.
    nuage_mm       : points du nuage dense (mm) : sert seulement à repérer les surfaces inventées.
    """
    avertissements, notes = [], []
    V, F, nettoyage = _nettoyer(V, F)
    z0 = float(z_coupe_mm) if (repere_plateau and z_coupe_mm is not None) else None
    if z0 is not None and z0 < 0:
        raise ErreurMesure("z_coupe_mm doit être positif ou nul.")

    if z0 is not None:
        # Garde-fou : si le modèle est du mauvais côté du plan, la coupe le viderait
        part_dessus = float((V[:, 2] > z0).mean())
        if part_dessus < 0.05:
            raise ErreurMesure(
                f"Seulement {100 * part_dessus:.1f} % du modèle est au-dessus du plan du plateau "
                f"(z = {z0:g} mm). Le modèle est sous le plan ou le repère est inversé : "
                f"refaire la mise à l'échelle (echelle.py) avec une cible bien visible.")
        V, F = decouper_au_plan(V, F, z0)
        if len(F) == 0:
            raise ErreurMesure("Plus rien au-dessus du plan de coupe : réduire z_coupe_mm.")

    # Fragments flottants : on garde la plus grosse pièce
    etiquettes = composantes(F, len(V))
    nb_comp = int(etiquettes.max()) + 1
    aires = _aires_faces(V, F)
    part_ecartee = 0.0
    if nb_comp > 1 and not garder_toutes:
        surface_par_comp = np.bincount(etiquettes, weights=aires)
        principale = int(np.argmax(surface_par_comp))
        part_ecartee = 100.0 * (1.0 - surface_par_comp[principale] / surface_par_comp.sum())
        F = F[etiquettes == principale]
        V, F, _ = _nettoyer(V, F)
        if part_ecartee > PART_ECARTEE_MAX_PCT:
            avertissements.append(
                f"{nb_comp - 1} fragment(s) écarté(s) = {part_ecartee:.1f} % de la surface : l'objet est "
                f"peut-être en plusieurs morceaux (option --toutes-composantes).")
        else:
            notes.append(f"{nb_comp - 1} petit(s) fragment(s) flottant(s) écarté(s) ({part_ecartee:.2f} % de la surface).")

    part_loin, espacement_nuage = None, None
    if nuage_mm is not None and len(nuage_mm) > 10:
        part_loin, espacement_nuage = part_loin_du_nuage(V, nuage_mm)

    # Fermeture : si le volume sort négatif les normales sont à l'envers, on retourne le maillage
    ar, aretes_plan, boucles_plan, boucles_trous, ouvertes = _classer_bords(V, F, z0)
    volume_au_dessus, surface_comblee = volume_ferme(V, F, aretes_plan, boucles_trous, z0)
    inverse = False
    if volume_au_dessus < 0:
        F = F[:, ::-1].copy()
        inverse = True
        ar, aretes_plan, boucles_plan, boucles_trous, ouvertes = _classer_bords(V, F, z0)
        volume_au_dessus, surface_comblee = volume_ferme(V, F, aretes_plan, boucles_trous, z0)
        notes.append("Normales retournées (le maillage était orienté vers l'intérieur).")

    # Orientation incohérente (quelques faces retournées) : beaucoup d'arêtes déséquilibrées hors coupe
    reparee = False
    n_autres = sum(len(b) for b in boucles_trous)
    if n_autres > 0.001 * len(F):
        try:
            import trimesh
            m = trimesh.Trimesh(V.copy(), F.copy(), process=False)
            trimesh.repair.fix_winding(m)
            V2, F2 = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
            r2 = _classer_bords(V2, F2, z0)
            if sum(len(b) for b in r2[3]) < n_autres:
                V, F = V2, F2
                ar, aretes_plan, boucles_plan, boucles_trous, ouvertes = r2
                volume_au_dessus, surface_comblee = volume_ferme(V, F, aretes_plan, boucles_trous, z0)
                if volume_au_dessus < 0:
                    F = F[:, ::-1].copy()
                    ar, aretes_plan, boucles_plan, boucles_trous, ouvertes = _classer_bords(V, F, z0)
                    volume_au_dessus, surface_comblee = volume_ferme(V, F, aretes_plan, boucles_trous, z0)
                reparee = True
                notes.append("Orientation des faces réparée (faces retournées détectées).")
        except Exception:  # networkx absent ou maillage trop dégradé : on garde l'état d'origine
            avertissements.append("Orientation des faces incohérente et non réparable automatiquement.")

    # Mesures
    surface_laterale = float(_aires_faces(V, F).sum())
    aire_base, perimetre = 0.0, 0.0
    if len(boucles_plan):
        aire_base = abs(sum(_aire_signee_xy(V[b][:, :2]) for b in boucles_plan))
        perimetre = float(np.linalg.norm(V[aretes_plan[:, 0]] - V[aretes_plan[:, 1]], axis=1).sum())
    prisme = aire_base * z0 if (prolonger and z0) else 0.0
    volume = volume_au_dessus + prisme
    surface_totale = surface_laterale + aire_base + (perimetre * z0 if (prolonger and z0) else 0.0) + surface_comblee

    mini, maxi = V.min(axis=0), V.max(axis=0)
    mesures = {
        "volume_mm3": volume,
        "volume_ml": volume / 1000.0,
        "surface_mm2": surface_totale,
        "boite_axes_mm": (maxi - mini).tolist(),
    }
    if repere_plateau:
        longueur, largeur, angle = rectangle_min(V[:, :2])
        mesures.update({
            "hauteur_mm": float(maxi[2]),     # le plateau est à z = 0 ; point le plus haut (biaisé par le bruit)
            "hauteur_robuste_mm": hauteur_robuste(V, F),
            "longueur_mm": longueur,
            "largeur_mm": largeur,
            "angle_empreinte_deg": angle,
        })
        if aire_base > 0:
            mesures["aire_base_mm2"] = aire_base
            mesures["diametre_equivalent_base_mm"] = 2.0 * math.sqrt(aire_base / math.pi)
    else:
        # Sans plateau connu, pas de « hauteur » : on donne la boîte selon les axes principaux
        centres = V - V.mean(axis=0)
        _, _, axes = np.linalg.svd(centres[:: max(1, len(V) // 200000)], full_matrices=False)
        proj = centres @ axes.T
        mesures["boite_principale_mm"] = sorted((proj.max(axis=0) - proj.min(axis=0)).tolist(), reverse=True)

    qualite = {
        **nettoyage,
        "sommets": int(len(V)), "faces": int(len(F)),
        "composantes_trouvees": nb_comp, "part_surface_ecartee_pct": part_ecartee,
        "aretes_non_manifold": ar["non_manifold"],
        "boucles_de_coupe": len(boucles_plan), "boucles_de_trou": len(boucles_trous),
        "bords_ouverts_non_fermables": ouvertes,
        "surface_comblee_mm2": surface_comblee,
        "part_surface_comblee_pct": 100.0 * surface_comblee / surface_totale if surface_totale else 0.0,
        "normales_retournees": inverse, "orientation_reparee": reparee,
        "z_coupe_mm": z0, "prolongement_jusqu_au_plateau": bool(prolonger and z0),
        "volume_au_dessus_de_la_coupe_mm3": volume_au_dessus,
        "volume_prisme_sous_la_coupe_mm3": prisme,
        # Si la coupe est à z0 + dz, le volume change de A * dz : à connaître pour régler z_coupe_mm
        "sensibilite_volume_mm3_par_mm_de_coupe": aire_base,
        "part_sommets_loin_du_nuage_pct": part_loin,
        "espacement_nuage_mm": espacement_nuage,
    }

    if ouvertes:
        avertissements.append(
            f"{ouvertes} bord(s) ouvert(s) impossible(s) à fermer : le volume n'est pas fiable. "
            f"Maillage trop abîmé ? Refaire le maillage (profondeur Poisson plus haute, nuage plus dense).")
    if qualite["part_surface_comblee_pct"] > PART_COMBLEE_MAX_PCT:
        avertissements.append(
            f"{qualite['part_surface_comblee_pct']:.2f} % de la surface vient de trous bouchés "
            f"(seuil {PART_COMBLEE_MAX_PCT} %) : le volume dépend de ce bouchage.")
    # Un trou qui descend jusqu'au plan de coupe se confond avec le bord de la coupe : la fermeture par
    # un cône n'est alors plus une approximation locale, l'erreur de volume est sans rapport avec la
    # taille du trou. On ne tente pas de réparer, on le dit.
    if z0 is not None:
        touchent = sum(1 for b in boucles_trous if np.any(np.abs(V[b][:, 2] - z0) < 1e-6))
        if touchent:
            avertissements.append(
                f"{touchent} trou(s) du maillage touche(nt) le plan de coupe : le contour de la base est "
                f"incomplet, le volume n'est PAS fiable. Refaire le maillage (voir le modèle avant mesure).")
    if z0 is not None and not boucles_plan:
        avertissements.append(
            "Aucune coupe au plan du plateau : l'objet ne touche pas le plan. Plan mal placé, ou objet "
            "déjà refermé par Poisson sous le plateau : volume à vérifier.")
    if z0 is None and repere_plateau is False and (len(boucles_trous) or ouvertes):
        avertissements.append("Maillage non fermé et plateau inconnu : le volume dépend du bouchage des trous.")
    if part_loin is not None and part_loin > PART_LOIN_NUAGE_MAX_PCT:
        avertissements.append(
            f"{part_loin:.1f} % des sommets sont à plus de {FACTEUR_DISTANCE_NUAGE:g}x l'espacement du nuage "
            f"dense : surfaces probablement inventées par Poisson (bulles). Voir le maillage avant de se fier au volume.")
    if ar["non_manifold"]:
        notes.append(f"{ar['non_manifold']} arête(s) partagée(s) par plus de 2 faces (sans effet sur le volume).")

    # Maillage à exporter : fermé, base au plateau
    Ve, Fe, notes_export = construire_maillage_ferme(V, F, boucles_plan, boucles_trous, z0, prolonger)
    notes.extend(notes_export)
    return mesures, qualite, avertissements, notes, (Ve, Fe)


# ----------------------------------------------------------------------
# Fichiers : lecture, rapports, historique
# ----------------------------------------------------------------------
Z_COUPE_MM_DEFAUT = 1.0
# Un objet scanné ici fait entre quelques mm et quelques mètres : hors de cette plage, l'échelle est fausse
DIMENSION_MIN_MM = 1.0
DIMENSION_MAX_MM = 5000.0
COLONNES_HISTORIQUE = ["date", "nom", "volume_mm3", "volume_ml", "hauteur_mm", "longueur_mm", "largeur_mm",
                       "diametre_equivalent_base_mm", "surface_mm2", "mm_par_unite",
                       "incertitude_echelle_pct", "fiable", "fichier_maillage"]


def _en_json(objet):
    """Convertit récursivement les types numpy pour json.dump."""
    if isinstance(objet, dict):
        return {k: _en_json(v) for k, v in objet.items()}
    if isinstance(objet, (list, tuple)):
        return [_en_json(v) for v in objet]
    if isinstance(objet, np.ndarray):
        return _en_json(objet.tolist())
    if isinstance(objet, (np.floating, np.integer)):
        return objet.item()
    if isinstance(objet, np.bool_):
        return bool(objet)
    return objet


def _nombre_fr(x):
    """Nombre au format du CSV pour Excel français : virgule décimale."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, (int, float, np.floating, np.integer)):
        return f"{float(x):.7g}".replace(".", ",") if not float(x).is_integer() else str(int(x))
    return str(x)


def _verifier_reconstruction(echelle, chemin_echelle, avertissements, notes):
    """Compare l'empreinte enregistrée dans echelle.json à la reconstruction actuelle (voisine du fichier)."""
    attendue = (echelle.get("reconstruction") or {}).get("empreinte")
    if not attendue:
        return          # échelle manuelle, ou fichier plus ancien : rien à comparer
    dossier = os.path.join(os.path.dirname(os.path.abspath(chemin_echelle)), "sparse_final")
    if not os.path.isdir(dossier):
        notes.append("Reconstruction (sparse_final) introuvable à côté de echelle.json : "
                     "cohérence échelle/modèle non vérifiable.")
    elif empreinte_reconstruction(dossier) != attendue:
        avertissements.append(
            "La reconstruction COLMAP a changé depuis le calcul de l'échelle : cette échelle est celle d'un autre "
            "modèle, les mesures sont FAUSSES. Relancer echelle.py sur la reconstruction actuelle.")


def _charger_nuage(chemin, echelle):
    """Points d'un nuage PLY, passés à l'échelle. Renvoie None si le fichier ne donne aucun point."""
    points = None
    # Open3D d'abord : c'est lui qui a écrit/lu ces nuages dans le reste du projet, et trimesh
    # échoue sur certains PLY d'OpenMVS ("PLY is unexpected length")
    try:
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(chemin).points, dtype=np.float64)
    except ImportError:
        pass
    if points is None or len(points) == 0:
        import trimesh
        points = np.asarray(getattr(trimesh.load(chemin, process=False), "vertices", []), dtype=np.float64)
    if len(points) == 0:
        return None
    return appliquer_echelle(points, echelle) if echelle else points


def mesurer(chemin_maillage, chemin_echelle=None, dossier_sortie=None, nom=None,
            z_coupe_mm=Z_COUPE_MM_DEFAUT, sans_coupe=False, prolonger=True, garder_toutes=False,
            chemin_nuage=None, formats=("stl",), autoriser_sans_echelle=False):
    """Mesure complète d'un fichier maillage. Renvoie le rapport (dict).

    Si dossier_sortie est donné, écrit modele_mm.<format>, rapport.json, rapport.txt
    et ajoute une ligne à ../historique.csv.
    """
    V, F, nettoyage = charger_maillage(chemin_maillage)

    echelle = None
    if chemin_echelle is not None:
        try:
            echelle = charger_echelle(chemin_echelle)
        except ErreurEchelle as e:
            if not autoriser_sans_echelle:
                raise ErreurMesure(str(e))
    elif not autoriser_sans_echelle:
        raise ErreurMesure("Aucune échelle fournie : sans elle les résultats ne seraient pas en millimètres. "
                           "Lancer d'abord echelle.py.")

    avertissements, notes = [], []
    if echelle is not None:
        _verifier_reconstruction(echelle, chemin_echelle, avertissements, notes)
    if echelle is None:
        unite = "unite_colmap"
        avertissements.append("Aucune échelle : toutes les valeurs sont dans l'unité de COLMAP, PAS en millimètres.")
        V_mm, repere_plateau = V, False
    else:
        unite = "mm"
        V_mm = appliquer_echelle(V, echelle)
        repere_plateau = echelle.get("transformation_4x4") is not None
        if not echelle.get("fiable", True):
            avertissements.append("L'échelle utilisée est marquée « à vérifier » (voir echelle.json) : "
                                  "toutes les mesures en héritent.")
        if not repere_plateau and not sans_coupe:
            notes.append("Échelle sans repère de plateau (facteur manuel) : pas de coupe au plateau, "
                         "l'objet doit être complet dans le maillage.")
    z_coupe = None if (sans_coupe or not repere_plateau) else z_coupe_mm

    nuage_mm = None
    if chemin_nuage and os.path.isfile(chemin_nuage):
        try:
            nuage_mm = _charger_nuage(chemin_nuage, echelle)
        except Exception as e:     # un nuage illisible ne doit pas empêcher la mesure
            notes.append(f"Nuage dense non lu ({e}) : diagnostic « surfaces inventées » ignoré.")

    mesures, qualite, av, no, (Ve, Fe) = mesurer_maillage(
        V_mm, F, z_coupe_mm=z_coupe, repere_plateau=repere_plateau, prolonger=prolonger,
        garder_toutes=garder_toutes, nuage_mm=nuage_mm)
    avertissements += av
    notes += no
    qualite["faces_degenerees_supprimees"] += nettoyage["faces_degenerees_supprimees"]
    qualite["sommets_fichier"], qualite["faces_fichier"] = nettoyage["sommets_fichier"], nettoyage["faces_fichier"]

    if echelle is not None:
        plus_grande = max(mesures["boite_axes_mm"])
        if not DIMENSION_MIN_MM <= plus_grande <= DIMENSION_MAX_MM:
            avertissements.append(
                f"Dimensions aberrantes ({plus_grande:.3g} mm pour la plus grande) : échelle d'un autre scan ? "
                f"Refaire la mise à l'échelle (echelle.py) sur CETTE reconstruction.")

    e_qual = (echelle or {}).get("qualite") or {}
    u_echelle = e_qual.get("incertitude_echelle_pct")
    mesures_echelle = {
        "mm_par_unite": echelle["mm_par_unite"] if echelle else None,
        "methode": echelle["methode"] if echelle else None,
        "fiable": echelle.get("fiable") if echelle else False,
        "incertitude_echelle_pct": u_echelle,
        # Le volume varie comme le CUBE de l'échelle : une erreur de 0,1 % sur l'échelle donne 0,3 % sur le volume
        "incertitude_volume_due_a_l_echelle_pct": 3.0 * u_echelle if u_echelle is not None else None,
    }

    rapport = {
        "version": 1,
        "nom": nom or datetime.now().strftime("mesure_%Y%m%d_%H%M%S"),
        "date": datetime.now().isoformat(timespec="seconds"),
        "unite_longueur": unite,
        "fichier_maillage": os.path.abspath(chemin_maillage),
        "echelle": mesures_echelle,
        "parametres": {"z_coupe_mm": z_coupe, "prolonger_jusqu_au_plateau": prolonger,
                       "garder_toutes_composantes": garder_toutes},
        "mesures": mesures,
        "qualite": qualite,
        "fiable": len(avertissements) == 0,
        "avertissements": avertissements,
        "notes": notes,
    }
    rapport = _en_json(rapport)

    if dossier_sortie is not None:
        ecrire_rapport(rapport, dossier_sortie, (Ve, Fe), formats)
    return rapport


def resume_texte(rapport):
    """Rapport lisible (celui du rapport.txt et de la console)."""
    m, q = rapport["mesures"], rapport["qualite"]
    u = "mm" if rapport["unite_longueur"] == "mm" else "u"
    lignes = [f"Mesure « {rapport['nom']} » du {rapport['date']}",
              f"Maillage : {rapport['fichier_maillage']}",
              f"Unité : {'millimètres' if u == 'mm' else 'unité COLMAP (PAS des mm)'}"]
    if rapport["echelle"]["mm_par_unite"]:
        e = rapport["echelle"]
        lignes.append(f"Échelle : {e['mm_par_unite']:.6g} mm/unité ({e['methode']})"
                      + (f", incertitude {e['incertitude_echelle_pct']:.3f} %" if e["incertitude_echelle_pct"] is not None else ""))
    lignes.append("")
    lignes.append(f"{'Volume':21s}: {m['volume_mm3']:.1f} {u}³" + (f"  ({m['volume_ml']:.2f} mL)" if u == "mm" else ""))
    for cle, titre in [("hauteur_mm", "Hauteur (point haut)"), ("hauteur_robuste_mm", "Hauteur (lissée)"),
                       ("longueur_mm", "Longueur (empreinte)"),
                       ("largeur_mm", "Largeur (empreinte)"), ("diametre_equivalent_base_mm", "Diamètre équivalent base")]:
        if cle in m:
            lignes.append(f"{titre:21s}: {m[cle]:.2f} {u}")
    lignes.append(f"{'Surface':21s}: {m['surface_mm2']:.1f} {u}²")
    lignes.append("")
    coupe = f"coupe à z = {q['z_coupe_mm']:g} {u}" if q["z_coupe_mm"] is not None else "pas de coupe"
    lignes.append(f"Qualité : {q['faces']} triangles, {coupe}, "
                  f"{q['boucles_de_coupe']} contour(s) de coupe, {q['boucles_de_trou']} trou(s) bouché(s)")
    if q["sensibilite_volume_mm3_par_mm_de_coupe"]:
        lignes.append(f"          Décaler la coupe de 1 mm change le volume de {q['sensibilite_volume_mm3_par_mm_de_coupe']:.0f} {u}³")
    lignes.append("Fiabilité : " + ("OK" if rapport["fiable"] else "À VÉRIFIER"))
    lignes += [f"  ! {a}" for a in rapport["avertissements"]] + [f"  - {n}" for n in rapport["notes"]]
    return "\n".join(lignes)


def ecrire_rapport(rapport, dossier_sortie, maillage_export, formats=("stl",)):
    """Écrit modèle exporté, rapport.json, rapport.txt, et ajoute la ligne d'historique."""
    import trimesh

    os.makedirs(dossier_sortie, exist_ok=True)
    Ve, Fe = maillage_export
    base = "modele_mm" if rapport["unite_longueur"] == "mm" else "modele_unites_colmap"
    fichiers = []
    for fmt in formats:
        chemin = os.path.join(dossier_sortie, f"{base}.{fmt}")
        trimesh.Trimesh(Ve, Fe, process=False).export(chemin)
        if not os.path.isfile(chemin) or os.path.getsize(chemin) == 0:
            # trimesh peut produire un fichier vide sans erreur : même piège que write_triangle_mesh d'Open3D
            raise ErreurMesure(f"Échec de l'export {chemin} (fichier vide).")
        fichiers.append(chemin)
    rapport["fichiers_exportes"] = [os.path.abspath(f) for f in fichiers]
    with open(os.path.join(dossier_sortie, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dossier_sortie, "rapport.txt"), "w", encoding="utf-8") as f:
        f.write(resume_texte(rapport) + "\n")
    ajouter_historique(os.path.join(os.path.dirname(os.path.abspath(dossier_sortie)), "historique.csv"), rapport)


def ajouter_historique(chemin_csv, rapport):
    """Ajoute une ligne au tableau de toutes les mesures (séparateur ';', virgule décimale : Excel FR)."""
    m = rapport["mesures"]
    e = rapport["echelle"]
    valeurs = {
        "date": rapport["date"], "nom": rapport["nom"],
        "volume_mm3": m.get("volume_mm3"), "volume_ml": m.get("volume_ml"),
        "hauteur_mm": m.get("hauteur_mm"), "longueur_mm": m.get("longueur_mm"),
        "largeur_mm": m.get("largeur_mm"),
        "diametre_equivalent_base_mm": m.get("diametre_equivalent_base_mm"),
        "surface_mm2": m.get("surface_mm2"), "mm_par_unite": e.get("mm_par_unite"),
        "incertitude_echelle_pct": e.get("incertitude_echelle_pct"),
        "fiable": rapport["fiable"], "fichier_maillage": rapport["fichier_maillage"],
    }
    nouveau = not os.path.isfile(chemin_csv)
    # BOM au tout début seulement : Excel s'en sert pour reconnaître l'UTF-8
    with open(chemin_csv, "a", encoding="utf-8-sig" if nouveau else "utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if nouveau:
            w.writerow(COLONNES_HISTORIQUE)
        w.writerow([_nombre_fr(valeurs[c]) for c in COLONNES_HISTORIQUE])


def main(argv=None):
    p = argparse.ArgumentParser(description="Mesures en mm (volume, cotes) d'un maillage mis à l'échelle.")
    p.add_argument("--travail", default="colmap_workspace")
    p.add_argument("--maillage", default=None, help="défaut : <travail>/modele_pour_impression.stl")
    p.add_argument("--echelle", default=None, help="défaut : <travail>/echelle.json")
    p.add_argument("--nuage", default=None, help="défaut : <travail>/openmvs/scene_dense.ply s'il existe")
    p.add_argument("--nom", default=None, help="nom de la mesure (dossier de sortie), ex : gateau_avant_01")
    p.add_argument("--z-coupe-mm", type=float, default=Z_COUPE_MM_DEFAUT,
                   help="hauteur de coupe au-dessus du plateau (défaut 1 mm)")
    p.add_argument("--sans-coupe", action="store_true", help="ne pas couper au plateau (objet complet)")
    p.add_argument("--sans-prolongement", action="store_true",
                   help="ne pas ajouter la tranche entre le plateau et la coupe")
    p.add_argument("--toutes-composantes", action="store_true", help="garder tous les morceaux, pas seulement le plus gros")
    p.add_argument("--formats", nargs="+", default=["stl"], choices=["stl", "obj", "ply"])
    p.add_argument("--sans-echelle", action="store_true", help="autoriser une mesure en unités COLMAP (déconseillé)")
    args = p.parse_args(argv)

    maillage = args.maillage or os.path.join(args.travail, "modele_pour_impression.stl")
    echelle = args.echelle or os.path.join(args.travail, "echelle.json")
    nuage = args.nuage or os.path.join(args.travail, "openmvs", "scene_dense.ply")
    nom = args.nom or datetime.now().strftime("mesure_%Y%m%d_%H%M%S")
    try:
        rapport = mesurer(
            maillage, echelle, os.path.join(args.travail, "mesures", nom), nom,
            z_coupe_mm=args.z_coupe_mm, sans_coupe=args.sans_coupe, prolonger=not args.sans_prolongement,
            garder_toutes=args.toutes_composantes, chemin_nuage=nuage, formats=args.formats,
            autoriser_sans_echelle=args.sans_echelle)
    except (ErreurMesure, ErreurEchelle) as e:
        print(f"ERREUR : {e}")
        return 1
    print(resume_texte(rapport))
    print(f"\nFichiers dans : {os.path.join(args.travail, 'mesures', nom)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
