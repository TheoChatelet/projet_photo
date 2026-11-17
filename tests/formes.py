"""Formes simples à volume connu exactement, pour tester les mesures."""
import numpy as np
import trimesh


def _aire_polygone_regulier(n, r):
    return 0.5 * n * r * r * np.sin(2 * np.pi / n)


def cylindre(r=25.0, h=40.0, n=64, decalage=(0.0, 0.0), z_base=0.0):
    """Prisme à n pans posé sur z = z_base. Renvoie (V, F, volume_exact)."""
    m = trimesh.creation.cylinder(radius=r, height=h, sections=n)
    m.apply_translation([decalage[0], decalage[1], z_base + h / 2])
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), _aire_polygone_regulier(n, r) * h


def boite(a=30.0, b=20.0, c=10.0, origine=(0.0, 0.0, 0.0)):
    m = trimesh.creation.box(extents=(a, b, c))
    m.apply_translation(np.array(origine) + [a / 2, b / 2, c / 2])
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), a * b * c


def anneau(r_ext=30.0, r_int=12.0, h=15.0, n=48):
    """Rondelle : la base a deux contours (extérieur et trou)."""
    m = trimesh.creation.annulus(r_min=r_int, r_max=r_ext, height=h, sections=n)
    m.apply_translation([0, 0, h / 2])
    v = (_aire_polygone_regulier(n, r_ext) - _aire_polygone_regulier(n, r_int)) * h
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), v


def fusionner(*formes):
    """Réunit plusieurs formes disjointes (ou accolées) en un seul maillage."""
    V, F, vol, base = [], [], 0.0, 0
    for v, f, volume in formes:
        V.append(v)
        F.append(f + base)
        base += len(v)
        vol += volume
    return np.vstack(V), np.vstack(F), vol


def prisme_profil_xz(profil, largeur_y):
    """Extrusion selon y d'un profil simple dessiné dans le plan (x, z). Renvoie (V, F, volume)."""
    import mapbox_earcut as earcut

    P = np.asarray(profil, float)
    n = len(P)
    V = np.vstack([np.column_stack([P[:, 0], np.zeros(n), P[:, 1]]),
                   np.column_stack([P[:, 0], np.full(n, largeur_y), P[:, 1]])])
    tri = np.asarray(earcut.triangulate_float64(P, np.array([n], dtype=np.uint32))).reshape(-1, 3)
    F = [tri, tri[:, ::-1] + n]
    for i in range(n):
        j = (i + 1) % n
        F.append(np.array([[i, j, j + n], [i, j + n, i + n]]))
    F = np.vstack(F)
    aire = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    # Orientation cohérente garantie par trimesh (indépendant du code testé), puis normales vers l'extérieur
    m = trimesh.Trimesh(V, F, process=False)
    trimesh.repair.fix_winding(m)
    F = np.asarray(m.faces)
    if np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() < 0:
        F = F[:, ::-1]
    return V, F.copy(), abs(aire) * largeur_y


def pont_en_u():
    """Deux pieds de 10 mm reliés par un tablier : 8000 mm3, DEUX contours à la base (un seul morceau)."""
    profil = [(0, 0), (10, 0), (10, 20), (30, 20), (30, 0), (40, 0), (40, 30), (0, 30)]
    return prisme_profil_xz(profil, 10.0)


def dome(r=40.0, subdivisions=4):
    """Sphère centrée sur le plateau : la partie utile est l'hémisphère au-dessus."""
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64), (2.0 / 3.0) * np.pi * r ** 3


def retourner(V, F):
    return V.copy(), F[:, ::-1].copy()
