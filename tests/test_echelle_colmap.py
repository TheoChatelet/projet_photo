"""Adaptateur pycolmap : vérifié sur la vraie reconstruction du dépôt."""
import os

import numpy as np
import pytest

import echelle as e

pycolmap = pytest.importorskip("pycolmap")
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARSE = os.path.join(RACINE, "colmap_workspace", "sparse_final")

pytestmark = pytest.mark.lourd


@pytest.fixture(scope="module")
def reconstruction():
    if not os.path.isdir(SPARSE):
        pytest.skip("reconstruction du dépôt absente")
    return pycolmap.Reconstruction(SPARSE)


def test_les_vues_ont_des_poses_valides(reconstruction):
    vues = e.vues_depuis_reconstruction(reconstruction)
    assert len(vues) == reconstruction.num_reg_images()
    for v in vues:
        R = v.cam_from_world[:, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0
        assert np.all(np.isfinite(v.centre()))


def test_coordonnees_normalisees_recollent_avec_les_points_3d(reconstruction):
    """Un vrai point 3D de COLMAP, projeté avec NOTRE matrice de pose, doit retomber sur le pixel
    observé une fois NOTRE normalisation appliquée : valide à la fois la pose, la distorsion
    et la convention de pixel (centre du premier pixel en 0,5)."""
    vues = {v.nom: v for v in e.vues_depuis_reconstruction(reconstruction)}
    ecarts = []
    for p3d in list(reconstruction.points3D.values())[::40][:200]:
        for el in p3d.track.elements[:3]:
            image = reconstruction.images[el.image_id]
            v = vues[image.name]
            observe = np.array(image.points2D[el.point2D_idx].xy)
            attendu = v.cam_from_world @ np.append(p3d.xyz, 1.0)
            norm_projete = attendu[:2] / attendu[2]
            ecarts.append(np.linalg.norm(v.normaliser(observe)[0] - norm_projete) * v.focale_px)
    assert len(ecarts) > 100
    # L'erreur de reprojection moyenne de COLMAP est de ~0,43 px : on doit être du même ordre
    assert np.median(ecarts) < 1.0
    assert np.percentile(ecarts, 95) < 3.0


def test_une_image_sans_pose_est_ignoree(reconstruction):
    n = len(e.vues_depuis_reconstruction(reconstruction))
    assert n == 34


# ----------------------------------------------------------------------------------------
# lancer_echelle de bout en bout : vraie reconstruction pycolmap écrite sur disque, images rendues
# ----------------------------------------------------------------------------------------
import json

import cv2
import synth
from generer_cible_echelle import construire_mise_en_page

MEP = construire_mise_en_page()


def _fabriquer_dossier_de_travail(racine, nb=12, rendre=True, s=137.0):
    """colmap_workspace synthétique : sparse_final + images_redimensionnees, vérité connue."""
    vrai = synth.SimilitudeVraie(s, axe="x", angle=70.0)
    rec = pycolmap.Reconstruction()
    dossier_images = os.path.join(racine, "images_redimensionnees")
    os.makedirs(dossier_images)
    for n, c in enumerate(synth.positions_autour(nb, graine=4), start=1):
        cam = pycolmap.Camera.create_from_model_id(
            n, pycolmap.CameraModelId.SIMPLE_RADIAL, synth.FOCALE, synth.LARGEUR, synth.HAUTEUR)
        rec.add_camera_with_trivial_rig(cam)
        R, t = synth.camera_regarde(c, (0, 0, 0))
        img = pycolmap.Image(name=f"{n:02d}.png", camera_id=n, image_id=n)
        rec.add_image_with_trivial_frame(img, pycolmap.Rigid3d(vrai.cam_from_world_colmap(R, t)))
        image = synth.rendre_image(MEP, R, t, graine=n) if rendre else np.full((synth.HAUTEUR, synth.LARGEUR), 160, np.uint8)
        ok, tampon = cv2.imencode(".png", image)
        tampon.tofile(os.path.join(dossier_images, f"{n:02d}.png"))
    sparse = os.path.join(racine, "sparse_final")
    os.makedirs(sparse)
    rec.write(sparse)
    return vrai


def _ecrire_mep(chemin, mep=MEP):
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(mep, f)
    return chemin


def test_lancer_echelle_de_bout_en_bout(tmp_path, capsys):
    racine = str(tmp_path / "Théo dé")           # accents + espace : cas réel sous Windows
    os.makedirs(racine)
    vrai = _fabriquer_dossier_de_travail(racine)
    mep = _ecrire_mep(str(tmp_path / "cible.json"))
    r = e.lancer_echelle(racine, mep)
    assert abs(r["mm_par_unite"] / 137.0 - 1) < 0.003
    assert r["fiable"], r["avertissements"]
    relu = e.charger_echelle(e.chemin_echelle(racine))        # le fichier écrit est relisible
    assert relu["mm_par_unite"] == pytest.approx(r["mm_par_unite"])
    assert "images analysées" in capsys.readouterr().out


def test_lancer_echelle_ecrit_les_images_de_debug(tmp_path):
    racine = str(tmp_path / "w")
    os.makedirs(racine)
    _fabriquer_dossier_de_travail(racine, nb=6)
    e.lancer_echelle(racine, _ecrire_mep(str(tmp_path / "c.json")), debug=True)
    assert len(os.listdir(os.path.join(racine, "echelle_debug"))) == 6


def test_lancer_echelle_corrige_la_taille_imprimee(tmp_path):
    """--taille-marqueur-mm : la feuille imprimée mesure 29,85 mm au lieu de 30."""
    racine = str(tmp_path / "w")
    os.makedirs(racine)
    _fabriquer_dossier_de_travail(racine)
    mep = _ecrire_mep(str(tmp_path / "c.json"))
    sans = e.lancer_echelle(racine, mep)["mm_par_unite"]
    avec = e.lancer_echelle(racine, mep, taille_marqueur_mm=29.85)["mm_par_unite"]
    assert avec / sans == pytest.approx(29.85 / 30.0, rel=1e-6)


def test_lancer_echelle_sans_marqueur_dans_les_photos(tmp_path):
    racine = str(tmp_path / "w")
    os.makedirs(racine)
    _fabriquer_dossier_de_travail(racine, nb=5, rendre=False)
    with pytest.raises(e.ErreurEchelle, match="Aucun marqueur"):
        e.lancer_echelle(racine, _ecrire_mep(str(tmp_path / "c.json")))
    assert not os.path.exists(e.chemin_echelle(racine))      # rien d'écrit en cas d'échec


def test_lancer_echelle_messages_d_erreur(tmp_path):
    racine = str(tmp_path / "w")
    os.makedirs(racine)
    mep = _ecrire_mep(str(tmp_path / "c.json"))
    with pytest.raises(e.ErreurEchelle, match="etape 1|étape 1"):
        e.lancer_echelle(racine, mep)
    os.makedirs(os.path.join(racine, "sparse_final"))
    with pytest.raises(e.ErreurEchelle, match="images"):
        e.lancer_echelle(racine, mep)
    os.makedirs(os.path.join(racine, "images_redimensionnees"))
    with pytest.raises(FileNotFoundError, match="generer_cible_echelle"):
        e.lancer_echelle(racine, str(tmp_path / "absent.json"))
