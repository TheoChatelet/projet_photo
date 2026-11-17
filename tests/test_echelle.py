"""Tests de la mise à l'échelle : on retrouve une échelle et un repère connus à l'avance."""
import json
import math
import os

import numpy as np
import pytest

import echelle as e
import synth
from generer_cible_echelle import construire_mise_en_page, mise_en_page_a_echelle

cv2 = pytest.importorskip("cv2")

S_VRAIE = 137.0  # mm par unité COLMAP
MEP = construire_mise_en_page()


# ---------------------------------------------------------------- Umeyama
def _nuage(n=20, plan=False, graine=3):
    rng = np.random.default_rng(graine)
    pts = rng.normal(0, 50, (n, 3))
    if plan:
        pts[:, 2] = 0
    return pts


@pytest.mark.parametrize("plan", [False, True])
def test_similitude_retrouvee_exactement(plan):
    X = _nuage(plan=plan)
    R = synth.rotation("x", 40) @ synth.rotation("z", -70)
    Y = 2.7 * (R @ X.T).T + [10, -20, 5]
    s, Rr, t = e.estimer_similitude(X, Y)
    assert s == pytest.approx(2.7, rel=1e-10)
    assert np.allclose(Rr, R, atol=1e-10)
    assert np.linalg.det(Rr) == pytest.approx(1.0)
    assert np.allclose(t, [10, -20, 5], atol=1e-8)


def test_similitude_ne_renvoie_jamais_de_miroir():
    """Une cible plane vue en miroir doit donner une vraie rotation, pas une symétrie."""
    X = _nuage(plan=True)
    Y = X * [1, -1, 1]  # symétrie : n'a pas de solution exacte, mais la rotation doit rester directe
    _, R, _ = e.estimer_similitude(X, Y)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_similitude_refuse_les_cas_degeneres():
    with pytest.raises(e.ErreurEchelle):
        e.estimer_similitude(np.zeros((5, 3)), _nuage(5))
    with pytest.raises(e.ErreurEchelle):
        e.estimer_similitude(_nuage(2), _nuage(2))


# ----------------------------------------------------------- triangulation
def test_triangulation_sans_bruit():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    X_mm = np.array([12.0, -7.0, 3.0])
    obs = []
    for c in synth.positions_autour(6):
        R, t = synth.camera_regarde(c, (0, 0, 0))
        P = vrai.cam_from_world_colmap(R, t)
        cam = P[:, :3] @ vrai.colmap_depuis_mm(X_mm) + P[:, 3]
        obs.append((P, cam[:2] / cam[2], synth.FOCALE, -P[:, :3].T @ P[:, 3]))
    X, erreur, angle, nb = e.trianguler_point(obs)
    assert np.allclose(X, vrai.colmap_depuis_mm(X_mm), atol=1e-7)
    assert nb == 6 and angle > 30 and erreur < 1e-3


def test_triangulation_retire_une_vue_aberrante():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    X_mm = np.array([0.0, 0.0, 0.0])
    obs = []
    for c in synth.positions_autour(8):
        R, t = synth.camera_regarde(c, (0, 0, 0))
        P = vrai.cam_from_world_colmap(R, t)
        cam = P[:, :3] @ vrai.colmap_depuis_mm(X_mm) + P[:, 3]
        obs.append([P, cam[:2] / cam[2], synth.FOCALE, -P[:, :3].T @ P[:, 3]])
    obs[3][1] = obs[3][1] + [40 / synth.FOCALE, 0]  # une détection fausse de 40 px
    X, _, _, nb = e.trianguler_point([tuple(o) for o in obs])
    assert nb == 7
    assert np.allclose(X, vrai.colmap_depuis_mm(X_mm), atol=1e-6)


def test_triangulation_une_seule_vue_est_impossible():
    P = np.hstack([np.eye(3), np.zeros((3, 1))])
    assert e.trianguler_point([(P, np.array([0.1, 0.1]), 1000.0, np.zeros(3))]) is None


# ------------------------------------------------- calcul de bout en bout
def _controle_repere(resultat, vrai, centres):
    """La transformation doit renvoyer les caméras COLMAP à leur vraie position en mm."""
    T = np.array(resultat["transformation_4x4"])
    for c_mm, vue_centre in zip(centres, [vrai.colmap_depuis_mm(c) for c in centres]):
        retrouve = (T @ np.append(vue_centre, 1.0))[:3]
        assert np.linalg.norm(retrouve - c_mm) < 1.5, (retrouve, c_mm)


def test_echelle_sans_bruit_est_exacte():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    centres = synth.positions_autour(12)
    r = e.calculer_echelle_cible(synth.vues_synthetiques(MEP, vrai, centres), MEP)
    assert r["mm_par_unite"] == pytest.approx(S_VRAIE, rel=1e-6)
    assert r["fiable"] and r["qualite"]["residu_rms_mm"] < 1e-4
    _controle_repere(r, vrai, centres)


@pytest.mark.parametrize("bruit_px, k1", [(0.2, 0.0), (0.3, 0.15), (0.5, -0.2)])
def test_echelle_avec_bruit_et_distorsion(bruit_px, k1):
    vrai = synth.SimilitudeVraie(S_VRAIE, axe="y", angle=-110.0)
    centres = synth.positions_autour(14, jitter=0.05)
    vues = synth.vues_synthetiques(MEP, vrai, centres, bruit_px=bruit_px, k1=k1)
    r = e.calculer_echelle_cible(vues, MEP)
    assert abs(r["mm_par_unite"] / S_VRAIE - 1) < 0.003          # moins de 0,3 %
    assert r["fiable"], r["avertissements"]
    # Les caméras sont AU-DESSUS du plan de la cible : z positif, donc le repère est « z vers le haut »
    T = np.array(r["transformation_4x4"])
    z = [(T @ np.append(vrai.colmap_depuis_mm(c), 1.0))[2] for c in centres]
    assert min(z) > 200


def test_le_bruit_se_traduit_en_incertitude():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    centres = synth.positions_autour(12)
    calme = e.calculer_echelle_cible(synth.vues_synthetiques(MEP, vrai, centres, bruit_px=0.05), MEP)
    bruite = e.calculer_echelle_cible(synth.vues_synthetiques(MEP, vrai, centres, bruit_px=1.0, graine=5), MEP)
    assert bruite["qualite"]["residu_rms_mm"] > calme["qualite"]["residu_rms_mm"]
    assert bruite["qualite"]["incertitude_echelle_pct"] > calme["qualite"]["incertitude_echelle_pct"]


def test_un_marqueur_incoherent_entre_les_photos_est_elimine():
    """Détecté 25 px trop loin partout : les rayons ne se coupent plus, il est rejeté dès la triangulation."""
    vrai = synth.SimilitudeVraie(S_VRAIE)
    centres = synth.positions_autour(12)
    vues = synth.vues_synthetiques(MEP, vrai, centres, bruit_px=0.2)
    for v in vues:
        if 3 in v.marqueurs:
            v.marqueurs[3] = v.marqueurs[3] + [25.0, -15.0]
    r = e.calculer_echelle_cible(vues, MEP)
    assert abs(r["mm_par_unite"] / S_VRAIE - 1) < 0.004
    assert 3 not in r["cible"]["marqueurs_utilises"]
    assert any("non exploités" in n for n in r["notes"])


def test_un_marqueur_mal_place_sur_la_feuille_est_ecarte():
    """Cohérent entre les photos mais pas à la position annoncée par la mise en page (collé de travers)."""
    vrai = synth.SimilitudeVraie(S_VRAIE)
    vues = synth.vues_synthetiques(MEP, vrai, synth.positions_autour(12), bruit_px=0.2)
    mep_fausse = json.loads(json.dumps(MEP))
    for m in mep_fausse["marqueurs"]:
        if m["id"] == 3:
            m["centre_mm"] = [m["centre_mm"][0] + 9.0, m["centre_mm"][1] - 7.0]
    r = e.calculer_echelle_cible(vues, mep_fausse)
    assert abs(r["mm_par_unite"] / S_VRAIE - 1) < 0.004
    assert 3 not in r["cible"]["marqueurs_utilises"]
    assert any("Marqueur 3 écarté" in n for n in r["notes"])


def test_peu_de_marqueurs_donne_un_avertissement():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    vues = synth.vues_synthetiques(MEP, vrai, synth.positions_autour(12), ids_vus={0, 4})
    r = e.calculer_echelle_cible(vues, MEP)
    assert abs(r["mm_par_unite"] / S_VRAIE - 1) < 0.001  # deux marqueurs, sans bruit : exact
    assert r["qualite"]["nb_marqueurs"] == 2
    assert r["qualite"]["incertitude_echelle_pct"] is None  # jackknife impossible à 2 marqueurs
    assert not r["fiable"]


def test_aucun_marqueur():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    vues = synth.vues_synthetiques(MEP, vrai, synth.positions_autour(6), ids_vus=set())
    with pytest.raises(e.ErreurEchelle, match="Aucun marqueur"):
        e.calculer_echelle_cible(vues, MEP)


def test_une_seule_photo_ne_suffit_pas():
    vrai = synth.SimilitudeVraie(S_VRAIE)
    vues = synth.vues_synthetiques(MEP, vrai, synth.positions_autour(1))
    with pytest.raises(e.ErreurEchelle, match="exploitable"):
        e.calculer_echelle_cible(vues, MEP)


def test_vues_trop_proches_donnent_une_erreur_claire():
    """Toutes les photos du même côté : les rayons sont presque parallèles."""
    vrai = synth.SimilitudeVraie(S_VRAIE)
    centres = [[420 * math.cos(math.radians(a)), 420 * math.sin(math.radians(a)), 250] for a in (0, 0.5, 1.0, 1.5)]
    vues = synth.vues_synthetiques(MEP, vrai, centres)
    with pytest.raises(e.ErreurEchelle, match="angle|exploitable"):
        e.calculer_echelle_cible(vues, MEP)


def test_taille_d_impression_corrigee_change_l_echelle():
    """Marqueurs imprimés 0,5 % trop petits : sans correction, l'échelle est fausse du même 0,5 %."""
    vrai = synth.SimilitudeVraie(S_VRAIE)
    mep_reel = mise_en_page_a_echelle(MEP, 30.0 * 0.995)      # ce qui est vraiment sur la feuille
    vues = synth.vues_synthetiques(mep_reel, vrai, synth.positions_autour(12))
    naif = e.calculer_echelle_cible(vues, MEP)["mm_par_unite"]
    corrige = e.calculer_echelle_cible(vues, mep_reel)["mm_par_unite"]
    assert corrige == pytest.approx(S_VRAIE, rel=1e-6)
    # Sans correction, l'erreur d'impression se retrouve intégralement dans l'échelle
    assert naif / S_VRAIE == pytest.approx(1 / 0.995, rel=1e-4)


# --------------------------------------------- avec de vraies images simulées
def _normaliseur_sans_distorsion(pixels):
    return (np.asarray(pixels, float).reshape(-1, 2) - [synth.LARGEUR / 2, synth.HAUTEUR / 2]) / synth.FOCALE


def test_chaine_complete_sur_images_rendues():
    """Rendu d'images -> détection ArUco -> triangulation -> échelle : le test le plus proche de la réalité."""
    vrai = synth.SimilitudeVraie(S_VRAIE, axe="x", angle=75.0)
    centres = synth.positions_autour(12, graine=2)
    vues = []
    for n, c in enumerate(centres):
        R, t = synth.camera_regarde(c, (0, 0, 0))
        image = synth.rendre_image(MEP, R, t, graine=n)
        v = e.Vue(f"img_{n}.png", vrai.cam_from_world_colmap(R, t), _normaliseur_sans_distorsion, synth.FOCALE)
        trouves = e.detecter_marqueurs(image, MEP["dictionnaire"], {m["id"] for m in MEP["marqueurs"]})
        v.marqueurs = {i: c_ + 0.5 for i, c_ in trouves.items()}
        vues.append(v)
    assert sum(len(v.marqueurs) for v in vues) >= 12 * 4, "trop peu de marqueurs détectés"
    r = e.calculer_echelle_cible(vues, MEP)
    assert abs(r["mm_par_unite"] / S_VRAIE - 1) < 0.003, r["mm_par_unite"]
    assert r["fiable"], r["avertissements"]
    _controle_repere(r, vrai, centres)


def test_detection_ignore_les_marqueurs_hors_cible():
    R, t = synth.camera_regarde(synth.positions_autour(1)[0], (0, 0, 0))
    image = synth.rendre_image(MEP, R, t)
    tous = e.detecter_marqueurs(image, MEP["dictionnaire"])
    seulement = e.detecter_marqueurs(image, MEP["dictionnaire"], ids_autorises={0, 1})
    assert len(tous) > 2 and set(seulement) <= {0, 1}


def test_detection_image_vide_ou_absente():
    assert e.detecter_marqueurs(None, "DICT_4X4_50") == {}
    assert e.detecter_marqueurs(np.zeros((0, 0), np.uint8), "DICT_4X4_50") == {}
    assert e.detecter_marqueurs(np.full((200, 200), 128, np.uint8), "DICT_4X4_50") == {}


def test_detection_accepte_les_images_couleur():
    R, t = synth.camera_regarde(synth.positions_autour(1)[0], (0, 0, 0))
    gris = synth.rendre_image(MEP, R, t)
    couleur = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    assert set(e.detecter_marqueurs(couleur, MEP["dictionnaire"])) == set(e.detecter_marqueurs(gris, MEP["dictionnaire"]))


def test_aucun_faux_positif_sur_du_bruit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        image = rng.integers(0, 255, (400, 400), dtype=np.uint8)
        assert e.detecter_marqueurs(image, "DICT_4X4_50") == {}


@pytest.mark.lourd
def test_aucun_faux_positif_sur_les_vraies_photos():
    """Les photos du dépôt ne contiennent pas de cible : on ne doit rien « inventer »."""
    dossier = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "colmap_workspace", "images_redimensionnees")
    if not os.path.isdir(dossier):
        pytest.skip("photos du dépôt absentes")
    from masquage_fond import lire_image, lister_images
    nb = 0
    for chemin in lister_images(dossier)[::3]:
        assert e.detecter_marqueurs(lire_image(chemin), "DICT_4X4_50", ids_autorises=set(range(8))) == {}, chemin
        nb += 1
    assert nb >= 5


# ------------------------------------------------------------ echelle.json
def test_facteur_manuel():
    r = e.echelle_depuis_facteur(137.5)
    assert r["mm_par_unite"] == 137.5 and r["transformation_4x4"] is None and r["fiable"]
    r2 = e.echelle_depuis_distance(0.25, 50.0)
    assert r2["mm_par_unite"] == pytest.approx(200.0) and r2["methode"] == "distance_connue"


@pytest.mark.parametrize("mauvais", [0, -3.0, float("nan"), float("inf"), "12", None])
def test_facteur_invalide(mauvais):
    with pytest.raises(e.ErreurEchelle):
        e.echelle_depuis_facteur(mauvais)


@pytest.mark.parametrize("a, b", [(0, 5), (5, 0), (-1, 4)])
def test_distance_invalide(a, b):
    with pytest.raises(e.ErreurEchelle):
        e.echelle_depuis_distance(a, b)


def test_aller_retour_echelle_json(tmp_path):
    vrai = synth.SimilitudeVraie(S_VRAIE)
    r = e.calculer_echelle_cible(synth.vues_synthetiques(MEP, vrai, synth.positions_autour(10)), MEP)
    chemin = e.ecrire_echelle(str(tmp_path), r)
    relu = e.charger_echelle(chemin)
    assert relu["mm_par_unite"] == pytest.approx(r["mm_par_unite"])
    assert np.allclose(relu["transformation_4x4"], r["transformation_4x4"])
    assert "Fiabilité" in e.resumer(relu)


def _ecrire(tmp_path, contenu):
    chemin = tmp_path / "echelle.json"
    chemin.write_text(contenu if isinstance(contenu, str) else json.dumps(contenu), encoding="utf-8")
    return str(chemin)


def test_chargement_refuse_les_fichiers_invalides(tmp_path):
    with pytest.raises(e.ErreurEchelle, match="introuvable"):
        e.charger_echelle(str(tmp_path / "rien.json"))
    with pytest.raises(e.ErreurEchelle, match="illisible"):
        e.charger_echelle(_ecrire(tmp_path, "{pas du json"))
    with pytest.raises(e.ErreurEchelle, match="mm_par_unite"):
        e.charger_echelle(_ecrire(tmp_path, {"mm_par_unite": -2}))
    with pytest.raises(e.ErreurEchelle, match="4x4"):
        e.charger_echelle(_ecrire(tmp_path, {"mm_par_unite": 2, "transformation_4x4": [[1, 0], [0, 1]]}))
    miroir = np.diag([2.0, 2.0, -2.0, 1.0]).tolist()
    with pytest.raises(e.ErreurEchelle, match="similitude"):
        e.charger_echelle(_ecrire(tmp_path, {"mm_par_unite": 2, "transformation_4x4": miroir}))
    deforme = np.diag([2.0, 3.0, 2.0, 1.0]).tolist()
    with pytest.raises(e.ErreurEchelle, match="similitude"):
        e.charger_echelle(_ecrire(tmp_path, {"mm_par_unite": 2, "transformation_4x4": deforme}))


def test_cli_facteur(tmp_path, capsys):
    assert e.main(["--travail", str(tmp_path), "--facteur", "42"]) == 0
    assert e.charger_echelle(e.chemin_echelle(str(tmp_path)))["mm_par_unite"] == 42
    assert e.main(["--travail", str(tmp_path), "--facteur", "-1"]) == 1
    assert e.main(["--travail", str(tmp_path), "--distance-mm", "5"]) == 1
    assert "ERREUR" in capsys.readouterr().out
