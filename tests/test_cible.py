"""Tests du générateur de cible ArUco."""
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import generer_cible_echelle as g

cv2 = pytest.importorskip("cv2")


def test_mise_en_page_par_defaut_tient_sur_a4():
    mep = g.construire_mise_en_page()
    assert len(mep["marqueurs"]) == 8
    assert mep["taille_marqueur_mm"] == 30.0
    ids = [m["id"] for m in mep["marqueurs"]]
    assert ids == list(range(8))


@pytest.mark.parametrize("options, mot", [
    (dict(rayon_mm=95.0), "ne tient pas"),
    (dict(nb=20), "se touchent"),
    (dict(nb=1), "au moins 2"),
    (dict(taille_mm=-5), "positifs"),
    (dict(page="B5"), "inconnue"),
    (dict(nom_dictionnaire="DICT_NOPE"), "inconnu"),
    (dict(nb=60, rayon_mm=70, taille_mm=3), "n'a que"),
])
def test_mise_en_page_refuse_les_cas_impossibles(options, mot):
    with pytest.raises(ValueError, match=mot):
        g.construire_mise_en_page(**options)


def test_coins_dans_l_ordre_opencv():
    mep = {"dictionnaire": "DICT_4X4_50", "taille_marqueur_mm": 30.0,
           "marqueurs": [{"id": 0, "centre_mm": [10.0, 20.0]}]}
    coins = g.coins_marqueur_mm(mep, 0)
    attendu = [[-5, 35, 0], [25, 35, 0], [25, 5, 0], [-5, 5, 0]]  # haut-gauche, haut-droit, bas-droit, bas-gauche
    assert np.allclose(coins, attendu)
    with pytest.raises(KeyError):
        g.coins_marqueur_mm(mep, 7)


@pytest.mark.parametrize("identifiant", [0, 1, 17, 49])
def test_les_cases_redonnent_le_bon_identifiant(identifiant):
    """Le dessin produit (cases) doit être relu par OpenCV avec le même numéro."""
    cases = g.cases_du_marqueur("DICT_4X4_50", identifiant)
    image = np.kron(1 - cases, np.ones((20, 20))).astype(np.uint8) * 255
    image = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    from echelle import detecter_marqueurs
    trouves = detecter_marqueurs(image, "DICT_4X4_50")
    assert list(trouves) == [identifiant]
    # coin 0 = haut-gauche : le plus proche de (40, 40)
    assert np.allclose(trouves[identifiant][0], [40, 40], atol=1.0)


def test_identifiant_hors_dictionnaire():
    with pytest.raises(ValueError):
        g.cases_du_marqueur("DICT_4X4_50", 50)


def test_correction_d_impression_est_uniforme():
    mep = g.construire_mise_en_page()
    corrigee = g.mise_en_page_a_echelle(mep, 29.85)
    f = 29.85 / 30.0
    assert corrigee["taille_marqueur_mm"] == pytest.approx(29.85)
    assert corrigee["marqueurs"][1]["centre_mm"][0] == pytest.approx(mep["marqueurs"][1]["centre_mm"][0] * f)
    # l'original n'est pas modifié
    assert mep["taille_marqueur_mm"] == 30.0
    with pytest.raises(ValueError):
        g.mise_en_page_a_echelle(mep, 0)


def test_ecriture_svg_et_json(tmp_path):
    svg, js = g.generer(str(tmp_path), nb=6)
    racine = ET.parse(svg).getroot()          # SVG bien formé
    assert racine.attrib["width"] == "297.0mm"  # page A4 en paysage, dimensions réelles
    mep = g.charger_mise_en_page(js)
    assert len(mep["marqueurs"]) == 6
    # 6 marqueurs de 36 cases : entre 10 et 36 cases noires chacun (la bordure en fait 20)
    nb_rect = len(racine.findall(".//{http://www.w3.org/2000/svg}g[@shape-rendering='crispEdges']/{http://www.w3.org/2000/svg}rect"))
    assert 6 * 20 <= nb_rect <= 6 * 36


@pytest.mark.parametrize("contenu, mot", [
    ("pas du json", "JSON"),
    (json.dumps({"dictionnaire": "DICT_4X4_50"}), "manquant"),
    (json.dumps({"dictionnaire": "d", "taille_marqueur_mm": 30, "marqueurs": []}), "aucun marqueur"),
    (json.dumps({"dictionnaire": "d", "taille_marqueur_mm": 30,
                 "marqueurs": [{"id": 1, "centre_mm": [0, 0]}, {"id": 1, "centre_mm": [9, 9]}]}), "double"),
    (json.dumps({"dictionnaire": "d", "taille_marqueur_mm": -1,
                 "marqueurs": [{"id": 1, "centre_mm": [0, 0]}]}), "positive"),
])
def test_chargement_refuse_les_fichiers_invalides(tmp_path, contenu, mot):
    chemin = tmp_path / "c.json"
    chemin.write_text(contenu, encoding="utf-8")
    with pytest.raises(ValueError, match=mot):
        g.charger_mise_en_page(str(chemin))


def test_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="generer_cible_echelle"):
        g.charger_mise_en_page(str(tmp_path / "absent.json"))


def test_cli_renvoie_un_code_d_erreur(tmp_path, capsys):
    assert g.main(["--dossier", str(tmp_path), "--rayon-mm", "200"]) == 1
    assert "ERREUR" in capsys.readouterr().out
    assert g.main(["--dossier", str(tmp_path)]) == 0
