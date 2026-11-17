"""Tests du gonflement : calculs vérifiés à la main, cas limites, chaîne complète sur des gâteaux simulés."""
import csv
import json
import math
import os

import numpy as np
import pytest

import formes
import gonflement as g
import mesure
import synth
import validation_precision as vp

trimesh = pytest.importorskip("trimesh")


def _rapport(nom, volume, methode="cible_aruco", **autres):
    return {"nom": nom, "unite_longueur": "mm", "fiable": True, "avertissements": [],
            "echelle": {"methode": methode}, "mesures": {"volume_mm3": volume, **autres}}


AVANT = [_rapport(f"a{i}", v) for i, v in enumerate([1000.0, 1010.0, 990.0])]
APRES = [_rapport(f"b{i}", v) for i, v in enumerate([1300.0, 1310.0, 1290.0])]


def test_calcul_verifie_a_la_main():
    res = g.comparer(AVANT, APRES, k=2.0, sys_pct=1.0)
    stat = 10.0 / math.sqrt(3)                       # écart-type 10, 3 scans
    ua = math.hypot(stat, 10.0)                      # + 1 % de 1000
    ub = math.hypot(stat, 13.0)                      # + 1 % de 1300
    assert res["avant"]["volume_moyen_mm3"] == 1000.0 and res["apres"]["volume_moyen_mm3"] == 1300.0
    assert res["delta_volume_mm3"] == pytest.approx(300.0)
    assert res["rapport_volumes"] == pytest.approx(1.3) and res["gonflement_pct"] == pytest.approx(30.0)
    assert res["incertitude_delta_volume_mm3"] == pytest.approx(math.hypot(ua, ub))
    assert res["incertitude_elargie_gonflement_pct"] == pytest.approx(
        100 * 2 * 1.3 * math.sqrt((ua / 1000) ** 2 + (ub / 1300) ** 2))
    assert res["significatif"] is True and res["avertissements"] == []


def test_l_erreur_commune_donne_une_borne_plus_optimiste():
    res = g.comparer(AVANT, APRES, sys_pct=2.0)
    assert res["incertitude_elargie_gonflement_erreur_commune_pct"] < res["incertitude_elargie_gonflement_pct"]
    # sans part systématique les deux bornes sont identiques
    res0 = g.comparer(AVANT, APRES, sys_pct=0.0)
    assert res0["incertitude_elargie_gonflement_erreur_commune_pct"] == pytest.approx(res0["incertitude_elargie_gonflement_pct"])


def test_variation_non_significative():
    proche = [_rapport(f"b{i}", v) for i, v in enumerate([1008.0, 1012.0, 1004.0])]
    res = g.comparer(AVANT, proche, sys_pct=2.0)
    assert res["gonflement_pct"] == pytest.approx(0.8, abs=0.01) and res["significatif"] is False
    assert "NON significative" in g.resume_texte(res)


def test_sans_incertitude_systematique_l_incertitude_est_declaree_sous_estimee():
    res = g.comparer(AVANT, APRES)
    assert res["incertitude_elargie_gonflement_pct"] is not None
    assert sum("SOUS-ESTIMÉE" in a for a in res["avertissements"]) == 2          # avant et après
    assert res["avant"]["incertitude_totale_mm3"] == pytest.approx(10.0 / math.sqrt(3))


def test_un_seul_scan_utilise_la_repetabilite_de_la_validation():
    validation = {"repetabilite_pct": 0.5, "systematique_pct": 1.0, "n_validation": 10, "fichier": "v.json"}
    res = g.comparer([_rapport("a", 1000.0)], [_rapport("b", 1200.0)], validation=validation)
    assert res["avant"]["incertitude_statistique_mm3"] == pytest.approx(5.0)       # 0,5 % de 1000
    assert res["avant"]["incertitude_systematique_mm3"] == pytest.approx(10.0)     # 1 % de 1000
    assert res["avant"]["ecart_type_mm3"] is None
    assert any("un seul scan : répétabilité tirée de la validation" in a for a in res["avertissements"])


def test_un_seul_scan_sans_validation_n_a_pas_d_incertitude():
    res = g.comparer([_rapport("a", 1000.0)], [_rapport("b", 1200.0)], sys_pct=1.0)
    assert res["incertitude_elargie_gonflement_pct"] is None and res["significatif"] is None
    assert "incertitude non calculable" in g.resume_texte(res)
    assert any("répétabilité inconnue" in a for a in res["avertissements"])


def test_la_validation_est_remplacee_par_la_valeur_donnee_a_la_main():
    validation = {"repetabilite_pct": 0.5, "systematique_pct": 5.0, "n_validation": 10, "fichier": "v.json"}
    res = g.comparer(AVANT, APRES, validation=validation, sys_pct=1.0)
    assert res["avant"]["incertitude_systematique_mm3"] == pytest.approx(10.0)       # 1 %, pas 5 %


def test_autres_grandeurs():
    avant = [_rapport(f"a{i}", 1000.0, hauteur_mm=h, longueur_mm=50.0) for i, h in enumerate([40.0, 40.2, 39.8])]
    apres = [_rapport(f"b{i}", 1300.0, hauteur_mm=h, longueur_mm=50.0) for i, h in enumerate([52.0, 52.2, 51.8])]
    avant[0]["mesures"]["largeur_mm"] = 30.0                  # présente dans un seul rapport : ignorée
    res = g.comparer(avant, apres, sys_pct=0.0)
    h = res["autres_grandeurs"]["hauteur_mm"]
    assert h["avant"] == pytest.approx(40.0) and h["apres"] == pytest.approx(52.0) and h["variation_pct"] == pytest.approx(30.0)
    assert h["significatif"] is True
    assert res["autres_grandeurs"]["longueur_mm"]["significatif"] is False         # écart-type nul, variation nulle
    assert "largeur_mm" not in res["autres_grandeurs"]


@pytest.mark.parametrize("avant, apres, mot", [
    ([], APRES, "au moins un scan"),
    (AVANT, [], "au moins un scan"),
    ([{"nom": "x", "unite_longueur": "mm", "mesures": {}}], APRES, "sans volume"),
    ([dict(_rapport("x", 5.0), unite_longueur="unite_colmap")], APRES, "pas en millimètres"),
    ([_rapport("x", 0.0)], APRES, "nul ou négatif"),
])
def test_entrees_invalides(avant, apres, mot):
    with pytest.raises(g.ErreurGonflement, match=mot):
        g.comparer(avant, apres)


def test_avertissements_de_coherence():
    enorme = g.comparer(AVANT, [_rapport("b", 5000.0)], sys_pct=1.0, validation={"repetabilite_pct": 1.0, "systematique_pct": 1.0,
                                                                                   "n_validation": 5, "fichier": "v"})
    assert any("multiplié par" in a for a in enorme["avertissements"])
    melange = g.comparer(AVANT, [_rapport("b", 1300.0, methode="facteur_manuel")], sys_pct=1.0,
                         validation={"repetabilite_pct": 1.0, "systematique_pct": 1.0, "n_validation": 5, "fichier": "v"})
    assert any("Méthodes d'échelle différentes" in a for a in melange["avertissements"])


# --------------------------------------------------------------- validation
def _fichier_validation(tmp_path, biais=40.0, u_ref=30.0, cv=0.6):
    v = {"resultats": [{"grandeur": "volume_mm3", "reference": 8000.0, "biais": biais, "incertitude_reference": u_ref,
                        "cv_pct": cv, "n": 10}]}
    chemin = tmp_path / "validation_resultats.json"
    chemin.write_text(json.dumps(v), encoding="utf-8")
    return str(chemin)


def test_lecture_de_la_validation(tmp_path):
    v = g.lire_validation(_fichier_validation(tmp_path))
    assert v["repetabilite_pct"] == 0.6 and v["n_validation"] == 10
    assert v["systematique_pct"] == pytest.approx(100 * math.hypot(40.0, 30.0) / 8000.0)       # 0,625 %
    with pytest.raises(g.ErreurGonflement, match="absent"):
        g.lire_validation(_fichier_validation(tmp_path), grandeur="hauteur_mm")
    with pytest.raises(g.ErreurGonflement, match="illisible"):
        g.lire_validation(str(tmp_path / "rien.json"))
    (tmp_path / "casse.json").write_text("{", encoding="utf-8")
    with pytest.raises(g.ErreurGonflement, match="illisible"):
        g.lire_validation(str(tmp_path / "casse.json"))


# ------------------------------------------------------------------ fichiers
def test_ecriture(tmp_path):
    res = g.comparer(AVANT, APRES, sys_pct=1.0)
    for _ in range(2):
        j, c = g.ecrire(res, str(tmp_path / "out"), "gateau_A")
    assert json.load(open(j, encoding="utf-8"))["nom"] == "gateau_A"
    brut = open(c, "rb").read()
    assert brut.startswith(b"\xef\xbb\xbf") and brut.count(b"\xef\xbb\xbf") == 1
    lignes = list(csv.reader(open(c, encoding="utf-8-sig"), delimiter=";"))
    assert len(lignes) == 3 and lignes[1][1] == "gateau_A" and lignes[1][6] == "30" and lignes[1][8] == "oui"
    assert "," in lignes[1][7]                                                     # virgule décimale


def test_cli(tmp_path, capsys):
    def ecrire_rapports(prefixe, volumes, fiable=True):
        for i, v in enumerate(volumes):
            d = tmp_path / prefixe / f"s{i}"
            d.mkdir(parents=True)
            r = _rapport(f"{prefixe}{i}", v)
            r["fiable"] = fiable
            (d / "rapport.json").write_text(json.dumps(r), encoding="utf-8")
        return str(tmp_path / prefixe)
    avant = ecrire_rapports("avant", [1000.0, 1010.0, 990.0])
    apres = ecrire_rapports("apres", [1300.0, 1310.0, 1290.0])
    sortie = str(tmp_path / "res")
    assert g.main(["--avant", avant, "--apres", apres, "--validation", _fichier_validation(tmp_path), "--nom", "essai",
                   "--sortie", sortie]) == 0
    sortie_texte = capsys.readouterr().out
    assert "GONFLEMENT : +30.00 %" in sortie_texte and "SIGNIFICATIVE" in sortie_texte
    assert os.path.isfile(os.path.join(sortie, "gonflement.csv"))
    # un état entièrement non fiable est exclu, donc il ne reste plus rien à comparer
    inutilisable = ecrire_rapports("mauvais", [1.0, 2.0], fiable=False)
    assert g.main(["--avant", inutilisable, "--apres", apres]) == 1
    assert "ERREUR" in capsys.readouterr().out
    assert g.main(["--avant", avant, "--apres", apres, "--validation", str(tmp_path / "absente.json")]) == 1


# ----------------------------------------- jumeau numérique : gâteau simulé
def test_gateau_simule_avant_apres(tmp_path):
    """Chaîne complète : maillages d'un « gâteau » (cylindre) avant/après, scans bruités et mis à l'échelle,
    mesurés puis comparés. Le gonflement vrai doit tomber dans l'incertitude annoncée."""
    rng = np.random.default_rng(3)

    def scans(prefixe, r, h, n_scans=4):
        V0, F, volume_vrai = formes.cylindre(r=r, h=h, n=96)
        for i in range(n_scans):
            vrai = synth.SimilitudeVraie(s=140 + 10 * rng.random(), axe="x", angle=60 + 30 * rng.random())
            e = synth.echelle_vraie(vrai)
            erreur = 1 + rng.normal(0, 0.001)
            e["transformation_4x4"] = (np.diag([erreur] * 3 + [1.0]) @ np.array(e["transformation_4x4"])).tolist()
            e["mm_par_unite"] *= erreur
            stl, chemin_e = str(tmp_path / f"{prefixe}{i}.stl"), str(tmp_path / f"{prefixe}{i}_e.json")
            trimesh.Trimesh(vrai.colmap_depuis_mm(V0 + rng.normal(0, 0.04, V0.shape)), F, process=False).export(stl)
            open(chemin_e, "w").write(json.dumps(e))
            mesure.mesurer(stl, chemin_e, str(tmp_path / prefixe / f"{prefixe}{i}"), f"{prefixe}{i}")
        return volume_vrai

    v_avant = scans("avant", 25.0, 40.0)
    v_apres = scans("apres", 26.0, 52.0)
    vrai_pct = 100 * (v_apres / v_avant - 1)
    avant, _ = vp.charger_rapports([str(tmp_path / "avant")])
    apres, _ = vp.charger_rapports([str(tmp_path / "apres")])
    res = g.comparer(avant, apres, sys_pct=0.5)
    assert res["significatif"] is True
    assert abs(res["gonflement_pct"] - vrai_pct) < res["incertitude_elargie_gonflement_pct"]      # la vérité est dans l'incertitude
    assert abs(res["gonflement_pct"] - vrai_pct) < 1.0                                           # et la mesure est juste à 1 point près
    assert res["autres_grandeurs"]["hauteur_robuste_mm"]["variation_pct"] == pytest.approx(30.0, abs=1.0)
