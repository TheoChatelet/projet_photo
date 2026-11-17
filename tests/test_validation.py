"""Tests de la validation de précision : statistiques vérifiées contre scipy, fichiers, ligne de commande."""
import csv
import json
import math
import os

import numpy as np
import pytest
from scipy import stats

import formes
import mesure
import synth
import validation_precision as vp


# ------------------------------------------------------------------ nombres
@pytest.mark.parametrize("texte, attendu", [("12,5", 12.5), (" 12.5 ", 12.5), ("1 234,5", 1234.5), ("1 234,5", 1234.5),
                                            ("-3", -3.0), ("1e-3", 0.001)])
def test_lire_nombre(texte, attendu):
    assert vp.lire_nombre(texte) == pytest.approx(attendu)


@pytest.mark.parametrize("mauvais", ["abc", "", "12,5,6", "nan", "inf", "-inf"])
def test_lire_nombre_refuse(mauvais):
    with pytest.raises(vp.ErreurValidation):
        vp.lire_nombre(mauvais)


# ----------------------------------------------------------------- référence
def _ecrire(tmp_path, texte, nom="ref.csv", encodage="utf-8"):
    chemin = tmp_path / nom
    chemin.write_bytes(texte.encode(encodage))
    return str(chemin)


def test_reference_excel_francais(tmp_path):
    """Séparateur ;, virgule décimale, BOM, commentaires et lignes vides : ce que sort Excel en français."""
    chemin = _ecrire(tmp_path, "﻿# commentaire\n\ngrandeur;valeur_ref;incertitude_ref;tolerance;commentaire\n"
                               "hauteur_mm;40,11;0,02;0,5;5 mesures\nvolume_mm3;78790,5;60;1500;\n")
    ref = vp.lire_reference(chemin)
    assert ref[0] == {"grandeur": "hauteur_mm", "valeur_ref": 40.11, "incertitude_ref": 0.02, "tolerance": 0.5,
                      "commentaire": "5 mesures"}
    assert ref[1]["valeur_ref"] == 78790.5 and ref[1]["commentaire"] == ""


def test_reference_separateur_virgule_et_point_decimal(tmp_path):
    ref = vp.lire_reference(_ecrire(tmp_path, "grandeur,valeur_ref,incertitude_ref\nhauteur_mm,40.11,0.02\n"))
    assert ref[0]["valeur_ref"] == 40.11 and ref[0]["tolerance"] is None


def test_reference_separateur_tabulation(tmp_path):
    ref = vp.lire_reference(_ecrire(tmp_path, "grandeur\tvaleur_ref\nhauteur_mm\t40,11\n"))
    assert ref[0]["valeur_ref"] == 40.11 and ref[0]["incertitude_ref"] is None


def test_la_virgule_separateur_n_est_pas_prise_pour_une_decimale(tmp_path):
    with pytest.raises(vp.ErreurValidation, match="3 champs pour 2 colonnes"):
        vp.lire_reference(_ecrire(tmp_path, "grandeur,valeur_ref\nhauteur_mm,40,11\n"))


@pytest.mark.parametrize("texte, mot", [
    ("", "en-tête"),
    ("grandeur;valeur_ref\n", "en-tête"),
    ("grandeur valeur_ref\nhauteur_mm 40\n", "séparateur"),
    ("grandeur;autre\nhauteur_mm;40\n", "valeur_ref"),
    ("grandeur;valeur_ref\nhauteur_mm;abc\n", "pas un nombre"),
    ("grandeur;valeur_ref\nhauteur_mm;-4\n", "positive"),
    ("grandeur;valeur_ref\nhauteur_mm;0\n", "positive"),
    ("grandeur;valeur_ref;incertitude_ref\nhauteur_mm;40;-1\n", "négative"),
    ("grandeur;valeur_ref;tolerance\nhauteur_mm;40;0\n", "tolérance nulle"),
    ("grandeur;valeur_ref\n;40\n", "vide"),
    ("grandeur;valeur_ref\nhauteur_mm;40\nhauteur_mm;41\n", "deux fois"),
])
def test_reference_invalide(tmp_path, texte, mot):
    with pytest.raises(vp.ErreurValidation, match=mot):
        vp.lire_reference(_ecrire(tmp_path, texte))


def test_reference_absente(tmp_path):
    with pytest.raises(vp.ErreurValidation, match="introuvable"):
        vp.lire_reference(str(tmp_path / "rien.csv"))


# ------------------------------------------------------------ statistiques
X = np.array([10.12, 10.21, 9.93, 10.04, 10.08, 10.17, 9.99, 10.11])


def test_statistiques_contre_scipy():
    ref = 10.0
    s = vp.statistiques(X, ref, u_ref=0.0, k=2.0, tolerance=0.5)
    n = len(X)
    assert s["n"] == n and s["moyenne"] == pytest.approx(X.mean())
    assert s["biais"] == pytest.approx(X.mean() - ref) and s["biais_pct"] == pytest.approx(100 * (X.mean() - ref) / ref)
    assert s["ecart_type"] == pytest.approx(X.std(ddof=1))
    assert s["cv_pct"] == pytest.approx(100 * X.std(ddof=1) / X.mean())
    assert s["limite_repetabilite"] == pytest.approx(2.77 * X.std(ddof=1))
    assert s["eqm"] == pytest.approx(math.sqrt(np.mean((X - ref) ** 2)))
    assert s["erreur_max_absolue"] == pytest.approx(np.abs(X - ref).max())
    # intervalle de confiance et p-valeur : mêmes résultats que scipy
    ic = stats.t.interval(0.95, n - 1, loc=X.mean(), scale=stats.sem(X))
    assert s["ic95_moyenne"] == pytest.approx(list(ic))
    assert s["p_biais"] == pytest.approx(stats.ttest_1samp(X, ref).pvalue, rel=1e-9)
    assert s["biais_significatif"] == bool(stats.ttest_1samp(X, ref).pvalue < 0.05)


def test_incertitude_elargie_et_conformite():
    s = vp.statistiques(X, 10.0, u_ref=0.03, k=2.0, tolerance=0.5)
    u = math.sqrt(X.std(ddof=1) ** 2 + (X.mean() - 10.0) ** 2 + 0.03 ** 2)
    assert s["incertitude_scan"] == pytest.approx(u) and s["incertitude_elargie"] == pytest.approx(2 * u)
    assert s["incertitude_elargie_pct"] == pytest.approx(100 * 2 * u / 10.0)
    assert s["conforme"] is True
    assert vp.statistiques(X, 10.0, 0.03, 2.0, tolerance=0.05)["conforme"] is False       # tolérance trop serrée
    assert vp.statistiques(X, 10.0, 0.03, 2.0, tolerance=None)["conforme"] is None
    assert vp.statistiques(X, 10.0, 0.03, 3.0)["incertitude_elargie"] == pytest.approx(3 * u)


def test_l_incertitude_de_la_reference_rend_le_biais_moins_significatif():
    biaise = np.array([10.30, 10.28, 10.33, 10.29, 10.31, 10.32])
    p_sans = vp.statistiques(biaise, 10.0, 0.0)["p_biais"]
    p_avec = vp.statistiques(biaise, 10.0, 0.5)["p_biais"]       # référence très incertaine : on ne peut plus conclure
    assert p_sans < 0.001 and p_avec > 0.05


def test_un_biais_net_est_detecte_et_pas_un_biais_nul():
    rng = np.random.default_rng(1)
    juste = 40.0 + rng.normal(0, 0.05, 12)
    biaise = 40.3 + rng.normal(0, 0.05, 12)
    assert not vp.statistiques(juste, 40.0, 0.01)["biais_significatif"]
    assert vp.statistiques(biaise, 40.0, 0.01)["biais_significatif"]


def test_cas_limites_des_statistiques():
    un = vp.statistiques([10.2], 10.0, 0.01, tolerance=0.5)
    assert un["n"] == 1 and un["ecart_type"] is None and un["incertitude_elargie"] is None and un["conforme"] is None
    assert un["biais"] == pytest.approx(0.2)
    deux = vp.statistiques([10.0, 10.2], 10.0)
    assert deux["ecart_type"] == pytest.approx(math.sqrt(0.02))
    identiques = vp.statistiques([10.2] * 5, 10.0, 0.0)            # écart-type nul
    assert identiques["ecart_type"] == 0 and identiques["p_biais"] == 0.0 and identiques["biais_significatif"]
    parfaits = vp.statistiques([10.0] * 5, 10.0, 0.0)
    assert parfaits["p_biais"] == 1.0 and not parfaits["biais_significatif"]
    with pytest.raises(vp.ErreurValidation):
        vp.statistiques([], 10.0)


def test_valeurs_atypiques():
    assert vp.valeurs_atypiques([10, 10.1, 9.9, 10.05, 10.0, 10.02, 13.0]) == [6]
    assert vp.valeurs_atypiques([10, 10.1, 9.9, 20.0]) == []               # n < 5 : on ne juge pas
    assert vp.valeurs_atypiques([10.0] * 8) == []                           # MAD nulle
    assert vp.valeurs_atypiques(X) == []                                    # dispersion normale


# ---------------------------------------------------- rapports et analyse
def _rapport(nom, **mesures):
    return {"nom": nom, "unite_longueur": "mm", "fiable": True, "avertissements": [], "mesures": mesures}


def _dossier_rapports(tmp_path, rapports):
    for r in rapports:
        d = tmp_path / "mesures" / r["nom"]
        d.mkdir(parents=True)
        (d / "rapport.json").write_text(json.dumps(r), encoding="utf-8")
    return str(tmp_path / "mesures")


def test_chargement_des_rapports_et_exclusions(tmp_path):
    bons = [_rapport(f"s{i}", hauteur_mm=40.0 + i / 100) for i in range(3)]
    non_fiable = dict(_rapport("douteux", hauteur_mm=41), fiable=False, avertissements=["échelle à vérifier"])
    sans_echelle = dict(_rapport("brut", hauteur_mm=0.4), unite_longueur="unite_colmap")
    dossier = _dossier_rapports(tmp_path, bons + [non_fiable, sans_echelle])
    (tmp_path / "mesures" / "casse").mkdir()
    (tmp_path / "mesures" / "casse" / "rapport.json").write_text("{pas du json", encoding="utf-8")
    (tmp_path / "mesures" / "autre").mkdir()
    (tmp_path / "mesures" / "autre" / "rapport.json").write_text('{"truc": 1}', encoding="utf-8")

    retenus, ignores = vp.charger_rapports([dossier])
    assert sorted(r["nom"] for r in retenus) == ["s0", "s1", "s2"]
    raisons = " | ".join(r for _, r in ignores)
    assert "non fiable" in raisons and "pas en millimètres" in raisons and "illisible" in raisons and "pas un rapport" in raisons
    assert len(ignores) == 4
    # avec l'option, le scan non fiable est gardé
    retenus2, _ = vp.charger_rapports([dossier], inclure_non_fiables=True)
    assert "douteux" in [r["nom"] for r in retenus2]
    # motifs et fichiers directs, sans doublon
    g, _ = vp.charger_rapports([os.path.join(dossier, "s*", "rapport.json"), os.path.join(dossier, "s0", "rapport.json")])
    assert sorted(r["nom"] for r in g) == ["s0", "s1", "s2"]


def test_motif_qui_designe_des_dossiers(tmp_path):
    """Le cas du protocole : colmap_workspace_scan*/mesures (PowerShell ne développe pas le *)."""
    for i in range(3):
        d = tmp_path / f"colmap_workspace_scan{i}" / "mesures" / f"scan{i}"
        d.mkdir(parents=True)
        (d / "rapport.json").write_text(json.dumps(_rapport(f"scan{i}", hauteur_mm=40.0)), encoding="utf-8")
    (tmp_path / "colmap_workspace_autre" / "mesures").mkdir(parents=True)          # ne doit pas être pris
    motif = str(tmp_path / "colmap_workspace_scan*" / "mesures")
    retenus, ignores = vp.charger_rapports([motif])
    assert sorted(r["nom"] for r in retenus) == ["scan0", "scan1", "scan2"] and not ignores
    # motif + dossier qui contient déjà les mêmes rapports : pas de doublon
    retenus2, _ = vp.charger_rapports([motif, str(tmp_path / "colmap_workspace_scan1")])
    assert len(retenus2) == 3


def test_aucun_rapport(tmp_path):
    with pytest.raises(vp.ErreurValidation, match="Aucun rapport"):
        vp.charger_rapports([str(tmp_path)])


def test_analyse_et_avertissements():
    rng = np.random.default_rng(0)
    rapports = [dict(_rapport(f"s{i}", hauteur_mm=40.1 + rng.normal(0, 0.04), volume_mm3=78800 + rng.normal(0, 90)),
                     _fichier=f"s{i}.json") for i in range(12)]
    rapports[3]["mesures"]["hauteur_mm"] = 41.4                      # un scan raté
    refs = [{"grandeur": "hauteur_mm", "valeur_ref": 40.11, "incertitude_ref": 0.02, "tolerance": 0.5, "commentaire": ""},
            {"grandeur": "volume_mm3", "valeur_ref": 78790.0, "incertitude_ref": None, "tolerance": None, "commentaire": ""},
            {"grandeur": "largeur_mm", "valeur_ref": 50.0, "incertitude_ref": 0.02, "tolerance": None, "commentaire": ""}]
    a = vp.analyser(refs, rapports)
    assert [r["grandeur"] for r in a["resultats"]] == ["hauteur_mm", "volume_mm3"]
    h = a["resultats"][0]
    assert h["atypiques"] == ["s3"] and h["unite"] == "mm" and h["n"] == 12
    texte = " | ".join(a["avertissements"])
    assert "atypique" in texte and "largeur_mm" in texte and "incertitude de la référence non renseignée" in texte
    assert len(a["detail"]) == 24 and sum(d["atypique"] for d in a["detail"]) == 1


def test_analyse_signale_les_grandeurs_manquantes_dans_certains_rapports():
    rapports = [dict(_rapport(f"s{i}", hauteur_mm=40.0), _fichier=f"s{i}") for i in range(5)]
    del rapports[0]["mesures"]["hauteur_mm"]
    a = vp.analyser([{"grandeur": "hauteur_mm", "valeur_ref": 40.0, "incertitude_ref": 0.02, "tolerance": None,
                      "commentaire": ""}], rapports)
    assert a["resultats"][0]["n"] == 4 and any("absent de 1 rapport" in w for w in a["avertissements"])


def test_analyse_sans_grandeur_commune_est_une_erreur():
    rapports = [dict(_rapport("s", hauteur_mm=40.0), _fichier="s")]
    with pytest.raises(vp.ErreurValidation, match="Aucune grandeur"):
        vp.analyser([{"grandeur": "volume_mm3", "valeur_ref": 1.0, "incertitude_ref": 0, "tolerance": None,
                      "commentaire": ""}], rapports)
    with pytest.raises(vp.ErreurValidation, match="Aucun rapport"):
        vp.analyser([], [])


def test_alerte_quand_la_reference_limite_la_validation():
    rapports = [dict(_rapport(f"s{i}", hauteur_mm=40.0 + 0.001 * i), _fichier=f"s{i}") for i in range(5)]
    a = vp.analyser([{"grandeur": "hauteur_mm", "valeur_ref": 40.0, "incertitude_ref": 0.05, "tolerance": 0.5,
                      "commentaire": ""}], rapports)
    assert any("limitée par la référence" in w for w in a["avertissements"])


# ------------------------------------------------------------------ sorties
def _analyse_type(tmp_path):
    rng = np.random.default_rng(2)
    rapports = [dict(_rapport(f"s{i}", hauteur_mm=40.1 + rng.normal(0, 0.04), diametre_equivalent_base_mm=50.0 + rng.normal(0, 0.05)),
                     _fichier=f"s{i}.json") for i in range(10)]
    refs = [{"grandeur": "hauteur_mm", "valeur_ref": 40.11, "incertitude_ref": 0.02, "tolerance": 0.5, "commentaire": ""},
            {"grandeur": "diametre_equivalent_base_mm", "valeur_ref": 50.02, "incertitude_ref": 0.02, "tolerance": 0.05, "commentaire": ""}]
    return vp.analyser(refs, rapports, condition="pièce repositionnée entre scans")


def test_ecriture_des_fichiers(tmp_path):
    a = _analyse_type(tmp_path)
    fichiers = vp.ecrire_resultats(a, str(tmp_path / "out"), ignores=[("x.json", "non fiable")])
    noms = {os.path.basename(f) for f in fichiers}
    assert {"validation_resultats.csv", "validation_detail.csv", "validation_resultats.json",
            "validation_rapport.md", "validation_graphique.png"} <= noms
    assert all(os.path.getsize(f) > 0 for f in fichiers)
    brut = open(tmp_path / "out" / "validation_resultats.csv", "rb").read()
    assert brut.startswith(b"\xef\xbb\xbf")
    lignes = list(csv.reader(open(tmp_path / "out" / "validation_resultats.csv", encoding="utf-8-sig"), delimiter=";"))
    assert lignes[0] == vp.COLONNES_RESULTATS and len(lignes) == 3
    ligne = dict(zip(lignes[0], lignes[1]))
    assert "," in ligne["moyenne"] and ligne["verdict"] in ("conforme", "NON conforme")
    assert json.load(open(tmp_path / "out" / "validation_resultats.json", encoding="utf-8"))["nb_scans"] == 10
    detail = list(csv.reader(open(tmp_path / "out" / "validation_detail.csv", encoding="utf-8-sig"), delimiter=";"))
    assert len(detail) == 1 + 20


def test_rapport_markdown():
    a = _analyse_type(None)
    md = vp.rapport_markdown(a, ignores=[("x.json", "non fiable")])
    for mot in ("# Validation", "| Grandeur |", "hauteur_mm", "Justesse", "Répétabilité", "Incertitude élargie",
                "pièce repositionnée entre scans", "Rapport exclu", "Méthode"):
        assert mot in md, mot
    assert md.count("\n|---|") == 1


def test_graphique_sans_matplotlib(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert vp.tracer_graphique(_analyse_type(None), str(tmp_path / "g.png")) is False
    assert not (tmp_path / "g.png").exists()


# --------------------------------------------------------- volume étalon
def test_volume_de_reference():
    v, u = vp.volume_reference("cylindre", {"diametre": 50.0, "hauteur": 40.0}, 0.02)
    assert v == pytest.approx(math.pi * 50 ** 2 * 40 / 4)
    assert u / v == pytest.approx(math.sqrt((2 * 0.02 / 50) ** 2 + (0.02 / 40) ** 2))
    v2, u2 = vp.volume_reference("parallelepipede", {"a": 30.0, "b": 20.0, "c": 10.0}, 0.02)
    assert v2 == 6000 and u2 / v2 == pytest.approx(math.sqrt((0.02 / 30) ** 2 + (0.02 / 20) ** 2 + (0.02 / 10) ** 2))
    v3, u3 = vp.volume_reference("sphere", {"diametre": 30.0}, 0.02)
    assert v3 == pytest.approx(math.pi * 30 ** 3 / 6) and u3 / v3 == pytest.approx(3 * 0.02 / 30)
    assert vp.volume_reference("cylindre", {"diametre": 50.0, "hauteur": 40.0}, 0)[1] == 0
    with pytest.raises(vp.ErreurValidation):
        vp.volume_reference("cube", {}, 0.02)
    with pytest.raises(vp.ErreurValidation):
        vp.volume_reference("cylindre", {"diametre": -1.0, "hauteur": 40.0}, 0.02)
    with pytest.raises(vp.ErreurValidation):
        vp.volume_reference("cylindre", {"diametre": 5.0, "hauteur": 40.0}, -0.02)


# --------------------------------------------------------- ligne de commande
def test_cli_modele_puis_analyse(tmp_path, capsys):
    modele = str(tmp_path / "ref.csv")
    assert vp.main(["modele", "--sortie", modele]) == 0
    refs = vp.lire_reference(modele)                    # le modèle fourni est lui-même valide
    assert {r["grandeur"] for r in refs} >= {"hauteur_mm", "volume_mm3"}
    rapports = [_rapport(f"s{i}", hauteur_mm=40.11 + 0.02 * (i - 3), diametre_equivalent_base_mm=50.02, volume_mm3=78790.0 + 10 * i)
                for i in range(7)]
    dossier = _dossier_rapports(tmp_path, rapports)
    sortie = str(tmp_path / "res")
    assert vp.main(["analyser", "--reference", modele, "--rapports", dossier, "--sortie", sortie, "--condition", "test"]) == 0
    assert os.path.isfile(os.path.join(sortie, "validation_rapport.md"))
    assert "Validation de la précision" in capsys.readouterr().out
    # erreurs : référence absente, aucun rapport, sous-commande manquante
    assert vp.main(["analyser", "--reference", str(tmp_path / "x.csv"), "--rapports", dossier]) == 1
    assert vp.main(["analyser", "--reference", modele, "--rapports", str(tmp_path / "vide")]) == 1
    assert vp.main([]) == 1


def test_cli_ref_volume(capsys):
    assert vp.main(["ref-volume", "cylindre", "--diametre", "50,02", "--hauteur", "40,11", "--u", "0,02"]) == 0
    sortie = capsys.readouterr().out.split()[0].split(";")
    attendu, u = vp.volume_reference("cylindre", {"diametre": 50.02, "hauteur": 40.11}, 0.02)
    assert sortie[0] == "volume_mm3"
    assert float(sortie[1].replace(",", ".")) == pytest.approx(attendu, rel=1e-7)
    assert float(sortie[2].replace(",", ".")) == pytest.approx(u, rel=1e-2)       # 3 chiffres significatifs
    assert vp.main(["ref-volume", "cylindre", "--diametre", "50"]) == 1          # hauteur manquante
    assert vp.main(["ref-volume", "cylindre", "--diametre", "abc", "--hauteur", "4"]) == 1


# ------------------------------------------------- jumeau numérique du protocole
def test_protocole_complet_sur_scans_simules(tmp_path):
    """Le protocole entier sur des « scans » simulés : N maillages d'un cylindre étalon, chacun avec un bruit de surface et
    une erreur d'échelle aléatoires, mesurés par mesure.py puis validés contre les dimensions exactes."""
    r_mm, h_mm, n_pans = 25.0, 40.0, 96
    V0, F, vol_vrai = formes.cylindre(r=r_mm, h=h_mm, n=n_pans)
    rng = np.random.default_rng(42)
    dossier = tmp_path / "mesures"
    for i in range(10):
        vrai = synth.SimilitudeVraie(s=130 + 20 * rng.random(), axe="x", angle=60 + 30 * rng.random())
        bruit = rng.normal(0, 0.05, V0.shape)                       # rugosité de reconstruction 0,05 mm
        erreur_echelle = 1 + rng.normal(0, 0.001)                   # 0,1 % d'erreur d'échelle propre à chaque scan
        V = vrai.colmap_depuis_mm(V0 + bruit)
        import trimesh
        stl = str(tmp_path / f"scan{i}.stl")
        trimesh.Trimesh(V, F, process=False).export(stl)
        echelle = synth.echelle_vraie(vrai)
        echelle["mm_par_unite"] *= erreur_echelle
        # une erreur d'échelle multiplie toutes les longueurs : x_mm' = e * x_mm
        echelle["transformation_4x4"] = (np.diag([erreur_echelle] * 3 + [1.0]) @ np.array(echelle["transformation_4x4"])).tolist()
        chemin_e = tmp_path / f"echelle{i}.json"
        chemin_e.write_text(json.dumps(echelle), encoding="utf-8")
        mesure.mesurer(stl, str(chemin_e), str(dossier / f"scan{i}"), f"scan{i}")
    refs = [{"grandeur": "hauteur_mm", "valeur_ref": h_mm, "incertitude_ref": 0.02, "tolerance": 0.5, "commentaire": ""},
            {"grandeur": "hauteur_robuste_mm", "valeur_ref": h_mm, "incertitude_ref": 0.02, "tolerance": 0.5, "commentaire": ""},
            {"grandeur": "volume_mm3", "valeur_ref": vol_vrai, "incertitude_ref": 0.002 * vol_vrai, "tolerance": 0.02 * vol_vrai,
             "commentaire": ""}]
    rapports, ignores = vp.charger_rapports([str(dossier)])
    assert len(rapports) == 10 and not ignores
    a = vp.analyser(refs, rapports)
    h, hr, v = a["resultats"]
    # Le bruit de surface (0,05 mm) biaise la hauteur brute vers le haut (extrême de ~100 sommets du dessus plat)
    # et la hauteur lissée corrige l'essentiel de ce biais : c'est ce que la validation doit faire apparaître
    assert h["biais"] > 0.05 and h["biais_significatif"]
    assert abs(hr["biais"]) < abs(h["biais"]) / 2 and abs(hr["biais_pct"]) < 0.15
    # Le scanner simulé est juste en volume ; la dispersion injectée (0,1 % d'échelle) se retrouve dans la répétabilité
    assert abs(v["biais_pct"]) < 0.5
    assert 0.05 < v["cv_pct"] < 1.0          # 0,1 % d'échelle = 0,3 % de volume
    assert h["conforme"] is True and hr["conforme"] is True and v["conforme"] is True
    fichiers = vp.ecrire_resultats(a, str(tmp_path / "validation"))
    assert any(f.endswith("validation_rapport.md") for f in fichiers)
