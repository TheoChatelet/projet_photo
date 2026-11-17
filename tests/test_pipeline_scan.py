"""Orchestrateur d'un scan : on vérifie l'ordre, les dossiers et la reprise, avec des étapes simulées
(COLMAP, OpenMVS et Open3D ne sont pas nécessaires : chaque étape a ses propres tests)."""
import os
import sys
import types

import pytest

import pipeline_scan as ps


@pytest.fixture
def etapes(monkeypatch, tmp_path):
    appels = []

    def faux_module(nom, **fonctions):
        mod = types.ModuleType(nom)
        for k, f in fonctions.items():
            setattr(mod, k, f)
        monkeypatch.setitem(sys.modules, nom, mod)
        return mod

    faux_module("main", executer_pipeline_securise=lambda **kw: appels.append(("reconstruction", kw)))
    mvs = faux_module("densification_openmvs", OPENMVS_BIN="defaut",
                      densifier=lambda **kw: appels.append(("densification", kw)))
    faux_module("maillage", mailler_et_preparer_impression=lambda **kw: appels.append(("maillage", kw)))
    faux_module("echelle", lancer_echelle=lambda *a, **kw: appels.append(("echelle", a)))
    faux_module("mesure", mesurer=lambda *a, **kw: appels.append(("mesure", a, kw)) or {"nom": a[3]},
                         resume_texte=lambda r: f"résumé {r['nom']}")
    (tmp_path / "photos").mkdir()
    return appels, mvs, tmp_path


def test_ordre_et_dossiers(etapes):
    appels, mvs, tmp = etapes
    r = ps.lancer(str(tmp / "photos"), "scan01", racine_travail=str(tmp), openmvs_bin=r"D:\OpenMVS")
    assert [a[0] for a in appels] == ["reconstruction", "densification", "maillage", "echelle", "mesure"]
    travail = str(tmp / "colmap_workspace_scan01")
    assert appels[0][1]["dossier_travail"] == travail and appels[0][1]["dossier_source"] == str(tmp / "photos")
    assert appels[0][1]["repartir_de_zero"] is True and appels[0][1]["camera_unique"] is True
    assert appels[1][1]["dossier_travail"] == travail and mvs.OPENMVS_BIN == r"D:\OpenMVS"
    assert appels[2][1]["afficher_interface"] is False          # jamais de fenêtre dans une série de scans
    assert appels[3][1][0] == travail
    mesure_args = appels[4][1]
    assert mesure_args[0] == os.path.join(travail, "modele_pour_impression.stl")
    assert mesure_args[1] == os.path.join(travail, "echelle.json")
    assert mesure_args[2] == os.path.join(travail, "mesures", "scan01") and mesure_args[3] == "scan01"
    assert r == {"nom": "scan01"}


def test_deux_scans_ont_des_dossiers_distincts(etapes):
    appels, _, tmp = etapes
    ps.lancer(str(tmp / "photos"), "scanA", racine_travail=str(tmp))
    ps.lancer(str(tmp / "photos"), "scanB", racine_travail=str(tmp))
    dossiers = {a[1]["dossier_travail"] for a in appels if a[0] == "reconstruction"}
    assert len(dossiers) == 2


@pytest.mark.parametrize("apres, restantes", [
    ("reconstruction", ["densification", "maillage", "echelle", "mesure"]),
    ("densification", ["maillage", "echelle", "mesure"]),
    ("maillage", ["echelle", "mesure"]),
    ("echelle", ["mesure"]),
    ("mesure", ["mesure"]),         # la mesure est toujours refaite : c'est elle qui produit le rapport
])
def test_reprise(etapes, apres, restantes):
    appels, _, tmp = etapes
    ps.lancer(str(tmp / "photos"), "s", racine_travail=str(tmp), reprendre_apres=apres)
    assert [a[0] for a in appels] == restantes


def test_parametres_transmis(etapes):
    appels, _, tmp = etapes
    ps.lancer(str(tmp / "photos"), "s", racine_travail=str(tmp), taille_marqueur_mm=29.9, z_coupe_mm=2.5,
              profondeur_poisson=9, camera_unique=False, formats=("stl", "obj"))
    d = {a[0]: a for a in appels}
    assert d["echelle"][1][2] == 29.9
    assert d["mesure"][2]["z_coupe_mm"] == 2.5 and d["mesure"][2]["formats"] == ("stl", "obj")
    assert d["maillage"][1]["profondeur_poisson"] == 9 and d["reconstruction"][1]["camera_unique"] is False


def test_erreurs_de_parametres(etapes, tmp_path):
    with pytest.raises(ValueError, match="Étape inconnue"):
        ps.lancer(str(tmp_path / "photos"), "s", reprendre_apres="magie")
    with pytest.raises(FileNotFoundError, match="introuvable"):
        ps.lancer(str(tmp_path / "absent"), "s")


def test_main_ne_plante_pas_une_serie(etapes, capsys, monkeypatch):
    appels, _, tmp = etapes
    assert ps.main(["--photos", str(tmp / "photos"), "--nom", "ok", "--racine-travail", str(tmp)]) == 0

    def casse(**kw):
        raise RuntimeError("COLMAP n'a pas pu positionner les caméras")
    sys.modules["main"].executer_pipeline_securise = casse
    assert ps.main(["--photos", str(tmp / "photos"), "--nom", "rate", "--racine-travail", str(tmp)]) == 1
    sortie = capsys.readouterr().out
    assert "ERREUR sur « rate »" in sortie and "positionner les caméras" in sortie
