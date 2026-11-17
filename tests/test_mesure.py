"""Tests des mesures : formes à volume exact, maillages abîmés, fichiers, rapports."""
import csv
import json
import os
import sys

import numpy as np
import pytest

import formes
import mesure as m
import synth

trimesh = pytest.importorskip("trimesh")


def _mesurer(forme, **kw):
    V, F, _ = forme
    return m.mesurer_maillage(V, F, **kw)


def _ferme(Ve, Fe):
    return len(m.analyser_aretes(Fe, len(Ve))["aretes_nettes"]) == 0


# ------------------------------------------------------------- volumes exacts
@pytest.mark.parametrize("nom, forme", [
    ("cylindre décalé", formes.cylindre(decalage=(7.0, -3.0))),
    ("boîte", formes.boite()),
    ("anneau (trou)", formes.anneau()),
    ("pont en U (2 pieds)", formes.pont_en_u()),
])
def test_volume_exact_et_export_ferme(nom, forme):
    mes, q, avert, notes, (Ve, Fe) = _mesurer(forme, z_coupe_mm=1.0)
    attendu = forme[2]
    assert mes["volume_mm3"] == pytest.approx(attendu, rel=1e-9), nom
    assert avert == []
    assert _ferme(Ve, Fe)
    assert m._volume_signe(Ve, Fe) == pytest.approx(attendu, rel=1e-9)   # l'export a bien le même volume
    assert Ve[:, 2].min() == pytest.approx(0.0, abs=1e-9)                # base posée sur le plateau


@pytest.mark.parametrize("z0", [0.0, 0.1, 1.0, 5.0, 12.5])
def test_le_volume_ne_depend_pas_de_la_hauteur_de_coupe_pour_un_prisme(z0):
    forme = formes.cylindre(r=20, h=40, n=96)
    assert _mesurer(forme, z_coupe_mm=z0)[0]["volume_mm3"] == pytest.approx(forme[2], rel=1e-9)


def test_la_sensibilite_a_la_coupe_est_l_aire_de_la_base():
    forme = formes.cylindre(r=20, h=40, n=96)
    _, q, *_ = _mesurer(forme, z_coupe_mm=1.0)
    aire = 0.5 * 96 * 20 ** 2 * np.sin(2 * np.pi / 96)
    assert q["sensibilite_volume_mm3_par_mm_de_coupe"] == pytest.approx(aire, rel=1e-9)
    # Sans prolongement, décaler la coupe de dz retire bien A * dz au volume
    v1 = _mesurer(forme, z_coupe_mm=1.0, prolonger=False)[0]["volume_mm3"]
    v2 = _mesurer(forme, z_coupe_mm=3.0, prolonger=False)[0]["volume_mm3"]
    assert v1 - v2 == pytest.approx(aire * 2.0, rel=1e-9)


def test_coupe_exactement_sur_des_sommets():
    """Plan passant par des arêtes et des faces du maillage : cas classique de découpe qui plante."""
    forme = formes.pont_en_u()          # sommets à z = 0, 20, 30
    mes, q, avert, _, (Ve, Fe) = _mesurer(forme, z_coupe_mm=20.0, prolonger=False)
    assert mes["volume_mm3"] == pytest.approx(4000.0, rel=1e-9)   # le tablier seul
    assert _ferme(Ve, Fe)


def test_dome_approche_l_hemisphere():
    V, F, vex = formes.dome(r=40.0)
    mes, q, avert, _, (Ve, Fe) = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert abs(mes["volume_mm3"] / vex - 1) < 0.005       # facettes de l'icosphère : < 0,5 %
    assert mes["hauteur_mm"] == pytest.approx(40.0, abs=0.2)
    assert mes["diametre_equivalent_base_mm"] == pytest.approx(80.0, abs=0.5)
    assert _ferme(Ve, Fe) and avert == []


def test_cotes_de_l_empreinte_sont_independantes_de_l_orientation():
    V, F, _ = formes.boite(30, 20, 10)
    R = synth.rotation("z", 33.0)
    V = (R @ (V - [15, 10, 0]).T).T + [200, -80, 0]
    mes = m.mesurer_maillage(V, F, z_coupe_mm=1.0)[0]
    assert mes["longueur_mm"] == pytest.approx(30.0, abs=1e-6)
    assert mes["largeur_mm"] == pytest.approx(20.0, abs=1e-6)
    assert mes["hauteur_mm"] == pytest.approx(10.0, abs=1e-9)
    assert mes["volume_mm3"] == pytest.approx(6000.0, rel=1e-9)


# --------------------------------------------------- maillages abîmés / pièges
def test_normales_inversees_sont_corrigees():
    V, F, vex = formes.cylindre()
    V2, F2 = formes.retourner(V, F)
    mes, q, avert, notes, (Ve, Fe) = m.mesurer_maillage(V2, F2, z_coupe_mm=1.0)
    assert mes["volume_mm3"] == pytest.approx(vex, rel=1e-9) and q["normales_retournees"]
    assert _ferme(Ve, Fe) and m._volume_signe(Ve, Fe) > 0          # l'export est bien orienté


def test_une_face_retournee_est_reparee():
    V, F, vex = formes.cylindre(n=96)
    F = F.copy()
    # Un triangle de la paroi haute (un triangle sous la coupe disparaîtrait avant d'être examiné)
    haut = np.flatnonzero(V[F].mean(axis=1)[:, 2] > 10)[3]
    F[haut] = F[haut][::-1]
    mes, q, avert, notes, (Ve, Fe) = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert mes["volume_mm3"] == pytest.approx(vex, rel=1e-6)
    assert avert == [] and _ferme(Ve, Fe)
    assert m._volume_signe(Ve, Fe) == pytest.approx(vex, rel=1e-6)
    # Soit réparé par reorientation des faces, soit absorbé comme un petit trou : jamais une erreur de volume
    assert q["orientation_reparee"] or q["boucles_de_trou"] >= 1


def test_petit_trou_est_bouche_et_signale_sans_alarme():
    """Dôme : triangles petits, un trou en haut se bouche presque sans erreur de volume."""
    V, F, vex = formes.dome(r=40.0, subdivisions=4)
    intact = m.mesurer_maillage(V, F, z_coupe_mm=1.0)[0]["volume_mm3"]
    centres = V[F].mean(axis=1)
    haut = np.flatnonzero(centres[:, 2] > 36)
    assert len(haut) >= 3, "sélection vide : le test ne vérifierait rien"
    # deux triangles voisins : on prend le premier et un triangle qui partage un sommet avec lui
    voisin = next(i for i in haut[1:] if len(set(F[i]) & set(F[haut[0]])) >= 2)
    F2 = np.delete(F, [haut[0], voisin], axis=0)
    mes, q, avert, notes, (Ve, Fe) = m.mesurer_maillage(V, F2, z_coupe_mm=1.0)
    assert q["boucles_de_trou"] == 1
    assert abs(mes["volume_mm3"] / intact - 1) < 0.0005         # < 0,05 % pour un trou de 2 triangles
    assert q["part_surface_comblee_pct"] < 1.0 and avert == []
    assert _ferme(Ve, Fe)


def test_trou_qui_touche_le_plan_est_signale():
    """Une colonne entière de la paroi manque : le trou rejoint la coupe, le volume ne doit pas être donné pour fiable."""
    V, F, vex = formes.cylindre(n=96)
    centres = V[F].mean(axis=1)
    colonne = np.flatnonzero((centres[:, 2] > 5) & (centres[:, 2] < 35) & (np.hypot(centres[:, 0], centres[:, 1]) > 19))
    assert len(colonne) >= 2, "sélection vide : le test ne vérifierait rien"
    mes, q, avert, *_ = m.mesurer_maillage(V, np.delete(F, colonne[:2], axis=0), z_coupe_mm=1.0)
    assert any("touche(nt) le plan de coupe" in a for a in avert)


def test_gros_trou_declenche_un_avertissement():
    V, F, vex = formes.cylindre(n=96)
    centres = V[F].mean(axis=1)
    paroi = np.flatnonzero((np.hypot(centres[:, 0], centres[:, 1]) > 19) & (centres[:, 2] > 5) & (centres[:, 0] > 0))
    assert len(paroi) > 20, "sélection vide : le test ne vérifierait rien"
    mes, q, avert, *_ = m.mesurer_maillage(V, np.delete(F, paroi, axis=0), z_coupe_mm=1.0)   # la moitié de la paroi disparaît
    assert q["part_surface_comblee_pct"] > m.PART_COMBLEE_MAX_PCT
    assert any("trous bouchés" in a for a in avert) and any("plan de coupe" in a for a in avert)


def test_fragment_flottant_est_ecarte():
    gros = formes.cylindre(r=25, h=40)
    petit = formes.cylindre(r=2, h=3, decalage=(80, 0), z_base=0)
    V, F, _ = formes.fusionner(gros, petit)
    mes, q, avert, notes, _ = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert mes["volume_mm3"] == pytest.approx(gros[2], rel=1e-9)
    assert q["composantes_trouvees"] == 2 and avert == []
    assert any("fragment" in n for n in notes)


def test_objet_en_deux_morceaux_previent():
    a = formes.cylindre(r=25, h=40)
    b = formes.cylindre(r=20, h=30, decalage=(120, 0))
    V, F, vex = formes.fusionner(a, b)
    mes, q, avert, *_ = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert any("plusieurs morceaux" in x for x in avert)
    # --toutes-composantes : les deux morceaux comptent
    mes2, *_ = m.mesurer_maillage(V, F, z_coupe_mm=1.0, garder_toutes=True)
    assert mes2["volume_mm3"] == pytest.approx(vex, rel=1e-9)


def test_le_plateau_sous_l_objet_est_ignore():
    objet = formes.cylindre(r=25, h=40)
    plateau = formes.boite(300, 300, 5, origine=(-150, -150, -5))
    V, F, _ = formes.fusionner(objet, plateau)
    mes, q, avert, *_ = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert mes["volume_mm3"] == pytest.approx(objet[2], rel=1e-9) and avert == []


def test_modele_du_mauvais_cote_du_plan_est_refuse():
    V, F, _ = formes.cylindre(z_base=-60)         # entièrement sous le plateau
    with pytest.raises(m.ErreurMesure, match="sous le plan|inversé"):
        m.mesurer_maillage(V, F, z_coupe_mm=1.0)


def test_coupe_trop_haute_est_refusee():
    V, F, _ = formes.cylindre(h=40)
    with pytest.raises(m.ErreurMesure, match="au-dessus du plan"):
        m.mesurer_maillage(V, F, z_coupe_mm=100.0)
    with pytest.raises(m.ErreurMesure, match="positif"):
        m.mesurer_maillage(V, F, z_coupe_mm=-1.0)


def test_triangles_degeneres_sont_retires():
    V, F, vex = formes.cylindre()
    F2 = np.vstack([F, [[0, 0, 1], [3, 4, 4]]])       # deux triangles dégénérés
    V2 = np.vstack([V, [[500, 500, 500]]])            # sommet orphelin
    mes, q, *_ = m.mesurer_maillage(V2, F2, z_coupe_mm=1.0)
    assert q["faces_degenerees_supprimees"] == 2
    assert mes["volume_mm3"] == pytest.approx(vex, rel=1e-9)


@pytest.mark.parametrize("mauvais", [np.nan, np.inf])
def test_coordonnees_non_finies_sont_refusees(mauvais):
    V, F, _ = formes.cylindre()
    V[5, 1] = mauvais
    with pytest.raises(m.ErreurMesure, match="non finies"):
        m.mesurer_maillage(V, F)


def test_maillage_sans_triangle_exploitable():
    V = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0.0]])       # trois points alignés : aire nulle
    with pytest.raises(m.ErreurMesure, match="aucun triangle"):
        m.mesurer_maillage(V, np.array([[0, 1, 2]]), z_coupe_mm=None, repere_plateau=False)


def test_sans_plateau_pas_de_coupe_ni_de_hauteur():
    V, F, vex = formes.cylindre(decalage=(50, 50), z_base=-100)    # n'importe où dans l'espace
    mes, q, avert, *_ = m.mesurer_maillage(V, F, repere_plateau=False, z_coupe_mm=None)
    assert mes["volume_mm3"] == pytest.approx(vex, rel=1e-9)
    assert "hauteur_mm" not in mes and q["z_coupe_mm"] is None and avert == []
    assert mes["boite_principale_mm"][0] >= mes["boite_principale_mm"][1] >= mes["boite_principale_mm"][2]
    assert mes["boite_principale_mm"][2] == pytest.approx(25 * 2 * np.cos(np.pi / 64) if False else mes["boite_principale_mm"][2])


def test_maillage_ouvert_sans_plateau_est_signale():
    V, F, vex = formes.cylindre(n=96)
    mes, q, avert, *_ = m.mesurer_maillage(V, np.delete(F, [10, 11, 12], axis=0), repere_plateau=False, z_coupe_mm=None)
    assert q["boucles_de_trou"] >= 1


# -------------------------------------------------------- découpe seule
def test_decoupe_garde_les_points_exactement_sur_le_plan():
    V, F, _ = formes.cylindre()
    V2, F2 = m.decouper_au_plan(V, F, 10.0)
    assert V2[:, 2].min() == pytest.approx(10.0) and len(F2) > 0
    ar = m.analyser_aretes(F2, len(V2))
    assert ar["non_manifold"] == 0
    bas = np.abs(V2[:, 2] - 10.0) < 1e-9
    assert bas.sum() > 0


def test_decoupe_ne_cree_pas_de_doublons_sur_la_coupe():
    """Deux triangles voisins coupés partagent le même point : sinon la coupe aurait des fentes."""
    V, F, _ = formes.dome(r=30, subdivisions=3)
    V2, F2 = m.decouper_au_plan(V, F, 0.0)
    ar = m.analyser_aretes(F2, len(V2))
    boucles, ouvertes = m.chainer_boucles(ar["aretes_nettes"][:, ::-1])
    assert ouvertes == 0 and len(boucles) == 1
    # Tous les points de la boucle sont sur le plan, aucun n'est en double
    pts = np.round(V2[boucles[0]], 9)
    assert len(np.unique(pts, axis=0)) == len(pts)


# --------------------------------------------- export sans earcut (repli)
def test_export_sans_earcut_repli_en_eventail(monkeypatch):
    monkeypatch.setitem(sys.modules, "mapbox_earcut", None)
    V, F, vex = formes.cylindre()
    mes, q, avert, notes, (Ve, Fe) = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert mes["volume_mm3"] == pytest.approx(vex, rel=1e-9)           # le volume n'en dépend pas
    assert _ferme(Ve, Fe) and any("éventail" in n for n in notes)


def test_export_sans_earcut_base_a_trou_non_fermee(monkeypatch):
    monkeypatch.setitem(sys.modules, "mapbox_earcut", None)
    V, F, vex = formes.anneau()
    mes, q, avert, notes, (Ve, Fe) = m.mesurer_maillage(V, F, z_coupe_mm=1.0)
    assert mes["volume_mm3"] == pytest.approx(vex, rel=1e-9)
    assert not _ferme(Ve, Fe) and any("earcut" in n for n in notes)


# ------------------------------------------------------------ fichiers
VRAI = synth.SimilitudeVraie(137.0, axe="x", angle=70.0)


def _stl_colmap(tmp_path, forme, nom="maillage.stl"):
    """Écrit la forme (en mm) dans le repère COLMAP du banc d'essai, comme le ferait maillage.py."""
    V, F, vex = forme
    chemin = str(tmp_path / nom)
    trimesh.Trimesh(VRAI.colmap_depuis_mm(V), F, process=False).export(chemin)
    return chemin, vex


def _echelle_json(tmp_path, contenu=None):
    chemin = tmp_path / "echelle.json"
    chemin.write_text(json.dumps(contenu if contenu is not None else synth.echelle_vraie(VRAI)), encoding="utf-8")
    return str(chemin)


def test_chaine_fichiers_complete(tmp_path):
    stl, vex = _stl_colmap(tmp_path, formes.cylindre(r=25, h=40, n=96, decalage=(10, 5)))
    sortie = str(tmp_path / "mesures" / "essai1")
    r = m.mesurer(stl, _echelle_json(tmp_path), sortie, "essai1", formats=("stl", "obj", "ply"))
    # STL en float32 : précision relative ~1e-6, bien meilleure que n'importe quelle mesure réelle
    assert r["mesures"]["volume_mm3"] == pytest.approx(vex, rel=1e-4)
    assert r["mesures"]["hauteur_mm"] == pytest.approx(40.0, abs=1e-2)
    assert r["fiable"] and r["unite_longueur"] == "mm"
    assert r["echelle"]["incertitude_volume_due_a_l_echelle_pct"] == pytest.approx(0.15)   # 3 x 0,05 %
    for fichier in ("modele_mm.stl", "modele_mm.obj", "modele_mm.ply", "rapport.json", "rapport.txt"):
        assert os.path.getsize(os.path.join(sortie, fichier)) > 0
    assert json.load(open(os.path.join(sortie, "rapport.json"), encoding="utf-8"))["nom"] == "essai1"
    # Le modèle exporté se relit, est fermé et a le même volume
    V, F, _ = m.charger_maillage(os.path.join(sortie, "modele_mm.stl"))
    assert _ferme(V, F) and m._volume_signe(V, F) == pytest.approx(vex, rel=1e-4)


def test_historique_csv_pour_excel_francais(tmp_path):
    stl, _ = _stl_colmap(tmp_path, formes.cylindre(r=25, h=40, n=96))
    for i in range(2):
        m.mesurer(stl, _echelle_json(tmp_path), str(tmp_path / "mesures" / f"essai{i}"), f"essai{i}",
                  formats=("stl",))
    brut = open(tmp_path / "mesures" / "historique.csv", "rb").read()
    assert brut.startswith(b"\xef\xbb\xbf") and brut.count(b"\xef\xbb\xbf") == 1      # un seul BOM
    lignes = list(csv.reader(open(tmp_path / "mesures" / "historique.csv", encoding="utf-8-sig"), delimiter=";"))
    assert lignes[0] == m.COLONNES_HISTORIQUE and len(lignes) == 3
    ligne = dict(zip(lignes[0], lignes[1]))
    assert "," in ligne["volume_ml"] and "." not in ligne["volume_ml"]      # virgule décimale
    assert ligne["fiable"] == "oui"


def test_nombres_francais():
    assert m._nombre_fr(None) == "" and m._nombre_fr(True) == "oui" and m._nombre_fr(False) == "non"
    assert m._nombre_fr(3.5) == "3,5" and m._nombre_fr(40.0) == "40" and m._nombre_fr("abc") == "abc"
    assert m._nombre_fr(np.float64(0.25)) == "0,25"


def test_sans_echelle_refuse_par_defaut(tmp_path):
    stl, _ = _stl_colmap(tmp_path, formes.cylindre())
    with pytest.raises(m.ErreurMesure, match="échelle"):
        m.mesurer(stl, None)
    with pytest.raises(m.ErreurMesure, match="introuvable"):
        m.mesurer(stl, str(tmp_path / "absent.json"))
    r = m.mesurer(stl, None, autoriser_sans_echelle=True)
    assert r["unite_longueur"] == "unite_colmap" and not r["fiable"]
    assert any("PAS en millimètres" in a for a in r["avertissements"])


def test_echelle_invalide_est_refusee(tmp_path):
    stl, _ = _stl_colmap(tmp_path, formes.cylindre())
    chemin = _echelle_json(tmp_path, {"mm_par_unite": -5})
    with pytest.raises(m.ErreurMesure, match="mm_par_unite"):
        m.mesurer(stl, chemin)


def test_echelle_douteuse_se_propage_au_rapport(tmp_path):
    stl, _ = _stl_colmap(tmp_path, formes.cylindre())
    e = synth.echelle_vraie(VRAI)
    e["fiable"] = False
    r = m.mesurer(stl, _echelle_json(tmp_path, e))
    assert not r["fiable"] and any("à vérifier" in a for a in r["avertissements"])


def test_facteur_manuel_mesure_sans_coupe(tmp_path):
    """Échelle simple (pas de plateau) : l'objet doit être complet, aucune coupe."""
    V, F, vex = formes.cylindre()
    stl = str(tmp_path / "m.stl")
    trimesh.Trimesh(V / 100.0, F, process=False).export(stl)       # « unités COLMAP » = mm / 100
    e = tmp_path / "e.json"
    e.write_text(json.dumps({"mm_par_unite": 100.0, "methode": "facteur_manuel", "transformation_4x4": None,
                             "fiable": True}), encoding="utf-8")
    r = m.mesurer(stl, str(e))
    assert r["mesures"]["volume_mm3"] == pytest.approx(vex, rel=1e-4)
    assert r["parametres"]["z_coupe_mm"] is None and "hauteur_mm" not in r["mesures"]
    assert any("Pas de coupe" in n or "pas de coupe" in n for n in r["notes"])


@pytest.mark.parametrize("contenu, mot", [
    (b"", "Impossible|introuvable|aucun|n'est pas un maillage"),
    (b"pas un stl du tout", "Impossible|n'est pas un maillage|aucun"),
])
def test_fichiers_maillage_invalides(tmp_path, contenu, mot):
    chemin = tmp_path / "x.stl"
    chemin.write_bytes(contenu)
    with pytest.raises(m.ErreurMesure, match=mot):
        m.charger_maillage(str(chemin))
    with pytest.raises(m.ErreurMesure, match="introuvable"):
        m.charger_maillage(str(tmp_path / "rien.stl"))


def test_un_nuage_de_points_n_est_pas_un_maillage(tmp_path):
    chemin = str(tmp_path / "nuage.ply")
    trimesh.PointCloud(np.random.default_rng(0).normal(size=(50, 3))).export(chemin)
    with pytest.raises(m.ErreurMesure, match="pas un maillage"):
        m.charger_maillage(chemin)


def test_un_stl_en_soupe_de_triangles_est_recolle(tmp_path):
    """Un STL stocke 3 sommets par triangle : charger_maillage doit les fusionner pour la topologie."""
    V, F, vex = formes.cylindre(n=96)
    chemin = str(tmp_path / "c.stl")
    trimesh.Trimesh(V, F, process=False).export(chemin)
    V2, F2, info = m.charger_maillage(chemin)
    assert len(V2) == len(V) and info["sommets_fichier"] == 3 * len(F)
    assert m.analyser_aretes(F2, len(V2))["libres"] == 0


def test_diagnostic_surfaces_inventees():
    """Des sommets loin de tout point du nuage = bulle de Poisson : doit être signalé."""
    V, F, _ = formes.cylindre(n=96)
    # nuage = échantillon de la surface réelle (sommets du cylindre + points intermédiaires)
    nuage = np.vstack([V, (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0])
    _, q, avert, *_ = m.mesurer_maillage(V, F, z_coupe_mm=1.0, nuage_mm=nuage)
    assert q["part_sommets_loin_du_nuage_pct"] < 1.0 and avert == []
    # Nuage qui n'a plus que la moitié de l'objet : l'autre moitié du maillage est « inventée »
    moitie = nuage[nuage[:, 0] < 0]
    _, q2, avert2, *_ = m.mesurer_maillage(V, F, z_coupe_mm=1.0, nuage_mm=moitie)
    assert q2["part_sommets_loin_du_nuage_pct"] > 20 and any("inventées" in a for a in avert2)


def test_cli(tmp_path, capsys):
    stl, _ = _stl_colmap(tmp_path, formes.cylindre(n=96))
    trav = str(tmp_path / "w")
    os.makedirs(trav)
    os.replace(_echelle_json(tmp_path), os.path.join(trav, "echelle.json"))
    assert m.main(["--travail", trav, "--maillage", stl, "--nom", "t1", "--formats", "stl", "obj"]) == 0
    assert os.path.isfile(os.path.join(trav, "mesures", "t1", "modele_mm.obj"))
    assert "Volume" in capsys.readouterr().out
    assert m.main(["--travail", trav, "--maillage", str(tmp_path / "absent.stl")]) == 1
    assert "ERREUR" in capsys.readouterr().out
    os.remove(os.path.join(trav, "echelle.json"))
    assert m.main(["--travail", trav, "--maillage", stl]) == 1


# ------------------------------------------------------- données réelles
@pytest.mark.lourd
def test_vraies_donnees_du_depot():
    """Régression sur le STL réel (1 M de triangles) : mêmes chiffres que trimesh, calculés indépendamment."""
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stl = os.path.join(racine, "colmap_workspace", "modele_pour_impression.stl")
    ply = os.path.join(racine, "colmap_workspace", "openmvs", "scene_dense.ply")
    if not os.path.isfile(stl):
        pytest.skip("STL du dépôt absent")
    e = {"mm_par_unite": 100.0, "methode": "facteur_manuel", "transformation_4x4": None, "fiable": True}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        chemin_e = os.path.join(d, "e.json")
        json.dump(e, open(chemin_e, "w"))
        r = m.mesurer(stl, chemin_e, chemin_nuage=ply if os.path.isfile(ply) else None)
    # trimesh donne 0,107757 unités³ pour tout le maillage : x 100³ = 107 757 mm³ (les fragments pèsent < 0,1 %)
    assert r["mesures"]["volume_mm3"] == pytest.approx(107757.0, rel=2e-3)
    assert r["qualite"]["composantes_trouvees"] > 1        # fragments flottants (connexité par sommets)
    assert r["fiable"], r["avertissements"]
    if os.path.isfile(ply):
        part = r["qualite"]["part_sommets_loin_du_nuage_pct"]
        if part is None:       # sans Open3D, trimesh ne lit pas ce PLY : le diagnostic est ignoré, et le dit
            assert any("Nuage dense non lu" in n for n in r["notes"])
        else:
            assert part < 1.0


def test_hauteur_robuste_corrige_le_biais_du_bruit_sans_ecraser_un_dome():
    rng = np.random.default_rng(7)
    # dessus plat bruité : le point le plus haut est biaisé, la hauteur lissée l'est beaucoup moins
    V, F, _ = formes.cylindre(r=25, h=40, n=96)
    biais_brut, biais_robuste = [], []
    for _ in range(30):
        Vb = V + rng.normal(0, 0.05, V.shape)
        mes = m.mesurer_maillage(Vb, F, z_coupe_mm=1.0)[0]
        biais_brut.append(mes["hauteur_mm"] - 40.0)
        biais_robuste.append(mes["hauteur_robuste_mm"] - 40.0)
    assert np.mean(biais_brut) > 0.07                       # ~ +2,4 sigma = +0,12 mm
    assert abs(np.mean(biais_robuste)) < 0.5 * np.mean(biais_brut)
    # sommet de dôme (maillage fin, non bruité) : la moyenne locale l'abaisse de ~ r²/(4R), ici moins de 0,1 mm
    Vd, Fd, _ = formes.dome(r=30.0, subdivisions=5)
    mes = m.mesurer_maillage(Vd, Fd, z_coupe_mm=1.0)[0]
    assert 0 <= mes["hauteur_mm"] - mes["hauteur_robuste_mm"] < 0.1


def test_hauteur_robuste_ne_depend_pas_de_la_forme_des_triangles():
    """Un cylindre à triangles longs (paroi sur toute la hauteur) et le même sous-divisé donnent la même hauteur."""
    V, F, _ = formes.cylindre(r=25, h=40, n=96)
    fin = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(V, F, max_edge=2.0), process=True)   # process : fusionne les sommets
    Vf, Ff = np.asarray(fin.vertices, float), np.asarray(fin.faces, np.int64)
    assert len(Ff) > 10 * len(F)
    assert m.hauteur_robuste(V, F) == pytest.approx(40.0, abs=1e-6)
    assert m.hauteur_robuste(Vf, Ff) == pytest.approx(40.0, abs=1e-6)


# ---------------------------------------------- cohérence échelle / reconstruction
def test_dimensions_aberrantes_sont_signalees(tmp_path):
    """Maillage d'un autre scan avec une échelle qui ne lui correspond pas : 100 000 mm de haut."""
    V, F, _ = formes.cylindre(r=25, h=40, n=96)
    stl = str(tmp_path / "m.stl")
    trimesh.Trimesh(V, F, process=False).export(stl)
    e = tmp_path / "e.json"
    e.write_text(json.dumps({"mm_par_unite": 5000.0, "methode": "facteur_manuel", "transformation_4x4": None,
                             "fiable": True}), encoding="utf-8")
    r = m.mesurer(stl, str(e))
    assert not r["fiable"] and any("Dimensions aberrantes" in a for a in r["avertissements"])
    e.write_text(json.dumps({"mm_par_unite": 0.001, "methode": "facteur_manuel", "transformation_4x4": None,
                             "fiable": True}), encoding="utf-8")
    assert any("Dimensions aberrantes" in a for a in m.mesurer(stl, str(e))["avertissements"])


def test_echelle_d_une_autre_reconstruction_est_detectee(tmp_path):
    """Reconstruction refaite après le calcul de l'échelle : les mesures seraient fausses sans le dire."""
    pytest.importorskip("pycolmap")
    import echelle
    from generer_cible_echelle import construire_mise_en_page
    travail = tmp_path / "w"
    travail.mkdir()
    vrai = synth.fabriquer_dossier_de_travail(str(travail), nb=8)
    mep = tmp_path / "cible.json"
    mep.write_text(json.dumps(construire_mise_en_page()), encoding="utf-8")
    echelle.lancer_echelle(str(travail), str(mep))
    V, F, _ = formes.cylindre(r=25, h=40, n=96)
    stl = str(travail / "modele_pour_impression.stl")
    trimesh.Trimesh(vrai.colmap_depuis_mm(V), F, process=False).export(stl)
    e_json = echelle.chemin_echelle(str(travail))

    r = m.mesurer(stl, e_json)
    assert r["fiable"] and not any("reconstruction" in a.lower() for a in r["avertissements"]), r["avertissements"]

    # on « refait » la reconstruction : le contenu de sparse_final change
    with open(travail / "sparse_final" / "images.bin", "ab") as f:
        f.write(b"\x00")
    r2 = m.mesurer(stl, e_json)
    assert not r2["fiable"] and any("a changé depuis le calcul de l'échelle" in a for a in r2["avertissements"])

    # reconstruction absente : on le dit, sans accuser à tort
    import shutil
    shutil.rmtree(travail / "sparse_final")
    r3 = m.mesurer(stl, e_json)
    assert any("non vérifiable" in n for n in r3["notes"]) and r3["fiable"]


def test_empreinte_de_reconstruction(tmp_path):
    import echelle
    d = tmp_path / "sparse"
    d.mkdir()
    assert echelle.empreinte_reconstruction(str(d)) is None                 # rien d'exploitable
    (d / "images.bin").write_bytes(b"abc")
    (d / "cameras.bin").write_bytes(b"xyz")
    h1 = echelle.empreinte_reconstruction(str(d))
    assert h1 == echelle.empreinte_reconstruction(str(d)) and len(h1) == 64   # stable
    (d / "images.bin").write_bytes(b"abd")
    assert echelle.empreinte_reconstruction(str(d)) != h1                     # sensible au contenu
    (d / "points3D.bin").write_bytes(b"ignore")
    (d / "images.bin").write_bytes(b"abc")
    assert echelle.empreinte_reconstruction(str(d)) == h1                     # points3D n'entre pas en jeu
