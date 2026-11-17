"""Interface graphique : pilotée pour de vrai (boutons appelés, thread de calcul, journal), sous écran virtuel.
Ignorés si Tkinter ou un écran manque (xvfb-run -a python -m pytest tests/test_interface.py)."""
import json
import time

import pytest

import formes
import synth

tk = pytest.importorskip("tkinter")
pytest.importorskip("pycolmap")
trimesh = pytest.importorskip("trimesh")

pytestmark = pytest.mark.lourd


@pytest.fixture
def app(monkeypatch):
    try:
        racine = tk.Tk()
    except tk.TclError:
        pytest.skip("pas d'écran disponible pour Tkinter")
    try:
        import interface_pipeline as ip
    except ImportError as e:         # open3d absent, par exemple
        racine.destroy()
        pytest.skip(f"interface non importable : {e}")
    messages = {"erreurs": [], "infos": [], "questions": []}
    monkeypatch.setattr(ip.messagebox, "showerror", lambda titre, texte: messages["erreurs"].append((titre, texte)))
    monkeypatch.setattr(ip.messagebox, "showinfo", lambda titre, texte: messages["infos"].append((titre, texte)))
    monkeypatch.setattr(ip.messagebox, "askyesno", lambda titre, texte: messages["questions"].append((titre, texte)) or messages.get("reponse", True))
    fenetre = ip.InterfacePipeline(racine)
    fenetre.messages = messages
    yield fenetre
    # Annule les boucles after() de l'interface (journal, rafraîchissement des boutons) avant de détruire la fenêtre :
    # sinon Tcl affiche « invalid command name » pour chacune à l'itération suivante
    for identifiant in racine.tk.splitlist(racine.tk.call("after", "info")):
        racine.after_cancel(identifiant)
    racine.destroy()


def attendre(app, delai=120):
    """Fait tourner la vraie boucle Tk jusqu'à la fin de l'étape en cours.

    Il faut une vraie mainloop : le thread de calcul rend la main à la fenêtre par racine.after(),
    que Tkinter n'autorise depuis un autre thread que si le thread principal est dans mainloop().
    """
    fin = time.time() + delai

    def verifier():
        if not app.en_cours or time.time() > fin:
            app.racine.quit()
        else:
            app.racine.after(30, verifier)

    app.racine.after(30, verifier)
    app.racine.mainloop()
    assert not app.en_cours, "l'étape ne s'est pas terminée"


def journal(app):
    # La console est alimentée par une file lue toutes les 150 ms : on la vide d'abord pour ne pas lire en avance
    while not app.file_journal.empty():
        app._ecrire_journal(app.file_journal.get_nowait())
    return app.zone_journal.get("1.0", "end")


def test_construction_et_valeurs_par_defaut(app):
    # 4 boutons existants + « Calculer l'échelle » + « Mesurer »
    assert len(app.boutons_lancement) == 6
    assert app.var_camera_unique.get() is True and app.var_z_coupe.get() == "1.0"
    assert app.var_mise_en_page.get() == "cible_echelle.json"


def test_creer_la_cible(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app._creer_cible()
    assert (tmp_path / "cible_echelle.svg").is_file() and (tmp_path / "cible_echelle.json").is_file()
    assert app.var_mise_en_page.get().endswith("cible_echelle.json")
    assert app.messages["infos"] and "100 %" in app.messages["infos"][0][1]
    assert "Cible créée" in journal(app)


@pytest.mark.parametrize("nom", ["", "   ", "a/b", "a\\b", "x:y", 'q"r', "a*b", "a?b", "<a>", "a|b"])
def test_nom_de_mesure_invalide(app, nom):
    app.var_nom_mesure.set(nom)
    app._lancer_etape4b()
    assert len(app.messages["erreurs"]) == 1 and "Nom de la mesure" in app.messages["erreurs"][0][1]
    assert not app.en_cours


@pytest.mark.parametrize("champ, valeur", [("var_z_coupe", "abc"), ("var_z_coupe", "-2"), ("var_taille_marqueur", "0"),
                                           ("var_taille_marqueur", "trente")])
def test_parametres_numeriques_invalides(app, champ, valeur):
    app.var_nom_mesure.set("ok")
    getattr(app, champ).set(valeur)
    (app._lancer_etape4b if champ == "var_z_coupe" else app._lancer_etape4a)()
    assert len(app.messages["erreurs"]) == 1 and not app.en_cours


def test_la_virgule_decimale_est_acceptee(app):
    assert app._lire_nombre_optionnel("29,9", "x", True) == 29.9
    assert app._lire_nombre_optionnel("  ", "x", True) is None
    assert app._lire_nombre_optionnel("0", "x", False) == 0.0
    assert app._lire_nombre_optionnel("0", "x", True) is False


def test_une_etape_en_erreur_remet_l_interface_en_etat(app, tmp_path):
    app.var_dossier_travail.set(str(tmp_path / "vide"))
    app._lancer_etape4a()
    attendre(app)
    assert "etape 1" in journal(app).lower() or "étape 1" in journal(app).lower()
    assert "[ERREUR]" in journal(app) and app.messages["erreurs"]
    # le bouton est de nouveau utilisable
    assert all(str(b.cget("state")) == "normal" for b in app.boutons_lancement)


def test_mesure_existante_demande_confirmation(app, tmp_path):
    (tmp_path / "mesures" / "deja").mkdir(parents=True)
    app.var_dossier_travail.set(str(tmp_path))
    app.var_nom_mesure.set("deja")
    app.messages["reponse"] = False
    app._lancer_etape4b()
    assert app.messages["questions"] and not app.en_cours              # refusé : rien n'est lancé


def test_chaine_complete_par_l_interface(app, tmp_path):
    """Cible -> échelle -> mesure, uniquement avec les boutons, sur une reconstruction et des photos simulées."""
    travail = tmp_path / "colmap_workspace"
    travail.mkdir()
    vrai = synth.fabriquer_dossier_de_travail(str(travail))
    V, F, volume_vrai = formes.cylindre(r=25, h=40, n=96, decalage=(5, 5))
    trimesh.Trimesh(vrai.colmap_depuis_mm(V), F, process=False).export(str(travail / "modele_pour_impression.stl"))

    from generer_cible_echelle import construire_mise_en_page
    mep = tmp_path / "cible_echelle.json"
    mep.write_text(json.dumps(construire_mise_en_page()), encoding="utf-8")
    app.var_dossier_travail.set(str(travail))
    app.var_mise_en_page.set(str(mep))

    app._lancer_etape4a()
    attendre(app)
    assert (travail / "echelle.json").is_file(), journal(app)
    assert "Échelle :" in journal(app) and not app.messages["erreurs"], journal(app)

    app.var_nom_mesure.set("gui01")
    app._lancer_etape4b()
    attendre(app)
    rapport = json.load(open(travail / "mesures" / "gui01" / "rapport.json", encoding="utf-8"))
    assert rapport["fiable"], rapport["avertissements"]
    assert rapport["mesures"]["volume_mm3"] == pytest.approx(volume_vrai, rel=0.015)
    assert rapport["mesures"]["hauteur_robuste_mm"] == pytest.approx(40.0, abs=0.5)
    assert (travail / "mesures" / "gui01" / "modele_mm.stl").stat().st_size > 0
    assert "Volume" in journal(app) and "Fichiers dans" in journal(app)
    assert (travail / "mesures" / "historique.csv").is_file()
