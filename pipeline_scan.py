"""
Un scan de bout en bout : photos -> reconstruction -> densification -> maillage -> échelle -> mesures.

Pourquoi : la validation demande une dizaine de scans de la même pièce (voir PROTOCOLE_VALIDATION.md), et
lancer cinq étapes à la main dix fois, c'est dix occasions de se tromper de dossier. Ce script enchaîne tout
pour UN dossier de photos et range le résultat dans un dossier de travail à part, pour que les scans ne se
mélangent pas.

    python pipeline_scan.py --photos Photos_scan01 --nom scan01
    python pipeline_scan.py --photos Photos_scan01 --nom scan01 --reprendre-apres maillage

Chaque scan a son dossier de travail : colmap_workspace_<nom>/ . Les mesures sont dans
colmap_workspace_<nom>/mesures/<nom>/rapport.json.

Pour traiter plusieurs scans d'affilée, lancer la commande pour chacun (une boucle dans PowerShell suffit) :
    foreach ($i in 1..10) { python pipeline_scan.py --photos "Photos_scan$i" --nom "scan$i" }
"""

import argparse
import os
import sys

# Ordre des étapes : permet de reprendre après l'une d'elles sans tout recalculer
ETAPES = ("reconstruction", "densification", "maillage", "echelle", "mesure")


def lancer(dossier_photos, nom, mise_en_page="cible_echelle.json", racine_travail=".",
           taille_marqueur_mm=None, z_coupe_mm=1.0, openmvs_bin=None, reprendre_apres=None,
           profondeur_poisson=10, cible_triangles=1500000, camera_unique=True, formats=("stl",)):
    """Enchaîne toutes les étapes et renvoie le rapport de mesure (dict).

    reprendre_apres : nom d'une étape déjà faite (voir ETAPES) : on repart de la suivante, sans rien effacer.
    """
    if reprendre_apres is not None and reprendre_apres not in ETAPES:
        raise ValueError(f"Étape inconnue : {reprendre_apres} (choix : {', '.join(ETAPES)})")
    if not os.path.isdir(dossier_photos):
        raise FileNotFoundError(f"Dossier de photos introuvable : {dossier_photos}")
    dossier_travail = os.path.join(racine_travail, f"colmap_workspace_{nom}")
    deja_fait = ETAPES.index(reprendre_apres) + 1 if reprendre_apres else 0

    def a_faire(etape):
        return ETAPES.index(etape) >= deja_fait

    # Imports ici : le module reste importable (et testable) sans COLMAP, OpenMVS ni Open3D
    if a_faire("reconstruction"):
        from main import executer_pipeline_securise
        print(f"\n===== [{nom}] Étape 1/5 : reconstruction (COLMAP) =====")
        executer_pipeline_securise(dossier_source=dossier_photos, dossier_travail=dossier_travail,
                                   repartir_de_zero=True, camera_unique=camera_unique)
    if a_faire("densification"):
        import densification_openmvs
        if openmvs_bin:
            densification_openmvs.OPENMVS_BIN = openmvs_bin
        print(f"\n===== [{nom}] Étape 2/5 : densification (OpenMVS) =====")
        densification_openmvs.densifier(dossier_travail=dossier_travail, repartir_de_zero=True)
    if a_faire("maillage"):
        from maillage import mailler_et_preparer_impression
        print(f"\n===== [{nom}] Étape 3/5 : maillage =====")
        mailler_et_preparer_impression(dossier_travail=dossier_travail, profondeur_poisson=profondeur_poisson,
                                       cible_triangles=cible_triangles, afficher_interface=False)
    if a_faire("echelle"):
        from echelle import lancer_echelle
        print(f"\n===== [{nom}] Étape 4/5 : mise à l'échelle (cible ArUco) =====")
        lancer_echelle(dossier_travail, mise_en_page, taille_marqueur_mm)
    print(f"\n===== [{nom}] Étape 5/5 : mesures =====")
    from mesure import mesurer, resume_texte
    rapport = mesurer(
        os.path.join(dossier_travail, "modele_pour_impression.stl"),
        os.path.join(dossier_travail, "echelle.json"),
        os.path.join(dossier_travail, "mesures", nom), nom, z_coupe_mm=z_coupe_mm,
        chemin_nuage=os.path.join(dossier_travail, "openmvs", "scene_dense.ply"), formats=formats)
    print(resume_texte(rapport))
    return rapport


def main(argv=None):
    p = argparse.ArgumentParser(description="Un scan complet, des photos au rapport de mesure.")
    p.add_argument("--photos", required=True, help="dossier des photos de ce scan")
    p.add_argument("--nom", required=True, help="nom du scan (ex : scan01) : sert à nommer le dossier de travail")
    p.add_argument("--mise-en-page", default="cible_echelle.json")
    p.add_argument("--taille-marqueur-mm", type=float, default=None, help="côté d'un marqueur mesuré sur la feuille imprimée")
    p.add_argument("--z-coupe-mm", type=float, default=1.0)
    p.add_argument("--openmvs-bin", default=None, help="dossier des .exe d'OpenMVS (sinon celui de densification_openmvs.py)")
    p.add_argument("--reprendre-apres", choices=ETAPES, default=None, help="reprendre après cette étape (déjà faite)")
    p.add_argument("--racine-travail", default=".")
    args = p.parse_args(argv)
    try:
        lancer(args.photos, args.nom, args.mise_en_page, args.racine_travail, args.taille_marqueur_mm, args.z_coupe_mm,
               args.openmvs_bin, args.reprendre_apres)
    except Exception as e:      # un scan raté ne doit pas faire planter une série : message clair, code d'erreur
        print(f"\nERREUR sur « {args.nom} » : {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
