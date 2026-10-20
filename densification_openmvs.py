"""
Étape 2 du pipeline : densification du nuage de points via OpenMVS (sur CPU).

À lancer APRÈS main.py.

Pourquoi cette étape : le nuage de main.py ne contient que quelques milliers
de points (seulement les points SIFT), trop peu pour un maillage imprimable.
La densification utilise tous les pixels des photos et monte à plusieurs
centaines de milliers de points.

Pourquoi OpenMVS et pas COLMAP : la densification de COLMAP
(patch_match_stereo) exige une carte graphique Nvidia (CUDA), qu'on n'a pas.
La version standard d'OpenMVS calcule sur le processeur.

Prérequis :
1. Télécharger les binaires Windows d'OpenMVS :
   https://github.com/cdcseacave/openMVS/releases
   et les dézipper, ex : C:\\OpenMVS
2. Mettre ce chemin dans OPENMVS_BIN ci-dessous (ou le choisir dans l'interface).
"""

import os
import shutil
import subprocess
import sys

# Chemin écrit en dur car il dépend de l'endroit où OpenMVS a été dézippé
# sur chaque PC : à adapter (l'interface permet aussi de le changer)
OPENMVS_BIN = r"C:\OpenMVS\vc17\x64\Release"


def _lancer(nom_exe, arguments, dossier_travail_courant):
    """Lance un exécutable OpenMVS et lève une erreur claire s'il échoue."""
    chemin_exe = os.path.join(OPENMVS_BIN, nom_exe)
    if not os.path.exists(chemin_exe):
        raise FileNotFoundError(
            f"Introuvable : {chemin_exe}\n"
            f"Vérifie que OPENMVS_BIN pointe bien vers le dossier contenant "
            f"les .exe d'OpenMVS (téléchargés depuis github.com/cdcseacave/openMVS/releases)."
        )

    # Arguments passés en liste et pas en une seule chaîne : les espaces dans
    # les chemins ne cassent pas la commande
    
    commande = [chemin_exe] + arguments
    print(f"\n>>> {' '.join(commande)}\n")
    # cwd = dossier openmvs, parce qu'OpenMVS écrit ses fichiers de sortie
    # dans le dossier courant.
    # capture_output : sinon la sortie s'affiche en direct et se fait couper
    # par la fenêtre du terminal, et on perd le message d'erreur utile.
    resultat = subprocess.run(
        commande,
        cwd=dossier_travail_courant,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    # Copie sur disque pour pouvoir relire l'erreur après coup
    sortie_complete = (resultat.stdout or "") + (resultat.stderr or "")
    chemin_log = os.path.join(dossier_travail_courant, f"{nom_exe}.log")
    with open(chemin_log, "w", encoding="utf-8") as f:
        f.write(sortie_complete)

    if resultat.returncode != 0:
        print(sortie_complete)
        # On pointe aussi vers les logs horodatés d'OpenMVS, parce que notre
        # .log est souvent vide : OpenMVS écrit surtout dans ses propres fichiers
        raise RuntimeError(
            f"{nom_exe} a échoué (code {resultat.returncode}).\n"
            f"Log capturé : {chemin_log}\n"
            f"Si ce fichier est vide, regarde le log horodaté d'OpenMVS "
            f"({os.path.splitext(nom_exe)[0]}-xxxx.log) dans {dossier_travail_courant}\n"
            f"(les lignes contenant 'error' sont généralement la cause réelle)"
        )
    else:
        # Les dernières lignes contiennent le bilan (nombre de points, etc.),
        # pratique pour comparer les essais sans ouvrir les logs
        print("\n".join(sortie_complete.strip().splitlines()[-10:]))


def densifier(dossier_travail="colmap_workspace", repartir_de_zero=True):
    """
    Enchaîne : undistort (pycolmap) -> InterfaceCOLMAP -> DensifyPointCloud
    -> ReconstructMesh -> TextureMesh.

    repartir_de_zero : True = vide le dossier openmvs/ avant de commencer.
    Mettre False seulement pour comparer plusieurs essais dans le même dossier.

    Retourne le chemin du nuage dense (.ply) et du maillage texturé.
    """
    # Import dans la fonction et pas en haut du fichier : comme ça on peut
    # importer ce module (par ex. pour lire OPENMVS_BIN) même sans pycolmap
    import pycolmap

    chemin_sparse_final = os.path.join(dossier_travail, "sparse_final")
    dossier_images = os.path.join(dossier_travail, "images_redimensionnees")
    dossier_openmvs = os.path.join(dossier_travail, "openmvs")
    dossier_undistort = os.path.join(dossier_openmvs, "undistorted")

    if not os.path.exists(chemin_sparse_final):
        raise FileNotFoundError(
            f"{chemin_sparse_final} introuvable. Lance d'abord main.py "
            f"(executer_pipeline_securise) pour générer la reconstruction sparse."
        )

    # Les anciens .dmap et .mvs seraient réutilisés par OpenMVS et fausseraient
    # le nouvel essai
    if repartir_de_zero and os.path.exists(dossier_openmvs):
        print(f"Nettoyage de {dossier_openmvs} (essai précédent)...")
        shutil.rmtree(dossier_openmvs)

    os.makedirs(dossier_openmvs, exist_ok=True)

    # OpenMVS ne sait pas gérer la déformation de l'objectif : il faut lui
    # donner des images déjà "redressées" (modèle sténopé, sans distorsion)
    print("Étape 1/4 : undistortion des images (CPU, via pycolmap)...")
    pycolmap.undistort_images(
        output_path=dossier_undistort,
        input_path=chemin_sparse_final,
        image_path=dossier_images,
    )

    # OpenMVS ne lit pas le format COLMAP directement, d'où la conversion en .mvs
    print("Étape 2/4 : conversion vers le format OpenMVS...")
    chemin_scene_mvs = "scene.mvs"
    _lancer(
        "InterfaceCOLMAP.exe",
        ["-i", os.path.abspath(dossier_undistort),
         "-o", chemin_scene_mvs,
         "--image-folder", os.path.join(os.path.abspath(dossier_undistort), "images")],
        dossier_travail_courant=dossier_openmvs,
    )

    # --tower-mode 0 : ce mode est fait pour les scènes de pylônes ou
    # bâtiments hauts, il n'apporte rien pour un petit objet posé sur une table
    print("Étape 3/4 : densification du nuage de points")
    _lancer(
        "DensifyPointCloud.exe",
        [chemin_scene_mvs, "--tower-mode", "0"],
        dossier_travail_courant=dossier_openmvs,
    )
    chemin_nuage_dense = os.path.join(dossier_openmvs, "scene_dense.ply")

    # Maillage + texture OpenMVS : pas utilisé pour le STL, mais gardé pour
    # comparer avec le maillage Poisson d'Open3D.
    # "-p scene_dense.ply" est indispensable : OpenMVS v2 ne range PAS le nuage
    # dense dans scene_dense.mvs (seulement les ~12 000 points de COLMAP).
    # Sans "-p", le maillage était fait sur ces points épars : grossier et
    # texture trouée. Voir github.com/cdcseacave/openMVS/wiki/Usage.
    print("Étape 4/4 : reconstruction du maillage + texturation...")
    _lancer("ReconstructMesh.exe",
            ["scene_dense.mvs", "-p", "scene_dense.ply",
             "-o", "scene_dense_mesh.mvs", "--archive-type", "2"],
            dossier_travail_courant=dossier_openmvs)
    _lancer("TextureMesh.exe", ["scene_dense_mesh.mvs", "--archive-type", "2"],
            dossier_travail_courant=dossier_openmvs)
    chemin_mesh_texture = os.path.join(dossier_openmvs, "scene_dense_mesh_texture.ply")

    print("\nSUCCÈS :")
    print(f"  Nuage dense       : {chemin_nuage_dense}")
    print(f"  Maillage texturé  : {chemin_mesh_texture}")
    print("Étape suivante : maillage.py (maillage Poisson + export STL)")

    return chemin_nuage_dense, chemin_mesh_texture


if __name__ == "__main__":
    # On vérifie le chemin avant de lancer, sinon on attend toute
    # l'undistortion pour découvrir à la fin qu'OpenMVS est introuvable
    if not os.path.isdir(OPENMVS_BIN):
        print(f"ERREUR : OPENMVS_BIN ('{OPENMVS_BIN}') n'existe pas.")
        print("Télécharge OpenMVS ici : https://github.com/cdcseacave/openMVS/releases")
        print("puis modifie la variable OPENMVS_BIN en haut de ce fichier.")
        sys.exit(1)

    densifier(dossier_travail="colmap_workspace")
