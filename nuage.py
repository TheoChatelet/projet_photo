"""
Visualiseur du nuage de points dense (scene_dense.ply).

Lancé par le bouton "Afficher le nuage dense en 3D" de interface_pipeline.py.
Il ne fait QUE de l'affichage : l'ancienne version de ce fichier était une
copie de maillage.py, donc cliquer sur le bouton recalculait tout le maillage
et écrasait le STL.

Lancement seul : python nuage.py [dossier_de_travail]
"""

import os
import sys

import open3d as o3d


def afficher_nuage(dossier_travail="colmap_workspace"):
    chemin_ply = os.path.join(dossier_travail, "openmvs", "scene_dense.ply")
    print(f"Chargement du nuage dense : {chemin_ply}")
    pcd = o3d.io.read_point_cloud(chemin_ply)
    # Open3D renvoie un nuage vide au lieu d'une erreur si le fichier manque :
    # sans ce test on ouvrirait une fenêtre noire sans comprendre pourquoi
    if pcd.is_empty():
        print("Erreur : nuage introuvable ou vide (lance d'abord la densification).")
        return
    print(f"{len(pcd.points)} points chargés")

    # Même fenêtre que pour le maillage, pour avoir les mêmes réglages et
    # pouvoir comparer nuage et maillage facilement
    o3d.visualization.gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer("Nuage de points dense", 1280, 720)
    vis.show_settings = True
    vis.add_geometry({"name": "Nuage_dense", "geometry": pcd})
    # Sinon la caméra démarre au hasard et le nuage peut être hors de l'écran
    vis.reset_camera_to_default()
    vis.scene.show_axes(True)
    o3d.visualization.gui.Application.instance.add_window(vis)
    o3d.visualization.gui.Application.instance.run()


if __name__ == "__main__":
    # L'interface passe le dossier de travail choisi par l'utilisateur ;
    # sans argument on prend le dossier par défaut
    dossier = sys.argv[1] if len(sys.argv) > 1 else "colmap_workspace"
    afficher_nuage(dossier)
