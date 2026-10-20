"""
Visualiseur du maillage final (modele_pour_impression.stl).

Lancé par le bouton "Afficher l'objet" de interface_pipeline.py. C'est un
fichier séparé de maillage.py pour que regarder l'objet ne relance pas le
calcul (avant, le bouton lançait maillage.py et écrasait le STL fait avec
les réglages de l'utilisateur).

Lancement seul : python interface_maillage.py [dossier_de_travail]
"""

import os
import sys

import open3d as o3d

# Dossier passé par l'interface plutôt qu'un chemin écrit en dur, pour
# afficher le bon fichier quand on change de dossier de travail
dossier_travail = sys.argv[1] if len(sys.argv) > 1 else "colmap_workspace"
chemin_modele = os.path.join(dossier_travail, "modele_pour_impression.stl")

# Open3D refuse de créer une fenêtre tant que son moteur graphique n'est pas initialisé
o3d.visualization.gui.Application.instance.initialize()

# O3DVisualizer plutôt que la fenêtre simple d'Open3D, pour avoir le panneau
# de réglages (éclairage, matériaux, affichage des arêtes...)
vis = o3d.visualization.O3DVisualizer("Logiciel de Photogrammétrie - Maillage 3D", 1280, 720)
vis.show_settings = True

print(f"Chargement du maillage depuis : {chemin_modele}...")
mesh = o3d.io.read_triangle_mesh(chemin_modele)

# Open3D renvoie un maillage vide au lieu d'une erreur si le fichier manque :
# on s'arrête plutôt que d'ouvrir une fenêtre vide
if mesh.is_empty():
    print("Erreur : Le fichier de maillage est introuvable ou vide.")
    sys.exit(1)

print(f"Succès : maillage chargé ({len(mesh.triangles)} triangles).")

# Sans normales, l'éclairage ne marche pas et l'objet apparaît tout plat
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

vis.add_geometry({"name": "Maillage_Final", "geometry": mesh})

# Sinon la caméra démarre au hasard et l'objet peut être hors de l'écran
vis.reset_camera_to_default()

# Fond et axes : aident à se repérer quand on fait tourner l'objet
vis.scene.show_skybox(True)
vis.scene.show_axes(True)

o3d.visualization.gui.Application.instance.add_window(vis)

print("Ouverture de l'interface graphique avancée...")
o3d.visualization.gui.Application.instance.run()
