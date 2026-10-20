"""
Étape 3 du pipeline : maillage Poisson + export STL pour l'impression 3D.

Pourquoi : une imprimante 3D a besoin d'une surface fermée (un maillage),
pas d'un nuage de points.

Entrée  : colmap_workspace/openmvs/scene_dense.ply (nuage dense d'OpenMVS)
Sortie  : colmap_workspace/modele_pour_impression.stl

Paramètres de la fonction :
    - profondeur_poisson : finesse du maillage (plus grand = plus de détails, plus long)
    - cible_triangles    : nombre de triangles visé après simplification (None = pas de simplification)
    - reparer_etancheite : True = tente de boucher les trous (voir l'avertissement plus bas)
    - afficher_interface : True = ouvre la fenêtre 3D à la fin

Lancement seul : python maillage.py [dossier_de_travail]
"""

import os
import sys

import numpy as np
import open3d as o3d


def mailler_et_preparer_impression(dossier_travail="colmap_workspace",
                                   profondeur_poisson=10,
                                   cible_triangles=1500000,
                                   reparer_etancheite=False,
                                   afficher_interface=True):

    chemin_ply = os.path.join(dossier_travail, "openmvs", "scene_dense.ply")
    print(f"Chargement du fichier : {chemin_ply}")
    pcd = o3d.io.read_point_cloud(chemin_ply)
    # read_point_cloud ne lève pas d'erreur si le fichier manque, il renvoie
    # un nuage vide : sans ce test on aurait une erreur incompréhensible plus loin
    if pcd.is_empty():
        raise FileNotFoundError(
            f"{chemin_ply} introuvable ou vide (as-tu lancé la densification ?)")
    print(f"{len(pcd.points)} points chargés")

    # Les points isolés dans le vide font pousser des "excroissances" sur la
    # surface Poisson, donc on les enlève avant (std_ratio=2 : on ne supprime
    # que les points vraiment loin de leurs voisins)
    print("Nettoyage des points isolés sur CPU")
    pcd_propre, _ = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)

    # Poisson a besoin de normales orientées VERS L'EXTÉRIEUR, sinon la
    # surface sort à l'envers.
    # On garde celles d'OpenMVS, parce qu'il les oriente avec la position des
    # caméras, donc correctement. Il ne faut PAS les orienter vers (0,0,0) :
    # dans le repère COLMAP, ce point tombe au centre de l'objet, donc toutes
    # les normales pointaient vers l'intérieur. Vérifié sur le dé : 99,9 % des
    # triangles étaient retournés.
    if pcd_propre.has_normals():
        print("Normales fournies par OpenMVS conservées (déjà orientées vers l'extérieur)")
    else:
        # Cas de secours si un jour le nuage n'a pas de normales
        print("Calcul des normales")
        pcd_propre.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )
        # Open3D sait seulement orienter VERS un point : on oriente vers le
        # centre de l'objet puis on inverse pour pointer vers l'extérieur.
        # Ça marche pour une forme compacte (dé, pale), moins pour une forme en U.
        centre = pcd_propre.get_center()
        pcd_propre.orient_normals_towards_camera_location(camera_location=centre)
        pcd_propre.normals = o3d.utility.Vector3dVector(-np.asarray(pcd_propre.normals))

    # Poisson plutôt qu'un maillage direct des points, parce qu'il donne une
    # surface fermée et lisse, ce qui est ce qu'il faut pour imprimer
    print(f"Génération du maillage 3D (Poisson, profondeur={profondeur_poisson})...")
    mesh, densites = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_propre, depth=profondeur_poisson
    )

    # Désactivé pour l'instant : enlever les zones de faible densité supprime
    # les "bulles" que Poisson invente là où il n'y a pas de points, mais ça
    # ouvre des trous et le maillage n'est plus étanche
    # print("Nettoyage des bordures du maillage...")
    # triangles_a_enlever = densites < np.quantile(densites, 0.001)
    # mesh.remove_vertices_by_mask(triangles_a_enlever)

    # Simplification pour garder un STL d'une taille raisonnable pour le
    # logiciel d'impression, avec une perte de détail faible
    if cible_triangles is not None and len(mesh.triangles) > cible_triangles:
        nb_avant = len(mesh.triangles)
        print(f"Simplification du maillage : {nb_avant} -> {cible_triangles} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=cible_triangles)
        print(f"Simplification terminée : {len(mesh.triangles)} triangles.")
    else:
        print(f"Simplification ignorée ({len(mesh.triangles)} triangles)")

    # L'imprimante a besoin d'un maillage étanche (sans trou) pour savoir ce
    # qui est "dedans" et le remplir.
    # Le résultat est gardé dans une variable parce que ce test est très lent
    # (plusieurs minutes sur ~500 000 triangles) : avant il était fait deux fois.
    print("Vérification de l'étanchéité (peut être long sur un gros maillage)...")
    etanche = mesh.is_watertight()
    print(f"Maillage étanche : {etanche}")

    if not etanche and reparer_etancheite:
        # Désactivé par défaut : fill_holes() a fait planter Python sans
        # message sur un maillage dense simplifié, sur le PC du stage.
        # Pistes : l'essayer AVANT la simplification, ou réparer le STL
        # dans Blender ou Meshmixer.
        print("Réparation des trous (peut prendre un moment sur un maillage dense)...")
        mesh_tenseur = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_tenseur = mesh_tenseur.fill_holes()
        mesh = mesh_tenseur.to_legacy()
        print(f"Maillage étanche après réparation : {mesh.is_watertight()}")
    elif not etanche:
        print("Réparation automatique désactivée (reparer_etancheite=False) — "
              "le .stl exporté n'est donc pas garanti étanche. Pour l'impression "
              "finale : réessaie avec reparer_etancheite=True, ou répare le "
              "fichier dans Blender/Meshmixer avant de slicer.")

    # Obligatoire AVANT l'export : sans normales, l'export STL écrit un
    # fichier de 0 octet sans aucun message d'erreur
    mesh.compute_vertex_normals()

    # STL parce que c'est le format que lisent tous les logiciels d'impression
    # (Cura, PrusaSlicer...)
    chemin_stl = os.path.join(dossier_travail, "modele_pour_impression.stl")
    ok = o3d.io.write_triangle_mesh(chemin_stl, mesh)
    # write_triangle_mesh ne lève pas d'exception en cas d'échec, il renvoie
    # juste False : il faut donc tester nous-mêmes
    if not ok:
        raise RuntimeError(f"Échec de l'export STL vers {chemin_stl} (write_triangle_mesh a renvoyé False).")
    taille_octets = os.path.getsize(chemin_stl)
    print(f"Fichier STL exporté pour impression 3D : {chemin_stl} ({taille_octets/1e6:.1f} Mo)")

    # Désactivable parce que la fenêtre bloque le programme jusqu'à sa
    # fermeture : l'interface graphique appelle donc cette fonction sans affichage
    if afficher_interface:
        print("Affichage de la surface 3D maillée dans l'interface avancée...")
        o3d.visualization.gui.Application.instance.initialize()
        vis = o3d.visualization.O3DVisualizer("Maillage", 1280, 720)
        vis.show_settings = True
        vis.add_geometry({"name": "Maillage_Poisson", "geometry": mesh})
        vis.reset_camera_to_default()
        vis.scene.show_skybox(True)
        vis.scene.show_axes(True)
        o3d.visualization.gui.Application.instance.add_window(vis)
        o3d.visualization.gui.Application.instance.run()    
    return chemin_stl


if __name__ == "__main__":
    # Dossier en argument, pour pouvoir travailler sur un autre dossier que
    # celui par défaut sans modifier le code
    dossier = sys.argv[1] if len(sys.argv) > 1 else "colmap_workspace"
    mailler_et_preparer_impression(dossier_travail=dossier)
