"""
Étape 1 du pipeline : reconstruction SfM ("Structure from Motion").

Retrouve la position de l'appareil pour chaque photo et un premier nuage
de points épars. Ce nuage est trop pauvre pour imprimer, mais les positions
des caméras sont indispensables à la densification (densification_openmvs.py).

Lancement seul : python main.py  (les photos doivent être dans "Photos/")
"""

import os
import shutil

import cv2
import pycolmap

from masquage_fond import ecrire_image, generer_masques, lire_image, lister_images


def executer_pipeline_securise(
    dossier_source="Photos",
    dossier_travail="colmap_workspace",
    masquer_fond=False,
    repartir_de_zero=True,
):
    """Lance toute la partie COLMAP (SfM).

    Paramètres :
        dossier_source   : dossier qui contient les photos d'origine
        dossier_travail  : dossier où tout est généré (base COLMAP, modèles...)
        masquer_fond     : True = génère des masques GrabCut pour ignorer le fond
        repartir_de_zero : True = supprime le dossier de travail avant de commencer

    Retourne le dossier des images redimensionnées (réutilisé par la densification).
    """

    # Si on garde les fichiers d'un ancien essai, COLMAP mélange les anciennes
    # et les nouvelles images et on ne sait plus d'où vient le résultat
    if repartir_de_zero and os.path.exists(dossier_travail):
        print(f"Nettoyage de {dossier_travail}, l'essai précédent")
        shutil.rmtree(dossier_travail)

    dossier_redim = os.path.join(dossier_travail, "images_redimensionnees")
    os.makedirs(dossier_redim, exist_ok=True)

    fichiers_photos = lister_images(dossier_source)

    # Sans ce test, un mauvais dossier fait planter pycolmap bien plus loin
    # avec un message qui ne parle pas du tout des photos
    if not fichiers_photos:
        raise FileNotFoundError(
            f"Aucune photo trouvée dans '{os.path.abspath(dossier_source)}'"
            f"Dossier photo à vérifier"
        )

    print(f"Étape 0 : Optimisation de {len(fichiers_photos)} images pour la RAM")
    # Avec les photos en pleine résolution (12 MP), le PC du stage saturait
    # en RAM. 1600 px reste assez fin pour retrouver les points d'intérêt.
    max_dimension = 1600
    nb_ok = 0
    for chemin in fichiers_photos:
        # lire_image et pas cv2.imread : cv2.imread échoue sous Windows dès
        # qu'il y a un accent dans le chemin
        img = lire_image(chemin)
        if img is None:
            print(f" Ignorée (Ilisible) : {chemin}")
            continue
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            echelle = max_dimension / float(max(h, w))
            img = cv2.resize(
                img,
                (int(w * echelle), int(h * echelle)),
                # INTER_AREA car c'est la méthode qui garde le plus de
                # netteté quand on RÉDUIT une image
                interpolation=cv2.INTER_AREA,
            )
        nom_fichier = os.path.basename(chemin)
        ecrire_image(os.path.join(dossier_redim, nom_fichier), img)
        nb_ok += 1

    # En dessous de 3 photos, COLMAP ne peut rien trianguler : autant
    # prévenir tout de suite
    if nb_ok < 3:
        raise RuntimeError(f"Seulement {nb_ok} photo(s) lisible(s) : il en faut beaucoup plus (40-50 minimum).")

    chemin_bdd = os.path.join(dossier_travail, "database.db")
    chemin_sparse = os.path.join(dossier_travail, "sparse")
    os.makedirs(chemin_sparse, exist_ok=True)

    # Une base restée d'un essai interrompu peut être corrompue, et COLMA
    # y ajouterait les nouvelles images au lieu de repartir de zéro
    
    if os.path.exists(chemin_bdd):  
        os.remove(chemin_bdd)

    options_lecture = pycolmap.ImageReaderOptions()
    if masquer_fond:
        print("Étape 0bis : génération des masques de fond (GrabCut)...")
        dossier_masques = os.path.join(dossier_travail, "masques")
        # Masques faits sur les images redimensionnées et pas sur les
        # originales : COLMAP exige qu'ils aient exactement la même taille
        generer_masques(dossier_redim, dossier_masques)
        options_lecture.mask_path = dossier_masques

    print("Étape 1/3 : Extraction des points d'intérêt (SIFT)...")
    pycolmap.extract_features(chemin_bdd, dossier_redim, reader_options=options_lecture)

    # Appariement exhaustif (toutes les paires) : avec ~50 photos ça reste
    # rapide, et on ne rate aucune paire qui se recouvre. Pour des centaines
    # de photos il faudrait passer à un appariement séquentiel.
    # guided_matching : récupère des correspondances en plus grâce à la
    # géométrie déjà validée -> plus de points au final, pour un peu plus de temps.
    print("Étape 2/3 : Mise en correspondance des points entre les images...")
    options_matching = pycolmap.FeatureMatchingOptions()
    options_matching.guided_matching = True
    pycolmap.match_exhaustive(chemin_bdd, matching_options=options_matching)

    print("Étape 3/3 : Reconstruction 3D (SfM) en cours...")
    reconstructions = pycolmap.incremental_mapping(
        chemin_bdd, dossier_redim, chemin_sparse
    )

    if not reconstructions:
        raise RuntimeError(
            "Aucune reconstruction calculée. Vérifiez le chevauchement des photos."
        )

    # COLMAP sort plusieurs modèles (0, 1, 2...) quand certaines photos ne
    # se relient pas aux autres : le plus gros est forcément le bon objet
    meilleure_rec = max(reconstructions.values(), key=lambda r: r.num_points3D())
    print(
        f"SUCCÈS : {meilleure_rec.num_points3D()} points 3D et "
        f"{meilleure_rec.num_reg_images()} caméras positionnées "
        f"(sur {nb_ok} photos) !"
    )
    # Photos non positionnées = presque toujours un manque de recouvrement :
    # le signaler aide à corriger la prise de vue la fois suivante
    if meilleure_rec.num_reg_images() < nb_ok:
        print(f"  Attention : {nb_ok - meilleure_rec.num_reg_images()} photo(s) "
              f"non positionnée(s) (recouvrement insuffisant ?)")

    # On recopie le modèle retenu dans un dossier fixe, sinon la densification
    # devrait deviner lequel de sparse/0, sparse/1... est le bon
    chemin_sparse_final = os.path.join(dossier_travail, "sparse_final")
    os.makedirs(chemin_sparse_final, exist_ok=True)
    meilleure_rec.write(chemin_sparse_final)
    print(f"Modèle sparse retenu écrit dans : {chemin_sparse_final}")

    # Export du nuage épars juste pour un aperçu rapide : il est trop pauvre
    # pour le maillage, c'est le nuage dense d'OpenMVS qui sert pour le STL
    chemin_ply = os.path.join(dossier_travail, "modele_3d_propre.ply")
    meilleure_rec.export_PLY(chemin_ply)
    print(f"Aperçu du nuage épars exporté : {chemin_ply}")
    print(
        "\nProchaine étape : lance densification_openmvs.py pour densifier "
        "ce nuage avant le maillage."
    )

    # La densification doit travailler sur les MÊMES images que COLMAP
    # (même taille), d'où le renvoi de ce dossier
    return dossier_redim


if __name__ == "__main__":
    executer_pipeline_securise(dossier_source="Photos")
