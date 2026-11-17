"""
Interface graphique du pipeline de photogrammétrie.

Interface Tkinter volontairement "classique" (thème par défaut du système,
aucune couleur imposée) permettant de :
  - lancer chaque étape du pipeline (reconstruction, densification,
    maillage optionnel, mise à l'échelle et mesures en mm) avec ses
    paramètres modifiables,
  - suivre la progression dans une console de journal intégrée,
  - visualiser le nuage de points ou le maillage final en 3D, mais
    uniquement sur demande explicite (bouton), jamais automatiquement :
    la fenêtre 3D s'ouvre dans un processus séparé quand tu cliques sur
    "Afficher...", une fois le fichier correspondant disponible.

Lancement : python interface_pipeline.py
(depuis le dossier du projet, car les chemins par défaut sont relatifs à ce dossier)
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Chaque étape reste dans son propre fichier pour pouvoir la lancer et la
# tester seule, sans l'interface
import densification_openmvs
import echelle
import generer_cible_echelle
import mesure
from maillage import mailler_et_preparer_impression
from main import executer_pipeline_securise


class RedirecteurConsole:
    """Redirige les print() du pipeline (exécuté dans un thread séparé)
    vers une file d'attente, relue en toute sécurité par le thread
    principal de Tkinter (on ne doit jamais toucher un widget depuis un
    autre thread que celui de la boucle graphique)."""

    def __init__(self, file_attente):
        self.file_attente = file_attente

    def write(self, texte):
        if texte:
            self.file_attente.put(texte)

    def flush(self):
        pass


class InterfacePipeline:
    def __init__(self, racine):
        self.racine = racine
        racine.title("Pipeline Photogrammétrie")
        racine.minsize(900, 700)

        # File d'attente parce que le thread de calcul n'a pas le droit
        # d'écrire lui-même dans la fenêtre (Tkinter n'est pas thread-safe)
        self.file_journal = queue.Queue()
        # Une seule étape à la fois : deux étapes en parallèle écriraient
        # dans le même dossier de travail
        self.en_cours = False

        self.var_dossier_source = tk.StringVar(value="Photos")
        self.var_dossier_travail = tk.StringVar(value="colmap_workspace")

        self.var_masquer_fond = tk.BooleanVar(value=False)
        self.var_repartir_zero_1 = tk.BooleanVar(value=True)
        self.var_camera_unique = tk.BooleanVar(value=True)

        self.var_openmvs_bin = tk.StringVar(value=densification_openmvs.OPENMVS_BIN)
        self.var_repartir_zero_2 = tk.BooleanVar(value=True)

        self.var_profondeur_poisson = tk.IntVar(value=10)
        self.var_limiter_triangles = tk.BooleanVar(value=True)
        self.var_cible_triangles = tk.StringVar(value="1500000")
        self.var_reparer_etancheite = tk.BooleanVar(value=False)

        self.var_mise_en_page = tk.StringVar(value="cible_echelle.json")
        self.var_taille_marqueur = tk.StringVar(value="")
        self.var_z_coupe = tk.StringVar(value="1.0")
        self.var_nom_mesure = tk.StringVar(value="mesure_01")

        self.boutons_lancement = []
        self.boutons_affichage = {}

        self._construire_interface()
        # Les boutons "Afficher" doivent suivre le dossier choisi : un autre
        # dossier n'a pas forcément de nuage ou de STL
        self.var_dossier_travail.trace_add(
            "write", self._verifier_fichiers_et_mettre_a_jour
        )
        self._rafraichir_boutons_affichage()
        self.racine.after(150, self._lire_file_journal)

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------
    def _construire_interface(self):
        conteneur = ttk.Frame(self.racine, padding=8)
        conteneur.pack(fill="both", expand=True)
        conteneur.columnconfigure(0, weight=1)
        # Seule la console grandit avec la fenêtre : c'est elle qu'on a
        # besoin de lire en entier, les réglages ont une taille fixe
        conteneur.rowconfigure(5, weight=1)

        self._section_dossiers(conteneur).grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )
        self._section_etape1(conteneur).grid(row=1, column=0, sticky="ew", pady=6)
        self._section_etape2(conteneur).grid(row=2, column=0, sticky="ew", pady=6)
        self._section_etape3(conteneur).grid(row=3, column=0, sticky="ew", pady=6)
        self._section_etape4(conteneur).grid(row=4, column=0, sticky="ew", pady=6)
        self._section_journal(conteneur).grid(
            row=5, column=0, sticky="nsew", pady=(6, 0)
        )

        # Barre "indeterminate" (qui défile) car COLMAP et OpenMVS ne donnent
        # pas d'avancement en % : impossible d'afficher une vraie progression
        bas = ttk.Frame(conteneur)
        bas.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        bas.columnconfigure(0, weight=1)
        self.etiquette_statut = ttk.Label(bas, text="Prêt.", anchor="w")
        self.etiquette_statut.grid(row=0, column=0, sticky="ew")
        self.barre_progression = ttk.Progressbar(bas, mode="indeterminate", length=160)
        self.barre_progression.grid(row=0, column=1, sticky="e", padx=(6, 0))

    def _parcourir_dossier(self, variable):
        chemin = filedialog.askdirectory()
        if chemin:
            variable.set(chemin)

    def _section_dossiers(self, parent):
        cadre = ttk.LabelFrame(parent, text="Dossiers du projet")
        cadre.columnconfigure(1, weight=1)

        ttk.Label(cadre, text="Dossier photos :").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Entry(cadre, textvariable=self.var_dossier_source).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(
            cadre,
            text="Parcourir",
            command=lambda: self._parcourir_dossier(self.var_dossier_source),
        ).grid(row=0, column=2, padx=4, pady=3)

        ttk.Label(cadre, text="Dossier de travail :").grid(
            row=1, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Entry(cadre, textvariable=self.var_dossier_travail).grid(
            row=1, column=1, sticky="ew", padx=4
        )
        ttk.Button(
            cadre,
            text="Parcourir",
            command=lambda: self._parcourir_dossier(self.var_dossier_travail),
        ).grid(row=1, column=2, padx=4, pady=3)

        return cadre

    def _section_etape1(self, parent):
        cadre = ttk.LabelFrame(parent, text="Reconstruction (SfM, main.py)")
        cadre.columnconfigure(3, weight=1)

        ttk.Checkbutton(
            cadre, text="Masquer le fond", variable=self.var_masquer_fond
        ).grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(
            cadre, text="Repartir de zéro", variable=self.var_repartir_zero_1
        ).grid(row=0, column=1, sticky="w", padx=4, pady=3)

        ttk.Checkbutton(
            cadre,
            text="Une seule caméra (mêmes réglages pour toutes les photos)",
            variable=self.var_camera_unique,
        ).grid(row=0, column=2, sticky="w", padx=4, pady=3)

        bouton = ttk.Button(cadre, text="Lancer", command=self._lancer_etape1)
        bouton.grid(row=0, column=4, sticky="e", padx=4, pady=3)
        self.boutons_lancement.append(bouton)

        return cadre

    def _section_etape2(self, parent):
        cadre = ttk.LabelFrame(parent, text="Densification (OpenMVS)")
        cadre.columnconfigure(1, weight=1)

        ttk.Label(cadre, text="Dossier OpenMVS :").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Entry(cadre, textvariable=self.var_openmvs_bin).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(
            cadre,
            text="Parcourir",
            command=lambda: self._parcourir_dossier(self.var_openmvs_bin),
        ).grid(row=0, column=2, padx=4, pady=3)
        ttk.Checkbutton(
            cadre, text="Repartir de zéro", variable=self.var_repartir_zero_2
        ).grid(row=1, column=0, sticky="w", padx=4, pady=3)

        ligne_boutons = ttk.Frame(cadre)
        ligne_boutons.grid(row=1, column=1, columnspan=2, sticky="e", padx=4, pady=3)
        bouton_lancer = ttk.Button(
            ligne_boutons, text="Lancer", command=self._lancer_etape2
        )
        bouton_lancer.pack(side="left", padx=(0, 6))
        self.boutons_lancement.append(bouton_lancer)

        bouton_pipeline_complet = ttk.Button(
            cadre, text="Lancer pipeline complet", command=self._lancer_pipeline_complet
        )
        bouton_pipeline_complet.grid(
            row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 3)
        )
        self.boutons_lancement.append(bouton_pipeline_complet)

        bouton_afficher_nuage = ttk.Button(
            cadre,
            text="Afficher le nuage dense en 3D",
            command=self._afficher_nuage,
            state="disabled",
        )
        bouton_afficher_nuage.grid(row=2, column=2, sticky="e", padx=4, pady=(6, 3))
        self.boutons_affichage["nuage"] = bouton_afficher_nuage

        return cadre

    def _section_etape3(self, parent):
        cadre = ttk.LabelFrame(parent, text="Maillage Poisson")
        for i in range(6):
            cadre.columnconfigure(i, weight=1 if i == 5 else 0)

        ttk.Label(cadre, text="Profondeur Poisson :").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Spinbox(
            cadre, from_=4, to=14, width=5, textvariable=self.var_profondeur_poisson
        ).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Checkbutton(
            cadre,
            text="Limiter le nombre de triangles :",
            variable=self.var_limiter_triangles,
            command=self._basculer_champ_triangles,
        ).grid(row=0, column=2, sticky="w", padx=(16, 4))
        self.champ_cible_triangles = ttk.Entry(
            cadre, textvariable=self.var_cible_triangles, width=10
        )
        self.champ_cible_triangles.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Checkbutton(
            cadre,
            text="Réparer l'étanchéité (pas fini)",
            variable=self.var_reparer_etancheite,
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=4, pady=3)

        bouton_lancer = ttk.Button(cadre, text="Lancer", command=self._lancer_etape3)
        bouton_lancer.grid(row=1, column=7, sticky="e", padx=4, pady=3)
        self.boutons_lancement.append(bouton_lancer)

        bouton_afficher_maillage = ttk.Button(
            cadre,
            text="Afficher l'objet",
            command=self._afficher_maillage,
            state="disabled",
        )
        bouton_afficher_maillage.grid(row=1, column=5, sticky="e", padx=4, pady=3)
        self.boutons_affichage["maillage"] = bouton_afficher_maillage

        return cadre

    def _section_etape4(self, parent):
        cadre = ttk.LabelFrame(parent, text="Échelle et mesures (en mm)")
        cadre.columnconfigure(1, weight=1)

        ttk.Label(cadre, text="Cible d'échelle :").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Entry(cadre, textvariable=self.var_mise_en_page).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(
            cadre, text="Créer la cible à imprimer", command=self._creer_cible
        ).grid(row=0, column=2, padx=4, pady=3)

        ttk.Label(cadre, text="Côté d'un marqueur mesuré (mm, facultatif) :").grid(
            row=1, column=0, sticky="w", padx=4, pady=3
        )
        ttk.Entry(cadre, textvariable=self.var_taille_marqueur, width=10).grid(
            row=1, column=1, sticky="w", padx=4
        )
        bouton_echelle = ttk.Button(
            cadre, text="Calculer l'échelle", command=self._lancer_etape4a
        )
        bouton_echelle.grid(row=1, column=2, sticky="e", padx=4, pady=3)
        self.boutons_lancement.append(bouton_echelle)

        ligne = ttk.Frame(cadre)
        ligne.grid(row=2, column=0, columnspan=3, sticky="ew", pady=3)
        ttk.Label(ligne, text="Nom de la mesure :").pack(side="left", padx=4)
        ttk.Entry(ligne, textvariable=self.var_nom_mesure, width=18).pack(side="left", padx=4)
        ttk.Label(ligne, text="Coupe au plateau (mm) :").pack(side="left", padx=(16, 4))
        ttk.Entry(ligne, textvariable=self.var_z_coupe, width=6).pack(side="left", padx=4)
        bouton_mesure = ttk.Button(ligne, text="Mesurer", command=self._lancer_etape4b)
        bouton_mesure.pack(side="right", padx=4)
        self.boutons_lancement.append(bouton_mesure)

        return cadre

    def _section_journal(self, parent):
        cadre = ttk.LabelFrame(parent, text="Console")
        cadre.columnconfigure(0, weight=1)
        cadre.rowconfigure(0, weight=1)
        self.zone_journal = scrolledtext.ScrolledText(
            cadre, height=16, state="disabled", font=("Courier New", 9), wrap="word"
        )
        self.zone_journal.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        return cadre

    def _basculer_champ_triangles(self):
        self.champ_cible_triangles.configure(
            state="normal" if self.var_limiter_triangles.get() else "disabled"
        )

    # ------------------------------------------------------------------
    # Journal / file d'attente
    # ------------------------------------------------------------------
    def _ecrire_journal(self, texte):
        self.zone_journal.configure(state="normal")
        self.zone_journal.insert("end", texte)
        self.zone_journal.see("end")
        self.zone_journal.configure(state="disabled")

    def _lire_file_journal(self):
        try:
            while True:
                texte = self.file_journal.get_nowait()
                self._ecrire_journal(texte)
        except queue.Empty:
            pass
        self.racine.after(150, self._lire_file_journal)

    # ------------------------------------------------------------------
    # Exécution en tâche de fond
    # ------------------------------------------------------------------
    def _definir_occupe(self, occupe, texte_statut):
        self.en_cours = occupe
        etat = "disabled" if occupe else "normal"
        for bouton in self.boutons_lancement:
            bouton.configure(state=etat)
        self.etiquette_statut.configure(text=texte_statut)
        if occupe:
            self.barre_progression.start(12)
        else:
            self.barre_progression.stop()
            # Simple vérification et pas _rafraichir_boutons_affichage() :
            # celle-ci relance une boucle de 2 s en plus de celle qui tourne
            # déjà, et les boucles s'accumulaient après chaque étape
            self._verifier_fichiers_et_mettre_a_jour()

    # Calcul dans un thread : sinon la fenêtre est figée ("ne répond pas")
    # pendant les plusieurs minutes de calcul
    def _lancer_dans_thread(self, texte_statut, fonction_cible):
        if self.en_cours:
            messagebox.showinfo("Pipeline", "Une étape est déjà en cours.")
            return
        self._definir_occupe(True, texte_statut)
        self._ecrire_journal(f"\n{texte_statut}\n")

        def travail():
            # Redirection des print() pour voir la progression dans la
            # fenêtre sans modifier les scripts des étapes ; le finally
            # remet la sortie normale même si l'étape plante
            ancien_stdout, ancien_stderr = sys.stdout, sys.stderr
            redirection = RedirecteurConsole(self.file_journal)
            sys.stdout = redirection
            sys.stderr = redirection
            erreur = None
            try:
                fonction_cible()
            # Toutes les erreurs sont attrapées exprès : sinon le thread
            # meurt en silence et l'interface reste bloquée sur "en cours".
            # noqa dit à Ruff que c'est voulu.
            except Exception:  # noqa: BLE001
                erreur = traceback.format_exc()
            finally:
                sys.stdout, sys.stderr = ancien_stdout, ancien_stderr
            # after(0, ...) : la fin doit être faite par le thread de la
            # fenêtre, car on n'a pas le droit de la toucher depuis ce thread
            self.racine.after(0, self._fin_thread, erreur)

        threading.Thread(target=travail, daemon=True).start()

    def _fin_thread(self, erreur):
        self._definir_occupe(
            False, "Terminé." if erreur is None else "Erreur, voir les logs."
        )
        if erreur:
            self._ecrire_journal(f"\n[ERREUR]\n{erreur}\n")
            messagebox.showerror("Échec de l'étape", "Erreur, détails dans les logs.")

    # ------------------------------------------------------------------
    # Actions des boutons "Lancer"
    # ------------------------------------------------------------------
    def _lancer_etape1(self):
        dossier_source = self.var_dossier_source.get()
        dossier_travail = self.var_dossier_travail.get()
        masquer_fond = self.var_masquer_fond.get()
        repartir_zero = self.var_repartir_zero_1.get()
        camera_unique = self.var_camera_unique.get()

        def cible():
            executer_pipeline_securise(
                dossier_source=dossier_source,
                dossier_travail=dossier_travail,
                masquer_fond=masquer_fond,
                repartir_de_zero=repartir_zero,
                camera_unique=camera_unique,
            )

        self._lancer_dans_thread("Étape 1 : reconstruction SfM", cible)

    def _lancer_etape2(self):
        dossier_travail = self.var_dossier_travail.get()
        repartir_zero = self.var_repartir_zero_2.get()
        openmvs_bin = self.var_openmvs_bin.get()

        def cible():
            if not os.path.isdir(openmvs_bin):
                raise FileNotFoundError(
                    f"Dossier OpenMVS introuvable : {openmvs_bin}\n"
                    f"Télécharger les extensions ici : https://github.com/cdcseacave/openMVS/releases"
                )
            densification_openmvs.OPENMVS_BIN = openmvs_bin
            densification_openmvs.densifier(
                dossier_travail=dossier_travail, repartir_de_zero=repartir_zero
            )

        self._lancer_dans_thread("Étape 2 : densification", cible)

    def _lancer_pipeline_complet(self):
        dossier_source = self.var_dossier_source.get()
        dossier_travail = self.var_dossier_travail.get()
        masquer_fond = self.var_masquer_fond.get()
        repartir_zero_1 = self.var_repartir_zero_1.get()
        repartir_zero_2 = self.var_repartir_zero_2.get()
        openmvs_bin = self.var_openmvs_bin.get()
        camera_unique = self.var_camera_unique.get()

        def cible():
            if not os.path.isdir(openmvs_bin):
                raise FileNotFoundError(
                    f"Dossier OpenMVS introuvable : {openmvs_bin}\n"
                    f"Télécharger les extensions ici : https://github.com/cdcseacave/openMVS/releases"
                )
            executer_pipeline_securise(
                dossier_source=dossier_source,
                dossier_travail=dossier_travail,
                masquer_fond=masquer_fond,
                repartir_de_zero=repartir_zero_1,
                camera_unique=camera_unique,
            )
            densification_openmvs.OPENMVS_BIN = openmvs_bin
            densification_openmvs.densifier(
                dossier_travail=dossier_travail, repartir_de_zero=repartir_zero_2
            )

        self._lancer_dans_thread("Pipeline complet en cours", cible)

    def _lancer_etape3(self):
        dossier_travail = self.var_dossier_travail.get()
        # Si la case contient autre chose qu'un entier, .get() lève une
        # TclError qui, sans ce try, fait que le bouton ne réagit pas du tout
        try:
            profondeur = self.var_profondeur_poisson.get()
        except tk.TclError:
            messagebox.showerror(
                "Paramètre invalide",
                "« Profondeur Poisson » doit être un nombre entier (entre 4 et 14).",
            )
            return
        reparer = self.var_reparer_etancheite.get()
        if self.var_limiter_triangles.get():
            try:
                cible_triangles = int(self.var_cible_triangles.get())
            except ValueError:
                messagebox.showerror(
                    "Paramètre invalide",
                    "« Cible triangles » doit être un nombre entier.",
                )
                return
        else:
            cible_triangles = None

        def cible():
            mailler_et_preparer_impression(
                dossier_travail=dossier_travail,
                profondeur_poisson=profondeur,
                cible_triangles=cible_triangles,
                reparer_etancheite=reparer,
                afficher_interface=False,
            )

        self._lancer_dans_thread("Étape 3 : maillage", cible)

    def _creer_cible(self):
        # Rapide (quelques fichiers) : pas besoin d'un thread. On écrit dans le dossier courant,
        # là où le champ « Cible d'échelle » va la chercher par défaut.
        try:
            svg, js = generer_cible_echelle.generer(".")
        except (ValueError, ImportError, OSError) as e:
            messagebox.showerror("Cible d'échelle", f"Impossible de créer la cible :\n{e}")
            return
        self.var_mise_en_page.set(js)
        self._ecrire_journal(f"\nCible créée : {svg}\n")
        messagebox.showinfo(
            "Cible d'échelle",
            f"Feuille à imprimer :\n{svg}\n\nImprimer à 100 % (pas « ajuster à la page »), "
            f"vérifier la barre de 100 mm au pied à coulisse, puis poser la feuille à plat "
            f"à côté de l'objet, bien visible sur les photos.",
        )

    def _lire_nombre_optionnel(self, texte, nom, strictement_positif):
        """Nombre saisi (virgule ou point). None si vide. Affiche l'erreur et renvoie False si invalide."""
        texte = texte.strip().replace(",", ".")
        if not texte:
            return None
        try:
            valeur = float(texte)
            if not (valeur > 0 if strictement_positif else valeur >= 0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Paramètre invalide", f"« {nom} » doit être un nombre positif.")
            return False
        return valeur

    def _lancer_etape4a(self):
        dossier_travail = self.var_dossier_travail.get()
        mise_en_page = self.var_mise_en_page.get()
        taille = self._lire_nombre_optionnel(
            self.var_taille_marqueur.get(), "Côté d'un marqueur", True
        )
        if taille is False:
            return

        def cible():
            echelle.lancer_echelle(dossier_travail, mise_en_page, taille)

        self._lancer_dans_thread("Étape 4a : mise à l'échelle", cible)

    def _lancer_etape4b(self):
        dossier_travail = self.var_dossier_travail.get()
        nom = self.var_nom_mesure.get().strip()
        # Le nom devient un nom de dossier : pas de caractère interdit sous Windows
        if not nom or any(c in nom for c in '\\/:*?"<>|'):
            messagebox.showerror(
                "Paramètre invalide",
                "« Nom de la mesure » est vide ou contient un caractère interdit (\\ / : * ? \" < > |).",
            )
            return
        z_coupe = self._lire_nombre_optionnel(self.var_z_coupe.get(), "Coupe au plateau", False)
        if z_coupe is False:
            return
        if z_coupe is None:
            z_coupe = mesure.Z_COUPE_MM_DEFAUT
        sortie = os.path.join(dossier_travail, "mesures", nom)
        if os.path.isdir(sortie) and not messagebox.askyesno(
            "Mesure existante", f"« {nom} » existe déjà et sera remplacée. Continuer ?"
        ):
            return

        def cible():
            rapport = mesure.mesurer(
                os.path.join(dossier_travail, "modele_pour_impression.stl"),
                echelle.chemin_echelle(dossier_travail),
                sortie,
                nom,
                z_coupe_mm=z_coupe,
                chemin_nuage=os.path.join(dossier_travail, "openmvs", "scene_dense.ply"),
            )
            print(mesure.resume_texte(rapport))
            print(f"\nFichiers dans : {os.path.abspath(sortie)}")

        self._lancer_dans_thread("Étape 4b : mesures", cible)

    # ------------------------------------------------------------------
    # Visualisation 3D à la demande (processus séparé, jamais automatique)
    # ------------------------------------------------------------------
    def _chemin_nuage_dense(self):
        return os.path.join(
            self.var_dossier_travail.get(), "openmvs", "scene_dense.ply"
        )

    def _chemin_stl(self):
        return os.path.join(
            self.var_dossier_travail.get(), "modele_pour_impression.stl"
        )

    def _verifier_fichiers_et_mettre_a_jour(self, *_ignore):
        self.boutons_affichage["nuage"].configure(
            state="normal" if os.path.isfile(self._chemin_nuage_dense()) else "disabled")
        self.boutons_affichage["maillage"].configure(
            state="normal" if os.path.isfile(self._chemin_stl()) else "disabled")

    # Vérification toutes les 2 s : les fichiers peuvent apparaître ou
    # disparaître en dehors de l'interface (script lancé à la main, dossier
    # supprimé). Ne l'appeler qu'une fois, au démarrage.
    def _rafraichir_boutons_affichage(self):
        self._verifier_fichiers_et_mettre_a_jour()
        self.racine.after(2000, self._rafraichir_boutons_affichage)

    def _lancer_script_affichage(self, nom_script, dossier_travail):
        # Autre processus Python : la boucle graphique d'Open3D et celle de
        # Tkinter ne peuvent pas tourner dans le même programme, et la
        # fenêtre 3D bloquerait l'interface
        chemin_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), nom_script)
        try:
            subprocess.Popen(
                [sys.executable, chemin_script, os.path.abspath(dossier_travail)])
            self._ecrire_journal(f"\nOuverture de {nom_script}\n")
        # OSError seulement : c'est la seule erreur que Popen peut lever ici
        # (script introuvable, droits), inutile d'en masquer d'autres
        except OSError as e:
            messagebox.showerror(
                "Affichage 3D", f"Impossible de lancer {nom_script} :\n{e}")

    def _afficher_nuage(self):
        self._lancer_script_affichage("nuage.py", self.var_dossier_travail.get())

    def _afficher_maillage(self):
        # interface_maillage.py et pas maillage.py : maillage.py recalcule
        # tout et écraserait le STL fait avec les réglages choisis
        self._lancer_script_affichage("interface_maillage.py", self.var_dossier_travail.get())


if __name__ == "__main__":
    racine = tk.Tk()
    app = InterfacePipeline(racine)
    racine.mainloop()
