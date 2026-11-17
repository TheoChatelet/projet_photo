"""
Validation de la précision du scanner : comparaison au pied à coulisse et répétabilité.

Pourquoi : un chiffre de volume ou de cote n'a de valeur que si l'on sait de combien il peut se tromper.
Ce module répond à deux questions, sur une pièce étalon dont on connaît les dimensions :
  1. Justesse     : en moyenne, le scanner donne-t-il la bonne valeur ? (biais par rapport au pied à coulisse)
  2. Répétabilité : en scannant plusieurs fois la même pièce, obtient-on toujours la même valeur ? (écart-type)
Il en tire une incertitude élargie pour UN scan, comparée à la tolérance voulue.

Entrées :
  - un fichier de référence (CSV) : les valeurs mesurées au pied à coulisse, avec leur incertitude
    (modèle : python validation_precision.py modele)
  - les rapport.json de N scans de la même pièce (écrits par mesure.py)
Sorties (dossier validation/resultats par défaut) :
  validation_rapport.md, validation_resultats.csv (Excel), validation_detail.csv, validation_resultats.json,
  validation_graphique.png

Méthode (voir PROTOCOLE_VALIDATION.md) :
  - biais         = moyenne des scans - référence
  - répétabilité  = écart-type s des scans ; limite de répétabilité r = 2,77 s (ISO 5725 : 95 % des écarts entre deux scans
                    dans les mêmes conditions sont inférieurs à r)
  - biais significatif ? test t de Student, en tenant compte de l'incertitude de la référence
  - incertitude d'un scan : u = racine(s² + biais² + u_ref²), élargie U = k·u (k = 2, ~95 %). Le biais est ajouté sans
    être corrigé : choix prudent. Si le biais est stable, on peut le corriger et U baisse.
  - conformité : U <= tolérance demandée

Lancement : python validation_precision.py analyser --reference ref.csv --rapports colmap_workspace/mesures
            python validation_precision.py modele
            python validation_precision.py ref-volume cylindre --diametre 50,02 --hauteur 40,11 --u 0,02
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

# Unité d'une grandeur d'après le suffixe de son nom dans rapport.json
UNITES = {"_mm": "mm", "_mm2": "mm²", "_mm3": "mm³", "_ml": "mL"}
GRANDEURS_CONNUES = ("hauteur_mm", "hauteur_robuste_mm", "longueur_mm", "largeur_mm", "diametre_equivalent_base_mm",
                     "aire_base_mm2", "surface_mm2", "volume_mm3")
FACTEUR_REPETABILITE = 2.77       # = 1,96 · racine(2), ISO 5725
SEUIL_ATYPIQUE = 3.5              # z modifié (Iglewicz-Hoaglin), au-delà : valeur signalée
N_RECOMMANDE = 10


class ErreurValidation(Exception):
    """Validation impossible. Le message dit quoi corriger."""


# ----------------------------------------------------------------------
# Lecture des fichiers
# ----------------------------------------------------------------------
def lire_nombre(texte, decimale_virgule=True):
    """'12,5' -> 12.5. Refuse tout ce qui n'est pas un nombre fini."""
    t = str(texte).strip().replace(" ", "").replace(" ", "")
    if decimale_virgule:
        t = t.replace(",", ".")
    try:
        x = float(t)
    except ValueError:
        raise ErreurValidation(f"« {texte} » n'est pas un nombre.")
    if not math.isfinite(x):
        raise ErreurValidation(f"« {texte} » n'est pas un nombre fini.")
    return x


def unite_de(grandeur):
    for suffixe, u in UNITES.items():
        if grandeur.endswith(suffixe):
            return u
    return ""


def lire_reference(chemin):
    """Lit le CSV des valeurs de référence. Renvoie une liste de dict.

    Colonnes : grandeur ; valeur_ref ; incertitude_ref ; tolerance ; commentaire (les 3 dernières facultatives).
    Accepte ; , ou tabulation comme séparateur, virgule ou point décimal (Excel français), BOM, lignes vides
    et lignes de commentaire commençant par #.
    """
    try:
        with open(chemin, "r", encoding="utf-8-sig") as f:
            lignes = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    except FileNotFoundError:
        raise ErreurValidation(f"Fichier de référence introuvable : {chemin} (modèle : validation_precision.py modele)")
    if len(lignes) < 2:
        raise ErreurValidation(f"{chemin} : il faut une ligne d'en-tête et au moins une ligne de données.")
    entete = lignes[0]
    separateur = max([";", "\t", ","], key=lambda s: (entete.count(s), s == ";"))
    if entete.count(separateur) == 0:
        raise ErreurValidation(f"{chemin} : séparateur introuvable dans l'en-tête (utiliser ; , ou tabulation).")
    lecteur = csv.reader(lignes, delimiter=separateur)
    colonnes = [c.strip().lower() for c in next(lecteur)]
    for requis in ("grandeur", "valeur_ref"):
        if requis not in colonnes:
            raise ErreurValidation(f"{chemin} : colonne « {requis} » manquante (colonnes trouvées : {', '.join(colonnes)}).")
    decimale_virgule = separateur != ","     # avec « , » comme séparateur, la virgule ne peut pas être décimale
    lignes_ref, vues = [], set()
    for numero, champs in enumerate(lecteur, start=2):
        if len(champs) > len(colonnes):
            # Typique d'une virgule décimale (« 40,11 ») dans un fichier séparé par des virgules : sans ce
            # contrôle, 40,11 serait lu comme 40 et le « 11 » ignoré, en silence
            raise ErreurValidation(
                f"{chemin}, ligne {numero} : {len(champs)} champs pour {len(colonnes)} colonnes. Une virgule décimale "
                f"dans un fichier séparé par des virgules ? Utiliser ; comme séparateur ou un point décimal.")
        ligne = dict(zip(colonnes, [c.strip() for c in champs]))
        grandeur = ligne.get("grandeur", "")
        if not grandeur:
            raise ErreurValidation(f"{chemin}, ligne {numero} : nom de grandeur vide.")
        if grandeur in vues:
            raise ErreurValidation(f"{chemin} : la grandeur « {grandeur} » est donnée deux fois.")
        vues.add(grandeur)
        try:
            valeur = lire_nombre(ligne["valeur_ref"], decimale_virgule)
            u_ref = lire_nombre(ligne["incertitude_ref"], decimale_virgule) if ligne.get("incertitude_ref") else None
            tol = lire_nombre(ligne["tolerance"], decimale_virgule) if ligne.get("tolerance") else None
        except ErreurValidation as e:
            raise ErreurValidation(f"{chemin}, ligne {numero} ({grandeur}) : {e}")
        if valeur <= 0:
            raise ErreurValidation(f"{chemin}, ligne {numero} ({grandeur}) : la valeur de référence doit être positive.")
        if (u_ref is not None and u_ref < 0) or (tol is not None and tol <= 0):
            raise ErreurValidation(f"{chemin}, ligne {numero} ({grandeur}) : incertitude négative ou tolérance nulle.")
        lignes_ref.append({"grandeur": grandeur, "valeur_ref": valeur, "incertitude_ref": u_ref,
                           "tolerance": tol, "commentaire": ligne.get("commentaire", "")})
    return lignes_ref


def _trouver_rapports(chemins):
    """Développe fichiers, dossiers et motifs (*) en liste de rapport.json, sans doublon.

    PowerShell ne développe pas les * pour les programmes : on reçoit le motif tel quel, qui peut
    désigner des fichiers comme des dossiers (colmap_workspace_scan*/mesures).
    """
    fichiers = []
    for c in chemins:
        correspondances = glob.glob(c, recursive=True) if any(ch in c for ch in "*?[") else [c]
        for m in correspondances:
            if os.path.isdir(m):
                fichiers += glob.glob(os.path.join(m, "**", "rapport.json"), recursive=True)
            else:
                fichiers.append(m)
    vus, uniques = set(), []
    for f in sorted(fichiers):
        cle = os.path.abspath(f)
        if cle not in vus:
            vus.add(cle)
            uniques.append(f)
    return uniques


def charger_rapports(chemins, inclure_non_fiables=False):
    """Charge les rapport.json. Renvoie (rapports retenus, [(fichier, raison d'exclusion)])."""
    fichiers = _trouver_rapports(chemins if isinstance(chemins, (list, tuple)) else [chemins])
    if not fichiers:
        raise ErreurValidation("Aucun rapport.json trouvé. Lancer d'abord mesure.py sur chaque scan.")
    retenus, ignores = [], []
    for f in fichiers:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            ignores.append((f, f"illisible ({e})"))
            continue
        if not isinstance(r, dict) or "mesures" not in r:
            ignores.append((f, "pas un rapport de mesure.py"))
        elif r.get("unite_longueur") != "mm":
            ignores.append((f, "pas en millimètres (mesure faite sans échelle)"))
        elif not r.get("fiable", False) and not inclure_non_fiables:
            ignores.append((f, "marqué non fiable : " + "; ".join(r.get("avertissements", []))[:120]))
        else:
            r["_fichier"] = f
            retenus.append(r)
    return retenus, ignores


# ----------------------------------------------------------------------
# Statistiques
# ----------------------------------------------------------------------
def valeurs_atypiques(x):
    """Indices des valeurs aberrantes (z modifié basé sur la médiane), pour n >= 5. Ne retire rien."""
    x = np.asarray(x, float)
    if len(x) < 5:
        return []
    mediane = np.median(x)
    mad = np.median(np.abs(x - mediane))
    if mad == 0:
        return []
    z = 0.6745 * (x - mediane) / mad
    return [int(i) for i in np.flatnonzero(np.abs(z) > SEUIL_ATYPIQUE)]


def statistiques(valeurs, ref, u_ref=None, k=2.0, tolerance=None):
    """Statistiques d'une grandeur : justesse, répétabilité, incertitude, conformité."""
    from scipy import stats

    x = np.asarray(valeurs, float)
    n = len(x)
    if n == 0:
        raise ErreurValidation("Aucune valeur à analyser.")
    u_ref_utilise = u_ref or 0.0
    moyenne = float(x.mean())
    biais = moyenne - ref
    erreurs = x - ref
    res = {
        "n": n, "reference": ref, "incertitude_reference": u_ref, "moyenne": moyenne,
        "biais": biais, "biais_pct": 100.0 * biais / ref,
        "eqm": float(np.sqrt(np.mean(erreurs ** 2))), "erreur_moyenne_absolue": float(np.mean(np.abs(erreurs))),
        "erreur_max_absolue": float(np.max(np.abs(erreurs))),
        "min": float(x.min()), "max": float(x.max()),
        "ecart_type": None, "cv_pct": None, "limite_repetabilite": None,
        "ic95_moyenne": None, "p_biais": None, "biais_significatif": None,
        "incertitude_scan": None, "incertitude_elargie": None, "incertitude_elargie_pct": None,
        "tolerance": tolerance, "k": k, "conforme": None,
    }
    if n >= 2:
        s = float(x.std(ddof=1))
        t975 = float(stats.t.ppf(0.975, n - 1))
        res["ecart_type"] = s
        res["cv_pct"] = 100.0 * s / moyenne if moyenne else None
        res["limite_repetabilite"] = FACTEUR_REPETABILITE * s
        res["ic95_moyenne"] = [moyenne - t975 * s / math.sqrt(n), moyenne + t975 * s / math.sqrt(n)]
        # Incertitude-type du biais : dispersion de la moyenne + incertitude de la référence
        u_biais = math.sqrt(s ** 2 / n + u_ref_utilise ** 2)
        if u_biais == 0:
            p = 0.0 if biais != 0 else 1.0
        else:
            # Degrés de liberté de Welch-Satterthwaite ; la référence (type B) est supposée de dof infini
            dof = u_biais ** 4 / ((s ** 2 / n) ** 2 / (n - 1)) if s > 0 else float("inf")
            p = float(2 * stats.t.sf(abs(biais / u_biais), dof))
        res["p_biais"] = p
        res["biais_significatif"] = bool(p < 0.05)
        u_scan = math.sqrt(s ** 2 + biais ** 2 + u_ref_utilise ** 2)
        res["incertitude_scan"] = u_scan
        res["incertitude_elargie"] = k * u_scan
        res["incertitude_elargie_pct"] = 100.0 * k * u_scan / ref
        if tolerance is not None:
            res["conforme"] = bool(k * u_scan <= tolerance)
    return res


def analyser(references, rapports, k=2.0, condition=None):
    """Analyse toutes les grandeurs de la référence. Renvoie le dict complet des résultats."""
    if not rapports:
        raise ErreurValidation("Aucun rapport exploitable (tous exclus : voir la liste des exclusions).")
    resultats, avertissements, detail = [], [], []
    for ref in references:
        g = ref["grandeur"]
        valeurs, noms, fichiers = [], [], []
        for r in rapports:
            v = r["mesures"].get(g)
            if isinstance(v, (int, float)) and math.isfinite(v):
                valeurs.append(float(v))
                noms.append(r.get("nom", ""))
                fichiers.append(r["_fichier"])
        if not valeurs:
            avertissements.append(f"« {g} » absent de tous les rapports : grandeur ignorée "
                                  f"(grandeurs disponibles : {', '.join(sorted(rapports[0]['mesures']))}).")
            continue
        if len(valeurs) < len(rapports):
            avertissements.append(f"« {g} » absent de {len(rapports) - len(valeurs)} rapport(s) sur {len(rapports)}.")
        u_ref = ref["incertitude_ref"]
        st = statistiques(valeurs, ref["valeur_ref"], u_ref, k, ref["tolerance"])
        atypiques = valeurs_atypiques(valeurs)
        st.update({"grandeur": g, "unite": unite_de(g), "atypiques": [noms[i] or fichiers[i] for i in atypiques],
                   "commentaire": ref["commentaire"]})
        n = st["n"]
        if u_ref is None:
            avertissements.append(f"« {g} » : incertitude de la référence non renseignée (prise nulle : prudence).")
        if n < 2:
            avertissements.append(f"« {g} » : un seul scan, pas de répétabilité calculable.")
        elif n < 3:
            avertissements.append(f"« {g} » : seulement {n} scans, statistiques très peu informatives.")
        elif n < N_RECOMMANDE:
            avertissements.append(f"« {g} » : {n} scans, {N_RECOMMANDE} ou plus recommandés pour une répétabilité solide.")
        if atypiques:
            avertissements.append(f"« {g} » : valeur(s) atypique(s) signalée(s) ({', '.join(st['atypiques'])}) : "
                                  f"à vérifier (photos floues ? mauvais éclairage ?). Aucune valeur n'est retirée.")
        if u_ref is not None and n >= 3 and st["ecart_type"] is not None and u_ref > st["ecart_type"]:
            avertissements.append(f"« {g} » : l'incertitude de la référence ({u_ref:g}) dépasse la dispersion des "
                                  f"scans ({st['ecart_type']:.3g}) : la validation est limitée par la référence, pas par le scanner.")
        resultats.append(st)
        for v, nom, fic in zip(valeurs, noms, fichiers):
            detail.append({"grandeur": g, "nom": nom, "fichier": fic, "valeur": v, "erreur": v - ref["valeur_ref"],
                           "erreur_pct": 100.0 * (v - ref["valeur_ref"]) / ref["valeur_ref"],
                           "atypique": (nom or fic) in st["atypiques"]})
    if not resultats:
        raise ErreurValidation("Aucune grandeur de la référence n'a été trouvée dans les rapports. "
                               + "; ".join(avertissements))
    return {"version": 1, "date": datetime.now().isoformat(timespec="seconds"), "condition": condition, "k": k,
            "nb_scans": len(rapports), "resultats": resultats, "detail": detail, "avertissements": avertissements}


# ----------------------------------------------------------------------
# Sorties
# ----------------------------------------------------------------------
def _fr(x, chiffres=4):
    """Nombre pour l'Excel français (virgule décimale)."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return "oui" if x else "non"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{chiffres}g}".replace(".", ",")


def _verdict(r):
    if r["conforme"] is True:
        return "conforme"
    if r["conforme"] is False:
        return "NON conforme"
    if r["incertitude_elargie"] is None:
        return "non évalué (n < 2)"
    return "non évalué (pas de tolérance)"


COLONNES_RESULTATS = ["grandeur", "unite", "n", "reference", "incertitude_reference", "moyenne", "ecart_type", "biais",
                      "biais_pct", "p_biais", "biais_significatif", "eqm", "erreur_max_absolue", "cv_pct",
                      "limite_repetabilite", "incertitude_elargie", "incertitude_elargie_pct", "tolerance", "verdict"]


def ecrire_csv(analyse, dossier):
    chemin = os.path.join(dossier, "validation_resultats.csv")
    with open(chemin, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLONNES_RESULTATS)
        for r in analyse["resultats"]:
            ligne = {**r, "verdict": _verdict(r)}
            w.writerow([ligne[c] if c in ("grandeur", "unite", "verdict") else _fr(ligne[c]) for c in COLONNES_RESULTATS])
    chemin2 = os.path.join(dossier, "validation_detail.csv")
    with open(chemin2, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["grandeur", "nom", "valeur", "erreur", "erreur_pct", "atypique", "fichier"])
        for d in analyse["detail"]:
            w.writerow([d["grandeur"], d["nom"], _fr(d["valeur"], 7), _fr(d["erreur"], 4), _fr(d["erreur_pct"], 4),
                        _fr(d["atypique"]), d["fichier"]])
    return chemin, chemin2


def rapport_markdown(analyse, ignores=()):
    """Texte du rapport de validation (à coller dans le rapport de stage)."""
    L = ["# Validation de la précision du scanner photogrammétrique", "",
         f"Date : {analyse['date']}  ",
         f"Nombre de scans analysés : {analyse['nb_scans']}  "]
    if analyse["condition"]:
        L.append(f"Conditions : {analyse['condition']}  ")
    L += [f"Facteur d'élargissement : k = {analyse['k']:g} (niveau de confiance d'environ 95 %)", "",
          "## Synthèse", "",
          "| Grandeur | n | Référence | Moyenne scans | Biais | Écart-type | Incertitude élargie U | Tolérance | Verdict |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in analyse["resultats"]:
        u = r["unite"]
        L.append("| {g} | {n} | {ref:.4g} {u} | {moy:.4g} {u} | {b:+.3g} {u} ({bp:+.2f} %) | {s} | {U} | {tol} | **{v}** |".format(
            g=r["grandeur"], n=r["n"], ref=r["reference"], u=u, moy=r["moyenne"], b=r["biais"], bp=r["biais_pct"],
            s=f"{r['ecart_type']:.3g} {u}" if r["ecart_type"] is not None else "—",
            U=f"±{r['incertitude_elargie']:.3g} {u} (±{r['incertitude_elargie_pct']:.2f} %)" if r["incertitude_elargie"] is not None else "—",
            tol=f"±{r['tolerance']:g} {u}" if r["tolerance"] is not None else "—", v=_verdict(r)))
    L += ["", "## Détail par grandeur", ""]
    for r in analyse["resultats"]:
        u = r["unite"]
        L.append(f"### {r['grandeur']}")
        L.append("")
        L.append(f"- Référence (pied à coulisse) : {r['reference']:.5g} {u}"
                 + (f" ± {r['incertitude_reference']:g} (incertitude-type)" if r["incertitude_reference"] is not None else ""))
        L.append(f"- {r['n']} scans : moyenne {r['moyenne']:.5g} {u}, de {r['min']:.5g} à {r['max']:.5g} {u}")
        L.append(f"- **Justesse** : biais {r['biais']:+.4g} {u} ({r['biais_pct']:+.3f} %) ; erreur maximale d'un scan {r['erreur_max_absolue']:.4g} {u} ; "
                 f"erreur quadratique moyenne {r['eqm']:.4g} {u}")
        if r["p_biais"] is not None:
            L.append(f"  - Le biais est **{'significatif' if r['biais_significatif'] else 'non significatif'}** (p = {r['p_biais']:.3g}, seuil 0,05)")
            L.append(f"- **Répétabilité** : écart-type {r['ecart_type']:.4g} {u} (coefficient de variation {r['cv_pct']:.3f} %) ; "
                     f"limite de répétabilité r = {r['limite_repetabilite']:.4g} {u}")
            L.append(f"- Intervalle de confiance à 95 % de la moyenne : [{r['ic95_moyenne'][0]:.5g} ; {r['ic95_moyenne'][1]:.5g}] {u}")
            L.append(f"- **Incertitude élargie d'un scan** : U = {r['incertitude_elargie']:.4g} {u} (±{r['incertitude_elargie_pct']:.3f} %)")
        L.append("")
    if analyse["avertissements"] or ignores:
        L += ["## Points d'attention", ""]
        L += [f"- {a}" for a in analyse["avertissements"]]
        L += [f"- Rapport exclu : `{f}` ({raison})" for f, raison in ignores]
        L.append("")
    L += ["## Méthode", "",
          "- Biais = moyenne des scans − référence. Répétabilité = écart-type des scans (même pièce, mêmes conditions). "
          "Limite de répétabilité r = 2,77 s (ISO 5725).",
          "- Le biais est testé par un test t de Student qui tient compte de l'incertitude de la référence.",
          "- Incertitude d'un scan : u = √(s² + biais² + u_ref²) ; U = k·u. Le biais est conservé dans l'incertitude "
          "(choix prudent). S'il est stable, il peut être corrigé et U diminue.",
          "- Verdict « conforme » : U inférieure ou égale à la tolérance demandée.", ""]
    return "\n".join(L)


def tracer_graphique(analyse, chemin_png):
    """Erreur relative de chaque scan par grandeur, avec la bande ±U. Renvoie False si matplotlib manque."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    grandeurs = [r["grandeur"] for r in analyse["resultats"]]
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(grandeurs) + 2), 4.5))
    for i, r in enumerate(analyse["resultats"]):
        pts = [d["erreur_pct"] for d in analyse["detail"] if d["grandeur"] == r["grandeur"]]
        ax.scatter([i] * len(pts), pts, color="#2a6fb0", alpha=0.7, zorder=3, label="un scan" if i == 0 else None)
        ax.hlines(r["biais_pct"], i - 0.25, i + 0.25, color="#c0392b", linewidth=2, zorder=4, label="biais moyen" if i == 0 else None)
        if r["incertitude_elargie_pct"] is not None:
            ax.add_patch(plt.Rectangle((i - 0.3, -r["incertitude_elargie_pct"]), 0.6, 2 * r["incertitude_elargie_pct"],
                                       color="#2a6fb0", alpha=0.12, zorder=1,
                                       label=f"±U (k={analyse['k']:g})" if i == 0 else None))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(grandeurs)))
    ax.set_xticklabels(grandeurs, rotation=20, ha="right")
    ax.set_ylabel("erreur relative par rapport au pied à coulisse (%)")
    ax.set_title("Précision du scanner : un point = un scan")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=150)
    plt.close(fig)
    return True


def ecrire_resultats(analyse, dossier, ignores=()):
    """Écrit tous les fichiers de sortie. Renvoie la liste des chemins."""
    os.makedirs(dossier, exist_ok=True)
    fichiers = list(ecrire_csv(analyse, dossier))
    chemin_json = os.path.join(dossier, "validation_resultats.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(analyse, f, indent=2, ensure_ascii=False)
    chemin_md = os.path.join(dossier, "validation_rapport.md")
    with open(chemin_md, "w", encoding="utf-8") as f:
        f.write(rapport_markdown(analyse, ignores))
    fichiers += [chemin_json, chemin_md]
    chemin_png = os.path.join(dossier, "validation_graphique.png")
    if tracer_graphique(analyse, chemin_png):
        fichiers.append(chemin_png)
    return fichiers


# ----------------------------------------------------------------------
# Aides : modèle de fichier, volume de référence d'un étalon
# ----------------------------------------------------------------------
MODELE_REFERENCE = """# Valeurs de référence mesurées au pied à coulisse sur la pièce étalon.
# Séparateur ; (virgule ou point acceptés pour les décimales). Les lignes commençant par # sont ignorées.
# grandeur        : nom d'une mesure de rapport.json (hauteur_mm, hauteur_robuste_mm, longueur_mm, largeur_mm,
#                   diametre_equivalent_base_mm, aire_base_mm2, surface_mm2, volume_mm3)
# valeur_ref      : moyenne de vos mesures au pied à coulisse (mm, mm², mm³)
# incertitude_ref : incertitude-type de cette référence, même unité (voir PROTOCOLE_VALIDATION.md)
# tolerance       : écart maximal accepté pour un scan, même unité (facultatif)
grandeur;valeur_ref;incertitude_ref;tolerance;commentaire
hauteur_mm;40,11;0,02;0,5;moyenne de 5 mesures, 3 positions
diametre_equivalent_base_mm;50,02;0,02;0,5;moyenne de 3 diamètres
volume_mm3;78790;60;1500;calculé avec : validation_precision.py ref-volume cylindre
"""


def volume_reference(forme, dimensions, u):
    """Volume (et son incertitude-type) d'un étalon d'après ses dimensions mesurées.

    dimensions : dict selon la forme (cylindre : diametre, hauteur ; parallelepipede : a, b, c ; sphere : diametre)
    u          : incertitude-type sur chaque dimension (mm)
    Les erreurs sur les dimensions s'additionnent en quadrature, pondérées par leur exposant dans la formule.
    """
    if u < 0:
        raise ErreurValidation("L'incertitude doit être positive ou nulle.")
    if any(v <= 0 for v in dimensions.values()):
        raise ErreurValidation("Toutes les dimensions doivent être positives.")
    if forme == "cylindre":
        d, h = dimensions["diametre"], dimensions["hauteur"]
        volume, rel = math.pi * d * d * h / 4.0, math.sqrt((2 * u / d) ** 2 + (u / h) ** 2)
    elif forme == "parallelepipede":
        a, b, c = dimensions["a"], dimensions["b"], dimensions["c"]
        volume, rel = a * b * c, math.sqrt((u / a) ** 2 + (u / b) ** 2 + (u / c) ** 2)
    elif forme == "sphere":
        d = dimensions["diametre"]
        volume, rel = math.pi * d ** 3 / 6.0, 3 * u / d
    else:
        raise ErreurValidation(f"Forme inconnue : {forme}")
    return volume, volume * rel


# ----------------------------------------------------------------------
# Ligne de commande
# ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Validation de la précision du scanner.")
    sp = p.add_subparsers(dest="commande")
    a = sp.add_parser("analyser", help="comparer les scans à la référence")
    a.add_argument("--reference", required=True, help="CSV des valeurs mesurées au pied à coulisse")
    a.add_argument("--rapports", nargs="+", required=True, help="rapport.json, dossiers ou motifs (*)")
    a.add_argument("--sortie", default=os.path.join("validation", "resultats"))
    a.add_argument("--k", type=float, default=2.0, help="facteur d'élargissement (défaut 2)")
    a.add_argument("--condition", default=None, help="ex : « répétabilité, pièce non déplacée » ou « repositionnée entre scans »")
    a.add_argument("--inclure-non-fiables", action="store_true", help="garder les scans marqués non fiables")
    sp.add_parser("modele", help="écrire un fichier de référence à remplir").add_argument(
        "--sortie", default=os.path.join("validation", "reference_modele.csv"))
    v = sp.add_parser("ref-volume", help="volume de référence d'un étalon d'après ses dimensions")
    v.add_argument("forme", choices=["cylindre", "parallelepipede", "sphere"])
    for nom in ("diametre", "hauteur", "a", "b", "c"):
        v.add_argument(f"--{nom}", default=None)
    v.add_argument("--u", default="0,02", help="incertitude-type de chaque dimension, en mm (défaut 0,02)")
    args = p.parse_args(argv)

    try:
        if args.commande == "modele":
            os.makedirs(os.path.dirname(os.path.abspath(args.sortie)), exist_ok=True)
            with open(args.sortie, "w", encoding="utf-8-sig") as f:
                f.write(MODELE_REFERENCE)
            print(f"Modèle écrit : {args.sortie}")
        elif args.commande == "ref-volume":
            noms = {"cylindre": ["diametre", "hauteur"], "parallelepipede": ["a", "b", "c"], "sphere": ["diametre"]}[args.forme]
            manquants = [n for n in noms if getattr(args, n) is None]
            if manquants:
                raise ErreurValidation(f"Il manque : {', '.join('--' + n for n in manquants)}")
            dims = {n: lire_nombre(getattr(args, n)) for n in noms}
            volume, incertitude = volume_reference(args.forme, dims, lire_nombre(args.u))
            print(f"volume_mm3;{_fr(volume, 8)};{_fr(incertitude, 3)}   (valeur_ref ; incertitude_ref à coller dans le CSV)")
        elif args.commande == "analyser":
            references = lire_reference(args.reference)
            rapports, ignores = charger_rapports(args.rapports, args.inclure_non_fiables)
            for f, raison in ignores:
                print(f"Exclu : {f} ({raison})")
            analyse = analyser(references, rapports, args.k, args.condition)
            fichiers = ecrire_resultats(analyse, args.sortie, ignores)
            print(rapport_markdown(analyse, ignores))
            print("Fichiers écrits :\n  " + "\n  ".join(fichiers))
        else:
            p.print_help()
            return 1
    except ErreurValidation as e:
        print(f"ERREUR : {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
