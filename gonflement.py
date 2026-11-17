"""
Gonflement d'un produit entre deux états (avant et après cuisson) : volume, hauteur, empreinte.

Pourquoi : le labo veut chiffrer de combien un gâteau gonfle à la cuisson. Un pourcentage de gonflement
sans incertitude ne dit rien : si le scanner se trompe de 2 %, un gonflement de 3 % est à peine démontré.
Ce module compare les mesures des deux états ET donne l'incertitude, en disant si la variation
est significative.

Entrées : les rapport.json de mesure.py. Plusieurs scans par état sont recommandés : leur moyenne a
une incertitude de répétabilité plus faible (divisée par la racine du nombre de scans).
Les scans doivent être mis à l'échelle (echelle.py) et marqués fiables par mesure.py.

Incertitude (k = 2 par défaut, ~95 %) :
  - part statistique : écart-type des scans / racine(n). Avec un seul scan, on prend la répétabilité
    mesurée à la validation (coefficient de variation du volume).
  - part systématique : biais et incertitude de la référence trouvés à la validation (validation_precision.py),
    ou valeur donnée à la main. Elle ne diminue pas quand on multiplie les scans.
  - les deux états sont supposés indépendants : estimation prudente. Si la même cible et le même banc servent
    aux deux, l'erreur d'échelle est en grande partie commune et s'annule dans le rapport des volumes :
    la valeur « erreur commune » donne cette borne optimiste.

Lancement : python gonflement.py --avant a1.json a2.json a3.json --apres b1.json b2.json b3.json
              [--validation validation/resultats/validation_resultats.json]
              [--incertitude-systematique-pct 1.5] [--nom gateau_marque_X]
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np

from validation_precision import ErreurValidation, charger_rapports

AUTRES_GRANDEURS = ("hauteur_mm", "hauteur_robuste_mm", "longueur_mm", "largeur_mm",
                    "diametre_equivalent_base_mm", "aire_base_mm2", "surface_mm2")
RAPPORT_VOLUMES_SUSPECT = 3.0     # au-delà, ce n'est probablement pas le même objet


class ErreurGonflement(Exception):
    """Comparaison impossible. Le message dit quoi corriger."""


def lire_validation(chemin, grandeur="volume_mm3"):
    """Extrait de validation_resultats.json la répétabilité et l'erreur systématique du volume (en %)."""
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            v = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ErreurGonflement(f"Validation illisible ({chemin}) : {e}")
    for r in v.get("resultats", []):
        if r.get("grandeur") == grandeur:
            ref = r["reference"]
            sys_rel = 100.0 * math.sqrt(r["biais"] ** 2 + (r.get("incertitude_reference") or 0.0) ** 2) / ref
            return {"repetabilite_pct": r.get("cv_pct"), "systematique_pct": sys_rel, "n_validation": r["n"],
                    "fichier": chemin}
    raise ErreurGonflement(f"« {grandeur} » absent de {chemin} : valider d'abord le volume sur une pièce étalon.")


def resumer_etat(rapports, validation=None, sys_pct=None):
    """Statistiques d'un état (avant ou après) : volume moyen et son incertitude-type en mm³."""
    volumes = np.array([r["mesures"]["volume_mm3"] for r in rapports], float)
    n = len(volumes)
    moyenne = float(volumes.mean())
    notes = []
    if n >= 2:
        s = float(volumes.std(ddof=1))
        stat = s / math.sqrt(n)
    elif validation and validation.get("repetabilite_pct") is not None:
        s = None
        stat = moyenne * validation["repetabilite_pct"] / 100.0          # un seul scan : répétabilité de la validation
        notes.append("un seul scan : répétabilité tirée de la validation")
    else:
        s, stat = None, None
        notes.append("un seul scan et pas de validation : répétabilité inconnue")
    if sys_pct is None and validation:
        sys_pct = validation["systematique_pct"]
    syst = moyenne * sys_pct / 100.0 if sys_pct is not None else None
    parties = [x for x in (stat, syst) if x is not None]
    return {"n": n, "volume_moyen_mm3": moyenne, "ecart_type_mm3": s,
            "incertitude_statistique_mm3": stat, "incertitude_systematique_mm3": syst,
            "incertitude_totale_mm3": math.sqrt(sum(x * x for x in parties)) if stat is not None else None,
            "volumes_mm3": volumes.tolist(), "noms": [r.get("nom", "") for r in rapports], "notes": notes}


def comparer(avant, apres, k=2.0, validation=None, sys_pct=None):
    """Compare deux listes de rapports. Renvoie le dict de résultats (voir les clés ci-dessous)."""
    if not avant or not apres:
        raise ErreurGonflement("Il faut au moins un scan avant et un scan après.")
    for nom, groupe in (("avant", avant), ("après", apres)):
        for r in groupe:
            if "volume_mm3" not in r.get("mesures", {}):
                raise ErreurGonflement(f"Scan « {r.get('nom', '?')} » ({nom}) sans volume.")
            if r.get("unite_longueur") != "mm":
                raise ErreurGonflement(f"Scan « {r.get('nom', '?')} » ({nom}) pas en millimètres : refaire la mise à l'échelle.")
    a = resumer_etat(avant, validation, sys_pct)
    b = resumer_etat(apres, validation, sys_pct)
    Va, Vb = a["volume_moyen_mm3"], b["volume_moyen_mm3"]
    if Va <= 0 or Vb <= 0:
        raise ErreurGonflement("Volume nul ou négatif : mesure invalide.")

    avertissements = []
    rapport_v = Vb / Va
    if rapport_v > RAPPORT_VOLUMES_SUSPECT or rapport_v < 1 / RAPPORT_VOLUMES_SUSPECT:
        avertissements.append(f"Le volume est multiplié par {rapport_v:.2f} : vérifier qu'il s'agit bien du même produit "
                              f"et que l'échelle est correcte dans les deux scans.")
    for nom, etat in (("avant", a), ("après", b)):
        for n in etat["notes"]:
            avertissements.append(f"État « {nom} » : {n}.")
        if etat["incertitude_systematique_mm3"] is None:
            avertissements.append(f"État « {nom} » : aucune incertitude systématique (pas de validation ni de valeur donnée) : "
                                  f"l'incertitude est SOUS-ESTIMÉE.")
    # Les deux méthodes d'échelle doivent être comparables : un facteur manuel d'un côté et une cible de l'autre est suspect
    methodes = {r.get("echelle", {}).get("methode") for r in avant + apres}
    if len(methodes) > 1:
        avertissements.append(f"Méthodes d'échelle différentes entre les scans ({', '.join(str(m) for m in methodes)}) : "
                              f"les deux états ne sont pas directement comparables.")

    res = {
        "version": 1, "date": datetime.now().isoformat(timespec="seconds"), "k": k,
        "avant": a, "apres": b,
        "delta_volume_mm3": Vb - Va, "rapport_volumes": rapport_v, "gonflement_pct": 100.0 * (rapport_v - 1.0),
        "validation": validation, "avertissements": avertissements,
        "incertitude_delta_volume_mm3": None, "incertitude_elargie_gonflement_pct": None,
        "incertitude_elargie_gonflement_erreur_commune_pct": None, "significatif": None, "autres_grandeurs": {},
    }
    if a["incertitude_totale_mm3"] is not None and b["incertitude_totale_mm3"] is not None:
        u_delta = math.sqrt(a["incertitude_totale_mm3"] ** 2 + b["incertitude_totale_mm3"] ** 2)
        u_rapport_rel = math.sqrt((a["incertitude_totale_mm3"] / Va) ** 2 + (b["incertitude_totale_mm3"] / Vb) ** 2)
        res["incertitude_delta_volume_mm3"] = u_delta
        res["incertitude_elargie_gonflement_pct"] = 100.0 * k * rapport_v * u_rapport_rel
        res["significatif"] = bool(abs(Vb - Va) > k * u_delta)
        # Borne optimiste : l'erreur systématique est la même dans les deux scans et s'annule dans le rapport
        stats_a, stats_b = a["incertitude_statistique_mm3"], b["incertitude_statistique_mm3"]
        if stats_a is not None and stats_b is not None:
            res["incertitude_elargie_gonflement_erreur_commune_pct"] = 100.0 * k * rapport_v * math.sqrt(
                (stats_a / Va) ** 2 + (stats_b / Vb) ** 2)

    for g in AUTRES_GRANDEURS:
        if all(g in r["mesures"] for r in avant + apres):
            xa = np.array([r["mesures"][g] for r in avant], float)
            xb = np.array([r["mesures"][g] for r in apres], float)
            u = None
            if len(xa) >= 2 and len(xb) >= 2:
                u = math.sqrt(xa.std(ddof=1) ** 2 / len(xa) + xb.std(ddof=1) ** 2 / len(xb))
            res["autres_grandeurs"][g] = {
                "avant": float(xa.mean()), "apres": float(xb.mean()), "variation": float(xb.mean() - xa.mean()),
                "variation_pct": 100.0 * float(xb.mean() / xa.mean() - 1.0) if xa.mean() else None,
                "incertitude_statistique": u,
                "significatif": bool(abs(xb.mean() - xa.mean()) > k * u) if u is not None else None}
    return res


def resume_texte(res, nom=None):
    a, b = res["avant"], res["apres"]
    L = [f"Gonflement{' : ' + nom if nom else ''}  ({res['date']})", "",
         f"Avant : {a['n']} scan(s), volume moyen {a['volume_moyen_mm3']:.1f} mm³ ({a['volume_moyen_mm3'] / 1000:.2f} mL)",
         f"Après : {b['n']} scan(s), volume moyen {b['volume_moyen_mm3']:.1f} mm³ ({b['volume_moyen_mm3'] / 1000:.2f} mL)", "",
         f"Variation de volume : {res['delta_volume_mm3']:+.1f} mm³ ({res['delta_volume_mm3'] / 1000:+.2f} mL)",
         f"Rapport des volumes : {res['rapport_volumes']:.4f}"]
    if res["incertitude_elargie_gonflement_pct"] is not None:
        L.append(f"GONFLEMENT : {res['gonflement_pct']:+.2f} %  ± {res['incertitude_elargie_gonflement_pct']:.2f} % (k = {res['k']:g})")
        if res["incertitude_elargie_gonflement_erreur_commune_pct"] is not None:
            L.append(f"  (± {res['incertitude_elargie_gonflement_erreur_commune_pct']:.2f} % si l'erreur d'échelle est la même "
                     f"dans les deux scans)")
        L.append("  -> variation " + ("SIGNIFICATIVE" if res["significatif"] else "NON significative (dans l'incertitude)"))
    else:
        L.append(f"GONFLEMENT : {res['gonflement_pct']:+.2f} %  (incertitude non calculable)")
    if res["autres_grandeurs"]:
        L += ["", "Autres grandeurs (moyennes) :"]
        for g, d in res["autres_grandeurs"].items():
            drapeau = "" if d["significatif"] is None else ("  significatif" if d["significatif"] else "  non significatif")
            L.append(f"  {g:30s} {d['avant']:.4g} -> {d['apres']:.4g}  ({d['variation_pct']:+.2f} %){drapeau}")
    if res["avertissements"]:
        L += [""] + [f"! {x}" for x in res["avertissements"]]
    return "\n".join(L)


def _fr(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return "oui" if x else "non"
    return f"{float(x):.6g}".replace(".", ",")


def ecrire(res, dossier, nom=None):
    """Écrit gonflement.json (dernier résultat) et ajoute une ligne à gonflement.csv."""
    os.makedirs(dossier, exist_ok=True)
    chemin_json = os.path.join(dossier, "gonflement.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump({**res, "nom": nom}, f, indent=2, ensure_ascii=False)
    chemin_csv = os.path.join(dossier, "gonflement.csv")
    nouveau = not os.path.isfile(chemin_csv)
    colonnes = ["date", "nom", "n_avant", "n_apres", "volume_avant_mm3", "volume_apres_mm3", "gonflement_pct",
                "incertitude_elargie_pct", "significatif"]
    with open(chemin_csv, "a", encoding="utf-8-sig" if nouveau else "utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        if nouveau:
            w.writerow(colonnes)
        w.writerow([res["date"], nom or "", res["avant"]["n"], res["apres"]["n"], _fr(res["avant"]["volume_moyen_mm3"]),
                    _fr(res["apres"]["volume_moyen_mm3"]), _fr(res["gonflement_pct"]),
                    _fr(res["incertitude_elargie_gonflement_pct"]), _fr(res["significatif"])])
    return chemin_json, chemin_csv


def main(argv=None):
    p = argparse.ArgumentParser(description="Gonflement entre deux états, avec incertitude.")
    p.add_argument("--avant", nargs="+", required=True, help="rapport.json, dossiers ou motifs de l'état avant")
    p.add_argument("--apres", nargs="+", required=True)
    p.add_argument("--validation", default=None, help="validation_resultats.json (répétabilité et biais du volume)")
    p.add_argument("--incertitude-systematique-pct", type=float, default=None,
                   help="erreur systématique relative du volume, en %% (remplace celle de la validation)")
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--nom", default=None)
    p.add_argument("--sortie", default=os.path.join("colmap_workspace", "mesures"))
    p.add_argument("--inclure-non-fiables", action="store_true")
    args = p.parse_args(argv)
    try:
        avant, ig1 = charger_rapports(args.avant, args.inclure_non_fiables)
        apres, ig2 = charger_rapports(args.apres, args.inclure_non_fiables)
        for f, raison in ig1 + ig2:
            print(f"Exclu : {f} ({raison})")
        validation = lire_validation(args.validation) if args.validation else None
        res = comparer(avant, apres, args.k, validation, args.incertitude_systematique_pct)
        print(resume_texte(res, args.nom))
        j, c = ecrire(res, args.sortie, args.nom)
        print(f"\nÉcrit : {j}\n        {c}")
    except (ErreurGonflement, ErreurValidation) as e:
        print(f"ERREUR : {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
