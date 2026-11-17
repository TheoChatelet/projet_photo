# Projet photogrammétrie

Reconstruction 3D par photogrammétrie, avec mesures en millimètres (cotes, volume) et validation de la précision.
Deux usages au labo : reconstituer la géométrie de pièces mécaniques sans plan, et mesurer le volume de produits
(gâteaux) avant/après cuisson.

## La chaîne, de bout en bout

| # | Étape | Commande | Entrée | Sortie |
|---|---|---|---|---|
| 1 | Reconstruction (COLMAP) | `python main.py` | `Photos/` | `colmap_workspace/sparse_final` |
| 2 | Densification (OpenMVS) | `python densification_openmvs.py` | `sparse_final` | `openmvs/scene_dense.ply` |
| 3 | Maillage (Poisson) | `python maillage.py` | `scene_dense.ply` | `modele_pour_impression.stl` (unité COLMAP) |
| 4a | **Échelle** | `python echelle.py` | `sparse_final`, photos, `cible_echelle.json` | `echelle.json` |
| 4b | **Mesures** | `python mesure.py --nom essai1` | STL + `echelle.json` | `mesures/essai1/` : `modele_mm.stl`, `rapport.json`, `rapport.txt` |
| 5 | **Validation** | `python validation_precision.py analyser ...` | rapports de N scans + référence | biais, répétabilité, incertitude |
| 6 | **Gonflement** | `python gonflement.py ...` | rapports avant/après | % de gonflement ± incertitude |

`python pipeline_scan.py --photos Photos_scan01 --nom scan01` enchaîne 1 à 4b pour un dossier de photos (utile pour les séries de validation).
`python interface_pipeline.py` ouvre l'interface graphique (étapes 1 à 4b).

## Installation

```
pip install -r requirements.txt
```
OpenMVS (les `.exe`) se télécharge à part : <https://github.com/cdcseacave/openMVS/releases>, puis renseigner `OPENMVS_BIN` dans `densification_openmvs.py` (ou le champ de l'interface).

## Pourquoi une échelle : la cible ArUco

COLMAP reconstruit la bonne forme mais dans une unité arbitraire. Pour mesurer en mm, on pose une feuille de **marqueurs ArUco** à plat à côté de l'objet.

1. `python generer_cible_echelle.py` crée `cible_echelle.svg` (et `cible_echelle.json`, sa description). Imprimer **à 100 %**.
2. Vérifier la barre de 100 mm dessinée sur la feuille et **mesurer le côté d'un marqueur** au pied à coulisse.
3. Poser la feuille sur le plateau, l'objet dessus ou à côté, les marqueurs visibles sur les photos depuis des directions variées.
4. Après l'étape 1 : `python echelle.py --taille-marqueur-mm 29.9` (la valeur mesurée).

`echelle.py` détecte les marqueurs, les triangule avec les positions de caméras de COLMAP, et en tire :
- l'**échelle** (mm par unité COLMAP), avec son incertitude estimée et un indicateur **fiable / à vérifier** ;
- le **plan du plateau** (z = 0, vers le haut), utilisé pour couper l'objet.

Sans cible : `python echelle.py --facteur 137.5` (échelle déjà connue) ou `--distance-unites X --distance-mm Y`. Dans ce cas il n'y a pas de plan de
plateau, donc pas de coupe : l'objet doit être entier dans le maillage. **Ne jamais mettre à l'échelle avec la cote de la pièce qu'on veut valider.**

## Ce que fait `mesure.py` et le « z de coupe »

Le volume d'un maillage n'a de sens que fermé. Or le **dessous de l'objet n'est pas visible** sur les photos : Poisson le referme avec une surface inventée.
`mesure.py` :

1. passe le maillage en mm dans le repère du plateau ;
2. **coupe au plan du plateau** (+ `z_coupe` = 1 mm par défaut, pour s'affranchir de l'épaisseur de la feuille et du bruit du plan) ;
3. garde la plus grosse pièce (les fragments flottants sont écartés et comptés) ;
4. **referme la coupe** et calcule le volume exact par la formule de la divergence ;
5. **ajoute la tranche** plateau → coupe comme un prisme (la paroi d'un objet posé est verticale sur ce premier millimètre) ;
6. mesure hauteur, longueur et largeur de l'empreinte (rectangle d'aire minimale, indépendant de l'orientation), diamètre équivalent de la base, surface ;
7. exporte un modèle **fermé, en mm, base à z = 0** (STL, OBJ ou PLY) ;
8. écrit `rapport.json` / `rapport.txt` et ajoute une ligne à `mesures/historique.csv` (s'ouvre dans Excel en français).

**Réglage du z de coupe.** Le rapport donne « décaler la coupe de 1 mm change le volume de X mm³ » (= l'aire de la base). Garder 1 mm en général.
Si la feuille ou le plateau apparaissent dans le modèle, monter à 1,5 ou 2 mm (`--z-coupe-mm 2`). Si un objet très bas est amputé, descendre.
Pour une paroi verticale en bas, le volume final ne dépend presque pas de ce réglage (prolongement du prisme) ; il en dépend pour une base arrondie.

**Lire un rapport.** `fiable` passe à *à vérifier* quand l'échelle est douteuse, qu'un trou touche le plan de coupe, que trop de surface a été
bouchée, que des surfaces semblent inventées loin des points (`--nuage`), ou que l'objet semble en plusieurs morceaux. Chaque avertissement dit quoi regarder.
Ne pas s'appuyer sur un volume marqué à vérifier sans avoir ouvert `modele_mm.stl`.

**Hauteur.** `hauteur_mm` est le point le plus haut : le bruit de surface la biaise vers le haut. `hauteur_robuste_mm` moyenne localement et corrige l'essentiel du biais.
La validation permet de choisir et de chiffrer.

## Valider la précision

Voir **[PROTOCOLE_VALIDATION.md](PROTOCOLE_VALIDATION.md)** : pièces étalons, mesures de référence au pied à coulisse (avec incertitude), séries de scans, lecture des résultats, critères d'acceptation.

## Tests

```
python -m pytest              # tous les tests (~15 s)
python -m pytest -m "not lourd"   # sans les tests sur les données réelles du dépôt
```
Les tests fabriquent leurs propres vérités (formes à volume exact, photos de cible simulées, reconstruction COLMAP synthétique, scans bruités).
Ils couvrent la chaîne échelle → mesure → validation → gonflement. Les tests de l'interface demandent un écran (ignorés sinon).

## Limites connues, à savoir

- **Rien n'a encore été testé sur de vraies photos d'une cible imprimée** : la détection ArUco, la triangulation et la mise à l'échelle sont validées sur des images simulées
  et sur la vraie reconstruction du dépôt (cohérence des poses), pas sur une vraie cible. C'est le premier essai à faire avec le banc.
- La densification OpenMVS (Windows, `.exe`) n'a pu être exécutée que sur ton PC : `pipeline_scan.py` est testé avec des étapes simulées.
- La coupe suppose que l'objet **repose sur le plan de la cible**. Un objet maintenu en l'air, ou une pièce scannée retournée, demande `--sans-coupe` et un maillage fermé.
- Un trou du maillage qui **touche le plan de coupe** rend le volume faux sans rapport avec la taille du trou : c'est détecté et signalé, pas réparé.
- Une surface brillante, transparente ou sans texture se reconstruit mal en photogrammétrie, quel que soit le réglage.
- L'incertitude annoncée n'est valable que pour les conditions de la validation (appareil, éclairage, taille et forme des pièces).

## Carte des fichiers

```
main.py, masquage_fond.py            étape 1 : reconstruction COLMAP (une seule caméra par défaut)
densification_openmvs.py             étape 2 : OpenMVS
maillage.py                          étape 3 : Poisson + STL
generer_cible_echelle.py             cible ArUco à imprimer
echelle.py                           échelle et repère du plateau
mesure.py                            volume, cotes, export en mm
validation_precision.py              biais, répétabilité, incertitude
gonflement.py                        avant/après cuisson
pipeline_scan.py                     un scan complet en une commande
interface_pipeline.py                interface graphique
nuage.py, interface_maillage.py      visionneuses 3D
tests/                               tests automatiques
```
