# Protocole de validation de la précision

But : démontrer, chiffres à l'appui, de combien le scanner photogrammétrique peut se tromper sur une cote ou un volume,
et si c'est assez bon pour les deux usages du labo (pièces mécaniques à réusiner, volume des gâteaux avant/après cuisson).

Deux questions, deux mesures :

| Question | Grandeur | Comment |
|---|---|---|
| Le scanner donne-t-il la **bonne valeur** ? (justesse) | biais = moyenne des scans − pied à coulisse | comparaison à une pièce étalon |
| Donne-t-il **toujours la même valeur** ? (répétabilité) | écart-type des scans | plusieurs scans de la même pièce |

Le résultat final est une **incertitude élargie U** (environ 95 %) pour un scan, comparée à la tolérance voulue.

> Les critères d'acceptation ci-dessous (§ 7) sont des propositions : à fixer avec Quentin Egloff selon l'usage visé.

---

## 1. Matériel

**Pièces étalons.** Le principe : des pièces simples, dont on peut mesurer les dimensions au pied à coulisse avec peu d'incertitude.

- **Cylindre** (par exemple Ø 50 × 40 mm) : proche d'un gâteau rond, donne diamètre, hauteur et volume.
- **Parallélépipède** (par exemple 60 × 40 × 30 mm) : cotes planes, arêtes vives.
- Si possible une 2ᵉ taille de cylindre, pour voir si l'erreur dépend de la taille (un gâteau est plus gros qu'un étalon).
- Matière : **mate et un peu texturée** (la photogrammétrie ne trouve rien à accrocher sur une surface lisse et uniforme ou brillante).
  Une pièce **imprimée en 3D en PLA mat** (avec Antoine) convient très bien : la qualité de l'impression n'a pas d'importance,
  car la référence est la mesure au pied à coulisse, pas le plan. Une pièce brillante se traite avec un spray matifiant.
  Une pièce unie se texture avec des points au feutre.

**Cible d'échelle.** Feuille de marqueurs ArUco : `python generer_cible_echelle.py` puis imprimer le SVG **à 100 %**.
Vérifier au pied à coulisse la barre de 100 mm et **mesurer le côté d'un marqueur** : cette valeur est donnée à `--taille-marqueur-mm`.
Une imprimante déforme parfois de quelques dixièmes de pour cent, et cette erreur se retrouve intégralement dans l'échelle
(donc trois fois dans le volume).

**Pied à coulisse.** Numérique, résolution 0,01 mm. L'erreur maximale tolérée est typiquement ±0,03 mm pour un 150 mm
(à vérifier sur la fiche de l'instrument ou son certificat d'étalonnage).

**Banc.** Plateau tournant, appareil sur trépied, éclairage diffus fixe, fond uni.

---

## 2. Mesures de référence (pied à coulisse)

Pour **chaque dimension utile** (hauteur, diamètre, côtés), faire **au moins 5 mesures** à des endroits différents
(pour un cylindre : 3 diamètres à 2 hauteurs différentes ; pour la hauteur : 4 positions autour). Noter chaque valeur.

- `valeur_ref` = moyenne des mesures.
- `incertitude_ref` = incertitude-type, qui combine trois sources :

  u_ref = √( (s/√n)² + (0,01/√12)² + (exactitude/√3)² )

  avec s l'écart-type de vos mesures, n leur nombre, 0,01 la résolution, et « exactitude » l'erreur maximale tolérée de l'instrument (0,03 mm).
  Ordre de grandeur habituel : **0,02 mm**.

- **Volume de référence** : calculé à partir des dimensions, avec propagation de l'incertitude :
  ```
  python validation_precision.py ref-volume cylindre --diametre 50,02 --hauteur 40,11 --u 0,02
  ```
  La commande affiche `volume_mm3;valeur;incertitude` à recopier dans le fichier de référence.

  Limite à connaître : ce volume suppose une forme parfaite (cylindre droit). Pour un **gâteau**, il n'y a pas de dimensions géométriques simples :
  une vérification indépendante du volume est la mesure par **déplacement** (immersion dans l'eau dans un sac étanche, ou méthode des
  graines de colza utilisée en boulangerie, norme AACC 10-05 : à confirmer avec le labo). À faire sur 2 ou 3 gâteaux,
  pour confirmer que l'erreur trouvée sur les étalons reste valable sur le vrai produit.

- Écrire le fichier de référence : `python validation_precision.py modele` crée `validation/reference_modele.csv` (à copier en `reference.csv`).
  Colonnes : `grandeur ; valeur_ref ; incertitude_ref ; tolerance ; commentaire`. Excel en français convient (séparateur `;`, virgule décimale).
  Noms de grandeurs : `hauteur_mm`, `hauteur_robuste_mm`, `longueur_mm`, `largeur_mm`, `diametre_equivalent_base_mm`, `aire_base_mm2`,
  `surface_mm2`, `volume_mm3`.

---

## 3. Prise de vue

À figer pour tous les scans (la validation ne vaut que pour ces conditions) :

- **Appareil et réglages fixes** : même objectif, **focale non modifiée**, mise au point verrouillée, sensibilité (ISO) basse, exposition manuelle
  ou verrouillée, déclenchement sans toucher l'appareil (c'est le rôle du banc automatisé). Une seule caméra est supposée par scan.
- **Nombre de photos : 40 minimum**, chevauchement entre photos voisines de **60 à 80 %**. Avec un plateau tournant : un pas de **9° ou moins**
  (360° / 40). Mieux : deux couronnes à des hauteurs de caméra différentes (par exemple 30° et 60° d'élévation) pour voir le dessus.
- **Cible d'échelle posée à plat sur le plateau**, à côté de l'objet, **tournant avec lui** (elle est fixe par rapport à l'objet).
  Chaque marqueur doit être net et visible dans une bonne partie des photos, vu de directions variées (au moins 3° d'écart entre les
  directions pour la triangulation).
- **Fond uni, sans motif**, ou option « Masquer le fond » de l'étape 1 : avec un plateau tournant et un appareil fixe, un décor fixe produit des points qui
  « ne bougent pas » et perturbent la reconstruction.
- **Lumière diffuse et stable** (pas de reflets, pas d'ombres dures qui changent d'une photo à l'autre).
- La pièce **repose sur le plateau, sur la feuille de cible** : c'est ce plan qui sert de plateau pour la coupe (§ « z de coupe » du README).
- **Ne pas utiliser une dimension de la pièce étalon pour mettre à l'échelle** : l'échelle vient uniquement de la cible, sinon la validation est truquée.

---

## 4. Série de scans

Trois niveaux, du moins au plus exigeant. À faire dans cet ordre, au minimum les deux premiers :

| Niveau | Quoi | Ce que l'on mesure |
|---|---|---|
| **A. Répétabilité** | 10 scans, **sans toucher** à la pièce ni au banc, entre deux prises complètes de photos | la dispersion due à l'acquisition et au calcul |
| **B. Reproductibilité** | 10 scans, **en retirant et reposant** la pièce chaque fois, si possible sur 2 jours | la dispersion réelle d'usage (placement, lumière) |
| **C. Plusieurs étalons** | A ou B sur chaque étalon (cylindre, parallélépipède, 2ᵉ taille) | l'effet de la forme et de la taille |

**10 scans au minimum** par série (en dessous de 10, l'écart-type est trop mal connu : l'outil le signale).

Pour chaque scan, un dossier de photos (`Photos_scan01`, `Photos_scan02`...) puis, avec **un seul ordre par scan** :

```
python pipeline_scan.py --photos Photos_scan01 --nom scan01 --taille-marqueur-mm 29.9
```

Il enchaîne reconstruction, densification, maillage, échelle et mesures, dans `colmap_workspace_scan01/`.
Pour une série, depuis PowerShell :

```
foreach ($i in 1..10) { $n = "{0:D2}" -f $i; python pipeline_scan.py --photos "Photos_scan$n" --nom "scan$n" --taille-marqueur-mm 29.9 }
```

Un scan qui échoue n'arrête pas la série. Garder le journal de chaque scan.

**Ne retirer aucun scan « mauvais » pour améliorer les statistiques.** Un scan manqué (cible non détectée, photos floues) est un résultat :
on le note, on explique la cause, et le protocole prévoit de l'exclure uniquement s'il est marqué non fiable par `mesure.py`
(échelle douteuse, maillage troué), avec la raison dans le rapport.

---

## 5. Analyse

```
python validation_precision.py analyser --reference validation/reference.csv --rapports "colmap_workspace_scan*/mesures" --condition "reproductibilité : pièce repositionnée entre scans"
```

Fichiers produits dans `validation/resultats/` :

- `validation_rapport.md` : tableau de synthèse et détail par grandeur, prêt à coller dans le rapport ;
- `validation_resultats.csv` et `validation_detail.csv` : pour Excel (un scan par ligne dans le détail) ;
- `validation_graphique.png` : erreur relative de chaque scan, biais moyen et bande ± U ;
- `validation_resultats.json` : lu ensuite par `gonflement.py`.

### Comment lire les résultats

- **Biais** (justesse) : moyenne des scans − référence. S'il est **significatif** (p < 0,05) et stable, il y a une erreur systématique :
  échelle (impression de la cible), calibration de l'objectif, seuil de coupe. Cause fréquente : **la hauteur « point haut »**
  est biaisée vers le haut par le bruit de surface (extrême de centaines de points). `hauteur_robuste_mm` moyenne localement et corrige l'essentiel :
  comparer les deux dans la validation.
- **Écart-type, coefficient de variation, limite de répétabilité r = 2,77 s** : la dispersion entre scans. Deux scans de la même pièce, dans les mêmes
  conditions, ne diffèrent de plus de r que dans 5 % des cas (ISO 5725).
- **Incertitude élargie U = k·√(s² + biais² + u_ref²)**, k = 2 : l'erreur qu'on peut attendre d'**un** scan. Le biais est conservé (choix prudent) ;
  si on le corrige, U diminue.
- Le volume dépend du **cube** de l'échelle : une erreur de 0,1 % sur l'échelle fait 0,3 % sur le volume. C'est pourquoi la mesure de la taille du marqueur compte.
- **Alertes** : la référence moins précise que le scanner (la validation est alors limitée par le pied à coulisse), valeurs atypiques (signalées,
  jamais retirées), nombre de scans insuffisant.

---

## 6. Gonflement des gâteaux

Une fois la validation faite sur le volume :

```
python gonflement.py --avant "colmap_workspace_av*/mesures" --apres "colmap_workspace_ap*/mesures" --validation validation/resultats/validation_resultats.json --nom gateau_X
```

Résultat : volume avant/après, **gonflement en % avec son incertitude élargie**, et si la variation est significative. Si les deux
états sont scannés avec la même cible et le même banc, l'erreur d'échelle est en grande partie commune et s'annule dans le rapport : l'outil donne aussi cette borne optimiste.
Plusieurs scans par état réduisent la part due à la répétabilité (divisée par la racine du nombre de scans), pas la part systématique.

---

## 7. Critères d'acceptation (à valider avec le tuteur)

Propositions de départ, à ajuster :

| Usage | Grandeur | Critère |
|---|---|---|
| Pièces mécaniques (réusinage) | cotes | U ≤ 0,5 mm, ou 1 % de la cote si plus grande |
| Gâteaux (gonflement) | volume | U ≤ 3 % |

Renseigner la colonne `tolerance` du fichier de référence (même unité que la grandeur) : le verdict « conforme » / « NON conforme » en découle.

## 8. Ce qui oblige à refaire la validation

Changement d'appareil, d'objectif ou de réglage de focale ; nouvelle impression de la cible (nouvelle taille réelle) ; changement d'éclairage
ou de fond ; changement de paramètres de reconstruction (résolution des photos, profondeur Poisson, réglage de coupe) ; nouveau type de produit (matière, couleur, taille très différente).

## 9. À mettre dans le rapport de stage

1. Les conditions de prise de vue (appareil, réglages, éclairage, nombre de photos, banc).
2. Les étalons, leurs dimensions de référence et l'incertitude de ces références.
3. Le nombre de scans et la condition (répétabilité ou reproductibilité).
4. Le tableau de synthèse (biais, écart-type, U, verdict) et le graphique.
5. Les scans exclus et pourquoi, les limites (formes testées, tailles testées, produits non testés).
6. La conclusion par usage : « pour des pièces de cet ordre de taille, la cote est connue à ± X mm ; le volume à ± Y % ».
