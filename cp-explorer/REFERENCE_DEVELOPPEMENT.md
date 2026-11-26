# HEAT CP Explorer — référence développement

## Objectif

Explorer **plusieurs décompositions CP simples** sur le corpus **canicule** (multi-langue),
pour juger quels tenseurs / configurations font ressortir des régimes de diffusion
potentiellement liés à de la manipulation (hubs, sync lexicale, objets viraux).

Template UI repris au propre depuis `hmpv_coupled_cp/index.html` :
**acteurs agrégés** (comportements corpus) + PCA + fiche composante + heatmap.
Pas de scorecard artificielle.

## Entrée

- CSV : `HEAT/01_corpus/exports_github/canicule_20260520_20260629/corpus_canicule_20260520_20260629.csv`
- Fenêtres : **P1** (20/05→08/06), **P2** (08/06→24/06), **P1P2** (20/05→24/06)

## Lancer

```bash
cd HEAT\03_analyses\pipelines\heat_pipelines\heat_cp_explorer

# Décompositions SIMPLES (défaut) — jour / heure / minute séparés
py -3.13 build_explorer.py --phase P1 --time-bin day
py -3.13 build_explorer.py --phase P1 --time-bin hour
py -3.13 build_explorer.py --phase P1 --time-bin minute

# Couplées en plus (optionnel)
py -3.13 build_explorer.py --phase P1 --time-bin day --with-coupled

# Sorties
#   run_p1_day_simple/index.html + rapport.html
#   run_p1_hour_simple/ …
#   run_p1_minute_simple/ …
start run_p1_day_simple\rapport.html
```

## Modalités (tenseurs)

Schéma commun : **USER × MID × MINUTE** → `log1p(count)`. Bin temporel = **1 minute** (`YYYY-MM-DD HH:MM` UTC).
Seul le MID change. Couplées = deux cubes avec **même facteur USER**.

| Clé | Modes | Lecture diffusion |
|-----|-------|-------------------|
| `uut` | user × cible RT × minute | amplification relationnelle |
| `uurlt` | user × url × minute | objets URL |
| `uudom` | user × domaine × minute | URL agrégée |
| `uht` | user × hashtag × minute | cadres lexicaux |
| `ubigram` | user × ngram_lemme × temps | n-grammes candidats n=2..4 (PMI+longest-match), tous tweets |
| `utrigram` | user × trigramme_lemme × temps | trigrammes glissants (contraste) |
| `ubigram_orig` | user × ngram × temps | **hors RT** — n-grammes candidats production propre |
| `utrigram_orig` | user × trigramme × temps | **hors RT** — trigrammes glissants |
| `uurlt_orig` | user × url × temps | **hors RT** — URLs (pendant de `uurlt`) |
| `uudom_orig` | user × domaine × temps | **hors RT** — domaines (pendant de `uudom`) |
| `uht_orig` | user × hashtag × temps | **hors RT** — hashtags (pendant de `uht`) |
| `umed_orig` | user × media_type × temps | **hors RT** — photo/video/gif |
| `umedurl_orig` | user × media_url × temps | **hors RT** — URL pièce jointe |
| `ucort` | user × publication cible co-RT × temps | **co-retweet** (MID = tweet original, ≥2 users) |
| `coupled_cort_ubigram_orig` | ubigram_orig ∥ ucort | n-grammes hors RT ∥ co-RT (**défaut**) |
| `coupled_cort_utrigram_orig` | utrigram_orig ∥ ucort | idem trigrammes (**défaut**) |
| `ucd` | user × creation_bin × temps | jour de création (snowflake YYYY-MM-DD) × activité |
| `ucdrt` | user × creation_bin × cible RT | cohortes de création × cibles RT (**sans** bin d’activité) |
| `ucdng` | user × creation_bin × n-gramme | cohortes de création × n-grammes lemmatisés (**sans** bin d’activité) |
| `ubio` / `ubio_ht` | user × bio_ngram\|tag × temps | n-grammes bio candidats (n=2..4, PMI+longest-match) / hashtags |
| `uname` | user × token display_name × temps | proxy profil (toujours dans le CSV) |
| `coupled_ucd_uut` | ucd ∥ uut | création ∥ RT-cible (**défaut**) |
| `coupled_ucd_ucort` | ucd ∥ ucort | création ∥ co-RT (**défaut**) |
| `coupled_ubio_ucort` | ubio ∥ ucort | bio ∥ co-RT (**défaut**) |
| `coupled_umed_ucort` | umed_orig ∥ ucort | média hors RT ∥ co-RT (**défaut**) |
| `coupled_umed_ubigram_orig` | umed_orig ∥ ubigram_orig | média ∥ n-grammes hors RT (**défaut**) |
| `coupled_uut_uurlt` | uurlt ∥ uut | RT-cible + liens (`--with-coupled`) |
| `coupled_uut_ubigram` | ubigram ∥ uut | RT-cible + n-grammes (`--with-coupled`) |

Bios : CSV intersection + cohort239 + `cache/user_bios.json` ; option `--scrape-bios`.
Création : pas de colonne `user_created_at` → snowflake (≥50 bits) → MID = jour `YYYY-MM-DD` (pas de bins annuels). Les comptes non datables (`unknown`) sont exclus de `ucd` / `ucdrt` / `ucdng` / couplés ucd.
Réseau Ck pour `ucdrt`/`ucdng` : co-occurrence sur le **mode 3** (cibles RT / n-grammes), pas sur les jours de création.

Voir aussi la section **§0 Configurations** dans `rapport.html`.

Valeurs : `log1p(count)`. Algo : CP-ALS / CoupledALS sparse (Python).

## N-grammes lemmatisés (multilingue)

- Colonne `language` du CSV → langue simplemma (`fr`, `en`, `es`, `it`, `de`, `pt`, …).
- Fallback `fr` (dominant sur P1).
- Nettoyage : strip `RT @…:`, URLs, mentions ; hashtags → token.
- Tokens ≥ 3 chars, stop-list FR/EN courte, POS non filtrée (simplemma sans spaCy).
- **`ubigram` / `ubigram_orig` / `ubio`** : n-grammes candidats n=2..4 (PMI≥0,8, df≥2, longest-match non chevauchant) ; fallback bigrammes si aucun candidat.
- **`utrigram` / `utrigram_orig`** : restent en fenêtres glissantes n=3 (contraste).
- Filtre mid : ≥ `min_users_ngram` users, top `max_bigrams` / `max_trigrams`.
- Tous tweets du pool (y compris RT nettoyés) — le signal de diffusion est aussi dans les RT.

## Acteurs & agrégats (focus analytique)

Pour chaque tenseur, `aggregate` dans le payload :

- **`top_actors`** — Σ |A| sur toutes les composantes ; profils corpus :
  n tweets / RT / propres, top cibles RT, hashtags, URLs précises (`host/path`, cliquables), domaines.
- **`top_mids`** — Σ |B| (cibles RT, URLs, domaines, hashtags, n-grammes…).
- **`top_mids2`** — si couplé : cibles RT de la 2ᵉ couche.

Onglet explorateur **Acteurs** = vue d’ensemble par tenseur.
Les tops par composante C1…CR restent disponibles (comportements attachés aux users).

## Sorties

- `index.html` + `data.js` + `payload.json` — explorateur (onglet Acteurs par défaut)
- **`rapport.html`** — **deux sections d’onglets** :
  1. **Temporalité journalière** — familles Méta / Objets / Hors RT / Profil / Couplés (inchangé)
  2. **Temporalité horaire** — mêmes familles tenseurs (sauf Méta), hash `#mod-hour-<clé>`
  Sous-onglets : **Temps** | **C1…CR** | **Homogénéité / diag**
- **`data.js`** — `window.HEAT_CP` (jour) + optionnel `window.HEAT_CP_HOUR` (fusion `--hour-out`)
- **`rapport_<modalité>.html`** — fiche dédiée (acteurs + objets agrégés en tête)
- **`rapports_index.html`** — hub des rapports
- `tensor_*.csv` · `*/user_explicit.csv` · `pipeline_config.json`

Régénérer les rapports sans relancer le CP :

```bash
# Jour seul
py -3.13 build_rapport.py --out run_p1_day_simple

# Jour + 2e section horaire (fusion payload hour dans data.js)
py -3.13 build_rapport.py --out run_p1_day_simple --hour-out run_p1_hour_simple
start run_p1_day_simple\rapport.html

# Publier sur GitHub Pages (mshubeco/heat-flux-continu/cp-explorer/)
.\sync_to_github_pages.ps1 -Push
```

### Légendes graphiques (ce qu’on lit / ce que ça dit)

| Graphique | On lit | Ça dit |
|---|---|---|
| Vue acteurs (cartes) | Comptes à forte masse CP + cibles/tags/domaines | Qui émerge, sur quels comportements |
| MID agrégés | Objets pondérés Σ composantes | Top cibles / hashtags / URLs / n-grammes du tenseur |
| Taille du tenseur (barres) | Effectifs users vs mid | Échelle acteurs vs objets |
| Temporalité | Poids par bin et par rang | Quand chaque régime est actif |
| Gini users/mid par rang | Concentration acteurs vs objets | Structure hub vs diffuse de chaque C |
| PCA user | Nuage 2D coloré par rang dominant | Groupes d’acteurs au profil proche |
| Heatmap user×rang | Intensité compte↔composante | Qui est typique de quel régime |
| Tops par composante | Acteurs + comportements + objets | Contenu interprétable de chaque C |


## Runs P1 — simples séparés jour / minute (2026-03-25)

Hub : `COMPARE_BINS.html`.

| Sortie | Bin | Modalités | Fit (logs CP, approx.) |
|---|---|---|---|
| `run_p1_day_simple/` | jour (19 bins) | 6 simples | ~0.77–0.94 selon tenseur |
| `run_p1_minute_simple/` | minute (4k–15k bins) | 6 simples | ~0.97–0.99 (très sparse) |

Minute : fit toujours mauvais (err ≥ 0.97) — utile en complément sync, pas en run principal.
Lecture prioritaire : **day simple** + onglet / sections acteurs.

Couplage : seulement avec `--with-coupled`.

## Historique

- 2026-11-26 — Table acteurs Ck : colonne **Création compte** (date+heure UTC via snowflake `user_id` ; aussi `account_created_at` dans le payload aux prochains runs).
- 2026-11-26 — UI Ck : **brief** (KPI + top5 acteurs/objets) · ordre Qui/quoi → Quand → comportements · signature/réseau/table en **`<details>`** (rendu différé à l’ouverture).
- 2026-11-26 — `rapport.html` : 2e section d’onglets **horaire** (jour intact) ; `build_rapport.py --hour-out run_*_hour_simple` → `HEAT_CP_HOUR`.
- 2026-11-26 — bin `hour` + runs `run_p1_hour_simple/` (HEAT) / `run_full_hour_simple/` (HMPV).
- 2026-11-26 — Hors RT : `uurlt_orig` / `uudom_orig` / `uht_orig` (pendants Objets sans RT ; pas de `uut_orig`).
- 2026-11-26 — Ck : **signature spatio-temporelle** (barres temps + heatmaps acteurs×t / objets×t, pic en pointillés).
- 2026-11-26 — Ck : acteurs filtrés au **même ordre de grandeur** que w₁ (`w ≥ 10^⌊log₁₀(w₁)⌋`) + bandeau tops agrégés cibles/mentions/hashtags/URLs.
- 2026-11-26 — jumeau HMPV FULL : `hmpv_cp_explorer/` → Pages `/cp-explorer-hmpv/` (couplé `coupled_cort_ubigram_orig`).
- 2026-11-26 — onglets Ck : graphe réseau d’échanges (RT / mentions) entre acteurs, survol détaillé.
- 2026-11-26 — `ubio` : n-grammes candidats (n=2..4, PMI≥0.8, df≥2, longest-match) à la place des bigrammes fixes.
- 2026-11-26 — onglets Ck : tableau acteurs (pubs, %RT, bio, top5 cibles/mentions/hashtags/domaines) ; `--enrich-profiles`.
- 2026-11-26 — libellé `ucort` : « status co-RT » → **publication cible co-RT**.
- 2026-11-26 — réseau Ck : poids/couleur/épaisseur des liens = **n_shared** (nb d’objets distincts), plus l’intensité MID (×count).
- 2026-11-26 — sync GitHub Pages : `sync_to_github_pages.ps1 -Push` → `mshubeco/heat-flux-continu` `/cp-explorer/`.
- 2026-11-26 — tenseurs `ucdrt` (émetteur × jour création × cible RT) et `ucdng` (× n-grammes), sans bin d’activité.
- 2026-11-26 — retrait onglet **Tendance** (ordre stable) par tenseur ; défaut = Temps.
- 2026-11-26 — publication GitHub Pages : `mshubeco/heat-flux-continu` → `/cp-explorer/` (rapports + data.js + scripts).
- 2026-11-26 — réseau Ck : survol lien = tooltip custom (hit-test segment, MID partagés) — Plotly `lines` seul ne capturait pas le hover.
- 2026-11-26 — réseau Ck : épaisseur lien ∝ n_shared (nb d’éléments MID) ; sans pastilles sur les arêtes ; couleur nœud ∝ poids CP.
- 2026-11-26 — réseau Ck : survol arête → liste des MID partagés (liés au tenseur) ; noverlap + taille ∝ degré.
- 2026-11-26 — réseau Ck : mono-modalité alignée au tenseur (co-MID révélés / RT pour `uut`), spatialisation AF/PCA 2D, zoom-pan Plotly.
- 2026-11-26 — `ubigram` / `ubigram_orig` (+ couplés) : n-grammes candidats n=2..4 (PMI+longest-match), comme `ubio` ; `utrigram*` reste glissant n=3.
- 2026-11-26 — tops acteurs : domaines → URLs précises (`host/path`, `top_urls`) cliquables (`a.web-link` → https) ; barres MID uurlt/uudom idem.
- 2026-11-26 — handles `@acteur` → X : barres acteurs en HTML (`a.x-link` + `href` natif), chips sous le réseau (Plotly non cliquable sur ticks).
- 2026-11-26 — `ucd` : MID = jour de création (YYYY-MM-DD), plus de bins multi-annuels.
- 2026-11-26 — onglets C1…CR : 3 graphes facteurs (temps × acteurs × objets) par composante.
- 2026-11-26 — bandeau tenseur `a × b × c` partout (onglets, Configs, overview, index, rapports mono) ; clé technique en secondaire.
- 2026-11-26 — UI rapport : familles + Tendance (ordre stable) / Temps / Homogénéité (PCA composantes).
- 2026-03-25 — tenseurs média hors RT : `umed_orig` / `umedurl_orig` + couplés umed∥ucort / umed∥bigram_orig.
- 2026-03-25 — tenseurs profil : `ucd` (création snowflake), `ubio`/`ubio_ht`, `uname` + couplés ucd/ubio.
- 2026-03-25 — tenseurs type pub. : `ubigram_orig` / `utrigram_orig` / `ucort` + couplés `coupled_cort_*_orig` (défaut).
- 2026-03-25 — rapport.html : navigation par onglets tenseurs (sticky) au lieu d’une liste.
- 2026-03-25 — scorecard retirée ; focus acteurs agrégés + comportements (RT/tags/domaines) par tenseur.
- 2026-03-25 — explorateur HEAT ; runs **day** et **minute** séparés, décomp. simples par défaut.
