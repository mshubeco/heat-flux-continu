# HMPV CP Explorer — référence développement

## Objectif

Rapport CP multi-tenseurs (UI type HEAT) sur **HMPV FULL Complété** (export PADMEN FR+EN),
avec entrée privilégiée sur `coupled_cort_ubigram_orig`
(n-grammes hors RT ∥ publication cible co-RT).

Fork de `heat_cp_explorer` : mêmes sous-onglets Temps / C1…CR / Homogénéité / réseaux.

## Entrée

- CSV : `C:\Users\th8992be\Desktop\cross_corpus\Export_padmen_hmpv_20250101-20250630_full.csv`
- Spécificités : colonne `url` (pas `tweet_url_url`) ; pas de `media_*` ;
  cibles RT souvent via `RT @user` car `retweeted_tweet_id` ∉ `tweet_id` du CSV.

## Lancer

```bash
cd HEAT\03_analyses\pipelines\heat_pipelines\hmpv_cp_explorer
py -3.13 build_explorer.py --phase FULL --time-bin day --max-users 3000
py -3.13 build_explorer.py --phase FULL --time-bin hour --max-users 3000
# Fusion jour + section onglets horaire dans le même rapport.html
py -3.13 build_rapport.py --out run_full_day_simple --hour-out run_full_hour_simple
start run_full_day_simple\rapport.html#mod-coupled_cort_ubigram_orig
# Horaire : #mod-hour-coupled_cort_ubigram_orig
```

## Particularités HMPV (corrections 2026-11-26)

1. **Dates ISO mixte** — `…42Z` vs `…42.000Z` : `pd.to_datetime(..., format="ISO8601")`
   sinon ~60 % NaT (surtout hors RT) → `ubigram_orig` vide.
2. **IDs snowflake** — `_id_str` sans `float()` (>2^53) + `dtype=string` à la lecture CSV.
3. **Pool hybride** — 50 % volume + 35 % producteurs hors RT (min 1 post propre) + complément URL/volume ;
   un top volume pur est quasi 100 % RT sur ce corpus.
4. **`ubigram_orig` / `utrigram_orig`** — `min_users` assoupli à `max(2, min_users_ngram-1)`.

## Run de référence (`run_full_day_simple`, day, max_users=3000)

| Tenseur | nnz | users | mid |
|---|---:|---:|---:|
| `ubigram_orig` | 22 154 | 1 739 | 2 000 |
| `ucort` | 13 209 | 1 619 | 1 421 |
| `coupled_cort_ubigram_orig` | CoupledALS OK | — | — |

Docs n-grammes : 33 052 (hors RT = 17 691). Médias / bios : vides ou trop clairsemés (skip).

## Publication GitHub Pages

```powershell
.\sync_to_github_pages.ps1 -Push
```

URL : https://mshubeco.github.io/heat-flux-continu/cp-explorer-hmpv/rapport.html#mod-coupled_cort_ubigram_orig

## Historique

- 2026-11-26 — `rapport.html` : 2e section d’onglets **horaire** (jour intact) via `--hour-out run_full_hour_simple`.
- 2026-11-26 — Hors RT : `uurlt_orig` / `uudom_orig` / `uht_orig`.
- 2026-11-26 — Ck : signature spatio-temporelle (temps + acteurs×t + objets×t).
- 2026-11-26 — Ck : filtre acteurs même ordre de grandeur que w₁ + tops agrégés
  cibles RT / mentions / hashtags / URLs (comme HEAT).
- 2026-11-26 — création fork HMPV FULL + Pages `/cp-explorer-hmpv/` ;
  fix ISO8601 / snowflake / pool hybride pour activer le couplé phare.
