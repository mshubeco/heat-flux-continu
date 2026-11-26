# HEAT · explorateur CP multi-tenseurs

Décomposition CP (canicule P1, bin **jour**) : rapports Plotly + explorateur interactif.

## Voir en ligne

- [Index des rapports](https://mshubeco.github.io/heat-flux-continu/cp-explorer/rapports_index.html)
- [Rapport complet](https://mshubeco.github.io/heat-flux-continu/cp-explorer/rapport.html)
- [Explorateur](https://mshubeco.github.io/heat-flux-continu/cp-explorer/index.html)

Fenêtre : **P1 20/05–07/06/2026** · tenseurs USER × MID × jour (sauf `ucdrt` / `ucdng` : création × objet, sans bin d’activité) · réseaux Ck mono-modalité.

## Mise à jour depuis le workspace

```powershell
cd Desktop\HEAT\03_analyses\pipelines\heat_pipelines\heat_cp_explorer
.\sync_to_github_pages.ps1 -Push
```

## Contenu publié

| Fichier | Rôle |
|---|---|
| `rapport.html` / `rapport_*.html` | Rapports par modalité |
| `index.html` | Explorateur interactif |
| `data.js` | Payload embarqué (`window.HEAT_CP`) |
| `scripts/build_*.py` | Reconstruction locale (nécessite corpus CSV + deps) |
| `REFERENCE_DEVELOPPEMENT.md` | Notes de développement |

Les CSV tenseurs bruts ne sont pas versionnés ici (regénérables via `scripts/`).

## Source workspace

Canonique : `Desktop\HEAT\03_analyses\pipelines\heat_pipelines\heat_cp_explorer\`
