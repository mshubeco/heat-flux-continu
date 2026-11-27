# -*- coding: utf-8 -*-
"""HMPV FULL · cascades de filiations discursives par n-grammes partagés.

Construit un graphe de publications (hors RT) reliées si |n-grammes ∩| ≥ min_shared
(défaut 4), puis une forêt de filiations orientée dans le temps (parent = voisin
antérieur au plus fort chevauchement). Visualisation HTML pour GitHub Pages.

Usage :
  py -3.13 build_ngram_filiation.py
  py -3.13 build_ngram_filiation.py --min-shared 4 --max-df 400 --min-tree 6
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_explorer import (
    DEFAULT_CSV,
    build_collocation_stats,
    lemmatize_tokens,
    map_lang,
    select_candidate_ngrams,
)

ROOT = Path(__file__).resolve().parent
OUT_DEFAULT = ROOT / "run_ngram_filiation"


def _id_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _s(val, default: str = "") -> str:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return default if s.lower() in {"", "nan", "<na>", "none"} else s


def load_publications(csv_path: Path) -> pd.DataFrame:
    usecols = [
        "tweet_id",
        "created_at",
        "tweet_user_id",
        "user_screen_name",
        "user_name",
        "text",
        "tweet_type",
        "language",
        "url",
        "retweet_count",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, dtype="string")
    df = df[df["tweet_type"].fillna("").str.lower().isin(["original", "reply", "quote"])].copy()
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True, errors="coerce")
    df = df.dropna(subset=["created_at", "text", "tweet_id"])
    df["tweet_id"] = df["tweet_id"].map(_id_str)
    df["tweet_user_id"] = df["tweet_user_id"].map(_id_str)
    df["retweet_count"] = pd.to_numeric(df["retweet_count"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset=["tweet_id"], keep="first")
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def extract_ngrams(df: pd.DataFrame) -> tuple[list[set[str]], dict]:
    toks_all: list[list[str]] = []
    for r in df.itertuples(index=False):
        toks_all.append(lemmatize_tokens(r.text or "", map_lang(r.language)))
    stats = build_collocation_stats(toks_all, n_max=4)
    ngrams: list[set[str]] = []
    for toks in toks_all:
        sel = select_candidate_ngrams(toks, stats, n_max=4, min_pmi=0.8, min_df=2)
        ngrams.append(set(sel))
    meta = {
        "n_docs_ngrams": sum(1 for s in ngrams if s),
        "mean_ngrams": round(sum(len(s) for s in ngrams) / max(len(ngrams), 1), 3),
        "n_unique_ngrams": len({g for s in ngrams for g in s}),
    }
    return ngrams, meta


def collapse_identical_ngram_sets(
    df: pd.DataFrame,
    ngrams: list[set[str]],
) -> tuple[pd.DataFrame, list[set[str]]]:
    """
    Fusionne les publications au même ensemble de n-grammes.
    Horodatage = plus ancien ; affichage = max RT parmi les 3 plus anciens.
    """
    buckets: dict[frozenset[str], list[int]] = defaultdict(list)
    for i, grams in enumerate(ngrams):
        if len(grams) < 1:
            continue
        buckets[frozenset(grams)].append(i)

    rows: list[dict] = []
    rep_ngrams: list[set[str]] = []
    for grams_key, idxs in buckets.items():
        idxs_sorted = sorted(idxs, key=lambda i: (df.at[i, "created_at"], i))
        earliest = idxs_sorted[0]
        head = idxs_sorted[: min(3, len(idxs_sorted))]
        show = max(head, key=lambda i: (int(df.at[i, "retweet_count"] or 0), -i))
        row = df.loc[earliest].to_dict()
        row["created_at"] = df.at[earliest, "created_at"]
        row["user_screen_name"] = df.at[show, "user_screen_name"]
        row["user_name"] = df.at[show, "user_name"]
        row["text"] = df.at[show, "text"]
        row["url"] = df.at[show, "url"]
        row["retweet_count"] = df.at[show, "retweet_count"]
        row["tweet_id"] = df.at[show, "tweet_id"]
        row["tweet_type"] = df.at[show, "tweet_type"]
        row["copies"] = len(idxs_sorted)
        rows.append(row)
        rep_ngrams.append(set(grams_key))

    before = list(zip(rows, rep_ngrams))
    before.sort(key=lambda t: t[0]["created_at"])
    rep_ngrams = [set(g) for _, g in before]
    out = pd.DataFrame([r for r, _ in before]).reset_index(drop=True)
    return out, rep_ngrams


def build_edges(
    ngrams: list[set[str]],
    *,
    min_shared: int,
    max_df: int,
) -> list[tuple[int, int, int, list[str]]]:
    """Arêtes (i, j, |∩|, ngrammes partagés) avec i < j (indices docs)."""
    inv: dict[str, list[int]] = defaultdict(list)
    for i, grams in enumerate(ngrams):
        for g in grams:
            inv[g].append(i)

    pair_hits: Counter[tuple[int, int]] = Counter()
    for g, idxs in inv.items():
        if len(idxs) < 2 or len(idxs) > max_df:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if i > j:
                    i, j = j, i
                pair_hits[(i, j)] += 1

    edges: list[tuple[int, int, int, list[str]]] = []
    for (i, j), c in pair_hits.items():
        if c < min_shared:
            continue
        shared = sorted(ngrams[i] & ngrams[j])
        if len(shared) < min_shared:
            continue
        # ignorer identité stricte (déjà collapsée en amont, filet de sécurité)
        if ngrams[i] == ngrams[j]:
            continue
        edges.append((i, j, len(shared), shared[:12]))
    return edges


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], rank: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def build_filiation_forest(
    n_docs: int,
    edges: list[tuple[int, int, int, list[str]]],
    times: list[pd.Timestamp],
) -> tuple[dict[int, int | None], dict[int, int], dict[tuple[int, int], tuple[int, list[str]]]]:
    """
    Parent[i] = voisin antérieur de poids max ; None = racine.
    level[i] = profondeur depuis la racine de sa cascade.
    """
    adj: dict[int, list[tuple[int, int, list[str]]]] = defaultdict(list)
    edge_meta: dict[tuple[int, int], tuple[int, list[str]]] = {}
    parent_uf = list(range(n_docs))
    rank = [0] * n_docs
    involved = set()

    for i, j, w, shared in edges:
        adj[i].append((j, w, shared))
        adj[j].append((i, w, shared))
        edge_meta[(i, j)] = (w, shared)
        edge_meta[(j, i)] = (w, shared)
        _union(parent_uf, rank, i, j)
        involved.add(i)
        involved.add(j)

    # degré entrant provisoire pour favoriser les ramifications (éviter les chaînes)
    child_count: Counter[int] = Counter()
    order_nodes = sorted(involved, key=lambda u: (times[u], u))
    parent: dict[int, int | None] = {i: None for i in involved}
    for i in order_nodes:
        earlier = [(j, w, shared) for j, w, shared in adj[i] if times[j] < times[i]]
        if not earlier:
            parent[i] = None
            continue
        max_w = max(t[1] for t in earlier)
        cands = [t for t in earlier if t[1] == max_w]
        # parmi poids max : parent le moins chargé (branching), puis le plus ancien
        cands.sort(key=lambda t: (child_count[t[0]], times[t[0]], t[0]))
        p = cands[0][0]
        parent[i] = p
        child_count[p] += 1

    # niveaux
    level: dict[int, int] = {}

    def depth(u: int) -> int:
        if u in level:
            return level[u]
        p = parent.get(u)
        if p is None:
            level[u] = 0
        else:
            level[u] = depth(p) + 1
        return level[u]

    for u in involved:
        depth(u)

    return parent, level, edge_meta


def collect_trees(
    parent: dict[int, int | None],
    level: dict[int, int],
    edge_meta: dict[tuple[int, int], tuple[int, list[str]]],
    df: pd.DataFrame,
    ngrams: list[set[str]],
    *,
    min_tree: int,
    max_trees: int,
    max_nodes: int,
) -> list[dict]:
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []
    for u, p in parent.items():
        if p is None:
            roots.append(u)
        else:
            children[p].append(u)

    def subtree_size(r: int) -> int:
        stack = [r]
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(children.get(u, []))
        return len(seen)

    def root_depth(r: int) -> int:
        stack = [r]
        seen: set[int] = set()
        md = 0
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            md = max(md, level.get(u, 0))
            stack.extend(children.get(u, []))
        # profondeur relative à la racine
        return max(0, md - level.get(r, 0))

    def branch_score(r: int) -> tuple[int, int, int]:
        """(niveaux avec ≥2 enfants, profondeur, taille) — favorise les vraies cascades."""
        size = subtree_size(r)
        depth = root_depth(r)
        branched = 0
        stack = [r]
        seen: set[int] = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            kids = children.get(u, [])
            if len(kids) >= 2:
                branched += 1
            stack.extend(kids)
        return branched, depth, size

    scored = sorted(
        ((branch_score(r), r) for r in roots),
        key=lambda x: (-x[0][0], -x[0][1], -x[0][2], df.at[x[1], "created_at"], x[1]),
    )

    trees: list[dict] = []
    nodes_budget = max_nodes
    for (branched, depth, size), root in scored:
        if size < min_tree:
            continue
        # écarter les pures chaînes linéaires sans ramification si on a mieux
        if branched == 0 and depth > 8 and len(trees) >= 8:
            continue
        if len(trees) >= max_trees or nodes_budget <= 0:
            break
        # BFS collect
        order: list[int] = []
        stack = [root]
        seen: set[int] = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            kids = sorted(
                children.get(u, []),
                key=lambda k: (df.at[k, "created_at"], -edge_meta.get((u, k), (0, []))[0], k),
            )
            stack.extend(reversed(kids))

        if len(order) > nodes_budget and trees:
            break

        nodes = []
        links = []
        for u in order:
            row = df.loc[u]
            text = _s(row.text)
            nodes.append(
                {
                    "id": _s(row.tweet_id),
                    "idx": u,
                    "level": int(level.get(u, 0)),
                    "t": row.created_at.isoformat(),
                    "user": _s(row.user_screen_name),
                    "name": _s(row.user_name),
                    "type": _s(row.tweet_type),
                    "rt": int(row.retweet_count or 0),
                    "copies": int(getattr(row, "copies", 1) or 1),
                    "url": _s(row.url),
                    "text": text[:420],
                    "n_ngrams": len(ngrams[u]),
                    "ngrams": sorted(ngrams[u])[:16],
                }
            )
            p = parent.get(u)
            if p is not None and p in seen:
                w, shared = edge_meta.get((p, u), (0, []))
                links.append(
                    {
                        "source": _s(df.at[p, "tweet_id"]),
                        "target": _s(row.tweet_id),
                        "shared": int(w),
                        "ngrams": shared[:10],
                    }
                )

        max_lvl = max((n["level"] for n in nodes), default=0)
        trees.append(
            {
                "id": f"t{len(trees):03d}",
                "root": _s(df.at[root, "tweet_id"]),
                "n_nodes": len(nodes),
                "n_links": len(links),
                "max_level": max_lvl,
                "t0": nodes[0]["t"] if nodes else None,
                "root_user": _s(df.at[root, "user_screen_name"]),
                "root_text": _s(df.at[root, "text"])[:180],
                "nodes": nodes,
                "links": links,
            }
        )
        nodes_budget -= len(order)

    return trees


def layout_tree(tree: dict) -> None:
    """Layout hiérarchique simple : y = niveau, x = feuilles équiréparties."""
    children: dict[str, list[str]] = defaultdict(list)
    node_by_id = {n["id"]: n for n in tree["nodes"]}
    roots = []
    child_targets = {l["target"] for l in tree["links"]}
    for n in tree["nodes"]:
        if n["id"] not in child_targets:
            roots.append(n["id"])
    for l in tree["links"]:
        children[l["source"]].append(l["target"])

    leaves_x: dict[str, float] = {}
    counter = [0]

    def place(u: str) -> float:
        kids = children.get(u, [])
        if not kids:
            x = float(counter[0])
            counter[0] += 1
            leaves_x[u] = x
            return x
        xs = [place(k) for k in kids]
        x = sum(xs) / len(xs)
        leaves_x[u] = x
        return x

    for r in roots:
        place(r)

    max_x = max(leaves_x.values()) if leaves_x else 1.0
    max_lvl = max((n["level"] for n in tree["nodes"]), default=0) or 1
    for n in tree["nodes"]:
        x = leaves_x.get(n["id"], 0.0)
        n["x"] = round(x / max(max_x, 1e-9), 5)
        n["y"] = round(n["level"] / max_lvl, 5) if max_lvl else 0.0


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>HMPV · Filiations discursives (n-grammes)</title>
<style>
:root{
  --bg:#0f1419; --panel:#171e26; --panel2:#1c2530; --border:#2a3542;
  --text:#e7eef6; --muted:#8b9aab; --accent:#6ec6ff;
  --font:"Segoe UI","IBM Plex Sans",system-ui,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font-family:var(--font)}
body{display:grid;grid-template-columns:320px 1fr 340px;grid-template-rows:auto 1fr;height:100vh}
header{grid-column:1/-1;display:flex;align-items:center;gap:1rem;padding:.65rem 1rem;border-bottom:1px solid var(--border);background:var(--panel)}
header h1{margin:0;font-size:.95rem;font-weight:650}
header .meta{color:var(--muted);font-size:.78rem}
.legend{margin-left:auto;display:flex;align-items:center;gap:.5rem;font-size:.75rem;color:var(--muted)}
.grad{width:120px;height:10px;border-radius:4px;background:linear-gradient(90deg,#f6e27a,#f0a45a,#e45858,#9b59b6,#3d348b)}
aside,#detail{background:var(--panel);display:flex;flex-direction:column;min-height:0}
aside{border-right:1px solid var(--border)}
#detail{border-left:1px solid var(--border)}
.panel-head{padding:.7rem .85rem .5rem;border-bottom:1px solid var(--border)}
.panel-head label{display:block;font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem}
input[type=search],select{width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:.4rem .5rem;font:inherit;font-size:.82rem}
#treeList{flex:1;overflow:auto;padding:.4rem}
.tree-item{display:block;width:100%;text-align:left;background:transparent;border:1px solid transparent;border-radius:8px;color:var(--text);padding:.55rem .6rem;margin-bottom:.35rem;cursor:pointer;font:inherit}
.tree-item:hover{background:var(--panel2);border-color:var(--border)}
.tree-item.active{background:#243041;border-color:#3d5a73}
.tree-item strong{display:block;font-size:.82rem;margin-bottom:.15rem}
.tree-item span{display:block;font-size:.72rem;color:var(--muted);line-height:1.35}
#stage{position:relative;min-width:0;min-height:0;background:radial-gradient(1200px 600px at 50% -10%,#1a2430 0%,var(--bg) 55%)}
svg#viz{width:100%;height:100%;display:block}
.node circle{stroke:#0f1419;stroke-width:1.5px;cursor:pointer}
.node text{font-size:10px;fill:var(--muted);pointer-events:none}
.link{fill:none;stroke-opacity:.55}
#hint{position:absolute;left:12px;bottom:10px;font-size:.72rem;color:var(--muted);background:rgba(15,20,25,.7);padding:.35rem .55rem;border-radius:6px}
#detailBody{flex:1;overflow:auto;padding:.85rem}
#detailBody h2{margin:0 0 .4rem;font-size:.95rem}
#detailBody .muted{color:var(--muted);font-size:.78rem}
#detailBody .box{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.65rem;margin:.6rem 0;font-size:.82rem;line-height:1.45;white-space:pre-wrap}
.tags{display:flex;flex-wrap:wrap;gap:.3rem;margin-top:.45rem}
.tag{font-size:.68rem;background:#243041;border:1px solid #354960;border-radius:999px;padding:.15rem .45rem;color:#c9d7e6}
a.x{color:var(--accent);text-decoration:none}
a.x:hover{text-decoration:underline}
.statrow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin:.5rem 0}
.stat{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.45rem .55rem}
.stat b{display:block;font-size:1rem}
.stat span{font-size:.68rem;color:var(--muted)}
@media (max-width:980px){
  body{grid-template-columns:1fr;grid-template-rows:auto auto 55vh auto}
  aside,#detail{border:none;border-bottom:1px solid var(--border);max-height:28vh}
}
</style>
</head>
<body>
<header>
  <h1>Filiations discursives · n-grammes</h1>
  <div class="meta" id="hdrMeta">HMPV FULL · chargement…</div>
  <div class="legend">racine <div class="grad" title="niveau 0 → profondeur max"></div> reprises</div>
</header>
<aside>
  <div class="panel-head">
    <label>Cascades (≥ __MIN_SHARED__ n-grammes partagés)</label>
    <input type="search" id="q" placeholder="Filtrer @user / texte / id…"/>
  </div>
  <div id="treeList"></div>
</aside>
<div id="stage">
  <svg id="viz"></svg>
  <div id="hint">Clic nœud · molette zoom · glisser pour pan. Axe vertical = niveau de reprise.</div>
</div>
<section id="detail">
  <div class="panel-head"><label>Détail</label></div>
  <div id="detailBody"><p class="muted">Sélectionnez une cascade, puis un nœud.</p></div>
</section>
<script>
const DATA = __PAYLOAD__;

const colorAt = (t) => {
  // t in [0,1] : jaune → orange → rouge → violet → indigo
  const stops = [
    [246,226,122],[240,164,90],[228,88,88],[155,89,182],[61,52,139]
  ];
  const x = Math.max(0, Math.min(1, t));
  const u = x * (stops.length - 1);
  const i = Math.floor(u);
  const f = u - i;
  const a = stops[i], b = stops[Math.min(i+1, stops.length-1)];
  const r = Math.round(a[0]*(1-f)+b[0]*f);
  const g = Math.round(a[1]*(1-f)+b[1]*f);
  const bl = Math.round(a[2]*(1-f)+b[2]*f);
  return `rgb(${r},${g},${bl})`;
};

const esc = (s) => String(s??'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

let activeTree = null;
let selectedId = null;
let transform = {x:0,y:0,k:1};

const svg = document.getElementById('viz');
const listEl = document.getElementById('treeList');
const detailEl = document.getElementById('detailBody');
const qEl = document.getElementById('q');

document.getElementById('hdrMeta').textContent =
  `${DATA.meta.corpus} · ${DATA.meta.window} · hors RT · min_shared=${DATA.meta.min_shared} · ${DATA.trees.length} cascades · ${DATA.meta.n_pubs_hors_rt.toLocaleString('fr-FR')} pubs`;

function renderList(filter='') {
  const f = filter.trim().toLowerCase();
  listEl.innerHTML = '';
  DATA.trees.forEach((t, idx) => {
    const blob = `${t.root_user} ${t.root_text} ${t.root}`.toLowerCase();
    if (f && !blob.includes(f) && !String(t.n_nodes).includes(f)) return;
    const btn = document.createElement('button');
    btn.className = 'tree-item' + (activeTree && activeTree.id===t.id ? ' active' : '');
    btn.innerHTML = `<strong>#${idx+1} · ${t.n_nodes} pubs · profondeur ${t.max_level}</strong>
      <span>@${esc(t.root_user)} · ${esc((t.root_text||'').slice(0,110))}</span>`;
    btn.onclick = () => selectTree(t);
    listEl.appendChild(btn);
  });
}

function selectTree(t) {
  activeTree = t;
  selectedId = t.root;
  transform = {x:0,y:0,k:1};
  renderList(qEl.value);
  draw();
  showNode(t.nodes.find(n => n.id===selectedId) || t.nodes[0]);
}

function showNode(n) {
  if (!n || !activeTree) return;
  selectedId = n.id;
  const inbound = activeTree.links.find(l => l.target===n.id);
  const outs = activeTree.links.filter(l => l.source===n.id);
  const url = n.url || `https://x.com/${encodeURIComponent(n.user)}/status/${n.id}`;
  detailEl.innerHTML = `
    <h2>@${esc(n.user)}</h2>
    <div class="muted">${esc(n.name)} · ${esc(n.type)} · niveau ${n.level} · ${esc(n.t).replace('T',' ').slice(0,19)} UTC</div>
    <div class="statrow">
      <div class="stat"><b>${n.rt}</b><span>retweets</span></div>
      <div class="stat"><b>${n.n_ngrams}</b><span>n-grammes</span></div>
      <div class="stat"><b>${n.copies||1}</b><span>copies même set</span></div>
    </div>
    <div class="box">${esc(n.text)}</div>
    <p class="muted"><a class="x" href="${esc(url)}" target="_blank" rel="noopener">Ouvrir sur X</a> · id ${esc(n.id)}</p>
    <label style="font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">N-grammes du nœud</label>
    <div class="tags">${(n.ngrams||[]).map(g=>`<span class="tag">${esc(g.replaceAll('_',' '))}</span>`).join('') || '<span class="muted">—</span>'}</div>
    ${inbound ? `<p style="margin-top:.8rem" class="muted">Reprise depuis parent · <b style="color:var(--text)">${inbound.shared}</b> n-grammes partagés</p>
      <div class="tags">${inbound.ngrams.map(g=>`<span class="tag">${esc(g.replaceAll('_',' '))}</span>`).join('')}</div>` : `<p style="margin-top:.8rem" class="muted">Racine de cascade</p>`}
    <p class="muted" style="margin-top:.8rem">${outs.length} reprise(s) descendante(s)</p>`;
  draw();
}

function draw() {
  if (!activeTree) return;
  const W = svg.clientWidth || 800;
  const H = svg.clientHeight || 600;
  const padX = 48, padY = 36;
  const maxLvl = Math.max(1, activeTree.max_level);
  const nodes = activeTree.nodes;
  const links = activeTree.links;
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));

  const X = (n) => padX + n.x * (W - 2*padX);
  const Y = (n) => padY + n.y * (H - 2*padY);

  let linksG = '', nodesG = '';
  for (const l of links) {
    const a = byId[l.source], b = byId[l.target];
    if (!a || !b) continue;
    const x1=X(a), y1=Y(a), x2=X(b), y2=Y(b);
    const my = (y1+y2)/2;
    const t = (a.level + b.level) / (2 * maxLvl);
    const col = colorAt(t);
    const sw = 1.2 + Math.min(4, l.shared) * 0.35;
    linksG += `<path class="link" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}" stroke="${col}" stroke-width="${sw}"/>`;
  }
  for (const n of nodes) {
    const t = n.level / maxLvl;
    const col = colorAt(t);
    const base = n.id===activeTree.root ? 8 : (n.id===selectedId ? 7 : 5);
    const r = base + Math.min(6, Math.log2(Math.max(1, n.copies||1)));
    const label = n.level===0 || n.id===selectedId ? `@${esc(n.user)}` : '';
    nodesG += `<g class="node" data-id="${esc(n.id)}" transform="translate(${X(n)},${Y(n)})">
      <circle r="${r}" fill="${col}" ${n.id===selectedId?'stroke="#fff" stroke-width="2"':''}/>
      ${label?`<text y="16" text-anchor="middle">${label}</text>`:''}
    </g>`;
  }

  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.innerHTML = `<g id="viewport" transform="translate(${transform.x},${transform.y}) scale(${transform.k})">${linksG}${nodesG}</g>`;
  svg.querySelectorAll('.node').forEach(g => {
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = g.getAttribute('data-id');
      showNode(byId[id]);
    });
  });
}

// pan / zoom
(function enablePanZoom(){
  let dragging=false, lx=0, ly=0;
  svg.addEventListener('mousedown', e => { dragging=true; lx=e.clientX; ly=e.clientY; });
  window.addEventListener('mouseup', () => dragging=false);
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    transform.x += e.clientX-lx; transform.y += e.clientY-ly;
    lx=e.clientX; ly=e.clientY; draw();
  });
  svg.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.08 : 0.92;
    transform.k = Math.max(0.4, Math.min(4, transform.k * factor));
    draw();
  }, {passive:false});
})();

qEl.addEventListener('input', () => renderList(qEl.value));
window.addEventListener('resize', () => draw());
renderList();
if (DATA.trees.length) selectTree(DATA.trees[0]);
</script>
</body>
</html>
"""


def write_html(payload: dict, out_html: Path, min_shared: int) -> None:
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__MIN_SHARED__", str(min_shared))
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cascades de filiations n-grammes HMPV FULL")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--min-shared", type=int, default=4, help="Seuil |n-grammes ∩| pour une arête")
    ap.add_argument("--max-df", type=int, default=400, help="Exclure n-grammes trop fréquents (anti-hub)")
    ap.add_argument("--min-tree", type=int, default=6, help="Taille mini d'une cascade exportée")
    ap.add_argument("--max-trees", type=int, default=40)
    ap.add_argument("--max-nodes", type=int, default=2500)
    args = ap.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"CSV → {args.csv}")
    df = load_publications(args.csv)
    print(f"  pubs hors RT = {len(df):,}")

    ngrams, ng_meta = extract_ngrams(df)
    print(f"  n-grammes : {ng_meta}")

    n_before = len(df)
    df, ngrams = collapse_identical_ngram_sets(df, ngrams)
    print(f"  après collapse ensembles identiques : {len(df):,} nœuds (était {n_before:,})")

    edges = build_edges(ngrams, min_shared=args.min_shared, max_df=args.max_df)
    print(f"  arêtes (≥{args.min_shared} ngrammes, max_df={args.max_df}) = {len(edges):,}")

    times = list(df["created_at"])
    parent, level, edge_meta = build_filiation_forest(len(df), edges, times)
    print(f"  nœuds dans graphe = {len(parent):,} · racines = {sum(1 for v in parent.values() if v is None):,}")

    trees = collect_trees(
        parent,
        level,
        edge_meta,
        df,
        ngrams,
        min_tree=args.min_tree,
        max_trees=args.max_trees,
        max_nodes=args.max_nodes,
    )
    for tr in trees:
        layout_tree(tr)
    print(f"  cascades exportées = {len(trees)} (min_tree={args.min_tree})")

    payload = {
        "meta": {
            "corpus": "HMPV",
            "window": "HMPV FULL 01/01–30/06/2025",
            "source_csv": str(args.csv.name),
            "scope": "original+reply+quote (hors retweet)",
            "ngram_method": "candidats n=2..4, PMI≥0.8, longest-match, lemmatisés",
            "min_shared": args.min_shared,
            "max_df": args.max_df,
            "min_tree": args.min_tree,
            "n_pubs_hors_rt": int(n_before),
            "n_nodes_collapsed": int(len(df)),
            "n_edges": int(len(edges)),
            "n_nodes_graph": int(len(parent)),
            "n_trees_exported": int(len(trees)),
            "collapse": "ensembles de n-grammes identiques fusionnés (copies)",
            **ng_meta,
            "built_at": pd.Timestamp.now("UTC").isoformat(),
            "elapsed_s": round(time.time() - t0, 1),
        },
        "trees": trees,
    }

    out_json = args.out / "payload_ngram_filiation.json"
    out_html = args.out / "ngram_filiation.html"
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    write_html(payload, out_html, args.min_shared)

    # index léger
    index = {
        "meta": payload["meta"],
        "trees": [
            {
                "id": t["id"],
                "n_nodes": t["n_nodes"],
                "max_level": t["max_level"],
                "root": t["root"],
                "root_user": t["root_user"],
                "root_text": t["root_text"],
            }
            for t in trees
        ],
    }
    (args.out / "index_ngram_filiation.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mb = out_html.stat().st_size / (1024 * 1024)
    print(f"OK → {out_html} ({mb:.2f} Mo) · {out_json.name}")
    print(f"Durée {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
