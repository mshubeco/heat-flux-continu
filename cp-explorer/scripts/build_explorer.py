# -*- coding: utf-8 -*-
"""HEAT · explorateur CP multi-tenseurs (canicule).

Décompositions simples user×objet×minute (+ couplages) pour comparer quels tenseurs
font ressortir des régimes de diffusion. N-grammes lemmatisés multilingues (simplemma).
Granularité temporelle : bin = 1 minute.

Usage :
  py -3.13 build_explorer.py --phase P1
  start index.html
"""
from __future__ import annotations

import argparse
import json
import math
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import simplemma
except ImportError as e:  # pragma: no cover
    raise SystemExit("Installer simplemma : py -3.13 -m pip install simplemma") from e

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = Path(
    r"C:\Users\th8992be\Desktop\HEAT\01_corpus\exports_github"
    r"\canicule_20260520_20260629\corpus_canicule_20260520_20260629.csv"
)
TWITTER_EPOCH_MS = 1_288_834_974_657
BIO_CACHE = ROOT / "cache" / "user_bios.json"
DEFAULT_BIO_CSV = Path(
    r"C:\Users\th8992be\Desktop\cross_corpus\outputs\intersection_4"
    r"\utilisateurs_intersection_bio_x.csv"
)
COHORT239_BIOS = Path(
    r"C:\Users\th8992be\Desktop\HEAT\03_analyses\pipelines\cib_p3\outputs\cohort239_bios.csv"
)

PHASES = {
    "P1": ("2026-05-20", "2026-06-08", "P1 20/05–07/06/2026"),
    "P2": ("2026-06-08", "2026-06-24", "P2 08/06–23/06/2026"),
    "P1P2": ("2026-05-20", "2026-06-24", "P1+P2 20/05–23/06/2026"),
}

MODALITY_META = {
    "uut": {
        "label": "RT · émetteur × cible × minute",
        "blurb": "Régimes d’amplification relationnelle (qui retweete qui, à quelle minute).",
        "mid_label": "Cibles RT",
        "coupled": False,
    },
    "uurlt": {
        "label": "URL · émetteur × url × minute",
        "blurb": "Circulation d’objets URL (qui partage quoi, à quelle minute).",
        "mid_label": "URLs",
        "coupled": False,
    },
    "uudom": {
        "label": "Domaine · émetteur × domaine × minute",
        "blurb": "Même logique URL, agrégée au domaine (moins de bruit viral).",
        "mid_label": "Domaines",
        "coupled": False,
    },
    "uht": {
        "label": "Hashtag · émetteur × tag × minute",
        "blurb": "Cadres lexicaux d’amplification (tags × temps minute).",
        "mid_label": "Hashtags",
        "coupled": False,
    },
    "ubigram": {
        "label": "N-grammes · émetteur × n-gramme × minute",
        "blurb": "MID = n-grammes candidats lemmatisés (n=2..4, PMI + longest-match). Clé « ubigram » conservée.",
        "mid_label": "N-grammes",
        "coupled": False,
    },
    "utrigram": {
        "label": "Trigramme · émetteur × trigramme × minute",
        "blurb": "Tenseur lexical : MID = trigramme lemmatisé (3 mots, fenêtre glissante). Clé technique « utrigram ».",
        "mid_label": "Trigrammes",
        "coupled": False,
    },
    "ubigram_orig": {
        "label": "N-grammes hors RT · émetteur × n-gramme × temps",
        "blurb": "N-grammes candidats (n=2..4, PMI + longest-match) sur publications propres seulement (hors RT).",
        "mid_label": "N-grammes (hors RT)",
        "coupled": False,
    },
    "utrigram_orig": {
        "label": "Trigramme hors RT · émetteur × trigramme × temps",
        "blurb": "Trigrammes lemmatisés sur publications propres seulement (hors retweet).",
        "mid_label": "Trigrammes (hors RT)",
        "coupled": False,
    },
    "uurlt_orig": {
        "label": "URL hors RT · émetteur × url × temps",
        "blurb": "URLs partagées en publication propre seulement (hors retweet).",
        "mid_label": "URLs (hors RT)",
        "coupled": False,
    },
    "uudom_orig": {
        "label": "Domaine hors RT · émetteur × domaine × temps",
        "blurb": "Domaines d’URL en publication propre seulement (hors retweet).",
        "mid_label": "Domaines (hors RT)",
        "coupled": False,
    },
    "uht_orig": {
        "label": "Hashtag hors RT · émetteur × tag × temps",
        "blurb": "Hashtags en publication propre seulement (hors retweet).",
        "mid_label": "Hashtags (hors RT)",
        "coupled": False,
    },
    "ucort": {
        "label": "Co-RT · émetteur × publication cible co-RT × temps",
        "blurb": "Qui retweete la même publication (MID = tweet original). Signal de co-retweet.",
        "mid_label": "Publication cible co-RT",
        "coupled": False,
    },
    "coupled_uut_uurlt": {
        "label": "Couplé RT × URL",
        "blurb": "Deux tenseurs (URL et RT) avec facteur USER partagé — pas deux types dans le même MID.",
        "mid_label": "URLs (tenseur 1)",
        "mid2_label": "Cibles RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_uut_ubigram": {
        "label": "Couplé RT × n-grammes",
        "blurb": "Deux tenseurs (n-grammes candidats et RT) avec facteur USER partagé.",
        "mid_label": "N-grammes (tenseur 1)",
        "mid2_label": "Cibles RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_cort_ubigram_orig": {
        "label": "B · Couplé co-RT × n-grammes hors RT",
        "blurb": "N-gramme × co-RT — USER partagé : production lexicale propre ∥ co-retweet des mêmes status.",
        "mid_label": "N-grammes hors RT (tenseur 1)",
        "mid2_label": "Publication cible co-RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_cort_utrigram_orig": {
        "label": "Couplé co-RT × trigramme hors RT",
        "blurb": "USER partagé : trigrammes hors RT ∥ co-retweet des mêmes status.",
        "mid_label": "Trigrammes hors RT (tenseur 1)",
        "mid2_label": "Publication cible co-RT (tenseur 2)",
        "coupled": True,
    },
    "ucd": {
        "label": "Création compte · user × jour création × temps",
        "blurb": "Jour de création (snowflake user_id, YYYY-MM-DD) × activité.",
        "mid_label": "Jours de création",
        "coupled": False,
    },
    "ucdrt": {
        "label": "Création × RT · émetteur × jour création × cible RT",
        "blurb": "Tenseur 3 modes sans bin d’activité : cohortes de création croisées aux cibles RT.",
        "mid_label": "Jours de création",
        "coupled": False,
    },
    "ucdng": {
        "label": "Création × n-grammes · émetteur × jour création × n-gramme",
        "blurb": "Tenseur 3 modes sans bin d’activité : cohortes de création croisées aux n-grammes lemmatisés.",
        "mid_label": "Jours de création",
        "coupled": False,
    },
    "ubio": {
        "label": "Bio · user × n-gramme bio × temps",
        "blurb": "N-grammes candidats de bio (n=2..4, PMI + longest-match), pondérés par l’activité.",
        "mid_label": "N-grammes bio",
        "coupled": False,
    },
    "ubio_ht": {
        "label": "Bio · user × hashtag bio × temps",
        "blurb": "Hashtags présents dans la bio, pondérés par l’activité.",
        "mid_label": "Hashtags bio",
        "coupled": False,
    },
    "uname": {
        "label": "Display name · user × token nom × temps",
        "blurb": "Tokens du user_name (toujours dans le CSV) — proxy lexical de profil si bio absente.",
        "mid_label": "Tokens display name",
        "coupled": False,
    },
    "umed_orig": {
        "label": "Média hors RT · user × type média × temps",
        "blurb": "Publications propres avec média (photo / video / gif) — hors retweet.",
        "mid_label": "Types média",
        "coupled": False,
    },
    "umedurl_orig": {
        "label": "Média hors RT · user × URL média × temps",
        "blurb": "URLs de pièces jointes médias sur tweets propres (hors RT).",
        "mid_label": "URLs média",
        "coupled": False,
    },
    "coupled_ucd_uut": {
        "label": "Couplé création × RT-cible",
        "blurb": "USER partagé : jours de création ∥ amplification relationnelle.",
        "mid_label": "Jours de création (tenseur 1)",
        "mid2_label": "Cibles RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_ucd_ucort": {
        "label": "Couplé création × co-RT",
        "blurb": "USER partagé : jours de création ∥ co-retweet des mêmes status.",
        "mid_label": "Jours de création (tenseur 1)",
        "mid2_label": "Publication cible co-RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_ubio_ucort": {
        "label": "Couplé bio × co-RT",
        "blurb": "USER partagé : n-grammes bio candidats ∥ co-retweet.",
        "mid_label": "N-grammes bio (tenseur 1)",
        "mid2_label": "Publication cible co-RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_umed_ucort": {
        "label": "Couplé média hors RT × co-RT",
        "blurb": "USER partagé : usage de médias en production propre ∥ co-retweet.",
        "mid_label": "Types média (tenseur 1)",
        "mid2_label": "Publication cible co-RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_umed_ubigram_orig": {
        "label": "D · Couplé média hors RT × n-grammes hors RT",
        "blurb": "Comportement × n-gramme — USER partagé : médias propres ∥ n-grammes candidats hors RT.",
        "mid_label": "Types média (tenseur 1)",
        "mid2_label": "N-grammes hors RT (tenseur 2)",
        "coupled": True,
    },
    # Hiérarchie expérimentations A–G (base USER × TEMPS ; couplage USER)
    "coupled_ubigram_uurlt": {
        "label": "A · Couplé n-grammes × URL",
        "blurb": "USER partagé : n-grammes candidats ∥ URLs (Coord. forte · Manip. forte).",
        "mid_label": "N-grammes (tenseur 1)",
        "mid2_label": "URLs (tenseur 2)",
        "coupled": True,
    },
    "coupled_ucort_uurlt_orig": {
        "label": "C · Couplé co-RT × URL hors RT",
        "blurb": "Bras du triptyque Co-RT × co-URL × co-texte (avec B et A hors RT). USER partagé.",
        "mid_label": "Publication cible co-RT (tenseur 1)",
        "mid2_label": "URLs hors RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_ucd_ubio": {
        "label": "E · Couplé création × bio",
        "blurb": "USER partagé : jours de création ∥ n-grammes bio (Activité × profil).",
        "mid_label": "Jours de création (tenseur 1)",
        "mid2_label": "N-grammes bio (tenseur 2)",
        "coupled": True,
    },
    "coupled_uurlt_ucort": {
        "label": "F · Couplé URL × co-RT",
        "blurb": "USER partagé : URLs ∥ co-retweet (Ressource × synchronisation).",
        "mid_label": "URLs (tenseur 1)",
        "mid2_label": "Publication cible co-RT (tenseur 2)",
        "coupled": True,
    },
    "coupled_uht_umed_orig": {
        "label": "G · Couplé hashtag × type média",
        "blurb": "USER partagé : hashtags ∥ types média hors RT (Frame × type de tweet).",
        "mid_label": "Hashtags (tenseur 1)",
        "mid2_label": "Types média hors RT (tenseur 2)",
        "coupled": True,
    },
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RT_PREFIX_RE = re.compile(r"^\s*RT\s+@\w+:\s*", re.I)
_TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôùûüçœæ0-9]{3,}", re.I)
_RT_AT = re.compile(r"^\s*RT\s+@([A-Za-z0-9_]{1,50})", re.I)

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "are", "was",
    "were", "been", "being", "their", "there", "what", "when", "where", "which", "will",
    "would", "could", "should", "about", "into", "just", "like", "than", "then", "them",
    "they", "your", "you", "not", "but", "all", "can", "out", "our", "how", "who",
    "les", "des", "une", "dans", "pour", "que", "qui", "est", "pas", "plus", "avec",
    "sur", "par", "sont", "aux", "ces", "cet", "cette", "dont", "comme", "tout", "tous",
    "fait", "faire", "être", "avoir", "http", "https", "www", "com", "amp", "via",
}

# simplemma language codes acceptés (sinon fallback fr)
_LANG_OK = {
    "fr", "en", "es", "it", "de", "pt", "nl", "ca", "ro", "sv", "da", "fi", "hu", "pl",
    "cs", "sk", "sl", "hr", "bg", "el", "ru", "uk", "tr", "id", "ga", "cy", "la",
}

NOISE_TAGS = {
    "canicule", "chaleur", "heatwave", "heat", "meteo", "météo", "climat", "climate",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _id_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def time_bin_label(ts: pd.Timestamp, time_bin: str) -> str:
    """Bin temporel UTC : day | hour | minute."""
    if time_bin == "day":
        return ts.strftime("%Y-%m-%d")
    if time_bin == "hour":
        return ts.strftime("%Y-%m-%d %H")
    if time_bin == "minute":
        return ts.strftime("%Y-%m-%d %H:%M")
    raise ValueError(f"time_bin inconnu: {time_bin}")


def minute_label(ts: pd.Timestamp) -> str:
    return time_bin_label(ts, "minute")


def day_label(ts: pd.Timestamp) -> str:
    return time_bin_label(ts, "day")


def time_bin_fr(time_bin: str) -> str:
    return {"day": "jour", "hour": "heure", "minute": "minute"}.get(time_bin, time_bin)


def modality_meta_for(time_bin: str) -> dict:
    """Libellés UI selon granularité temporelle."""
    t = time_bin_fr(time_bin)
    base = {
        "uut": (
            f"RT · émetteur × cible × {t}",
            f"Régimes d’amplification relationnelle (qui retweete qui, quand — bin {t}).",
            "Cibles RT",
        ),
        "uurlt": (
            f"URL · émetteur × url × {t}",
            f"Circulation d’objets URL (bin {t}).",
            "URLs",
        ),
        "uudom": (
            f"Domaine · émetteur × domaine × {t}",
            f"URL agrégée au domaine (bin {t}).",
            "Domaines",
        ),
        "uht": (
            f"Hashtag · émetteur × tag × {t}",
            f"Cadres lexicaux d’amplification (bin {t}).",
            "Hashtags",
        ),
        "ubigram": (
            f"N-grammes · émetteur × n-gramme × {t}",
            f"MID = n-grammes candidats (n=2..4, PMI≥0.8, longest-match). Bin {t}.",
            "N-grammes",
        ),
        "utrigram": (
            f"Trigramme · émetteur × trigramme × {t}",
            f"MID = trigramme lemmatisé (3 mots, fenêtre glissante). Bin {t}.",
            "Trigrammes",
        ),
        "ubigram_orig": (
            f"N-grammes hors RT · émetteur × n-gramme × {t}",
            f"N-grammes candidats (n=2..4, PMI + longest-match) sur tweets propres. Bin {t}.",
            "N-grammes (hors RT)",
        ),
        "utrigram_orig": (
            f"Trigramme hors RT · émetteur × trigramme × {t}",
            f"Trigrammes lemmatisés sur tweets propres seulement (hors retweet). Bin {t}.",
            "Trigrammes (hors RT)",
        ),
        "uurlt_orig": (
            f"URL hors RT · émetteur × url × {t}",
            f"URLs en publication propre seulement (hors RT). Bin {t}.",
            "URLs (hors RT)",
        ),
        "uudom_orig": (
            f"Domaine hors RT · émetteur × domaine × {t}",
            f"Domaines d’URL en publication propre (hors RT). Bin {t}.",
            "Domaines (hors RT)",
        ),
        "uht_orig": (
            f"Hashtag hors RT · émetteur × tag × {t}",
            f"Hashtags en publication propre seulement (hors RT). Bin {t}.",
            "Hashtags (hors RT)",
        ),
        "ucort": (
            f"Co-RT · émetteur × publication cible co-RT × {t}",
            f"Qui retweete la même publication (co-retweet). Bin {t}.",
            "Publication cible co-RT",
        ),
        "coupled_uut_uurlt": (
            "Couplé RT × URL",
            f"Deux tenseurs (URL ∥ RT cible), USER partagé, bin {t}.",
            "URLs (tenseur 1)",
        ),
        "coupled_uut_ubigram": (
            "Couplé RT × n-grammes",
            f"Deux tenseurs (n-grammes candidats ∥ RT cible), USER partagé, bin {t}.",
            "N-grammes (tenseur 1)",
        ),
        "coupled_cort_ubigram_orig": (
            "B · Couplé co-RT × n-grammes hors RT",
            f"N-gramme × co-RT — production lexicale propre ∥ co-retweet (USER partagé). Bin {t}.",
            "N-grammes hors RT (tenseur 1)",
        ),
        "coupled_cort_utrigram_orig": (
            "Couplé co-RT × trigramme hors RT",
            f"Trigrammes hors RT ∥ co-retweet (USER partagé). Bin {t}.",
            "Trigrammes hors RT (tenseur 1)",
        ),
        "ucd": (
            f"Création compte · user × jour création × {t}",
            f"Snowflake → jour de création (YYYY-MM-DD) × activité (bin {t}).",
            "Jours de création",
        ),
        "ucdrt": (
            "Création × RT · émetteur × jour création × cible RT",
            "Cohortes de création (snowflake YYYY-MM-DD) croisées aux cibles RT — pas de bin d’activité.",
            "Jours de création",
        ),
        "ucdng": (
            "Création × n-grammes · émetteur × jour création × n-gramme",
            "Cohortes de création croisées aux n-grammes candidats lemmatisés — pas de bin d’activité.",
            "Jours de création",
        ),
        "ubio": (
            f"Bio · user × n-gramme bio × {t}",
            f"N-grammes bio candidats (n=2..4, PMI≥0.8, longest-match) × activité (bin {t}).",
            "N-grammes bio",
        ),
        "ubio_ht": (
            f"Bio · user × hashtag bio × {t}",
            f"Hashtags de bio × activité (bin {t}).",
            "Hashtags bio",
        ),
        "uname": (
            f"Display name · user × token nom × {t}",
            f"Tokens du user_name × activité (bin {t}).",
            "Tokens display name",
        ),
        "umed_orig": (
            f"Média hors RT · user × type média × {t}",
            f"photo / video / gif sur tweets propres seulement (bin {t}).",
            "Types média",
        ),
        "umedurl_orig": (
            f"Média hors RT · user × URL média × {t}",
            f"URLs de médias attachés, hors retweet (bin {t}).",
            "URLs média",
        ),
        "coupled_ucd_uut": (
            "Couplé création × RT-cible",
            f"Jours de création ∥ cibles RT (USER partagé). Bin {t}.",
            "Jours de création (tenseur 1)",
        ),
        "coupled_ucd_ucort": (
            "Couplé création × co-RT",
            f"Jours de création ∥ co-RT (USER partagé). Bin {t}.",
            "Jours de création (tenseur 1)",
        ),
        "coupled_ubio_ucort": (
            "Couplé bio × co-RT",
            f"N-grammes bio candidats ∥ co-RT (USER partagé). Bin {t}.",
            "N-grammes bio (tenseur 1)",
        ),
        "coupled_umed_ucort": (
            "Couplé média hors RT × co-RT",
            f"Types média propres ∥ co-RT (USER partagé). Bin {t}.",
            "Types média (tenseur 1)",
        ),
        "coupled_umed_ubigram_orig": (
            "D · Couplé média hors RT × n-grammes hors RT",
            f"Comportement × n-gramme — types média propres ∥ n-grammes hors RT. Bin {t}.",
            "Types média (tenseur 1)",
        ),
        "coupled_ubigram_uurlt": (
            "A · Couplé n-grammes × URL",
            f"N-grammes candidats ∥ URLs (USER partagé). Bin {t}.",
            "N-grammes (tenseur 1)",
        ),
        "coupled_ucort_uurlt_orig": (
            "C · Couplé co-RT × URL hors RT",
            f"Bras URL du triptyque Co-RT × co-URL × co-texte. Bin {t}.",
            "Publication cible co-RT (tenseur 1)",
        ),
        "coupled_ucd_ubio": (
            "E · Couplé création × bio",
            f"Jours de création ∥ n-grammes bio (USER partagé). Bin {t}.",
            "Jours de création (tenseur 1)",
        ),
        "coupled_uurlt_ucort": (
            "F · Couplé URL × co-RT",
            f"URLs ∥ co-retweet (Ressource × synchronisation). Bin {t}.",
            "URLs (tenseur 1)",
        ),
        "coupled_uht_umed_orig": (
            "G · Couplé hashtag × type média",
            f"Hashtags ∥ types média hors RT (Frame × type de tweet). Bin {t}.",
            "Hashtags (tenseur 1)",
        ),
    }
    modes = {
        "uut": f"émetteur × cible RT × {t}",
        "uurlt": f"émetteur × URL × {t}",
        "uudom": f"émetteur × domaine × {t}",
        "uht": f"émetteur × hashtag × {t}",
        "ubigram": f"émetteur × n-gramme × {t}",
        "utrigram": f"émetteur × trigramme × {t}",
        "ubigram_orig": f"émetteur × n-gramme (hors RT) × {t}",
        "utrigram_orig": f"émetteur × trigramme (hors RT) × {t}",
        "uurlt_orig": f"émetteur × URL (hors RT) × {t}",
        "uudom_orig": f"émetteur × domaine (hors RT) × {t}",
        "uht_orig": f"émetteur × hashtag (hors RT) × {t}",
        "ucort": f"émetteur × publication cible co-RT × {t}",
        "ucd": f"émetteur × jour création × {t}",
        "ucdrt": "émetteur × jour création × cible RT",
        "ucdng": "émetteur × jour création × n-gramme",
        "ubio": f"émetteur × n-gramme bio × {t}",
        "ubio_ht": f"émetteur × hashtag bio × {t}",
        "uname": f"émetteur × token display name × {t}",
        "umed_orig": f"émetteur × type média (hors RT) × {t}",
        "umedurl_orig": f"émetteur × URL média (hors RT) × {t}",
        "coupled_uut_uurlt": f"(émetteur × URL × {t})  ∥  (émetteur × cible RT × {t})",
        "coupled_uut_ubigram": f"(émetteur × n-gramme × {t})  ∥  (émetteur × cible RT × {t})",
        "coupled_cort_ubigram_orig": f"(émetteur × n-gramme hors RT × {t})  ∥  (émetteur × publication cible co-RT × {t})",
        "coupled_cort_utrigram_orig": f"(émetteur × trigramme hors RT × {t})  ∥  (émetteur × publication cible co-RT × {t})",
        "coupled_ucd_uut": f"(émetteur × jour création × {t})  ∥  (émetteur × cible RT × {t})",
        "coupled_ucd_ucort": f"(émetteur × jour création × {t})  ∥  (émetteur × publication cible co-RT × {t})",
        "coupled_ubio_ucort": f"(émetteur × n-gramme bio × {t})  ∥  (émetteur × publication cible co-RT × {t})",
        "coupled_umed_ucort": f"(émetteur × type média hors RT × {t})  ∥  (émetteur × publication cible co-RT × {t})",
        "coupled_umed_ubigram_orig": f"(émetteur × type média hors RT × {t})  ∥  (émetteur × n-gramme hors RT × {t})",
        "coupled_ubigram_uurlt": f"(émetteur × n-gramme × {t})  ∥  (émetteur × URL × {t})",
        "coupled_ucort_uurlt_orig": f"(émetteur × publication cible co-RT × {t})  ∥  (émetteur × URL (hors RT) × {t})",
        "coupled_ucd_ubio": f"(émetteur × jour création × {t})  ∥  (émetteur × n-gramme bio × {t})",
        "coupled_uurlt_ucort": f"(émetteur × URL × {t})  ∥  (émetteur × publication cible co-RT × {t})",
        "coupled_uht_umed_orig": f"(émetteur × hashtag × {t})  ∥  (émetteur × type média (hors RT) × {t})",
    }
    mid2 = {
        "coupled_cort_ubigram_orig": "Publication cible co-RT (tenseur 2)",
        "coupled_cort_utrigram_orig": "Publication cible co-RT (tenseur 2)",
        "coupled_ucd_ucort": "Publication cible co-RT (tenseur 2)",
        "coupled_ubio_ucort": "Publication cible co-RT (tenseur 2)",
        "coupled_umed_ucort": "Publication cible co-RT (tenseur 2)",
        "coupled_uurlt_ucort": "Publication cible co-RT (tenseur 2)",
        "coupled_ucort_uurlt_orig": "URLs hors RT (tenseur 2)",
        "coupled_umed_ubigram_orig": "N-grammes hors RT (tenseur 2)",
        "coupled_ubigram_uurlt": "URLs (tenseur 2)",
        "coupled_ucd_uut": "Cibles RT (tenseur 2)",
        "coupled_ucd_ubio": "N-grammes bio (tenseur 2)",
        "coupled_uht_umed_orig": "Types média hors RT (tenseur 2)",
        "coupled_uut_uurlt": "Cibles RT (tenseur 2)",
        "coupled_uut_ubigram": "Cibles RT (tenseur 2)",
    }
    out = {}
    for k, (lab, blurb, mid) in base.items():
        out[k] = {
            "label": lab,
            "blurb": blurb,
            "mid_label": mid,
            "modes": modes.get(k, lab),
            "coupled": k.startswith("coupled"),
        }
        if k.startswith("coupled"):
            out[k]["mid2_label"] = mid2.get(k, "MID₂ (tenseur 2)")
    return out


def normalize_url(u: str) -> str | None:
    u = (u or "").strip().rstrip(").,;]}>'\"")
    if len(u) < 8:
        return None
    try:
        p = urlparse(u if "://" in u else f"http://{u}")
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host or host in {"t.co"}:
            return None
        path = (p.path or "").rstrip("/")
        return f"{host}{path}"[:240]
    except Exception:
        return None


def url_domain(norm: str) -> str | None:
    if not norm:
        return None
    host = norm.split("/", 1)[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def split_field(val) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    return [p.strip() for p in re.split(r"[,;|]", s) if p.strip()]


def parse_hashtags(val) -> list[str]:
    out = []
    for p in split_field(val):
        t = p.lstrip("#").strip().lower()
        if len(t) >= 2:
            out.append(t)
    return out


def parse_media_types(val) -> list[str]:
    """Types média (photo / video / animated_gif…), champs multi séparés , ; |."""
    out, seen = [], set()
    for p in split_field(val):
        t = p.strip().lower().replace(" ", "_")
        if not t or t in {"nan", "none", "null"}:
            continue
        if t in {"image", "img", "pic"}:
            t = "photo"
        if t in {"gif", "animatedgif"}:
            t = "animated_gif"
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_media_urls(val) -> list[str]:
    out, seen = [], set()
    for p in split_field(val):
        norm = normalize_url(p)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def map_lang(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "fr"
    s = str(raw).strip().lower()
    if s in {"und", "unk", "qam", "qme", "qht", "zxx", ""}:
        return "fr"
    if s in _LANG_OK:
        return s
    return "fr"


def clean_tweet_text(raw: str) -> str:
    s = _RT_PREFIX_RE.sub(" ", raw or "")
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(r" \1 ", s)
    return s


def lemmatize_tokens(raw: str, lang: str) -> list[str]:
    s = clean_tweet_text(raw).lower()
    out: list[str] = []
    for t in _TOKEN_RE.findall(s):
        t = t.lower()
        if t in _STOP or t.isdigit():
            continue
        try:
            lem = simplemma.lemmatize(t, lang=lang).lower().strip()
        except Exception:
            lem = t
        if len(lem) < 3 or lem in _STOP or lem.isdigit():
            continue
        out.append(lem)
    return out


def make_ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    out, seen = [], set()
    for i in range(len(tokens) - n + 1):
        gram = "_".join(tokens[i : i + n])
        if gram in seen:
            continue
        seen.add(gram)
        out.append(gram)
    return out


def build_collocation_stats(
    token_lists: list[list[str]],
    *,
    n_max: int = 4,
) -> dict:
    """Comptes unigrammes / bigrammes adjacents + df des n-grammes (n=2..n_max)."""
    uni: Counter[str] = Counter()
    bi: Counter[str] = Counter()
    df: Counter[str] = Counter()
    n_tok = 0
    for toks in token_lists:
        if not toks:
            continue
        n_tok += len(toks)
        for t in toks:
            uni[t] += 1
        for i in range(len(toks) - 1):
            bi[f"{toks[i]}_{toks[i + 1]}"] += 1
        seen: set[str] = set()
        for n in range(2, n_max + 1):
            if len(toks) < n:
                break
            for i in range(len(toks) - n + 1):
                g = "_".join(toks[i : i + n])
                if g not in seen:
                    seen.add(g)
                    df[g] += 1
    return {"uni": uni, "bi": bi, "df": df, "n_tok": max(n_tok, 1), "n_max": n_max}


def _adjacent_pmi(a: str, b: str, stats: dict) -> float:
    uni, bi, N = stats["uni"], stats["bi"], stats["n_tok"]
    ca, cb = uni.get(a, 0), uni.get(b, 0)
    cab = bi.get(f"{a}_{b}", 0)
    if ca <= 0 or cb <= 0 or cab <= 0:
        return float("-inf")
    # PMI = log ( P(a,b) / (P(a)P(b)) ) avec P estimés en fréquences
    return float(math.log((cab * N) / (ca * cb)))


def ngram_collocation_score(toks: list[str], start: int, n: int, stats: dict) -> float:
    """Score = PMI min des bigrammes adjacents (conservateur pour n>2)."""
    if n < 2 or start + n > len(toks):
        return float("-inf")
    if n == 2:
        return _adjacent_pmi(toks[start], toks[start + 1], stats)
    scores = [_adjacent_pmi(toks[j], toks[j + 1], stats) for j in range(start, start + n - 1)]
    return float(min(scores)) if scores else float("-inf")


def select_candidate_ngrams(
    toks: list[str],
    stats: dict,
    *,
    n_max: int = 4,
    min_pmi: float = 0.8,
    min_df: int = 2,
) -> list[str]:
    """
    N-grammes candidats (n=2..n_max) : PMI / df, puis longest-match non chevauchant.
    Fallback : tous les bigrammes si aucun candidat ne passe.
    """
    if len(toks) < 2:
        return []
    n_max = min(n_max, int(stats.get("n_max") or n_max))
    df_map: Counter = stats["df"]
    cands: list[tuple[int, float, int, int, str]] = []
    for n in range(2, n_max + 1):
        for i in range(len(toks) - n + 1):
            gram = "_".join(toks[i : i + n])
            if df_map.get(gram, 0) < min_df:
                continue
            score = ngram_collocation_score(toks, i, n, stats)
            if score < min_pmi:
                continue
            cands.append((n, score, i, i + n, gram))
    if not cands:
        return make_ngrams(toks, 2)
    # plus long d’abord, puis PMI décroissant
    cands.sort(key=lambda x: (-x[0], -x[1], x[2]))
    covered = [False] * len(toks)
    selected: list[str] = []
    seen: set[str] = set()
    for _n, _sc, s, e, gram in cands:
        if any(covered[s:e]):
            continue
        for j in range(s, e):
            covered[j] = True
        if gram not in seen:
            seen.add(gram)
            selected.append(gram)
    return selected or make_ngrams(toks, 2)


def snowflake_created_at(uid: str, *, min_year: int = 2018) -> datetime | None:
    """Date de création via snowflake Twitter (IDs ≥50 bits uniquement).

    `min_year=2018` : filtre tenseurs ucd* (évite IDs douteux).
    `min_year=2006` : affichage table acteurs (comptes Twitter historiques OK).
    """
    try:
        i = int(str(uid).strip())
    except (TypeError, ValueError):
        return None
    if i.bit_length() < 50:
        return None
    ms = (i >> 22) + TWITTER_EPOCH_MS
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    if dt < datetime(min_year, 1, 1, tzinfo=timezone.utc) or dt > datetime.now(timezone.utc):
        return None
    return dt


def creation_bin(dt: datetime | None) -> str:
    """Jour de création (YYYY-MM-DD) via snowflake — pas de regroupement annuel."""
    if dt is None:
        return "unknown"
    return dt.strftime("%Y-%m-%d")


def _save_bio_cache(cache: dict[str, dict]) -> None:
    BIO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BIO_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def load_bios_for_pool(
    pool: pd.DataFrame,
    *,
    bio_csv: Path | None = DEFAULT_BIO_CSV,
    scrape: bool = False,
    max_scrape: int = 800,
    delay_s: float = 0.45,
) -> dict[str, str]:
    """Bios indexées par user_id (sources caches + CSV + scrape fxtwitter optionnel)."""
    sn_to_uid = {
        str(s).lstrip("@").lower(): str(u)
        for u, s in zip(pool["user_id"], pool["screen_name"])
        if pd.notna(s) and str(s).strip()
    }
    cache: dict[str, dict] = {}
    if BIO_CACHE.is_file():
        try:
            cache = json.loads(BIO_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}

    def _put(sn: str, bio: str, source: str) -> None:
        sn = (sn or "").lstrip("@").lower().strip()
        bio = (bio or "").strip()
        if not sn or not bio:
            return
        prev = cache.get(sn) or {}
        if prev.get("bio") and len(prev["bio"]) >= len(bio):
            return
        cache[sn] = {"handle": sn, "bio": bio, "status": "ok", "source": source}

    if bio_csv and bio_csv.is_file():
        log(f"Bios — lecture {bio_csv.name}…")
        try:
            df = pd.read_csv(bio_csv, dtype=str, low_memory=False)
            cols = set(df.columns)
            bio_col = "x_bio" if "x_bio" in cols else ("bio" if "bio" in cols else None)
            sn_col = "screen_name" if "screen_name" in cols else None
            st_col = "x_status" if "x_status" in cols else ("bio_status" if "bio_status" in cols else None)
            if bio_col and sn_col:
                for i in range(len(df)):
                    st = str(df.iloc[i][st_col]).lower() if st_col else "ok"
                    if st_col and st not in {"ok", "nan", ""}:
                        continue
                    bio = df.iloc[i][bio_col]
                    sn = df.iloc[i][sn_col]
                    if pd.notna(bio) and str(bio).strip() and pd.notna(sn):
                        _put(str(sn), str(bio), bio_csv.name)
        except Exception as e:
            log(f"  bio CSV skip: {e}")

    if COHORT239_BIOS.is_file():
        try:
            df = pd.read_csv(COHORT239_BIOS, dtype=str)
            for sn, bio, st in zip(df.get("screen_name", []), df.get("bio", []), df.get("bio_status", [])):
                if str(st).lower() == "ok" and pd.notna(bio) and str(bio).strip():
                    _put(str(sn), str(bio), "cohort239")
        except Exception as e:
            log(f"  cohort239 bios skip: {e}")

    # scrape manquants
    missing = [
        sn for sn, uid in sn_to_uid.items()
        if not ((cache.get(sn) or {}).get("bio") or "").strip()
    ]
    if scrape and missing:
        to_fetch = missing[: max(0, int(max_scrape))]
        log(f"Bios — scrape fxtwitter {len(to_fetch)}/{len(missing)} manquants…")
        ctx = ssl.create_default_context()
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"

        def _fetch_bio(handle: str) -> tuple[str, str]:
            """Retourne (bio, status) ; status = ok | empty | not_found | error."""
            last_status = "error"
            for attempt in range(3):
                for tmpl in ("https://api.fxtwitter.com/{h}", "https://api.vxtwitter.com/{h}"):
                    url = tmpl.format(h=handle)
                    req = urllib.request.Request(
                        url, headers={"User-Agent": ua, "Accept": "application/json"}
                    )
                    try:
                        with urllib.request.urlopen(req, timeout=25, context=ctx) as r:
                            data = json.loads(r.read().decode("utf-8", "replace"))
                        user = data.get("user") or data.get("tweet", {}).get("author") or {}
                        bio = (
                            user.get("description")
                            or user.get("bio")
                            or user.get("desc")
                            or ""
                        ).strip()
                        if bio:
                            return bio, "ok"
                        # profil trouvé mais bio vide
                        if user.get("screen_name") or user.get("id"):
                            last_status = "empty"
                        else:
                            last_status = "error"
                    except urllib.error.HTTPError as e:
                        if e.code in {404, 410}:
                            return "", "not_found"
                        last_status = "error"
                    except Exception:
                        last_status = "error"
                if attempt < 2:
                    time.sleep(delay_s * (1.5 + attempt))
            return "", last_status

        n_ok = n_empty = n_nf = n_err = 0
        for i, sn in enumerate(to_fetch, 1):
            bio, st = _fetch_bio(sn)
            if st == "ok":
                n_ok += 1
                source = "fxtwitter"
            elif st == "not_found":
                n_nf += 1
                source = "fxtwitter_404"
            elif st == "empty":
                n_empty += 1
                source = "fxtwitter_empty"
            else:
                n_err += 1
                source = "fxtwitter_error"
            cache[sn] = {
                "handle": sn,
                "bio": bio,
                "status": st,
                "source": source,
            }
            if i % 40 == 0:
                _save_bio_cache(cache)
                log(f"  scrape {i}/{len(to_fetch)} · ok={n_ok} empty={n_empty} 404={n_nf} err={n_err}")
            time.sleep(delay_s)
        _save_bio_cache(cache)
        log(f"  scrape fini · ok={n_ok} empty={n_empty} 404={n_nf} err={n_err}")
    else:
        _save_bio_cache(cache)

    out: dict[str, str] = {}
    for sn, uid in sn_to_uid.items():
        bio = ((cache.get(sn) or {}).get("bio") or "").strip()
        if bio:
            out[uid] = bio
    log(f"Bios — {len(out):,}/{len(pool):,} comptes du pool couverts")
    return out


def bio_features(bio: str, lang: str = "fr") -> tuple[list[str], list[str]]:
    """Retourne (tokens lemmatisés, hashtags) — sélection n-grammes via stats corpus."""
    tags = [t.lower() for t in _HASHTAG_RE.findall(bio or "")]
    toks = lemmatize_tokens(bio or "", lang)
    return toks, tags


def bio_motifs_for_pool(
    bios: dict[str, str],
    *,
    n_max: int = 4,
    min_pmi: float = 0.8,
    min_df: int = 2,
    lang: str = "fr",
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict]:
    """
    Motifs bio = n-grammes candidats (n=2..n_max), PMI + longest-match.
    Retourne (motifs_par_uid, tags_par_uid, stats).
    """
    bio_toks: dict[str, list[str]] = {}
    bio_tags: dict[str, list[str]] = {}
    for uid, bio in bios.items():
        toks, tags = bio_features(bio, lang=lang)
        if toks:
            bio_toks[uid] = toks
        if tags:
            bio_tags[uid] = tags
    stats = build_collocation_stats(list(bio_toks.values()), n_max=n_max)
    motifs: dict[str, list[str]] = {}
    n_by_len: Counter[int] = Counter()
    for uid, toks in bio_toks.items():
        sel = select_candidate_ngrams(
            toks, stats, n_max=n_max, min_pmi=min_pmi, min_df=min_df
        )
        if sel:
            motifs[uid] = sel
            for g in sel:
                n_by_len[g.count("_") + 1] += 1
    stats["n_by_len"] = dict(n_by_len)
    stats["n_bios_tok"] = len(bio_toks)
    stats["n_bios_motifs"] = len(motifs)
    return motifs, bio_tags, stats


# ---------------------------------------------------------------------------
# Pool + tenseurs
# ---------------------------------------------------------------------------


def load_pool(
    csv_path: Path,
    *,
    t0: str,
    t1: str,
    min_user_tweets: int,
    max_users: int,
) -> pd.DataFrame:
    t0_ts = pd.Timestamp(t0, tz="UTC")
    t1_ts = pd.Timestamp(t1, tz="UTC")
    counts: Counter[str] = Counter()
    sn_map: dict[str, str] = {}
    url_users: set[str] = set()
    usecols = ["user_id", "user_screen_name", "created_at", "tweet_url_url"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk["created_at"] = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce")
        sub = chunk[(chunk["created_at"] >= t0_ts) & (chunk["created_at"] < t1_ts)]
        if sub.empty:
            continue
        for uid, sn, url in zip(sub["user_id"], sub["user_screen_name"], sub["tweet_url_url"]):
            u = _id_str(uid)
            if not u:
                continue
            counts[u] += 1
            if pd.notna(sn):
                sn_map.setdefault(u, str(sn).strip().lstrip("@"))
            if pd.notna(url) and str(url).strip():
                url_users.add(u)
    # pool hybride : volume + présence URL (évite pool 100 % RT)
    eligible = [u for u, n in counts.items() if n >= min_user_tweets]
    by_vol = sorted(eligible, key=lambda u: counts[u], reverse=True)
    n_vol = int(max_users * 0.7)
    picked = by_vol[:n_vol]
    rest = [u for u in by_vol[n_vol:] if u in url_users]
    for u in rest:
        if len(picked) >= max_users:
            break
        picked.append(u)
    if len(picked) < max_users:
        for u in by_vol:
            if u not in set(picked):
                picked.append(u)
            if len(picked) >= max_users:
                break
    rows = [
        {"user_id": u, "screen_name": sn_map.get(u, u), "n_tweets": counts[u], "has_url": int(u in url_users)}
        for u in picked
    ]
    return pd.DataFrame(rows)


def _filter_creation_object_dict(
    raw: dict[tuple[str, str, str], float],
    *,
    min_count: float,
    max_objects: int,
    min_users: int,
    object_name: str,
) -> pd.DataFrame:
    """Tenseur (user, creation_bin, objet) — filtre sur l’objet (mode 3)."""
    users_per: dict[str, set[str]] = defaultdict(set)
    totals: Counter[str] = Counter()
    for (u, _cbin, obj), v in raw.items():
        users_per[obj].add(u)
        totals[obj] += v
    allowed = {g for g, us in users_per.items() if len(us) >= min_users}
    ranked = [g for g, _ in totals.most_common() if g in allowed][:max_objects]
    keep = set(ranked)
    rows = [
        (u, cbin, obj, v)
        for (u, cbin, obj), v in raw.items()
        if obj in keep and v >= min_count
    ]
    return pd.DataFrame(rows, columns=["user_id", "creation_bin", object_name, "val"])


def _filter_ngram_dict(
    raw: dict[tuple[str, str, str], float],
    *,
    min_count: float,
    max_ngrams: int,
    min_users: int,
    mid_name: str,
) -> pd.DataFrame:
    users_per: dict[str, set[str]] = defaultdict(set)
    totals: Counter[str] = Counter()
    for (u, gram, _day), v in raw.items():
        users_per[gram].add(u)
        totals[gram] += v
    allowed = {g for g, us in users_per.items() if len(us) >= min_users}
    if max_ngrams > 0 and allowed:
        allowed = set(sorted(allowed, key=lambda g: totals[g], reverse=True)[:max_ngrams])
    rows = [(a, b, c, v) for (a, b, c), v in raw.items() if v >= min_count and b in allowed]
    return pd.DataFrame(rows, columns=["user_id", mid_name, "tbin", "val"])


def build_all_tensors(
    csv_path: Path,
    pool: pd.DataFrame,
    *,
    t0: str,
    t1: str,
    min_count: float,
    max_urls: int,
    max_domains: int,
    max_hashtags: int,
    max_targets: int,
    max_bigrams: int,
    max_trigrams: int,
    min_users_ngram: int,
    max_cort: int = 2500,
    min_users_cort: int = 2,
    with_ngrams: bool = True,
    time_bin: str = "day",
    bios: dict[str, str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, str]]:
    t0_ts = pd.Timestamp(t0, tz="UTC")
    t1_ts = pd.Timestamp(t1, tz="UTC")
    pool_ids = set(pool["user_id"].astype(str))
    sn_extra = {str(u): str(s) for u, s in zip(pool["user_id"], pool["screen_name"])}
    tcol = "tbin"
    bios = bios or {}
    creation_bins = {uid: creation_bin(snowflake_created_at(uid)) for uid in pool_ids}
    bio_motifs, bio_tags, bio_stats = bio_motifs_for_pool(bios, n_max=4, min_pmi=0.8, min_df=2)
    log(
        f"Profils — création snowflake: "
        f"{sum(1 for b in creation_bins.values() if b != 'unknown'):,}/{len(pool_ids):,} datés · "
        f"bios motifs={bio_stats.get('n_bios_motifs', 0):,}/{bio_stats.get('n_bios_tok', 0):,} · "
        f"n-len={bio_stats.get('n_by_len', {})} · tags={len(bio_tags):,}"
    )

    usecols = [
        "tweet_id",
        "user_id",
        "user_screen_name",
        "created_at",
        "tweet_type",
        "tweet_url_url",
        "hashtag",
        "text",
        "language",
        "retweeted_tweet_id",
        "user_name",
        "media_type",
        "media_url",
    ]

    log("Pass 1 — auteurs de tweets (résolution RT)…")
    tweet_author: dict[str, str] = {}
    for chunk in pd.read_csv(
        csv_path, usecols=["tweet_id", "user_id", "user_screen_name"], chunksize=250_000, low_memory=False
    ):
        for tid, uid, sn in zip(chunk["tweet_id"], chunk["user_id"], chunk["user_screen_name"]):
            t, u = _id_str(tid), _id_str(uid)
            if t and u:
                tweet_author[t] = u
            if u and pd.notna(sn):
                sn_extra.setdefault(u, str(sn).strip().lstrip("@"))

    uut: dict[tuple[str, str, str], float] = defaultdict(float)
    uurlt: dict[tuple[str, str, str], float] = defaultdict(float)
    uudom: dict[tuple[str, str, str], float] = defaultdict(float)
    uht: dict[tuple[str, str, str], float] = defaultdict(float)
    uurlt_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    uudom_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    uht_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    ubigram: dict[tuple[str, str, str], float] = defaultdict(float)
    utrigram: dict[tuple[str, str, str], float] = defaultdict(float)
    ubigram_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    utrigram_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    ucort: dict[tuple[str, str, str], float] = defaultdict(float)
    ucd: dict[tuple[str, str, str], float] = defaultdict(float)
    ucdrt: dict[tuple[str, str, str], float] = defaultdict(float)
    ucdng: dict[tuple[str, str, str], float] = defaultdict(float)
    ubio: dict[tuple[str, str, str], float] = defaultdict(float)
    ubio_ht: dict[tuple[str, str, str], float] = defaultdict(float)
    uname: dict[tuple[str, str, str], float] = defaultdict(float)
    umed_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    umedurl_orig: dict[tuple[str, str, str], float] = defaultdict(float)
    lang_counts: Counter[str] = Counter()
    n_ngram_docs = 0
    n_ngram_orig = 0
    n_media_orig = 0
    # docs lexicaux : (user, tbin, toks, is_orig) — sélection n-grammes après stats PMI
    ngram_docs: list[tuple[str, str, list[str], bool]] = []

    log("Pass 2 — agrégation tenseurs…")
    for chunk_idx, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols, chunksize=200_000, low_memory=False), start=1
    ):
        chunk["created_at"] = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce")
        sub = chunk[(chunk["created_at"] >= t0_ts) & (chunk["created_at"] < t1_ts)]
        if sub.empty:
            continue
        in_pool = sub[sub["user_id"].map(lambda x: _id_str(x) in pool_ids)]
        if in_pool.empty:
            continue
        tbins = in_pool["created_at"].map(lambda ts: time_bin_label(ts, time_bin))
        tt = in_pool["tweet_type"].astype(str).str.lower()

        for uid, sn in zip(in_pool["user_id"], in_pool["user_screen_name"]):
            u = _id_str(uid)
            if u and pd.notna(sn):
                sn_extra.setdefault(u, str(sn).strip().lstrip("@"))

        # création / bio / display name (attributs profil × activité)
        for uid, tbin, uname_raw in zip(in_pool["user_id"], tbins, in_pool["user_name"]):
            u = _id_str(uid)
            if not u:
                continue
            cbin = creation_bins.get(u, "unknown")
            # MID = jour YYYY-MM-DD uniquement (exclure unknown : hub qui écrase le CP)
            if cbin != "unknown":
                ucd[(u, cbin, tbin)] += 1.0
            for bg in bio_motifs.get(u, []):
                ubio[(u, bg, tbin)] += 1.0
            for tag in bio_tags.get(u, []):
                ubio_ht[(u, tag, tbin)] += 1.0
            if pd.notna(uname_raw):
                for tok in lemmatize_tokens(str(uname_raw), "fr")[:8]:
                    uname[(u, tok, tbin)] += 1.0

        for uid, raw_url, tbin, typ in zip(
            in_pool["user_id"], in_pool["tweet_url_url"], tbins, tt
        ):
            u = _id_str(uid)
            norm = normalize_url(str(raw_url) if pd.notna(raw_url) else "")
            if not u or not norm:
                continue
            uurlt[(u, norm, tbin)] += 1.0
            is_orig = typ != "retweet"
            if is_orig:
                uurlt_orig[(u, norm, tbin)] += 1.0
            dom = url_domain(norm)
            if dom:
                uudom[(u, dom, tbin)] += 1.0
                if is_orig:
                    uudom_orig[(u, dom, tbin)] += 1.0

        for uid, raw_ht, tbin, typ in zip(
            in_pool["user_id"], in_pool["hashtag"], tbins, tt
        ):
            u = _id_str(uid)
            is_orig = typ != "retweet"
            for tag in parse_hashtags(raw_ht):
                uht[(u, tag, tbin)] += 1.0
                if is_orig:
                    uht_orig[(u, tag, tbin)] += 1.0

        # médias hors RT (production propre avec pièce jointe)
        for uid, typ, mtype, murl, tbin in zip(
            in_pool["user_id"], tt, in_pool["media_type"], in_pool["media_url"], tbins
        ):
            if typ == "retweet":
                continue
            u = _id_str(uid)
            if not u:
                continue
            types = parse_media_types(mtype)
            urls = parse_media_urls(murl)
            if not types and urls:
                types = ["media"]
            if not types and not urls:
                continue
            n_media_orig += 1
            for mt in types:
                umed_orig[(u, mt, tbin)] += 1.0
            for url in urls:
                umedurl_orig[(u, url, tbin)] += 1.0

        if with_ngrams:
            for uid, text, lang, tbin, typ in zip(
                in_pool["user_id"], in_pool["text"], in_pool["language"], tbins, tt
            ):
                u = _id_str(uid)
                if not u or not pd.notna(text):
                    continue
                lg = map_lang(lang)
                lang_counts[lg] += 1
                toks = lemmatize_tokens(str(text), lg)
                if len(toks) < 2:
                    continue
                is_orig = typ != "retweet"
                n_ngram_docs += 1
                if is_orig:
                    n_ngram_orig += 1
                ngram_docs.append((u, tbin, toks, is_orig))
                # trigrammes : fenêtre glissante fixe (contraste vs n-grammes candidats)
                for tg in make_ngrams(toks, 3):
                    utrigram[(u, tg, tbin)] += 1.0
                if is_orig:
                    for tg in make_ngrams(toks, 3):
                        utrigram_orig[(u, tg, tbin)] += 1.0

        rt = in_pool[tt == "retweet"]
        for uid, rtid, tbin in zip(
            rt["user_id"],
            rt["retweeted_tweet_id"],
            rt["created_at"].map(lambda ts: time_bin_label(ts, time_bin)),
        ):
            u = _id_str(uid)
            rid = _id_str(rtid)
            target = tweet_author.get(rid, "")
            if u and target and target != u:
                uut[(u, target, tbin)] += 1.0
                cbin = creation_bins.get(u, "unknown")
                if cbin != "unknown":
                    ucdrt[(u, cbin, target)] += 1.0
            # co-RT : MID = publication cible co-RT (plusieurs users → même tweet)
            if u and rid:
                ucort[(u, rid, tbin)] += 1.0

        if chunk_idx % 4 == 0:
            log(f"  chunk {chunk_idx} · docs ngram={n_ngram_docs:,} · horsRT={n_ngram_orig:,}")

    if with_ngrams:
        log(
            f"  docs n-grammes={n_ngram_docs:,} (hors RT={n_ngram_orig:,}) · "
            f"langues={dict(lang_counts.most_common(8))}"
        )
        log("  sélection n-grammes candidats (n=2..4, PMI≥0.8, df≥2, longest-match)…")
        tweet_stats = build_collocation_stats([toks for _, _, toks, _ in ngram_docs], n_max=4)
        n_by_len: Counter[int] = Counter()
        n_by_len_orig: Counter[int] = Counter()
        for u, tbin, toks, is_orig in ngram_docs:
            sel = select_candidate_ngrams(toks, tweet_stats, n_max=4, min_pmi=0.8, min_df=2)
            cbin = creation_bins.get(u, "unknown")
            for g in sel:
                ubigram[(u, g, tbin)] += 1.0
                n_by_len[g.count("_") + 1] += 1
                if cbin != "unknown":
                    ucdng[(u, cbin, g)] += 1.0
                if is_orig:
                    ubigram_orig[(u, g, tbin)] += 1.0
                    n_by_len_orig[g.count("_") + 1] += 1
        log(f"  ubigram n-len={dict(n_by_len)} · ubigram_orig n-len={dict(n_by_len_orig)}")
        ngram_docs.clear()

    def to_df(d: dict, cols: tuple[str, str, str], max_mid: int = 0) -> pd.DataFrame:
        df = pd.DataFrame(
            [(a, b, c, v) for (a, b, c), v in d.items() if v >= min_count],
            columns=[cols[0], cols[1], cols[2], "val"],
        )
        if max_mid > 0 and not df.empty:
            mid = cols[1]
            top = df.groupby(mid)["val"].sum().sort_values(ascending=False).head(max_mid).index
            df = df[df[mid].isin(top)].copy()
        return df.reset_index(drop=True)

    tensors = {
        "uut": to_df(uut, ("user_id", "retweeted_user_id", tcol), max_targets),
        "uurlt": to_df(uurlt, ("user_id", "url", tcol), max_urls),
        "uudom": to_df(uudom, ("user_id", "domain", tcol), max_domains),
        "uht": to_df(uht, ("user_id", "hashtag", tcol), max_hashtags),
        "uurlt_orig": to_df(uurlt_orig, ("user_id", "url", tcol), max_urls),
        "uudom_orig": to_df(uudom_orig, ("user_id", "domain", tcol), max_domains),
        "uht_orig": to_df(uht_orig, ("user_id", "hashtag", tcol), max_hashtags),
        "ucd": to_df(ucd, ("user_id", "creation_bin", tcol), 0),
        "ucdrt": _filter_creation_object_dict(
            ucdrt,
            min_count=min_count,
            max_objects=max_targets,
            min_users=2,
            object_name="retweeted_user_id",
        ),
        "uname": _filter_ngram_dict(
            uname, min_count=min_count, max_ngrams=800, min_users=min_users_ngram, mid_name="name_token"
        ),
        # co-RT : ne garder que les status relayés par ≥ min_users_cort comptes du pool
        "ucort": _filter_ngram_dict(
            ucort,
            min_count=min_count,
            max_ngrams=max_cort,
            min_users=min_users_cort,
            mid_name="status_id",
        ),
        "umed_orig": to_df(umed_orig, ("user_id", "media_type", tcol), 0),
        "umedurl_orig": to_df(umedurl_orig, ("user_id", "media_url", tcol), max_urls),
    }
    log(f"  docs média hors RT={n_media_orig:,}")
    if bio_motifs:
        tensors["ubio"] = _filter_ngram_dict(
            ubio, min_count=min_count, max_ngrams=1500, min_users=2, mid_name="bio_ngram"
        )
    if bio_tags:
        tensors["ubio_ht"] = _filter_ngram_dict(
            ubio_ht, min_count=min_count, max_ngrams=600, min_users=2, mid_name="bio_hashtag"
        )
    if with_ngrams:
        tensors["ubigram"] = _filter_ngram_dict(
            ubigram, min_count=min_count, max_ngrams=max_bigrams, min_users=min_users_ngram, mid_name="ngram"
        )
        tensors["utrigram"] = _filter_ngram_dict(
            utrigram, min_count=min_count, max_ngrams=max_trigrams, min_users=min_users_ngram, mid_name="trigram"
        )
        tensors["ubigram_orig"] = _filter_ngram_dict(
            ubigram_orig,
            min_count=min_count,
            max_ngrams=max_bigrams,
            min_users=min_users_ngram,
            mid_name="ngram",
        )
        tensors["utrigram_orig"] = _filter_ngram_dict(
            utrigram_orig,
            min_count=min_count,
            max_ngrams=max_trigrams,
            min_users=min_users_ngram,
            mid_name="trigram",
        )
        tensors["ucdng"] = _filter_creation_object_dict(
            ucdng,
            min_count=min_count,
            max_objects=max_bigrams,
            min_users=min_users_ngram,
            object_name="ngram",
        )
        for k, mid in (
            ("ubigram", "ngram"),
            ("utrigram", "trigram"),
            ("ubigram_orig", "ngram"),
            ("utrigram_orig", "trigram"),
            ("ucort", "status_id"),
            ("ubio", "bio_ngram"),
            ("ubio_ht", "bio_hashtag"),
            ("uname", "name_token"),
            ("ucd", "creation_bin"),
            ("ucdrt", "creation_bin"),
            ("ucdng", "creation_bin"),
            ("umed_orig", "media_type"),
            ("umedurl_orig", "media_url"),
        ):
            df = tensors.get(k)
            if df is None or df.empty:
                continue
            log(
                f"  {k}: nnz={len(df):,} users={df['user_id'].nunique():,} "
                f"mid={df[mid].nunique():,}"
            )
    else:
        for k, mid in (
            ("ucort", "status_id"),
            ("ucd", "creation_bin"),
            ("ucdrt", "creation_bin"),
            ("uname", "name_token"),
            ("ubio", "bio_ngram"),
            ("umed_orig", "media_type"),
            ("umedurl_orig", "media_url"),
        ):
            df = tensors.get(k)
            if df is None or df.empty:
                continue
            log(
                f"  {k}: nnz={len(df):,} users={df['user_id'].nunique():,} mid={df[mid].nunique():,}"
            )
    return tensors, sn_extra, tweet_author


# ---------------------------------------------------------------------------
# CP-ALS
# ---------------------------------------------------------------------------


def encode_triples(df, cols, *, value_transform="log1p"):
    a_ids = sorted(df[cols[0]].astype(str).unique())
    b_ids = sorted(df[cols[1]].astype(str).unique())
    c_ids = sorted(df[cols[2]].astype(str).unique())
    a_map = {x: i for i, x in enumerate(a_ids)}
    b_map = {x: i for i, x in enumerate(b_ids)}
    c_map = {x: i for i, x in enumerate(c_ids)}
    ii = df[cols[0]].astype(str).map(a_map).to_numpy(dtype=np.int32)
    jj = df[cols[1]].astype(str).map(b_map).to_numpy(dtype=np.int32)
    kk = df[cols[2]].astype(str).map(c_map).to_numpy(dtype=np.int32)
    vv = df["val"].astype(float).to_numpy()
    if value_transform == "log1p":
        vv = np.log1p(vv)
    elif value_transform == "binary":
        vv = (vv > 0).astype(float)
    return ii, jj, kk, vv, a_ids, b_ids, c_ids


def _mttkrp(ii, jj, kk, vv, Fj, Fk, n_rows):
    out = np.zeros((n_rows, Fj.shape[1]), dtype=np.float64)
    np.add.at(out, ii, vv[:, None] * (Fj[jj] * Fk[kk]))
    return out


def _mttkrp_mode2(ii, jj, kk, vv, Fi, Fk, n_rows):
    out = np.zeros((n_rows, Fi.shape[1]), dtype=np.float64)
    np.add.at(out, jj, vv[:, None] * (Fi[ii] * Fk[kk]))
    return out


def _mttkrp_mode3(ii, jj, kk, vv, Fi, Fj, n_rows):
    out = np.zeros((n_rows, Fi.shape[1]), dtype=np.float64)
    np.add.at(out, kk, vv[:, None] * (Fi[ii] * Fj[jj]))
    return out


def _gram_inv(Fa, Fb):
    g = (Fa.T @ Fa) * (Fb.T @ Fb)
    g.flat[:: g.shape[0] + 1] += 1e-8
    return np.linalg.inv(g)


def _normalize_cols(F):
    norms = np.linalg.norm(F, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return F / norms, norms


def sparse_cp_als(ii, jj, kk, vv, dims, rank, n_iter, seed=0):
    rng = np.random.default_rng(seed)
    I, J, K = dims
    A = rng.normal(size=(I, rank))
    B = rng.normal(size=(J, rank))
    C = rng.normal(size=(K, rank))
    A, _ = _normalize_cols(A)
    B, _ = _normalize_cols(B)
    C, lam = _normalize_cols(C)
    for it in range(n_iter):
        A = _mttkrp(ii, jj, kk, vv, B, C, I) @ _gram_inv(B, C)
        A, _ = _normalize_cols(A)
        B = _mttkrp_mode2(ii, jj, kk, vv, A, C, J) @ _gram_inv(A, C)
        B, _ = _normalize_cols(B)
        C = _mttkrp_mode3(ii, jj, kk, vv, A, B, K) @ _gram_inv(A, B)
        C, lam = _normalize_cols(C)
        if (it + 1) % 10 == 0 or it == 0:
            pred = np.sum(A[ii] * B[jj] * C[kk] * lam, axis=1)
            err = float(np.linalg.norm(vv - pred) / (np.linalg.norm(vv) + 1e-12))
            log(f"  CP it {it+1}/{n_iter} · err_nnz={err:.4f}")
    return A, B, C, lam


def sparse_coupled_als(t1, t2, rank, n_iter, seed=0):
    (i1, j1, k1, v1, a1, b1, c1) = t1
    (i2, j2, k2, v2, a2, b2, c2) = t2
    common = sorted(set(a1) & set(a2))
    if len(common) < 10:
        raise RuntimeError(f"Intersection user trop petite ({len(common)})")
    cmap = {u: i for i, u in enumerate(common)}
    a1_arr, a2_arr = np.array(a1), np.array(a2)
    u1, u2 = a1_arr[i1], a2_arr[i2]
    keep1 = np.fromiter((u in cmap for u in u1), dtype=bool, count=len(u1))
    keep2 = np.fromiter((u in cmap for u in u2), dtype=bool, count=len(u2))
    i1c = np.array([cmap[u] for u in u1[keep1]], dtype=np.int32)
    j1c, k1c, v1c = j1[keep1], k1[keep1], v1[keep1]
    i2c = np.array([cmap[u] for u in u2[keep2]], dtype=np.int32)
    j2c, k2c, v2c = j2[keep2], k2[keep2], v2[keep2]
    I = len(common)
    J1, K1, J2, K2 = len(b1), len(c1), len(b2), len(c2)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(I, rank))
    B = rng.normal(size=(J1, rank))
    C = rng.normal(size=(K1, rank))
    D = rng.normal(size=(J2, rank))
    E = rng.normal(size=(K2, rank))
    for M in (A, B, C, D, E):
        M[:], _ = _normalize_cols(M)
    for it in range(n_iter):
        # user partagé : moyenne des deux MTTKRP
        A1 = _mttkrp(i1c, j1c, k1c, v1c, B, C, I) @ _gram_inv(B, C)
        A2 = _mttkrp(i2c, j2c, k2c, v2c, D, E, I) @ _gram_inv(D, E)
        A = 0.5 * (A1 + A2)
        A, _ = _normalize_cols(A)
        B = _mttkrp_mode2(i1c, j1c, k1c, v1c, A, C, J1) @ _gram_inv(A, C)
        B, _ = _normalize_cols(B)
        C = _mttkrp_mode3(i1c, j1c, k1c, v1c, A, B, K1) @ _gram_inv(A, B)
        C, _ = _normalize_cols(C)
        D = _mttkrp_mode2(i2c, j2c, k2c, v2c, A, E, J2) @ _gram_inv(A, E)
        D, _ = _normalize_cols(D)
        E = _mttkrp_mode3(i2c, j2c, k2c, v2c, A, D, K2) @ _gram_inv(A, D)
        E, _ = _normalize_cols(E)
        if (it + 1) % 10 == 0 or it == 0:
            p1 = np.sum(A[i1c] * B[j1c] * C[k1c], axis=1)
            p2 = np.sum(A[i2c] * D[j2c] * E[k2c], axis=1)
            e1 = float(np.linalg.norm(v1c - p1) / (np.linalg.norm(v1c) + 1e-12))
            e2 = float(np.linalg.norm(v2c - p2) / (np.linalg.norm(v2c) + 1e-12))
            log(f"  Coupled it {it+1}/{n_iter} · err1={e1:.4f} err2={e2:.4f}")
    return {
        "users": common,
        "A": A,
        "mid1": b1,
        "B": B,
        "day1": c1,
        "C": C,
        "mid2": b2,
        "D": D,
        "day2": c2,
        "E": E,
    }


def align_sign(A: np.ndarray) -> np.ndarray:
    out = A.copy()
    for r in range(out.shape[1]):
        if out[np.argmax(np.abs(out[:, r])), r] < 0:
            out[:, r] *= -1
    return out


def gini(weights) -> float:
    x = np.asarray(weights, dtype=float)
    x = x[x > 0]
    if x.size < 2:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n)


def top_entities(ids, F, r, k=15):
    col = F[:, r]
    order = np.argsort(-np.abs(col))[:k]
    return [{"id": ids[i], "w": float(abs(col[i]))} for i in order if abs(col[i]) > 1e-12]


def magnitude_threshold(w_top: float) -> float:
    """Seuil « même ordre de grandeur » : 10^floor(log10(w_top))."""
    import math

    if w_top <= 0:
        return 0.0
    return float(10.0 ** math.floor(math.log10(w_top)))


def top_entities_same_magnitude(ids, F, r, *, max_scan: int = 800) -> list[dict]:
    """Top acteurs Ck dont le poids est du même ordre de grandeur que le 1er."""
    col = np.abs(F[:, r])
    order = np.argsort(-col)
    if order.size == 0:
        return []
    w_top = float(col[order[0]])
    if w_top <= 1e-12:
        return []
    thr = magnitude_threshold(w_top)
    out: list[dict] = []
    for i in order[:max_scan]:
        w = float(col[i])
        if w < thr or w <= 1e-12:
            break
        out.append({"id": ids[i], "w": w})
    return out


def aggregate_component_behavior_tops(
    users: list[dict],
    profiles: dict[str, dict],
    sn: dict[str, str],
    *,
    k: int = 10,
) -> dict:
    """Tops cibles / mentions / hashtags / URLs agrégés sur les acteurs filtrés de Ck."""
    rt: Counter = Counter()
    ment: Counter = Counter()
    ht: Counter = Counter()
    urls: Counter = Counter()
    for u in users:
        p = profiles.get(u["id"]) or {}
        rt.update(p.get("rt_targets") or {})
        ment.update(p.get("mentions") or {})
        ht.update(p.get("hashtags") or {})
        urls.update(p.get("urls") or {})
        if not p.get("urls"):
            urls.update(p.get("domains") or {})
    w_top = float(users[0]["w"]) if users else 0.0
    thr = magnitude_threshold(w_top) if w_top else 0.0
    return {
        "n_actors": len(users),
        "w_top": round(w_top, 6),
        "w_min": round(float(users[-1]["w"]), 6) if users else 0.0,
        "magnitude_threshold": round(thr, 6),
        "top_rt_targets": _top_counter(rt, sn, k, as_account=True),
        "top_mentions": _top_counter(ment, sn, k, as_account=True),
        "top_hashtags": _top_counter(ht, sn, k),
        "top_urls": _top_counter(urls, sn, k, preserve_label=True),
    }


def time_series(time_ids, C, r, max_points: int = 900):
    """Série temporelle pour l’UI. Tenseur = toutes les minutes ; affichage = top poids."""
    col = np.abs(C[:, r])
    rows = [{"day": time_ids[i], "w": float(col[i])} for i in range(len(time_ids)) if col[i] > 1e-12]
    if len(rows) > max_points:
        rows = sorted(rows, key=lambda d: -d["w"])[:max_points]
    # "day" = clé JSON historique ; contenu = label minute YYYY-MM-DD HH:MM
    rows.sort(key=lambda d: d["day"])
    return rows


def classify_mid_item(item: str, key: str) -> str:
    s = (item or "").lower()
    if key == "uut":
        return "compte"
    if key == "ucort" or key.startswith("coupled_cort") or key == "coupled_ucort_uurlt_orig":
        return "status_x"
    if key in {
        "ubigram",
        "ubigram_orig",
        "coupled_uut_ubigram",
        "coupled_cort_ubigram_orig",
        "coupled_ubigram_uurlt",
        "ubio",
        "coupled_ubio_ucort",
    }:
        # n-grammes candidats : classer selon longueur
        n = s.count("_") + 1 if s else 0
        if n >= 4:
            return "4gramme"
        if n == 3:
            return "trigramme"
        return "bigramme"
    if key in {"utrigram", "utrigram_orig", "coupled_cort_utrigram_orig"}:
        return "trigramme"
    if key in {"ucd", "ucdrt", "ucdng"} or key.startswith("coupled_ucd"):
        return "creation_bin"
    if key == "ubio_ht":
        return "hashtag"
    if key == "uname":
        return "name_token"
    if key in {"umed_orig", "coupled_umed_ucort", "coupled_umed_ubigram_orig"}:
        return s if s in {"photo", "video", "animated_gif", "media"} else "media_type"
    if key == "umedurl_orig":
        return "media_url"
    if key in {"uht", "uht_orig", "coupled_uht_umed_orig"}:
        return "noise_tag" if s in NOISE_TAGS else "hashtag"
    if key in {"uudom", "uudom_orig"}:
        if s in {"x.com", "twitter.com", "t.co"}:
            return "plateforme_x"
        return "domaine"
    if any(x in s for x in ("lemonde.fr", "liberation.fr", "francetvinfo", "bfmtv", "lefigaro")):
        return "media_fr"
    if s.startswith("x.com/") or "/status/" in s:
        return "status_x"
    if "." in s or "/" in s:
        return "url_ext"
    return "autre"


def time_lag_value(time_a, time_b, time_bin: str):
    """Écart entre pics des deux couches (jours / heures / minutes)."""
    if not time_a or not time_b:
        return None
    pa = max(time_a, key=lambda d: d["w"])["day"]
    pb = max(time_b, key=lambda d: d["w"])["day"]
    try:
        # "YYYY-MM-DD HH" → Timestamp parseable
        ta = pd.Timestamp(pa if ":" in str(pa) or len(str(pa)) <= 10 else f"{pa}:00")
        tb = pd.Timestamp(pb if ":" in str(pb) or len(str(pb)) <= 10 else f"{pb}:00")
        delta = tb - ta
        if time_bin == "day":
            return float(delta.days)
        if time_bin == "hour":
            return float(delta.total_seconds() / 3600.0)
        return float(delta.total_seconds() / 60.0)
    except Exception:
        return None


def auto_label_component(*, key, mid, time, time2, gini_mid, classes: Counter, time_bin: str = "day"):
    peak = max(time, key=lambda d: d["w"])["day"] if time else "?"
    lag = time_lag_value(time, time2, time_bin) if time2 else None
    unit = {"day": "j", "hour": "h", "minute": "min"}.get(time_bin, "t")
    top_cls = classes.most_common(1)[0][0] if classes else "autre"
    if key.startswith("coupled") and lag is not None:
        layer = "n-gramme" if "ubigram" in key else "URL"
        if lag <= -1:
            return f"Import {layer} puis RT (lag {lag:.0f}{unit} · pic {peak})"
        if lag >= 1:
            return f"Amplification RT puis ancrage {layer} (lag +{lag:.0f}{unit} · pic {peak})"
        return f"Campagne sync RT×{layer} (pic {peak})"
    if gini_mid >= 0.65:
        return f"Relais de hub ({top_cls} · pic {peak})"
    if top_cls in {"media_fr", "science", "factcheck"}:
        return f"Circulation documentaire ({top_cls} · pic {peak})"
    if top_cls == "status_x":
        return f"Amplification de status X (pic {peak})"
    if top_cls in {"bigramme", "trigramme", "4gramme"}:
        return f"Motif lexical ({top_cls} · pic {peak})"
    if key in {"uht", "uht_orig"} and top_cls == "noise_tag":
        return f"Cadre lexical collecté (tags bruit · pic {peak})"
    return f"Régime diffus ({top_cls} · Gini={gini_mid:.2f} · pic {peak})"


def pca_users(user_ids, A, sn):
    A = align_sign(A)
    n, r = A.shape
    mass = np.abs(A).sum(axis=1)
    dom = np.argmax(np.abs(A), axis=1)
    if r >= 2 and n >= 3:
        coords = PCA(n_components=2, random_state=0).fit_transform(A)
        evr = PCA(n_components=min(3, r), random_state=0).fit(A).explained_variance_ratio_
    else:
        coords = np.column_stack([A[:, 0], np.zeros(n)])
        evr = [1.0]
    points = []
    for i, uid in enumerate(user_ids):
        wrow = np.abs(A[i])
        order = np.argsort(-wrow)
        points.append(
            {
                "id": uid,
                "sn": sn.get(uid, uid),
                "x": round(float(coords[i, 0]), 5),
                "y": round(float(coords[i, 1]), 5),
                "mass": round(float(mass[i]), 5),
                "dom": int(dom[i]) + 1,
                "dom2": int(order[1]) + 1 if r > 1 else None,
                "weights": [round(float(A[i, j]), 5) for j in range(r)],
            }
        )
    label_idx = list(np.argsort(-mass)[:12])
    return {
        "explained": [round(float(x), 4) for x in evr],
        "points": points,
        "label_ids": [user_ids[i] for i in label_idx],
    }


def collect_user_behaviors(
    csv_path: Path,
    *,
    t0: str,
    t1: str,
    pool_ids: set[str],
    tweet_author: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Profils comportementaux des comptes du pool sur la fenêtre."""
    t0_ts = pd.Timestamp(t0, tz="UTC")
    t1_ts = pd.Timestamp(t1, tz="UTC")
    profiles: dict[str, dict] = {
        u: {
            "n_tweets": 0,
            "n_rt": 0,
            "n_orig": 0,
            "rt_targets": Counter(),
            "mentions": Counter(),
            "hashtags": Counter(),
            "urls": Counter(),
            "domains": Counter(),
            "bio": "",
        }
        for u in pool_ids
    }
    usecols = [
        "user_id",
        "created_at",
        "tweet_type",
        "hashtag",
        "tweet_url_url",
        "retweeted_tweet_id",
        "text",
        "mentions",
    ]
    log("Profils comportementaux (cibles RT, mentions, hashtags, URLs)…")
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=200_000, low_memory=False):
        chunk["created_at"] = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce")
        sub = chunk[(chunk["created_at"] >= t0_ts) & (chunk["created_at"] < t1_ts)]
        if sub.empty:
            continue
        sub = sub[sub["user_id"].map(lambda x: _id_str(x) in pool_ids)]
        if sub.empty:
            continue
        tt = sub["tweet_type"].astype(str).str.lower()
        for uid, typ, ht, url, rtid, text, mentions in zip(
            sub["user_id"],
            tt,
            sub["hashtag"],
            sub["tweet_url_url"],
            sub["retweeted_tweet_id"],
            sub["text"],
            sub["mentions"],
        ):
            u = _id_str(uid)
            p = profiles[u]
            p["n_tweets"] += 1
            is_rt = typ == "retweet"
            mention_parts = split_field(mentions) if pd.notna(mentions) else []
            if is_rt:
                p["n_rt"] += 1
                target = ""
                if tweet_author is not None:
                    target = tweet_author.get(_id_str(rtid), "")
                if not target and pd.notna(text):
                    m = _RT_AT.match(str(text))
                    if m:
                        target = f"sn:{m.group(1).lower()}"
                if not target and mention_parts:
                    target = f"sn:{mention_parts[0].lstrip('@').lower()}"
                if target and target != u:
                    p["rt_targets"][target] += 1
                # mentions hors auteur RT (mentions additionnelles dans le RT)
                skip0 = mention_parts[0].lstrip("@").lower() if mention_parts else ""
                for part in mention_parts[1:]:
                    h = part.lstrip("@").lower()
                    if h and h != skip0:
                        p["mentions"][f"sn:{h}"] += 1
            else:
                p["n_orig"] += 1
                for part in mention_parts:
                    h = part.lstrip("@").lower()
                    if h:
                        p["mentions"][f"sn:{h}"] += 1
            for tag in parse_hashtags(ht):
                p["hashtags"][tag] += 1
            norm = normalize_url(str(url) if pd.notna(url) else "")
            if norm:
                p["urls"][norm] += 1
                dom = url_domain(norm)
                if dom:
                    p["domains"][dom] += 1
    return profiles


def _top_counter(
    counter: Counter,
    sn: dict[str, str],
    k: int = 5,
    as_account: bool = False,
    *,
    preserve_label: bool = False,
) -> list[dict]:
    out = []
    for item, n in counter.most_common(k):
        if as_account:
            lab = sn.get(item, item)
            if str(lab).startswith("sn:"):
                lab = str(lab)[3:]
            out.append({"id": item, "label": "@" + str(lab).lstrip("@"), "n": int(n)})
        else:
            raw = str(item)
            lab = raw if preserve_label else raw.replace("_", " ")
            out.append({"id": item, "label": lab, "n": int(n)})
    return out


def _handle_index(user_ids: list[str], sn: dict[str, str]) -> dict[str, str]:
    """handle lowercase → user_id (pour résoudre sn:xxx et labels)."""
    idx: dict[str, str] = {}
    for uid in user_ids:
        h = str(sn.get(uid, "") or "").lstrip("@").lower().strip()
        if h:
            idx[h] = uid
        idx[str(uid)] = uid
    return idx


def _resolve_peer(target: str, uid_set: set[str], handle_idx: dict[str, str], sn: dict[str, str]) -> str | None:
    t = str(target)
    if t in uid_set:
        return t
    if t.startswith("sn:"):
        return handle_idx.get(t[3:].lower())
    lab = str(sn.get(t, "") or "").lstrip("@").lower()
    if lab and lab in handle_idx:
        return handle_idx[lab]
    return handle_idx.get(t.lower())


def network_mode_for(key: str) -> dict:
    """Une seule modalité d’interaction, alignée sur le tenseur étudié."""
    k = key or ""
    if k == "uut" or k.endswith("_uut") or k == "coupled_ucd_uut":
        # co-cibles RT parmi les MID révélés (souvent plus dense que RT A→B internes)
        return {
            "kind": "co_mid",
            "label": "Co-cibles RT (MID Ck)",
            "edge_fr": "cibles RT partagées",
            "directed": False,
        }
    if "ubigram_orig" in k:
        return {
            "kind": "co_mid",
            "label": "Co-n-grammes hors RT (MID Ck)",
            "edge_fr": "n-grammes partagés",
            "directed": False,
        }
    if "ubigram" in k:
        return {
            "kind": "co_mid",
            "label": "Co-n-grammes (MID Ck)",
            "edge_fr": "n-grammes partagés",
            "directed": False,
        }
    if "utrigram_orig" in k:
        return {
            "kind": "co_mid",
            "label": "Co-trigrammes hors RT (MID Ck)",
            "edge_fr": "trigrammes partagés",
            "directed": False,
        }
    if "utrigram" in k:
        return {
            "kind": "co_mid",
            "label": "Co-trigrammes (MID Ck)",
            "edge_fr": "trigrammes partagés",
            "directed": False,
        }
    if "ubio" in k:
        return {
            "kind": "co_mid",
            "label": "Co-n-grammes bio (MID Ck)",
            "edge_fr": "motifs bio partagés",
            "directed": False,
        }
    if "uurlt_orig" in k:
        return {
            "kind": "co_mid",
            "label": "Co-URLs hors RT (MID Ck)",
            "edge_fr": "URLs hors RT partagées",
            "directed": False,
        }
    if "uurlt" in k or k == "coupled_uut_uurlt":
        return {
            "kind": "co_mid",
            "label": "Co-URLs (MID Ck)",
            "edge_fr": "URLs partagées",
            "directed": False,
        }
    if "uudom_orig" in k:
        return {
            "kind": "co_mid",
            "label": "Co-domaines hors RT (MID Ck)",
            "edge_fr": "domaines hors RT partagés",
            "directed": False,
        }
    if "uudom" in k:
        return {
            "kind": "co_mid",
            "label": "Co-domaines (MID Ck)",
            "edge_fr": "domaines partagés",
            "directed": False,
        }
    if k == "uht_orig" or k.endswith("uht_orig"):
        return {
            "kind": "co_mid",
            "label": "Co-hashtags hors RT (MID Ck)",
            "edge_fr": "hashtags hors RT partagés",
            "directed": False,
        }
    if k == "uht" or k.endswith("_uht"):
        return {
            "kind": "co_mid",
            "label": "Co-hashtags (MID Ck)",
            "edge_fr": "hashtags partagés",
            "directed": False,
        }
    if "ucort" in k:
        return {
            "kind": "co_mid",
            "label": "Co-publications cibles co-RT (MID Ck)",
            "edge_fr": "publications cibles co-RT partagées",
            "directed": False,
        }
    if k == "ucdrt":
        return {
            "kind": "co_mid",
            "label": "Co-cibles RT (mode 3 Ck)",
            "edge_fr": "cibles RT partagées",
            "directed": False,
        }
    if k == "ucdng":
        return {
            "kind": "co_mid",
            "label": "Co-n-grammes (mode 3 Ck)",
            "edge_fr": "n-grammes partagés",
            "directed": False,
        }
    if k == "ucd" or k.startswith("coupled_ucd"):
        return {
            "kind": "co_mid",
            "label": "Même jour de création (MID Ck)",
            "edge_fr": "création partagée",
            "directed": False,
        }
    if "uname" in k:
        return {
            "kind": "co_mid",
            "label": "Co-tokens display name (MID Ck)",
            "edge_fr": "tokens partagés",
            "directed": False,
        }
    if "umedurl" in k:
        return {
            "kind": "co_mid",
            "label": "Co-URLs média hors RT (MID Ck)",
            "edge_fr": "médias partagés",
            "directed": False,
        }
    if "umed" in k:
        return {
            "kind": "co_mid",
            "label": "Co-types média hors RT (MID Ck)",
            "edge_fr": "types média partagés",
            "directed": False,
        }
    return {
        "kind": "co_mid",
        "label": "Co-objets MID Ck",
        "edge_fr": "objets partagés",
        "directed": False,
    }


def user_mid_index(df: pd.DataFrame, user_col: str, mid_col: str) -> dict[str, Counter]:
    """user_id → Counter(mid) (somme sur les bins temps)."""
    out: dict[str, Counter] = defaultdict(Counter)
    if df is None or df.empty:
        return out
    for u, m, v in zip(df[user_col].astype(str), df[mid_col].astype(str), df["val"]):
        out[u][m] += float(v)
    return out


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """AF / PCA 2D sur lignes (acteurs × traits). Retourne coords n×2 centrées."""
    n, d = X.shape
    if n == 0:
        return np.zeros((0, 2))
    if n == 1 or d == 0:
        return np.zeros((n, 2))
    Xc = X - X.mean(axis=0, keepdims=True)
    # colonnes quasi-nulles
    std = Xc.std(axis=0)
    keep = std > 1e-12
    if not np.any(keep):
        return np.zeros((n, 2))
    Xc = Xc[:, keep]
    try:
        # économie : SVD sur Gram si n < d
        if n <= Xc.shape[1]:
            G = Xc @ Xc.T
            evals, evecs = np.linalg.eigh(G)
            order = np.argsort(evals)[::-1]
            coords = evecs[:, order[:2]] * np.sqrt(np.maximum(evals[order[:2]], 0.0))
        else:
            U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
            k = min(2, U.shape[1], len(S))
            coords = np.zeros((n, 2))
            coords[:, :k] = U[:, :k] * S[:k]
    except Exception:
        coords = np.zeros((n, 2))
    # échelle visuelle stable
    for j in range(2):
        span = float(np.ptp(coords[:, j])) or 1.0
        coords[:, j] = (coords[:, j] - coords[:, j].mean()) / span
    return coords


def _noverlap_2d(coords: np.ndarray, degrees: list[int] | np.ndarray, *, iters: int = 120) -> np.ndarray:
    """Écarte les nœuds qui se chevauchent (rayon ∝ √degré), conserve la structure AF."""
    n = int(coords.shape[0])
    if n < 2:
        return coords
    P = np.asarray(coords, dtype=float).copy()
    for j in range(2):
        span = float(np.ptp(P[:, j])) or 1.0
        P[:, j] = (P[:, j] - P[:, j].mean()) / span
    deg = np.asarray(degrees, dtype=float)
    max_d = float(deg.max()) if len(deg) else 1.0
    # rayons en espace normalisé (plus grands = plus d’écart)
    radii = 0.055 + 0.14 * np.sqrt(np.maximum(deg, 0.0) / max(max_d, 1.0))
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = P[i, 0] - P[j, 0]
                dy = P[i, 1] - P[j, 1]
                dist = math.hypot(dx, dy)
                min_d = float(radii[i] + radii[j])
                if dist >= min_d:
                    continue
                if dist < 1e-9:
                    # séparation aléatoire déterministe
                    ang = (i * 17 + j * 31) % 360 * math.pi / 180.0
                    dx, dy = math.cos(ang), math.sin(ang)
                    dist = 1e-9
                push = 0.55 * (min_d - dist)
                ux, uy = dx / dist, dy / dist
                P[i, 0] += ux * push
                P[i, 1] += uy * push
                P[j, 0] -= ux * push
                P[j, 1] -= uy * push
                moved = True
        if not moved:
            break
    # re-centrage
    P -= P.mean(axis=0, keepdims=True)
    return P


def build_component_network(
    users: list[dict],
    profiles: dict[str, dict],
    sn: dict[str, str],
    *,
    key: str = "",
    mid_items: list[dict] | None = None,
    user_mids: dict[str, Counter] | None = None,
    max_edges: int = 180,
) -> dict:
    """
    Graphe mono-modalité entre acteurs Ck, spatialisé en AF 2D.
    - uut : RT dirigés entre acteurs de la composante
    - sinon : co-occurrence des MID révélés par la décomposition (intersection)
    """
    mode = network_mode_for(key)
    uids = [str(u["id"]) for u in users]
    uid_set = set(uids)
    handle_idx = _handle_index(uids, sn)
    by_id = {str(u["id"]): u for u in users}
    mid_revealed = [str(m.get("id")) for m in (mid_items or []) if m.get("id") is not None]
    mid_set = set(mid_revealed)
    mid_lab = {
        str(m.get("id")): str(m.get("label") or m.get("id"))
        for m in (mid_items or [])
        if m.get("id") is not None
    }
    um = user_mids or {}

    edges: list[dict] = []

    if mode["kind"] == "rt":
        # RT dirigés A→B uniquement si A et B ∈ Ck
        for uid in uids:
            p = profiles.get(uid) or {}
            for t, n in (p.get("rt_targets") or Counter()).items():
                peer = _resolve_peer(str(t), uid_set, handle_idx, sn)
                if not peer or int(n) <= 0:
                    continue
                src_u, dst_u = by_id.get(uid) or {}, by_id.get(peer) or {}
                edges.append(
                    {
                        "source": uid,
                        "target": peer,
                        "source_label": src_u.get("label")
                        or ("@" + str(sn.get(uid, uid)).lstrip("@")),
                        "target_label": dst_u.get("label")
                        or ("@" + str(sn.get(peer, peer)).lstrip("@")),
                        "weight": int(n),
                        "kind": "rt",
                        "shared": [],
                        "n_shared": 0,
                    }
                )
    else:
        # Vecteurs acteurs × MID Ck (révélés) → co-occurrence
        vecs: dict[str, Counter] = {}
        for uid in uids:
            raw = um.get(uid) or Counter()
            if mid_set:
                filt = Counter({m: raw[m] for m in mid_set if raw.get(m, 0) > 0})
            else:
                filt = Counter(raw)
            vecs[uid] = filt
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                a, b = uids[i], uids[j]
                ca, cb = vecs[a], vecs[b]
                if not ca or not cb:
                    continue
                common = set(ca) & set(cb)
                if not common:
                    continue
                n_shared = len(common)
                # intensité MID (min des comptes) — informative seulement, pas le poids du lien
                intensity = int(sum(min(ca[m], cb[m]) for m in common))
                # liste : priorité aux MID les plus fréquents (affichage), poids du lien = |common|
                top = sorted(common, key=lambda m: -min(ca[m], cb[m]))[:10]
                shared = [mid_lab.get(m, str(m).replace("_", " ")) for m in top]
                edges.append(
                    {
                        "source": a,
                        "target": b,
                        "source_label": (by_id.get(a) or {}).get("label")
                        or ("@" + str(sn.get(a, a)).lstrip("@")),
                        "target_label": (by_id.get(b) or {}).get("label")
                        or ("@" + str(sn.get(b, b)).lstrip("@")),
                        # poids / couleur / épaisseur = nombre strict d’éléments partagés
                        "weight": n_shared,
                        "kind": "co_mid",
                        "shared": shared,
                        "n_shared": n_shared,
                        "intensity": intensity,
                    }
                )

    edges.sort(key=lambda d: (-d.get("n_shared", 0), -d.get("intensity", d.get("weight", 0))))
    edges = edges[:max_edges]
    linked = {e["source"] for e in edges} | {e["target"] for e in edges}

    # Matrice acteurs × MID Ck pour AF 2D
    feat_ids = mid_revealed[:40] if mid_revealed else []
    if not feat_ids and mode["kind"] == "rt":
        # traits = voisins RT entrants/sortants dans Ck
        feat_ids = list(uids)
    X = np.zeros((len(uids), max(len(feat_ids), 1)), dtype=float)
    if feat_ids:
        f_idx = {f: i for i, f in enumerate(feat_ids)}
        u_idx = {u: i for i, u in enumerate(uids)}
        if mode["kind"] == "rt":
            for e in edges:
                si = u_idx.get(e["source"], -1)
                ti = f_idx.get(e["target"], -1)
                if si >= 0 and ti >= 0:
                    X[si, ti] += float(e["weight"])
        else:
            for ui, uid in enumerate(uids):
                raw = um.get(uid) or Counter()
                for m, c in raw.items():
                    j = f_idx.get(m)
                    if j is not None:
                        X[ui, j] = float(c)
    coords = _pca_2d(X)
    degrees = [
        sum(1 for e in edges if e["source"] == uid or e["target"] == uid) for uid in uids
    ]
    # isolés : anneau extérieur pour ne pas saturer le cœur AF
    if any(d == 0 for d in degrees):
        isol_idx = [i for i, d in enumerate(degrees) if d == 0]
        for k, i in enumerate(isol_idx):
            ang = 2 * math.pi * k / max(len(isol_idx), 1) - math.pi / 2
            coords[i, 0] = 1.35 * math.cos(ang)
            coords[i, 1] = 1.35 * math.sin(ang)
    coords = _noverlap_2d(coords, degrees, iters=140)

    nodes = []
    for i, u in enumerate(users):
        uid = str(u["id"])
        deg = int(degrees[i]) if i < len(degrees) else 0
        nodes.append(
            {
                "id": uid,
                "label": u.get("label") or ("@" + str(sn.get(uid, uid)).lstrip("@")),
                "w": float(u.get("w") or 0),
                "n_tweets": int(u.get("n_tweets") or 0),
                "share_rt": float(u.get("share_rt") or 0),
                "n_orig": int(u.get("n_orig") or 0),
                "degree": deg,
                "isolated": uid not in linked,
                "x": float(coords[i, 0]) if i < len(coords) else 0.0,
                "y": float(coords[i, 1]) if i < len(coords) else 0.0,
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_isolated": sum(1 for n in nodes if n["isolated"]),
        "mode": mode["kind"],
        "mode_label": mode["label"],
        "edge_fr": mode["edge_fr"],
        "directed": bool(mode["directed"]),
        "n_mids": len(mid_set),
        "layout": "pca2d_noverlap",
        "size_by": "degree",
    }


def enrich_actor_metrics(
    uid: str,
    *,
    w: float,
    sn: dict[str, str],
    profiles: dict[str, dict],
    bios: dict[str, str] | None = None,
) -> dict:
    """Métriques caractéristiques d’un acteur pour tableaux UI."""
    p = profiles.get(uid) or {}
    n_tw = int(p.get("n_tweets") or 0)
    n_rt = int(p.get("n_rt") or 0)
    n_orig = int(p.get("n_orig") or 0)
    bio = ""
    if bios and uid in bios:
        bio = bios[uid]
    elif p.get("bio"):
        bio = str(p.get("bio") or "")
    lab = sn.get(uid, uid)
    created = snowflake_created_at(uid, min_year=2006)
    return {
        "id": uid,
        "label": "@" + str(lab).lstrip("@"),
        "w": float(w),
        "n_tweets": n_tw,
        "n_rt": n_rt,
        "n_orig": n_orig,
        "share_rt": round((n_rt / n_tw), 3) if n_tw else 0.0,
        # Création compte via snowflake user_id (UTC) — affichage (2006+)
        "account_created_at": created.strftime("%Y-%m-%d %H:%M:%S UTC") if created else None,
        "account_created_day": creation_bin(created),
        "bio": bio[:280],
        "top_rt_targets": _top_counter(p.get("rt_targets") or Counter(), sn, 5, as_account=True),
        "top_mentions": _top_counter(p.get("mentions") or Counter(), sn, 5, as_account=True),
        "top_hashtags": _top_counter(p.get("hashtags") or Counter(), sn, 5),
        # URLs normalisées host/path (plus riches que le seul domaine)
        "top_urls": _top_counter(p.get("urls") or Counter(), sn, 5, preserve_label=True),
        "top_domains": _top_counter(p.get("domains") or Counter(), sn, 5, preserve_label=True),
    }


def aggregate_modality(
    *,
    key: str,
    user_ids: list[str],
    A: np.ndarray,
    mid_ids: list[str],
    B: np.ndarray,
    sn: dict[str, str],
    profiles: dict[str, dict],
    components: list[dict],
    mid2_ids: list[str] | None = None,
    D: np.ndarray | None = None,
    top_k: int = 30,
) -> dict:
    """Agrège les composantes : tops acteurs / objets + comportements corpus."""
    A = align_sign(A)
    B = align_sign(B)
    mass_u = np.abs(A).sum(axis=1)
    mass_m = np.abs(B).sum(axis=1)
    # présence dans les tops composantes
    appear: dict[str, list[int]] = defaultdict(list)
    for c in components:
        cid = c["id"]
        for u in c.get("users") or []:
            appear[u["id"]].append(cid)

    actors = []
    order_u = np.argsort(-mass_u)[:top_k]
    for i in order_u:
        uid = user_ids[i]
        w = float(mass_u[i])
        if w <= 1e-12:
            continue
        dom = int(np.argmax(np.abs(A[i]))) + 1
        row = enrich_actor_metrics(uid, w=w, sn=sn, profiles=profiles)
        row["w_sum"] = round(w, 4)
        row.pop("w", None)
        row["dom_comp"] = dom
        row["in_comps"] = appear.get(uid, [dom])
        actors.append(row)

    def _status_label(mid: str) -> str:
        au = sn.get(f"author_of:{mid}")
        if au:
            return f"@{str(au).lstrip('@')} · …{str(mid)[-8:]}"
        return f"status …{str(mid)[-10:]}"

    def mid_rows(ids, F, *, as_account: bool = False, as_status: bool = False, as_ngram: bool = False):
        mass = np.abs(F).sum(axis=1)
        rows = []
        for i in np.argsort(-mass)[:top_k]:
            w = float(mass[i])
            if w <= 1e-12:
                continue
            mid = ids[i]
            if as_account:
                lab = sn.get(mid, mid)
                if str(lab).startswith("sn:"):
                    lab = str(lab)[3:]
                label = "@" + str(lab).lstrip("@")
            elif as_status:
                label = _status_label(mid)
            elif as_ngram:
                label = str(mid).replace("_", " ")
            else:
                label = str(mid)
            rows.append({"id": mid, "label": label, "w_sum": round(w, 4)})
        return rows

    ngram_keys = {
        "ubigram",
        "utrigram",
        "ubigram_orig",
        "utrigram_orig",
        "coupled_uut_ubigram",
        "coupled_cort_ubigram_orig",
        "coupled_cort_utrigram_orig",
        "coupled_ubigram_uurlt",
        "ubio",
        "coupled_ubio_ucort",
        "coupled_ucd_ubio",
        "uname",
    }
    mid_kind_map = {
        "uut": "cibles RT",
        "uurlt": "URLs",
        "uudom": "domaines",
        "uht": "hashtags",
        "uurlt_orig": "URLs hors RT",
        "uudom_orig": "domaines hors RT",
        "uht_orig": "hashtags hors RT",
        "ubigram": "n-grammes",
        "utrigram": "trigrammes",
        "ubigram_orig": "n-grammes hors RT",
        "utrigram_orig": "trigrammes hors RT",
        "ucort": "publication cible co-RT",
        "ucd": "jours de création",
        "ucdrt": "jours de création",
        "ucdng": "jours de création",
        "ubio": "n-grammes bio",
        "ubio_ht": "hashtags bio",
        "uname": "tokens display name",
        "umed_orig": "types média hors RT",
        "umedurl_orig": "URLs média hors RT",
        "coupled_uut_uurlt": "URLs",
        "coupled_uut_ubigram": "n-grammes",
        "coupled_cort_ubigram_orig": "n-grammes hors RT",
        "coupled_cort_utrigram_orig": "trigrammes hors RT",
        "coupled_ucd_uut": "jours de création",
        "coupled_ucd_ucort": "jours de création",
        "coupled_ubio_ucort": "n-grammes bio",
        "coupled_umed_ucort": "types média hors RT",
        "coupled_umed_ubigram_orig": "types média hors RT",
        "coupled_ubigram_uurlt": "n-grammes",
        "coupled_ucort_uurlt_orig": "publication cible co-RT",
        "coupled_ucd_ubio": "jours de création",
        "coupled_uurlt_ucort": "URLs",
        "coupled_uht_umed_orig": "hashtags",
    }
    agg = {
        "top_actors": actors,
        "top_mids": mid_rows(
            mid_ids,
            B,
            as_account=(key == "uut"),
            as_status=(key == "ucort"),
            as_ngram=(key in ngram_keys),
        ),
        "mid_kind": mid_kind_map.get(key, "objets"),
        "n_actors_shown": len(actors),
    }
    if mid2_ids is not None and D is not None:
        if key in {
            "coupled_cort_ubigram_orig",
            "coupled_cort_utrigram_orig",
            "coupled_ucd_ucort",
            "coupled_ubio_ucort",
            "coupled_umed_ucort",
        }:
            agg["top_mids2"] = mid_rows(mid2_ids, align_sign(D), as_status=True)
            agg["mid2_kind"] = "publication cible co-RT"
        elif key == "coupled_umed_ubigram_orig":
            agg["top_mids2"] = mid_rows(mid2_ids, align_sign(D), as_ngram=True)
            agg["mid2_kind"] = "n-grammes hors RT"
        else:
            agg["top_mids2"] = mid_rows(mid2_ids, align_sign(D), as_account=True)
            agg["mid2_kind"] = "cibles RT"
    return agg


def build_modality_payload(
    *,
    key,
    user_ids,
    A,
    mid_ids,
    B,
    day_ids,
    C,
    sn,
    mid2_ids=None,
    D=None,
    day2_ids=None,
    E=None,
    err_nnz=None,
    time_bin: str = "day",
    profiles: dict[str, dict] | None = None,
    bios: dict[str, str] | None = None,
    user_mids: dict[str, Counter] | None = None,
    top_users: int = 40,
):
    A, B, C = align_sign(A), align_sign(B), align_sign(C)
    if D is not None:
        D = align_sign(D)
    if E is not None:
        E = align_sign(E)
    rank = A.shape[1]
    meta = modality_meta_for(time_bin)[key]
    components = []
    labels = []
    profs = profiles or {}
    bios = bios or {}
    for uid, bio in bios.items():
        if uid in profs:
            profs[uid]["bio"] = bio
    for r in range(rank):
        if A[:, r].max() < -A[:, r].min() and A[:, r].min() < 0:
            for M in (A, B, C):
                M[:, r] *= -1
            if D is not None:
                D[:, r] *= -1
            if E is not None:
                E[:, r] *= -1
        # Acteurs : même ordre de grandeur que le 1er poids CP (pas un top-k arbitraire)
        users = [
            enrich_actor_metrics(u["id"], w=u["w"], sn=sn, profiles=profs, bios=bios)
            for u in top_entities_same_magnitude(user_ids, A, r)
        ]
        if not users:
            users = [
                enrich_actor_metrics(u["id"], w=u["w"], sn=sn, profiles=profs, bios=bios)
                for u in top_entities(user_ids, A, r, min(top_users, 15))
            ]
        behavior_tops = aggregate_component_behavior_tops(users, profs, sn, k=10)
        mid = top_entities(mid_ids, B, r, 15)
        for m in mid:
            mid_id = m["id"]
            if key == "uut":
                m["label"] = "@" + str(sn.get(mid_id, mid_id)).lstrip("@")
            elif key == "ucort":
                au = sn.get(f"author_of:{mid_id}")
                m["label"] = (
                    f"@{str(au).lstrip('@')} · …{str(mid_id)[-8:]}"
                    if au
                    else f"status …{str(mid_id)[-10:]}"
                )
            elif key in {
                "ubigram",
                "utrigram",
                "ubigram_orig",
                "utrigram_orig",
                "coupled_uut_ubigram",
                "coupled_cort_ubigram_orig",
                "coupled_cort_utrigram_orig",
                "coupled_ubigram_uurlt",
                "ubio",
                "coupled_ubio_ucort",
                "coupled_ucd_ubio",
                "uname",
            }:
                m["label"] = str(mid_id).replace("_", " ")
            elif key in {"coupled_ucort_uurlt_orig"}:
                au = sn.get(f"author_of:{mid_id}")
                m["label"] = (
                    f"@{str(au).lstrip('@')} · …{str(mid_id)[-8:]}"
                    if au
                    else f"status …{str(mid_id)[-10:]}"
                )
            else:
                m["label"] = mid_id
        tseries = time_series(day_ids, C, r)
        if key == "ucdrt":
            for row in tseries:
                hid = sn.get(row["day"], row["day"])
                row["day"] = "@" + str(hid).lstrip("@")
        elif key == "ucdng":
            for row in tseries:
                row["day"] = str(row["day"]).replace("_", " ")
        mid2 = tseries2 = None
        if mid2_ids is not None and D is not None:
            mid2 = top_entities(mid2_ids, D, r, 15)
            for m in mid2:
                mid_id = m["id"]
                if key in {
                    "coupled_cort_ubigram_orig",
                    "coupled_cort_utrigram_orig",
                    "coupled_ucd_ucort",
                    "coupled_ubio_ucort",
                    "coupled_umed_ucort",
                    "coupled_uurlt_ucort",
                }:
                    au = sn.get(f"author_of:{mid_id}")
                    m["label"] = (
                        f"@{str(au).lstrip('@')} · …{str(mid_id)[-8:]}"
                        if au
                        else f"status …{str(mid_id)[-10:]}"
                    )
                elif key in {
                    "coupled_umed_ubigram_orig",
                    "coupled_ucd_ubio",
                }:
                    m["label"] = str(mid_id).replace("_", " ")
                elif key == "coupled_uht_umed_orig":
                    m["label"] = str(mid_id)
                else:
                    m["label"] = str(mid_id) if "url" in key or "uurlt" in key else (
                        "@" + str(sn.get(mid_id, mid_id)).lstrip("@")
                    )
        if day2_ids is not None and E is not None:
            tseries2 = time_series(day2_ids, E, r)
        class_key = key
        if key == "coupled_uut_uurlt":
            class_key = "uurlt"
        elif key in {"coupled_uut_ubigram", "coupled_ubigram_uurlt"}:
            class_key = "ubigram"
        elif key == "coupled_cort_ubigram_orig":
            class_key = "ubigram_orig"
        elif key == "coupled_cort_utrigram_orig":
            class_key = "utrigram_orig"
        elif key == "coupled_ucort_uurlt_orig":
            class_key = "ucort"
        elif key.startswith("coupled_ucd"):
            class_key = "ucd"
        elif key == "coupled_ubio_ucort":
            class_key = "ubio"
        elif key == "coupled_uurlt_ucort":
            class_key = "uurlt"
        elif key == "coupled_uht_umed_orig":
            class_key = "uht"
        elif key.startswith("coupled_umed"):
            class_key = "umed_orig"
        classes = Counter(classify_mid_item(str(m["id"]), class_key) for m in mid)
        gm = gini([m["w"] for m in mid])
        lab = auto_label_component(
            key=key,
            mid=mid,
            time=tseries,
            time2=tseries2,
            gini_mid=gm,
            classes=classes,
            time_bin=time_bin,
        )
        labels.append(lab)
        peak = max(tseries, key=lambda d: d["w"])["day"] if tseries else None
        net_mid = mid
        if key in {"ucdrt", "ucdng"}:
            # Réseau sur les objets du mode 3 (cibles RT / n-grammes), pas les jours de création
            net_mid = top_entities(day_ids, C, r, 15)
            for m in net_mid:
                mid_id = m["id"]
                if key == "ucdrt":
                    m["label"] = "@" + str(sn.get(mid_id, mid_id)).lstrip("@")
                else:
                    m["label"] = str(mid_id).replace("_", " ")
        network = build_component_network(
            users,
            profs,
            sn,
            key=key,
            mid_items=net_mid,
            user_mids=user_mids,
        )
        components.append(
            {
                "id": r + 1,
                "label": lab,
                "metrics": {
                    "peak": peak,
                    "gini_users": round(gini([u["w"] for u in users]), 3),
                    "gini_mid": round(gm, 3),
                    "lag_days": time_lag_value(tseries, tseries2, time_bin),
                    "lag_unit": {"day": "j", "hour": "h", "minute": "min"}.get(time_bin, "t"),
                    "classes": dict(classes),
                    "n_actors_same_magnitude": behavior_tops["n_actors"],
                    "magnitude_threshold": behavior_tops["magnitude_threshold"],
                },
                "users": users,
                "behavior_tops": behavior_tops,
                "network": network,
                "mid": mid,
                "time": tseries,
                "mid2": mid2,
                "time2": tseries2,
            }
        )
    # heatmap sample
    mass = np.abs(A).sum(axis=1)
    top_n = min(80, len(user_ids))
    idx = np.argsort(-mass)[:top_n]
    heat = {
        "user_labels": ["@" + str(sn.get(user_ids[i], user_ids[i])).lstrip("@") for i in idx],
        "z": [[round(float(abs(A[i, r])), 4) for r in range(rank)] for i in idx],
    }
    stats = {
        "n_users": len(user_ids),
        "n_mid": len(mid_ids),
        "n_tbin": len(day_ids),
        "err_nnz": err_nnz,
        "labels": labels,
    }
    agg = aggregate_modality(
        key=key,
        user_ids=user_ids,
        A=A,
        mid_ids=mid_ids,
        B=B,
        sn=sn,
        profiles=profiles or {},
        components=components,
        mid2_ids=mid2_ids,
        D=D,
    )
    return {
        "key": key,
        "label": meta["label"],
        "blurb": meta["blurb"],
        "mid_label": meta["mid_label"],
        "mid2_label": meta.get("mid2_label"),
        "modes": meta.get("modes") or meta["label"],
        "coupled": meta["coupled"],
        "rank": rank,
        "n_users": len(user_ids),
        "n_mid": len(mid_ids),
        "n_days": len(day_ids),
        "components": components,
        "pca": pca_users(user_ids, A, sn),
        "heatmap": heat,
        "stats": stats,
        "aggregate": agg,
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML_TMPL = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HEAT · CP multi-tenseurs</title>
<script src="plotly.min.js"></script>
<style>
  :root {
    --bg:#eef1f4; --card:#fff; --ink:#15202b; --muted:#5a6570; --line:#d5dde5;
    --accent:#0f6e6a; --accent-soft:#d8efed; --rt:#1f4e79; --chip:#f4f7fa; --soft:#f7fafb;
  }
  * { box-sizing: border-box; }
  body { margin:0; color:var(--ink); background:var(--bg);
    font:15px/1.45 "Segoe UI","IBM Plex Sans",system-ui,sans-serif; }
  header {
    background:linear-gradient(135deg,#14323f 0%,#0f6e6a 55%,#1f4e79 100%);
    color:#f4faf9; padding:22px 28px 18px;
  }
  header h1 { margin:0 0 6px; font:600 26px/1.2 Georgia,"Source Serif 4",serif; letter-spacing:-.02em; }
  header .sub { margin:0; opacity:.88; max-width:920px; font-size:14px; }
  header .meta { margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; }
  .pill { display:inline-flex; gap:6px; align-items:center; background:rgba(255,255,255,.12);
    border:1px solid rgba(255,255,255,.22); border-radius:999px; padding:4px 10px; font-size:12px; }
  .layout { display:grid; grid-template-columns:280px minmax(0,1fr); min-height:calc(100vh - 120px); }
  nav { background:var(--card); border-right:1px solid var(--line); padding:16px 14px;
    position:sticky; top:0; align-self:start; max-height:100vh; overflow:auto; }
  nav h2 { margin:0 0 10px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
  .mod { width:100%; text-align:left; border:1px solid var(--line); background:var(--chip);
    border-radius:10px; padding:10px 12px; margin-bottom:8px; cursor:pointer; color:inherit; font:inherit; }
  .mod:hover { border-color:#9bb0c0; }
  .mod.on { border-color:var(--accent); background:var(--accent-soft); box-shadow:inset 3px 0 0 var(--accent); }
  .mod b { display:block; font-size:13.5px; margin-bottom:2px; }
  .mod span { display:block; color:var(--muted); font-size:12px; }
  .mod .sc { color:var(--muted); font-weight:500; font-size:11px; margin-top:4px; }
  .actor-card { border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin:8px 0; background:var(--soft); }
  .actor-card h4 { margin:0 0 4px; font-size:14px; }
  .actor-card .meta { font-size:12px; color:var(--muted); margin-bottom:6px; }
  .actor-card ul { margin:4px 0 0; padding-left:1.1em; font-size:12.5px; }
  .actor-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width:900px){ .actor-grid { grid-template-columns:1fr; } }
  main { padding:16px 18px 40px; }
  .toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; background:var(--card);
    border:1px solid var(--line); border-radius:12px; padding:10px 12px; margin-bottom:14px; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; }
  .tab { border:1px solid var(--line); background:#fff; border-radius:8px; padding:7px 12px;
    cursor:pointer; font:inherit; color:inherit; }
  .tab.on { background:var(--ink); color:#fff; border-color:var(--ink); }
  .comp-pills { display:flex; flex-wrap:wrap; gap:6px; margin-left:auto; align-items:center; }
  .hint { font-size:11px; color:var(--muted); margin-right:4px; }
  .cpill { border:1px solid var(--line); background:#fff; border-radius:999px; padding:5px 11px;
    cursor:pointer; font:600 12px/1 inherit; color:var(--muted); }
  .cpill.on { background:var(--accent); border-color:var(--accent); color:#fff; }
  .cpill.dim { opacity:.45; }
  .blurb { background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:12px 14px; margin-bottom:14px; color:var(--muted); font-size:13.5px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:8px; margin-bottom:14px; }
  .panel.pad { padding:12px 14px; }
  .grid2 { display:grid; grid-template-columns:1.1fr .9fr; gap:14px; }
  @media (max-width:1100px) {
    .layout { grid-template-columns:1fr; }
    .grid2 { grid-template-columns:1fr; }
    nav { position:relative; max-height:none; }
  }
  .plot { width:100%; height:420px; }
  .plot.tall { height:540px; }
  .plot.heat { height:560px; }
  .cap { color:var(--muted); font-size:12px; padding:4px 8px 8px; }
  .diag { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
  .diag .chip { background:var(--soft); border:1px solid var(--line); border-radius:8px;
    padding:6px 10px; font-size:12.5px; }
  .diag .chip b { color:var(--ink); }
  .diag .chip em { color:var(--accent); font-style:normal; font-weight:650; }
  table.tops, table.sc { width:100%; border-collapse:collapse; font-size:13px; }
  table.tops th, table.sc th { text-align:left; color:var(--muted); font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:.06em; padding:6px 8px; border-bottom:1px solid var(--line); }
  table.tops td, table.sc td { padding:6px 8px; border-bottom:1px solid #eef2f6; vertical-align:top; }
  table.tops tr:hover td, table.sc tr:hover td { background:#f7fafb; }
  .barwrap { background:#edf2f5; border-radius:99px; height:8px; overflow:hidden; min-width:70px; }
  .bar { height:100%; background:var(--accent); border-radius:99px; }
  .wnum { font-variant-numeric:tabular-nums; color:var(--muted); width:54px; text-align:right; }
  a { color:var(--rt); }
  a.x-link {
    color:var(--accent2, #1f4e79); font-weight:700; text-decoration:none;
    border-bottom:1px solid transparent;
  }
  a.x-link:hover { border-bottom-color:currentColor; }
  .pca-title { font:600 14px/1.3 inherit; padding:8px 10px 0; }
  .muted { color:var(--muted); font-size:12px; }
  .card { border:1px solid var(--line); border-radius:10px; padding:12px; margin:8px 0; background:var(--soft); }
  .card h3 { margin:0 0 6px; font-size:15px; }
</style>
</head>
<body>
<header>
  <h1>HEAT canicule — CP multi-tenseurs</h1>
  <p class="sub">Acteurs et objets révélés par chaque tenseur (agrégat multi-composantes + comportements corpus).
     <a href="rapport.html" style="color:#d8efed;font-weight:650">Rapport HTML complet</a>
     · <a href="rapports_index.html" style="color:#d8efed">Index des rapports</a></p>
  <div class="meta" id="meta"></div>
</header>
<div class="layout">
  <nav>
    <h2>Modalités</h2>
    <div id="mods"></div>
  </nav>
  <main>
    <div class="toolbar">
      <div class="tabs" id="tabs">
        <button class="tab on" data-view="actors">Acteurs révélés</button>
        <button class="tab" data-view="pca">PCA composantes</button>
        <button class="tab" data-view="comp">Fiche composante</button>
        <button class="tab" data-view="heat">Heatmap user × rang</button>
      </div>
      <div class="comp-pills" id="compPills"></div>
    </div>
    <div class="blurb" id="blurb"></div>
    <section id="view-actors">
      <div class="panel pad">
        <p class="muted" style="margin-top:0">Agrégat multi-composantes : comptes mis en avant (somme des poids CP),
           avec cibles RT / hashtags / domaines observés dans le corpus, et objets MID agrégés.</p>
        <div class="actor-grid">
          <div>
            <h3 style="margin:0 0 8px;font-size:14px">Top comptes révélés</h3>
            <div id="aggActors"></div>
          </div>
          <div>
            <h3 style="margin:0 0 8px;font-size:14px" id="aggMidTitle">Top objets (MID)</h3>
            <table class="tops" id="aggMidTable"><thead><tr><th>Objet</th><th></th><th class="wnum">Σ poids</th></tr></thead><tbody></tbody></table>
            <div id="aggMid2Wrap" hidden>
              <h3 style="margin:14px 0 8px;font-size:14px" id="aggMid2Title">Top MID₂</h3>
              <table class="tops" id="aggMid2Table"><thead><tr><th>Objet</th><th></th><th class="wnum">Σ poids</th></tr></thead><tbody></tbody></table>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section id="view-pca" hidden>
      <div class="panel">
        <div class="pca-title" id="pcaTitle"></div>
        <div id="pcaPlot" class="plot tall"></div>
        <p class="cap" id="pcaCap"></p>
      </div>
    </section>
    <section id="view-comp" hidden>
      <div class="panel pad" id="compDiag"></div>
      <div class="grid2">
        <div class="panel"><div id="timePlot" class="plot"></div><p class="cap">Temporalité du rang</p></div>
        <div class="panel pad" style="overflow:auto; max-height:520px">
          <table class="tops" id="userTable"><thead><tr><th>Acteurs</th><th></th><th class="wnum">poids</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
      <div class="grid2">
        <div class="panel pad" style="overflow:auto; max-height:420px">
          <table class="tops" id="midTable"><thead><tr><th id="midHead">Objets</th><th></th><th class="wnum">poids</th></tr></thead><tbody></tbody></table>
        </div>
        <div class="panel pad" style="overflow:auto; max-height:420px" id="mid2Wrap" hidden>
          <table class="tops" id="mid2Table"><thead><tr><th id="mid2Head">Cibles</th><th></th><th class="wnum">poids</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </section>
    <section id="view-heat" hidden>
      <div class="panel"><div id="heatPlot" class="plot heat"></div>
        <p class="cap">Users triés par masse — colonnes = rangs</p></div>
    </section>
  </main>
</div>
<script src="data.js"></script>
<script>
const DATA = window.HEAT_CP;
const COLORS = ["#0f6e6a","#1f4e79","#9a3412","#5b4b8a","#2f6b3a","#a15c00","#0e7490","#7f1d1d"];
let modKey = DATA.modalities[0].key;
let view = "actors";
let comp = 1;
let isolate = false;

function mod(){ return DATA.modalities.find(m => m.key === modKey); }
function fmt(n){ return Number(n).toLocaleString("fr-FR"); }

function tensorModes(m){
  if (m && m.modes) return m.modes;
  const t = (DATA.meta && (DATA.meta.time_bin_fr || DATA.meta.time_bin)) || "temps";
  const key = typeof m === "string" ? m : (m && m.key) || "";
  const map = {
    uut:`émetteur × cible RT × ${t}`, uurlt:`émetteur × URL × ${t}`,
    uudom:`émetteur × domaine × ${t}`, uht:`émetteur × hashtag × ${t}`,
    ubigram:`émetteur × n-gramme × ${t}`, utrigram:`émetteur × trigramme × ${t}`,
    ubigram_orig:`émetteur × n-gramme (hors RT) × ${t}`,
    utrigram_orig:`émetteur × trigramme (hors RT) × ${t}`,
    ucort:`émetteur × publication cible co-RT × ${t}`, ucd:`émetteur × jour création × ${t}`,
    ubio:`émetteur × n-gramme bio × ${t}`, ubio_ht:`émetteur × hashtag bio × ${t}`,
    uname:`émetteur × token display name × ${t}`,
    umed_orig:`émetteur × type média (hors RT) × ${t}`,
    umedurl_orig:`émetteur × URL média (hors RT) × ${t}`,
  };
  if (key.startsWith("coupled_") && !map[key]) {
    return m?.label || key;
  }
  return map[key] || m?.label || key;
}

function renderMods(){
  const box = document.getElementById("mods");
  box.innerHTML = DATA.modalities.map(m => {
    const top = (m.aggregate?.top_actors||[])[0];
    const modes = tensorModes(m);
    return `
    <button class="mod ${m.key===modKey?"on":""}" data-k="${m.key}" title="${modes}">
      <b>${modes}</b>
      <span><code>${m.key}</code> · ${fmt(m.n_users)} users · mid ${fmt(m.n_mid)} · R=${m.rank}</span>
      <div class="sc">top: ${top?top.label:"—"}</div>
    </button>`;
  }).join("");
  box.querySelectorAll(".mod").forEach(b => b.onclick = () => {
    modKey=b.dataset.k; comp=1; isolate=false; paint();
  });
}

function renderMeta(){
  const m = DATA.meta;
  document.getElementById("meta").innerHTML = [
    ["Fenêtre", m.window], ["Corpus", m.corpus], ["Pool", m.n_pool + " comptes"],
    ["Temps", m.time_bin_fr || m.time_bin || "—"],
    ["Rang", "R = " + m.rank], ["Lemmes", m.lemma], ["Algo", m.algo],
  ].map(([k,v]) => `<span class="pill"><strong>${k}</strong> ${v}</span>`).join("");
}

function renderPills(){
  const m = mod();
  const box = document.getElementById("compPills");
  if (view === "actors") { box.innerHTML = ""; return; }
  box.innerHTML = `<span class="hint">${isolate ? "isolé" : "clic = highlight · double-clic = isoler"}</span>` +
    Array.from({length:m.rank}, (_,i)=>i+1).map(r => {
      const on = r===comp ? "on" : (isolate ? "dim" : "");
      const lab = m.components[r-1]?.label ? ` title="${m.components[r-1].label.replace(/"/g,"&quot;")}"` : "";
      return `<button class="cpill ${on}" data-r="${r}"${lab}>C${r}</button>`;
    }).join("");
  box.querySelectorAll(".cpill").forEach(b => {
    b.onclick = () => { comp=+b.dataset.r; isolate=false; paint(); };
    b.ondblclick = (e) => { e.preventDefault(); comp=+b.dataset.r; isolate=true; paint(); };
  });
}

function xHandle(label){
  const m = String(label??"").trim().match(/^@([A-Za-z0-9_]{1,50})$/);
  return m ? m[1] : null;
}
function xLink(label){
  const h = xHandle(label);
  if (!h) return String(label??"");
  return `<a class="x-link" href="https://x.com/${encodeURIComponent(h)}" target="_blank" rel="noopener noreferrer">@${h}</a>`;
}

function fillTable(tbody, rows, maxW, wKey="w"){
  tbody.innerHTML = (rows||[]).map(r => {
    const lab = r.label || r.id;
    const wv = r[wKey] != null ? r[wKey] : r.w;
    const pct = maxW > 0 ? (100*wv/maxW) : 0;
    let link;
    if (xHandle(lab)) link = xLink(lab);
    else if (/[./]/.test(String(r.id||"")) && !String(lab).startsWith("@"))
      link = `<a href="https://${String(r.id).replace(/^https?:\/\//,"")}" target="_blank" rel="noopener">${lab}</a>`;
    else link = lab;
    return `<tr><td>${link}</td><td><div class="barwrap"><div class="bar" style="width:${pct}%"></div></div></td><td class="wnum">${Number(wv).toFixed(3)}</td></tr>`;
  }).join("");
}

function paintActors(){
  const m = mod();
  const agg = m.aggregate || {};
  const actors = agg.top_actors || [];
  document.getElementById("aggActors").innerHTML = actors.slice(0,20).map(a => {
    const comps = (a.in_comps||[]).join(", ");
    const rt = (a.top_rt_targets||[]).map(x=>`${xLink(x.label)} (${x.n})`).join(", ") || "—";
    const ht = (a.top_hashtags||[]).map(x=>`#${x.label} (${x.n})`).join(", ") || "—";
    const urls = (a.top_urls||a.top_domains||[]).map(x=>{
      const id = String(x.id||x.label||"");
      const href = id.match(/^https?:\/\//i) ? id : ("https://"+id.replace(/^\/+/,""));
      return `<a href="${href}" target="_blank" rel="noopener noreferrer">${x.label}</a> (${x.n})`;
    }).join(", ") || "—";
    return `<div class="actor-card">
      <h4>${xLink(a.label)} <span class="muted">Σw=${a.w_sum}</span></h4>
      <div class="meta">composante dominante C${a.dom_comp} · vu dans C[${comps}] ·
        tweets ${fmt(a.n_tweets)} (RT ${fmt(a.n_rt)} · propres ${fmt(a.n_orig)})</div>
      <ul>
        <li><b>Cibles RT</b> : ${rt}</li>
        <li><b>Hashtags</b> : ${ht}</li>
        <li><b>URLs</b> : ${urls}</li>
      </ul>
    </div>`;
  }).join("") || "<p class='muted'>Aucun acteur.</p>";

  document.getElementById("aggMidTitle").textContent = "Top " + (agg.mid_kind || "objets") + " (Σ composantes)";
  const mids = agg.top_mids || [];
  const mmax = Math.max(...mids.map(x=>x.w_sum), 1e-9);
  fillTable(document.querySelector("#aggMidTable tbody"), mids, mmax, "w_sum");
  const wrap = document.getElementById("aggMid2Wrap");
  if (agg.top_mids2 && agg.top_mids2.length) {
    wrap.hidden = false;
    document.getElementById("aggMid2Title").textContent = "Top " + (agg.mid2_kind || "MID₂");
    const m2max = Math.max(...agg.top_mids2.map(x=>x.w_sum), 1e-9);
    fillTable(document.querySelector("#aggMid2Table tbody"), agg.top_mids2, m2max, "w_sum");
  } else wrap.hidden = true;
}

function paintPca(){
  const m = mod();
  const pts = m.pca.points;
  const maxMass = Math.max(...pts.map(p => p.mass), 1e-9);
  const labelSet = new Set(m.pca.label_ids || []);
  const traces = [];
  for (let r=1; r<=m.rank; r++){
    const active = !isolate || r===comp;
    const hi = r===comp;
    const sub = pts.filter(p => p.dom === r);
    traces.push({
      type: "scattergl", mode: "markers", name: "C"+r,
      x: sub.map(p=>p.x), y: sub.map(p=>p.y),
      customdata: sub.map(p => ({ sn:p.sn, dom:p.dom, dom2:p.dom2, mass:p.mass,
        w:(p.weights||[]).map((x,i)=>`C${i+1}=${x}`).join(" · ") })),
      marker: {
        color: COLORS[(r-1)%COLORS.length],
        size: sub.map(p => 5 + 14 * Math.sqrt(p.mass / maxMass)),
        opacity: !active ? 0.08 : (hi ? 0.85 : 0.28),
        line: { width: hi ? 0.6 : 0, color: "#fff" }
      },
      hovertemplate: "@%{customdata.sn}<br>dominant C%{customdata.dom}<br>2ᵉ C%{customdata.dom2}<br>mass=%{customdata.mass}<br>%{customdata.w}<extra></extra>"
    });
  }
  const ann = pts.filter(p => labelSet.has(p.id) && p.dom===comp).slice(0,8).map(p => ({
    x:p.x, y:p.y, text:"@"+p.sn, showarrow:false, xanchor:"left", xshift:6,
    font:{size:11, color:"#15202b"}
  }));
  const ev = m.pca.explained || [];
  document.getElementById("pcaTitle").textContent =
    `PCA des facteurs user — PC1 ${((ev[0]||0)*100).toFixed(1)}% · PC2 ${((ev[1]||0)*100).toFixed(1)}%`;
  Plotly.newPlot("pcaPlot", traces, {
    margin:{t:20,r:20,b:40,l:40}, paper_bgcolor:"#fff", plot_bgcolor:"#fff",
    xaxis:{title:"PC1", zeroline:true, gridcolor:"#eef2f6"},
    yaxis:{title:"PC2", zeroline:true, gridcolor:"#eef2f6"},
    legend:{orientation:"h", y:1.08}, annotations:ann
  }, {responsive:true, displayModeBar:false});
  document.getElementById("pcaCap").textContent =
    `Couleur = composante dominante · taille = masse |C| · highlight C${comp}`;
}

function paintComp(){
  const m = mod();
  const c = m.components[comp-1];
  if (!c) return;
  const met = c.metrics || {};
  const behave = (c.users||[]).slice(0,5).map(u => {
    const rt = (u.top_rt_targets||[]).slice(0,3).map(x=>xLink(x.label)).join(", ") || "—";
    const ht = (u.top_hashtags||[]).slice(0,3).map(x=>"#"+x.label).join(", ") || "—";
    return `<div class="chip">${xLink(u.label)} RT→ ${rt} · ${ht}</div>`;
  }).join("");
  document.getElementById("compDiag").innerHTML = `
    <div class="diag">
      <div class="chip"><b>C${c.id}</b> <em>${c.label}</em></div>
      <div class="chip">Pic <b>${met.peak||"—"}</b></div>
      <div class="chip">Gini mid <b>${met.gini_mid??"—"}</b></div>
      <div class="chip">Gini users <b>${met.gini_users??"—"}</b></div>
      ${met.lag_days!=null?`<div class="chip">Lag <b>${met.lag_days} ${met.lag_unit||"j"}</b></div>`:""}
      ${behave}
    </div>`;
  const t1 = { type:"scatter", mode:"lines+markers", name:"couche 1",
    x:(c.time||[]).map(d=>d.day), y:(c.time||[]).map(d=>d.w),
    line:{color:COLORS[0], width:2}, marker:{size:6} };
  const traces = [t1];
  if (c.time2) traces.push({
    type:"scatter", mode:"lines+markers", name:"couche 2 (RT)",
    x:c.time2.map(d=>d.day), y:c.time2.map(d=>d.w),
    line:{color:COLORS[1], width:2, dash:"dot"}, marker:{size:6}
  });
  Plotly.newPlot("timePlot", traces, {
    margin:{t:10,r:10,b:40,l:40}, paper_bgcolor:"#fff", plot_bgcolor:"#fff",
    xaxis:{tickangle:-35}, yaxis:{title:"poids"}, showlegend:traces.length>1
  }, {responsive:true, displayModeBar:false});
  const umax = Math.max(...(c.users||[]).map(u=>u.w), 1e-9);
  fillTable(document.querySelector("#userTable tbody"), c.users, umax);
  document.getElementById("midHead").textContent = m.mid_label || "Objets";
  const mmax = Math.max(...(c.mid||[]).map(u=>u.w), 1e-9);
  fillTable(document.querySelector("#midTable tbody"), c.mid, mmax);
  const wrap = document.getElementById("mid2Wrap");
  if (c.mid2 && c.mid2.length) {
    wrap.hidden = false;
    document.getElementById("mid2Head").textContent = m.mid2_label || "Couche 2";
    const m2max = Math.max(...c.mid2.map(u=>u.w), 1e-9);
    fillTable(document.querySelector("#mid2Table tbody"), c.mid2, m2max);
  } else wrap.hidden = true;
}

function paintHeat(){
  const m = mod();
  const h = m.heatmap;
  Plotly.newPlot("heatPlot", [{
    type:"heatmap", z:h.z, y:h.user_labels,
    x: Array.from({length:m.rank},(_,i)=>"C"+(i+1)),
    colorscale:[[0,"#f7fafb"],[0.5,"#7eb8b4"],[1,"#0f6e6a"]],
    hovertemplate:"%{y} · %{x} = %{z}<extra></extra>"
  }], {
    margin:{t:10,r:20,b:40,l:120}, paper_bgcolor:"#fff",
    yaxis:{autorange:"reversed", tickfont:{size:10}}
  }, {responsive:true, displayModeBar:false});
}

function paint(){
  renderMods(); renderMeta(); renderPills();
  document.getElementById("blurb").textContent = mod().blurb;
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.dataset.view===view));
  ["actors","pca","comp","heat"].forEach(v => {
    document.getElementById("view-"+v).hidden = (v !== view);
  });
  if (view==="actors") paintActors();
  if (view==="pca") paintPca();
  if (view==="comp") paintComp();
  if (view==="heat") paintHeat();
}

document.getElementById("tabs").onclick = (e) => {
  const b = e.target.closest(".tab"); if (!b) return; view=b.dataset.view; paint();
};
paint();
</script>
</body>
</html>
"""


def write_html(out_dir: Path) -> None:
    (out_dir / "index.html").write_text(HTML_TMPL, encoding="utf-8")
    try:
        from build_rapport import write_reports

        paths = write_reports(out_dir)
        log(f"  rapports HTML : {len(paths)} fichiers (rapport.html + par modalité)")
    except Exception as e:
        log(f"  rapports HTML non générés : {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    time_bin = args.time_bin
    t_fr = time_bin_fr(time_bin)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV introuvable : {csv_path}")
    t0, t1, window_label = PHASES[args.phase]

    out_arg = Path(args.out).resolve()
    if out_arg == ROOT.resolve():
        out = ROOT / f"run_{args.phase.lower()}_{time_bin}_simple"
    else:
        out = out_arg
    out.mkdir(parents=True, exist_ok=True)

    log(f"=== CP simple · bin={time_bin} ({t_fr}) · {window_label} → {out.name} ===")
    pool = load_pool(
        csv_path,
        t0=t0,
        t1=t1,
        min_user_tweets=args.min_user_tweets,
        max_users=args.max_users,
    )
    pool.to_csv(out / "user_pool.csv", index=False)
    log(f"  pool → {len(pool)} comptes")

    bios = load_bios_for_pool(
        pool,
        bio_csv=Path(args.bios_csv) if args.bios_csv else DEFAULT_BIO_CSV,
        scrape=args.scrape_bios,
        max_scrape=args.max_scrape_bios,
    )
    tensors, sn_extra, tweet_author = build_all_tensors(
        csv_path,
        pool,
        t0=t0,
        t1=t1,
        min_count=args.min_count,
        max_urls=args.max_urls,
        max_domains=args.max_domains,
        max_hashtags=args.max_hashtags,
        max_targets=args.max_targets,
        max_bigrams=args.max_bigrams,
        max_trigrams=args.max_trigrams,
        min_users_ngram=args.min_users_ngram,
        max_cort=args.max_cort,
        min_users_cort=args.min_users_cort,
        with_ngrams=not args.no_ngrams,
        time_bin=time_bin,
        bios=bios,
    )
    sn = {str(u): str(s) for u, s in zip(pool["user_id"], pool["screen_name"])}
    sn.update(sn_extra)
    for uid, sname in sn_extra.items():
        sn.setdefault(str(uid), str(sname))
    # labels auteurs des publication cible co-RT
    if "ucort" in tensors and not tensors["ucort"].empty:
        for tid in tensors["ucort"]["status_id"].astype(str).unique():
            au = tweet_author.get(str(tid), "")
            if au:
                sn[f"author_of:{tid}"] = sn.get(au, au)
    profiles = collect_user_behaviors(
        csv_path,
        t0=t0,
        t1=t1,
        pool_ids=set(pool["user_id"].astype(str)),
        tweet_author=tweet_author,
    )
    for uid, bio in bios.items():
        if uid in profiles:
            profiles[uid]["bio"] = bio
    # labels pour cibles RT / mentions hors pool
    for p in profiles.values():
        for tid in list(p["rt_targets"]) + list(p.get("mentions") or []):
            if tid not in sn:
                if str(tid).startswith("sn:"):
                    sn[tid] = str(tid)[3:]
                else:
                    sn.setdefault(tid, tid)

    for name, df in tensors.items():
        if df.empty:
            log(f"  tensor {name}: VIDE")
            continue
        mode_cols = [c for c in df.columns if c not in {"user_id", "val"}]
        mid = mode_cols[0] if mode_cols else "?"
        df.to_csv(out / f"tensor_{name}.csv", index=False)
        if "tbin" in df.columns:
            third = f"{t_fr}s={df['tbin'].nunique():,}"
        elif len(mode_cols) >= 2:
            third = f"{mode_cols[1]}={df[mode_cols[1]].nunique():,}"
        else:
            third = "—"
        log(
            f"  tensor {name}: {len(df):,} nnz · users={df['user_id'].nunique():,} · "
            f"mid={df[mid].nunique():,} · {third}"
        )

    rank = max(5, int(args.rank))
    n_iter = int(args.iters)
    modalities: list[dict] = []
    encoded = {}

    # Simples : objets classiques + type de publication (hors RT / co-RT)
    specs = [
        ("uut", ("user_id", "retweeted_user_id", "tbin")),
        ("uurlt", ("user_id", "url", "tbin")),
        ("uudom", ("user_id", "domain", "tbin")),
        ("uht", ("user_id", "hashtag", "tbin")),
        ("ubigram", ("user_id", "ngram", "tbin")),
        ("utrigram", ("user_id", "trigram", "tbin")),
        ("ubigram_orig", ("user_id", "ngram", "tbin")),
        ("utrigram_orig", ("user_id", "trigram", "tbin")),
        ("uurlt_orig", ("user_id", "url", "tbin")),
        ("uudom_orig", ("user_id", "domain", "tbin")),
        ("uht_orig", ("user_id", "hashtag", "tbin")),
        ("ucort", ("user_id", "status_id", "tbin")),
        ("ucd", ("user_id", "creation_bin", "tbin")),
        ("ucdrt", ("user_id", "creation_bin", "retweeted_user_id")),
        ("ucdng", ("user_id", "creation_bin", "ngram")),
        ("ubio", ("user_id", "bio_ngram", "tbin")),
        ("ubio_ht", ("user_id", "bio_hashtag", "tbin")),
        ("uname", ("user_id", "name_token", "tbin")),
        ("umed_orig", ("user_id", "media_type", "tbin")),
        ("umedurl_orig", ("user_id", "media_url", "tbin")),
    ]
    for key, cols in specs:
        df = tensors.get(key)
        if df is None or df.empty or df["user_id"].nunique() < 20:
            log(f"  skip {key}: trop vide")
            continue
        log(f"CP-ALS {key} R={rank} · bin={time_bin}")
        enc = encode_triples(df, cols)
        encoded[key] = enc
        ii, jj, kk, vv, a_ids, b_ids, c_ids = enc
        A, B, C, lam = sparse_cp_als(
            ii, jj, kk, vv, (len(a_ids), len(b_ids), len(c_ids)), rank, n_iter, seed=42
        )
        pred = np.sum(A[ii] * B[jj] * C[kk] * lam, axis=1)
        err = float(np.linalg.norm(vv - pred) / (np.linalg.norm(vv) + 1e-12))
        fac_dir = out / key
        fac_dir.mkdir(exist_ok=True)
        pd.DataFrame(
            [{"entity": a_ids[i], "rank": r, "value": float(A[i, r])} for i in range(len(a_ids)) for r in range(rank)]
        ).to_csv(fac_dir / "user_explicit.csv", index=False)
        # Réseau : co-objets du mode 3 (cibles RT / n-grammes), pas les jours de création
        if key in {"ucdrt", "ucdng"}:
            umids = user_mid_index(df, cols[0], cols[2])
        else:
            umids = user_mid_index(df, cols[0], cols[1])
        modalities.append(
            build_modality_payload(
                key=key,
                user_ids=a_ids,
                A=A,
                mid_ids=b_ids,
                B=B,
                day_ids=c_ids,
                C=C,
                sn=sn,
                err_nnz=round(err, 4),
                time_bin=time_bin,
                profiles=profiles,
                bios=bios,
                user_mids=umids,
            )
        )

    # Couplés A–G (hiérarchie screen) + historiques ; chaque bras = USER × MID × temps
    default_coupled = [
        ("coupled_ubigram_uurlt", "ubigram", "uurlt", 18),          # A
        ("coupled_cort_ubigram_orig", "ubigram_orig", "ucort", 11),  # B
        ("coupled_ucort_uurlt_orig", "ucort", "uurlt_orig", 19),     # C (bras URL)
        ("coupled_umed_ubigram_orig", "umed_orig", "ubigram_orig", 17),  # D
        ("coupled_ucd_ubio", "ucd", "ubio", 20),                     # E
        ("coupled_uurlt_ucort", "uurlt", "ucort", 21),               # F
        ("coupled_uht_umed_orig", "uht", "umed_orig", 22),          # G
        ("coupled_cort_utrigram_orig", "utrigram_orig", "ucort", 12),
        ("coupled_ucd_uut", "ucd", "uut", 13),
        ("coupled_ucd_ucort", "ucd", "ucort", 14),
        ("coupled_ubio_ucort", "ubio", "ucort", 15),
        ("coupled_umed_ucort", "umed_orig", "ucort", 16),
    ]
    for coup_key, k1, k2, seed in default_coupled:
        if k1 not in encoded or k2 not in encoded:
            log(f"  skip {coup_key}: tenseurs manquants ({k1}/{k2})")
            continue
        log(f"CoupledALS {k1}×{k2} R={rank}")
        try:
            coup = sparse_coupled_als(encoded[k1], encoded[k2], rank, n_iter, seed=seed)
            df1 = tensors.get(k1)
            mid_col = [c for c in (df1.columns if df1 is not None else []) if c not in {"user_id", "tbin", "val"}]
            umids = user_mid_index(df1, "user_id", mid_col[0]) if df1 is not None and mid_col else None
            modalities.append(
                build_modality_payload(
                    key=coup_key,
                    user_ids=list(coup["users"]),
                    A=coup["A"],
                    mid_ids=list(coup["mid1"]),
                    B=coup["B"],
                    day_ids=list(coup["day1"]),
                    C=coup["C"],
                    sn=sn,
                    mid2_ids=list(coup["mid2"]),
                    D=coup["D"],
                    day2_ids=list(coup["day2"]),
                    E=coup["E"],
                    err_nnz=None,
                    time_bin=time_bin,
                    profiles=profiles,
                    bios=bios,
                    user_mids=umids,
                )
            )
        except Exception as e:
            log(f"  coupled {coup_key} failed: {e}")

    # Couplés historiques RT-cible (optionnels)
    if args.with_coupled:
        for coup_key, mid_key in (("coupled_uut_uurlt", "uurlt"), ("coupled_uut_ubigram", "ubigram")):
            if "uut" not in encoded or mid_key not in encoded:
                continue
            log(f"CoupledALS uut×{mid_key} R={rank}")
            try:
                coup = sparse_coupled_als(encoded[mid_key], encoded["uut"], rank, n_iter, seed=7)
                df1 = tensors.get(mid_key)
                mid_col = [c for c in (df1.columns if df1 is not None else []) if c not in {"user_id", "tbin", "val"}]
                umids = user_mid_index(df1, "user_id", mid_col[0]) if df1 is not None and mid_col else None
                modalities.append(
                    build_modality_payload(
                        key=coup_key,
                        user_ids=list(coup["users"]),
                        A=coup["A"],
                        mid_ids=list(coup["mid1"]),
                        B=coup["B"],
                        day_ids=list(coup["day1"]),
                        C=coup["C"],
                        sn=sn,
                        mid2_ids=list(coup["mid2"]),
                        D=coup["D"],
                        day2_ids=list(coup["day2"]),
                        E=coup["E"],
                        err_nnz=None,
                        time_bin=time_bin,
                        profiles=profiles,
                        bios=bios,
                        user_mids=umids,
                    )
                )
            except Exception as e:
                log(f"  coupled {coup_key} failed: {e}")

    # copy plotly into run dir if needed
    plotly_src = ROOT / "plotly.min.js"
    plotly_dst = out / "plotly.min.js"
    if plotly_src.is_file() and not plotly_dst.is_file():
        plotly_dst.write_bytes(plotly_src.read_bytes())

    payload = {
        "meta": {
            "corpus": "HEAT canicule (multi-langue)",
            "window": window_label,
            "phase": args.phase,
            "time_bin": time_bin,
            "time_bin_fr": t_fr,
            "t0": t0,
            "t1": t1,
            "n_pool": len(pool),
            "rank": rank,
            "lemma": "simplemma + colonne language (fallback fr)",
            "algo": "CP-ALS sparse · log1p + CoupledALS (co-RT × n-grammes hors RT)"
            + (" + CoupledALS RT-cible" if args.with_coupled else ""),
            "csv": str(csv_path),
        },
        "modalities": modalities,
    }
    (out / "data.js").write_text(
        "window.HEAT_CP = " + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8"
    )
    (out / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(out)
    cfg = {
        "csv": str(csv_path),
        "phase": args.phase,
        "time_bin": time_bin,
        "simple_only": False,
        "with_coupled_rt_cible": args.with_coupled,
        "window": window_label,
        "t0": t0,
        "t1": t1,
        "max_users": args.max_users,
        "rank": rank,
        "iters": n_iter,
        "modalities": [m["key"] for m in modalities],
        "err_nnz": {m["key"]: (m.get("stats") or {}).get("err_nnz") for m in modalities},
        "top_actor": {
            m["key"]: ((m.get("aggregate") or {}).get("top_actors") or [{}])[0].get("label")
            for m in modalities
        },
    }
    (out / "pipeline_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log(f"OK → {out / 'index.html'} ({len(modalities)} modalités simples, bin={time_bin})")
    for m in modalities:
        err = (m.get("stats") or {}).get("err_nnz")
        top = ((m.get("aggregate") or {}).get("top_actors") or [{}])[0].get("label", "—")
        err_s = f"err={err:.3f}" if err is not None else "err=—"
        log(f"  {m['key']:12s}  {err_s}  top_acteur={top}")


def enrich_existing_run(
    out_dir: Path,
    *,
    csv_path: Path | None = None,
    phase: str = "P1",
    top_users: int = 40,
    bios_csv: Path | None = None,
) -> None:
    """Réinjecte bio / mentions / tops dans un run existant (sans recalcul CP)."""
    out_dir = Path(out_dir)
    payload_path = out_dir / "payload.json"
    pool_path = out_dir / "user_pool.csv"
    cfg_path = out_dir / "pipeline_config.json"
    if not payload_path.is_file() or not pool_path.is_file():
        raise SystemExit(f"Run incomplet dans {out_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    csv_path = Path(csv_path or cfg.get("csv") or DEFAULT_CSV)
    phase = str(cfg.get("phase") or phase)
    t0, t1, _ = PHASES[phase]
    pool = pd.read_csv(pool_path, dtype={"user_id": str})
    pool_ids = set(pool["user_id"].astype(str))
    sn = {str(u): str(s) for u, s in zip(pool["user_id"], pool["screen_name"])}
    bios = load_bios_for_pool(
        pool,
        bio_csv=bios_csv or Path(cfg.get("bios_csv") or DEFAULT_BIO_CSV),
        scrape=False,
    )
    # auteurs RT pour résolution cibles
    tweet_author: dict[str, str] = {}
    log("Enrich — pass auteurs RT…")
    for chunk in pd.read_csv(
        csv_path, usecols=["tweet_id", "user_id", "user_screen_name"], chunksize=250_000, low_memory=False
    ):
        for tid, uid, sname in zip(chunk["tweet_id"], chunk["user_id"], chunk["user_screen_name"]):
            t, u = _id_str(tid), _id_str(uid)
            if t and u:
                tweet_author[t] = u
            if u and pd.notna(sname):
                sn.setdefault(u, str(sname).strip().lstrip("@"))
    profiles = collect_user_behaviors(
        csv_path, t0=t0, t1=t1, pool_ids=pool_ids, tweet_author=tweet_author
    )
    for uid, bio in bios.items():
        if uid in profiles:
            profiles[uid]["bio"] = bio
    for p in profiles.values():
        for tid in list(p["rt_targets"]) + list(p.get("mentions") or []):
            if tid not in sn and str(tid).startswith("sn:"):
                sn[tid] = str(tid)[3:]

    data = json.loads(payload_path.read_text(encoding="utf-8"))
    tensor_mid_cache: dict[str, dict[str, Counter]] = {}

    def _umids_for(key: str) -> dict[str, Counter] | None:
        # tenseur primaire pour les couplés
        src = {
            "coupled_cort_ubigram_orig": "ubigram_orig",
            "coupled_cort_utrigram_orig": "utrigram_orig",
            "coupled_ucd_uut": "ucd",
            "coupled_ucd_ucort": "ucd",
            "coupled_ucd_ubio": "ucd",
            "coupled_ubio_ucort": "ubio",
            "coupled_umed_ucort": "umed_orig",
            "coupled_umed_ubigram_orig": "umed_orig",
            "coupled_uut_uurlt": "uurlt",
            "coupled_uut_ubigram": "ubigram",
            "coupled_ubigram_uurlt": "ubigram",
            "coupled_ucort_uurlt_orig": "ucort",
            "coupled_uurlt_ucort": "uurlt",
            "coupled_uht_umed_orig": "uht",
        }.get(key, key)
        if src in tensor_mid_cache:
            return tensor_mid_cache[src]
        path = out_dir / f"tensor_{src}.csv"
        if not path.is_file():
            tensor_mid_cache[src] = {}
            return None
        tdf = pd.read_csv(path, low_memory=False)
        # ucdrt/ucdng : objets réseau = mode 3 (pas creation_bin)
        if src == "ucdrt" and "retweeted_user_id" in tdf.columns:
            um = user_mid_index(tdf, "user_id", "retweeted_user_id")
        elif src == "ucdng" and "ngram" in tdf.columns:
            um = user_mid_index(tdf, "user_id", "ngram")
        else:
            mid_col = [c for c in tdf.columns if c not in {"user_id", "tbin", "val"}]
            if not mid_col:
                tensor_mid_cache[src] = {}
                return None
            um = user_mid_index(tdf, "user_id", mid_col[0])
        tensor_mid_cache[src] = um
        return um

    for m in data.get("modalities") or []:
        key = m["key"]
        umids = _umids_for(key)
        fac = out_dir / key / "user_explicit.csv"
        if fac.is_file():
            df = pd.read_csv(fac, dtype={"entity": str})
            users = sorted(df["entity"].astype(str).unique().tolist())
            ranks = sorted(int(x) for x in df["rank"].unique().tolist())
            idx = {u: i for i, u in enumerate(users)}
            A = np.zeros((len(users), max(ranks) + 1), dtype=float)
            for ent, r, v in zip(df["entity"], df["rank"], df["value"]):
                A[idx[str(ent)], int(r)] = float(v)
            by_id = {int(c["id"]): c for c in (m.get("components") or [])}
            for r in ranks:
                rows = [
                    enrich_actor_metrics(u["id"], w=u["w"], sn=sn, profiles=profiles, bios=bios)
                    for u in top_entities(users, A, int(r), top_users)
                ]
                cid = int(r) + 1
                if cid in by_id:
                    by_id[cid]["users"] = rows
                    net_mid = by_id[cid].get("mid") or []
                    # ucdrt/ucdng : ne pas filtrer sur les jours de création (MID mode 2)
                    if key in {"ucdrt", "ucdng"}:
                        net_mid = []
                    by_id[cid]["network"] = build_component_network(
                        rows,
                        profiles,
                        sn,
                        key=key,
                        mid_items=net_mid,
                        user_mids=umids,
                    )
            m["components"] = [by_id[k] for k in sorted(by_id)]
        else:
            # couplés / sans facteurs disque : enrichir users existants
            for c in m.get("components") or []:
                enriched = []
                for u in c.get("users") or []:
                    enriched.append(
                        enrich_actor_metrics(
                            u["id"], w=float(u.get("w") or 0), sn=sn, profiles=profiles, bios=bios
                        )
                    )
                c["users"] = enriched
                net_mid = c.get("mid") or []
                if key in {"ucdrt", "ucdng"}:
                    net_mid = []
                c["network"] = build_component_network(
                    enriched,
                    profiles,
                    sn,
                    key=key,
                    mid_items=net_mid,
                    user_mids=umids,
                )
        # agrégat acteurs
        agg = m.get("aggregate") or {}
        actors = []
        for a in agg.get("top_actors") or []:
            row = enrich_actor_metrics(
                a["id"], w=float(a.get("w_sum") or a.get("w") or 0), sn=sn, profiles=profiles, bios=bios
            )
            row["w_sum"] = round(float(a.get("w_sum") or row["w"]), 4)
            row.pop("w", None)
            row["dom_comp"] = a.get("dom_comp")
            row["in_comps"] = a.get("in_comps") or []
            actors.append(row)
        if actors:
            agg["top_actors"] = actors
            m["aggregate"] = agg
        log(f"  enrich {key}: users/comp ≈ {len((m.get('components') or [{}])[0].get('users') or [])}")

    payload_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (out_dir / "data.js").write_text(
        "window.HEAT_CP = " + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8"
    )
    try:
        from build_rapport import write_reports

        write_reports(out_dir)
    except Exception as e:
        log(f"  rapports non régénérés: {e}")
    log(f"Enrich OK → {out_dir}")


# Specs encode pour tenseurs simples (colonnes CSV)
_TENSOR_COLS = {
    "uut": ("user_id", "retweeted_user_id", "tbin"),
    "uurlt": ("user_id", "url", "tbin"),
    "uudom": ("user_id", "domain", "tbin"),
    "uht": ("user_id", "hashtag", "tbin"),
    "ubigram": ("user_id", "ngram", "tbin"),
    "utrigram": ("user_id", "trigram", "tbin"),
    "ubigram_orig": ("user_id", "ngram", "tbin"),
    "utrigram_orig": ("user_id", "trigram", "tbin"),
    "uurlt_orig": ("user_id", "url", "tbin"),
    "uudom_orig": ("user_id", "domain", "tbin"),
    "uht_orig": ("user_id", "hashtag", "tbin"),
    "ucort": ("user_id", "status_id", "tbin"),
    "ucd": ("user_id", "creation_bin", "tbin"),
    "ubio": ("user_id", "bio_ngram", "tbin"),
    "umed_orig": ("user_id", "media_type", "tbin"),
}

# Couplés A–G (+ manquants) à injecter dans un run existant
HIERARCHY_COUPLED = [
    ("coupled_ubigram_uurlt", "ubigram", "uurlt", 18),
    ("coupled_cort_ubigram_orig", "ubigram_orig", "ucort", 11),
    ("coupled_ucort_uurlt_orig", "ucort", "uurlt_orig", 19),
    ("coupled_umed_ubigram_orig", "umed_orig", "ubigram_orig", 17),
    ("coupled_ucd_ubio", "ucd", "ubio", 20),
    ("coupled_uurlt_ucort", "uurlt", "ucort", 21),
    ("coupled_uht_umed_orig", "uht", "umed_orig", 22),
]


def add_hierarchy_coupled(
    out_dir: Path,
    *,
    csv_path: Path | None = None,
    phase: str = "P1",
    rank: int | None = None,
    n_iter: int | None = None,
    only_missing: bool = True,
) -> None:
    """CoupledALS A–G depuis tensor_*.csv d’un run existant ; merge payload + rapports."""
    out_dir = Path(out_dir)
    payload_path = out_dir / "payload.json"
    pool_path = out_dir / "user_pool.csv"
    cfg_path = out_dir / "pipeline_config.json"
    if not payload_path.is_file() or not pool_path.is_file():
        raise SystemExit(f"Run incomplet dans {out_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    csv_path = Path(csv_path or cfg.get("csv") or DEFAULT_CSV)
    phase = str(cfg.get("phase") or phase)
    t0, t1, _ = PHASES[phase]
    time_bin = str(cfg.get("time_bin") or "day")
    rank = int(rank if rank is not None else cfg.get("rank") or 5)
    n_iter = int(n_iter if n_iter is not None else cfg.get("iters") or 35)

    data = json.loads(payload_path.read_text(encoding="utf-8"))
    existing = {m["key"] for m in (data.get("modalities") or [])}
    wanted = [
        (ck, k1, k2, seed)
        for ck, k1, k2, seed in HIERARCHY_COUPLED
        if (not only_missing) or ck not in existing
    ]
    if not wanted:
        log("Aucun couplé A–G manquant — rien à faire.")
        return

    pool = pd.read_csv(pool_path, dtype={"user_id": str})
    pool_ids = set(pool["user_id"].astype(str))
    sn = {str(u): str(s) for u, s in zip(pool["user_id"], pool["screen_name"])}
    bios = load_bios_for_pool(pool, bio_csv=DEFAULT_BIO_CSV, scrape=False)
    tweet_author: dict[str, str] = {}
    log("Hierarchy coupled — auteurs RT…")
    for chunk in pd.read_csv(
        csv_path, usecols=["tweet_id", "user_id", "user_screen_name"], chunksize=250_000, low_memory=False
    ):
        for tid, uid, sname in zip(chunk["tweet_id"], chunk["user_id"], chunk["user_screen_name"]):
            t, u = _id_str(tid), _id_str(uid)
            if t and u:
                tweet_author[t] = u
            if u and pd.notna(sname):
                sn.setdefault(u, str(sname).strip().lstrip("@"))
    profiles = collect_user_behaviors(
        csv_path, t0=t0, t1=t1, pool_ids=pool_ids, tweet_author=tweet_author
    )
    for uid, bio in bios.items():
        if uid in profiles:
            profiles[uid]["bio"] = bio

    need_keys = sorted({k for _, k1, k2, _ in wanted for k in (k1, k2)})
    encoded: dict = {}
    tensors_df: dict = {}
    for key in need_keys:
        path = out_dir / f"tensor_{key}.csv"
        cols = _TENSOR_COLS.get(key)
        if not path.is_file() or not cols:
            log(f"  skip encode {key}: CSV/cols manquants")
            continue
        df = pd.read_csv(path, low_memory=False)
        if key == "ucort":
            for tid in df["status_id"].astype(str).unique():
                au = tweet_author.get(str(tid), "")
                if au:
                    sn[f"author_of:{tid}"] = sn.get(au, au)
        tensors_df[key] = df
        encoded[key] = encode_triples(df, cols)
        log(f"  encode {key}: nnz={len(df):,}")

    new_mods: list[dict] = []
    for coup_key, k1, k2, seed in wanted:
        if k1 not in encoded or k2 not in encoded:
            log(f"  skip {coup_key}: tenseurs manquants ({k1}/{k2})")
            continue
        log(f"CoupledALS {coup_key} ({k1}∥{k2}) R={rank}")
        try:
            coup = sparse_coupled_als(encoded[k1], encoded[k2], rank, n_iter, seed=seed)
            df1 = tensors_df.get(k1)
            mid_col = [c for c in (df1.columns if df1 is not None else []) if c not in {"user_id", "tbin", "val"}]
            umids = user_mid_index(df1, "user_id", mid_col[0]) if df1 is not None and mid_col else None
            mod = build_modality_payload(
                key=coup_key,
                user_ids=list(coup["users"]),
                A=coup["A"],
                mid_ids=list(coup["mid1"]),
                B=coup["B"],
                day_ids=list(coup["day1"]),
                C=coup["C"],
                sn=sn,
                mid2_ids=list(coup["mid2"]),
                D=coup["D"],
                day2_ids=list(coup["day2"]),
                E=coup["E"],
                err_nnz=None,
                time_bin=time_bin,
                profiles=profiles,
                bios=bios,
                user_mids=umids,
            )
            new_mods.append(mod)
            top = ((mod.get("aggregate") or {}).get("top_actors") or [{}])[0].get("label", "—")
            log(f"  OK {coup_key} · users={mod['n_users']} · top={top}")
        except Exception as e:
            log(f"  coupled {coup_key} failed: {e}")

    if not new_mods:
        log("Aucun nouveau couplé produit.")
        return

    # Remplacer si déjà présent, sinon insérer en tête des couplés (ordre A–G)
    by_key = {m["key"]: m for m in data["modalities"]}
    for m in new_mods:
        by_key[m["key"]] = m
    # Rafraîchir libellés A–G même pour couplés déjà calculés (B, D…)
    meta_bin = modality_meta_for(time_bin)
    for ck, *_ in HIERARCHY_COUPLED:
        if ck in by_key and ck in meta_bin:
            mm = meta_bin[ck]
            by_key[ck]["label"] = mm["label"]
            by_key[ck]["blurb"] = mm["blurb"]
            by_key[ck]["modes"] = mm["modes"]
            by_key[ck]["mid_label"] = mm.get("mid_label")
            by_key[ck]["mid2_label"] = mm.get("mid2_label")
    order_pref = [ck for ck, *_ in HIERARCHY_COUPLED]
    simples = [m for m in data["modalities"] if not str(m["key"]).startswith("coupled")]
    coupled_rest = [
        by_key[k]
        for k in by_key
        if str(k).startswith("coupled") and k not in order_pref
    ]
    coupled_hier = [by_key[k] for k in order_pref if k in by_key]
    data["modalities"] = simples + coupled_hier + coupled_rest
    if data.get("meta"):
        data["meta"]["algo"] = (
            "CP-ALS sparse · log1p + CoupledALS (hiérarchie A–G USER×temps)"
        )

    payload_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # Préserver HEAT_CP_HOUR s’il existe
    hour_blob = ""
    data_js = out_dir / "data.js"
    if data_js.is_file():
        txt = data_js.read_text(encoding="utf-8")
        if "window.HEAT_CP_HOUR" in txt:
            hour_blob = "\n" + txt[txt.index("window.HEAT_CP_HOUR") :]
    data_js.write_text(
        "window.HEAT_CP = " + json.dumps(data, ensure_ascii=False) + ";" + hour_blob,
        encoding="utf-8",
    )
    cfg["modalities"] = [m["key"] for m in data["modalities"]]
    cfg["top_actor"] = {
        m["key"]: ((m.get("aggregate") or {}).get("top_actors") or [{}])[0].get("label")
        for m in data["modalities"]
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        from build_rapport import write_reports

        write_reports(out_dir)
    except Exception as e:
        log(f"  rapports non régénérés: {e}")
    log(f"Hierarchy coupled OK → +{len(new_mods)} modalités dans {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HEAT CP multi-tenseurs → HTML")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=ROOT, help="Dossier sortie (défaut: run_<phase>_<bin>_simple)")
    p.add_argument("--phase", choices=list(PHASES), default="P1")
    p.add_argument("--time-bin", choices=["day", "hour", "minute"], default="day", dest="time_bin")
    p.add_argument(
        "--enrich-profiles",
        action="store_true",
        help="Réinjecte bio/mentions/tops dans un run existant (sans CP)",
    )
    p.add_argument(
        "--add-hierarchy-coupled",
        action="store_true",
        help="Ajoute les couplés A–G (screen) depuis tensor_*.csv d’un run existant",
    )
    p.add_argument(
        "--with-coupled",
        action="store_true",
        help="Ajoute aussi les couplés RT-cible historiques (URL∥uut, bigramme∥uut)",
    )
    p.add_argument("--rank", type=int, default=5)
    p.add_argument("--iters", type=int, default=35)
    p.add_argument("--max-users", type=int, default=2000)
    p.add_argument("--min-user-tweets", type=int, default=3)
    p.add_argument("--min-count", type=float, default=1.0)
    p.add_argument("--max-urls", type=int, default=2000)
    p.add_argument("--max-domains", type=int, default=800)
    p.add_argument("--max-hashtags", type=int, default=800)
    p.add_argument("--max-targets", type=int, default=2500)
    p.add_argument("--max-cort", type=int, default=2500, help="Top publication cible co-RT (MID)")
    p.add_argument(
        "--min-users-cort",
        type=int,
        default=2,
        help="Min comptes du pool ayant RT le même status",
    )
    p.add_argument("--max-bigrams", type=int, default=2000)
    p.add_argument("--max-trigrams", type=int, default=1500)
    p.add_argument("--min-users-ngram", type=int, default=3)
    p.add_argument("--no-ngrams", action="store_true")
    p.add_argument("--bios-csv", type=str, default=str(DEFAULT_BIO_CSV))
    p.add_argument(
        "--scrape-bios",
        action="store_true",
        help="Complète les bios manquantes du pool via api.fxtwitter.com (cache local)",
    )
    p.add_argument("--max-scrape-bios", type=int, default=600)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out).resolve()
    if out == ROOT.resolve():
        out = ROOT / f"run_{args.phase.lower()}_{args.time_bin}_simple"
    if args.enrich_profiles:
        enrich_existing_run(
            out,
            csv_path=args.csv,
            phase=args.phase,
            bios_csv=Path(args.bios_csv) if args.bios_csv else DEFAULT_BIO_CSV,
        )
    elif args.add_hierarchy_coupled:
        add_hierarchy_coupled(
            out,
            csv_path=args.csv,
            phase=args.phase,
            rank=args.rank,
            n_iter=args.iters,
        )
    else:
        run(args)
