# app/tools/preprocess.py
from __future__ import annotations
import re, hashlib
from typing import Iterable, Tuple, Literal, Dict, Any
import pandas as pd

try:
    from logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("pre")

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_WS = re.compile(r"\s+")
RE_EMOJI = re.compile(
    "[" "\U0001F300-\U0001F6FF" "\U0001F900-\U0001F9FF" "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF" "\U00002600-\U000026FF" "]+", flags=re.UNICODE
)

STOPWORDS_UK = {"і","й","та","або","але","що","це","я","ти","ви","ми","він","вона","воно","вони","же","би","не",
                "на","до","від","у","в","з","із","за","як","то","тільки","лише","ще","дуже"}
STOPWORDS_RU = {"и","а","но","что","это","я","ты","вы","мы","он","она","оно","они","же","бы","не",
                "на","до","от","у","в","с","со","за","как","то","только","лишь","ещё","очень"}
STOPWORDS_EN = {"the","a","an","and","or","but","that","this","it","i","you","we","they","he","she",
                "to","of","in","on","at","for","from","by","as","is","are","was","were","be","been","being"}

def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.replace("\u200b", "")
    t = RE_URL.sub(" ", t)
    t = RE_EMOJI.sub(" ", t)
    t = re.sub(r"[<>]", " ", t)
    t = RE_WS.sub(" ", t).strip()
    return t

def _normalize_for_hash(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = RE_WS.sub(" ", t).strip()
    return t

def text_hash(t: str) -> str:
    return hashlib.sha1(_normalize_for_hash(t).encode("utf-8")).hexdigest()

def detect_lang_series(texts: Iterable[str]) -> pd.Series:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    langs = []
    for s in texts:
        try:
            langs.append(detect(s) if isinstance(s, str) and len(s) >= 10 else "unknown")
        except Exception:
            langs.append("unknown")
    return pd.Series(langs)

def _is_mostly_stopwords(tokens: list[str], lang: str) -> bool:
    if not tokens:
        return True
    sw = STOPWORDS_EN if lang == "en" else STOPWORDS_UK if lang == "uk" else STOPWORDS_RU
    non_sw = [t for t in tokens if t.lower() not in sw and len(t) > 2]
    return len(non_sw) <= 1

def flag_spam_rule_based(s: str, *, aggressive_stopword_check: bool = False) -> int:
    if not s or len(s) < 6:
        return 1
    if re.search(r"(.)\1{6,}", s):
        return 1
    urls = RE_URL.findall(s)
    if len(urls) >= 1 and len(clean_text(s)) < 12:
        return 1
    if aggressive_stopword_check:
        toks = re.findall(r"\w+", s, flags=re.UNICODE)
        if _is_mostly_stopwords(toks, "uk") and _is_mostly_stopwords(toks, "ru") and _is_mostly_stopwords(toks, "en"):
            return 1
    return 0

def select_fast_batch(
    df: pd.DataFrame,
    *,
    mode: Literal["top_likes","newest"] = "top_likes",
    limit: int = 1200,
    include_replies: bool = False,
) -> pd.DataFrame:
    x = df.copy()
    if not include_replies and "is_reply" in x.columns:
        x = x[x["is_reply"] == 0]
    if mode == "top_likes":
        x = x.sort_values(["like_count","published_at"], ascending=[False, True])
    else:
        x = x.sort_values("published_at", ascending=False)
    return x.head(limit).reset_index(drop=True)

def preprocess_comments_df(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    min_chars: int = 12,  # м'якше для YouTube
    keep_langs: Tuple[str, ...] = ("uk", "ru", "en", "pl", "cs", "sk"),  # додали сусідні мови
    drop_spam: bool = True,
    deduplicate: bool = True,
    aggressive_stopword_check: bool = False,
    return_debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основний конвеєр. Якщо return_debug=True — повертає (df_clean, debug_df) із причинами відсіву.
    """
    x = df.copy()
    debug: Dict[str, Any] = {
        "n_in": len(x),
        "after_minlen": None,
        "after_lang": None,
        "after_spam": None,
        "after_dedup": None,
        "lang_counts": None,
        "dropped_reason": {"minlen":0,"lang":0,"spam":0,"dup":0}
    }

    # 1) clean
    x["text_clean"] = x[text_col].astype(str).map(clean_text)

    # 2) min length
    mask_len = x["text_clean"].str.len() >= min_chars
    debug["dropped_reason"]["minlen"] = int((~mask_len).sum())
    x = x[mask_len]
    debug["after_minlen"] = len(x)

    # 3) language
    x["lang"] = detect_lang_series(x["text_clean"].tolist())
    debug["lang_counts"] = x["lang"].value_counts().to_dict()
    if keep_langs:
        allowed = set(keep_langs) | {"unknown"}
        mask_lang = x["lang"].isin(allowed)
        debug["dropped_reason"]["lang"] = int((~mask_lang).sum())
        x = x[mask_lang]
    debug["after_lang"] = len(x)

    # 4) spam
    if drop_spam:
        spam_flags = x["text_clean"].map(lambda s: flag_spam_rule_based(s, aggressive_stopword_check=aggressive_stopword_check))
        debug["dropped_reason"]["spam"] = int(spam_flags.sum())
        x = x[spam_flags == 0]
    debug["after_spam"] = len(x)

    # 5) dedup
    if deduplicate:
        x["text_hash"] = x["text_clean"].map(text_hash)
        before = len(x)
        x = x.drop_duplicates(subset=["text_hash"])
        debug["dropped_reason"]["dup"] = int(before - len(x))
    debug["after_dedup"] = len(x)

    cols_keep = [
        "video_id","comment_id","parent_id","author","author_channel_id",
        "text","text_clean","like_count","reply_count","published_at","updated_at",
        "is_reply","lang"
    ]
    x = x[[c for c in cols_keep if c in x.columns]].reset_index(drop=True)

    if not return_debug:
        return x
    debug_df = pd.DataFrame([{
        "n_in": debug["n_in"],
        "after_minlen": debug["after_minlen"],
        "after_lang": debug["after_lang"],
        "after_spam": debug["after_spam"],
        "after_dedup": debug["after_dedup"],
        "dropped_minlen": debug["dropped_reason"]["minlen"],
        "dropped_lang": debug["dropped_reason"]["lang"],
        "dropped_spam": debug["dropped_reason"]["spam"],
        "dropped_dup": debug["dropped_reason"]["dup"],
        "lang_counts": debug["lang_counts"],
    }])
    return x, debug_df
