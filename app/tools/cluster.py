# app/tools/cluster.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd

# sklearn: metric='cosine' у нових версіях, affinity='cosine' у старіших
from sklearn.cluster import AgglomerativeClustering

def _agglomerative_cosine(distance_threshold: float):
    try:
        return AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )
    except TypeError:  # старіші sklearn
        return AgglomerativeClustering(
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None,
        )

def cluster_embeddings(
    X: np.ndarray,
    *,
    sim_threshold: float = 0.34,   # 0..1 (cosine). 0.34 ~ помірно щільні кластери
    min_cluster_size: int = 3,     # дрібні кластери відправляємо в -1 (misc)
) -> np.ndarray:
    """
    Повертає масив labels (0..K-1, -1 для дрібних/шуму).
    Очікує, що X нормалізовані (cosine == dot).
    """
    assert X.ndim == 2
    dist_thr = max(0.0, 1.0 - float(sim_threshold))
    model = _agglomerative_cosine(dist_thr)
    labels = model.fit_predict(X)

    # зливаємо кластери менш ніж min_cluster_size у -1
    out = labels.copy()
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < min_cluster_size:
            out[idx] = -1
    return out

def summarize_clusters(
    df: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    topk_quotes: int = 3,
) -> pd.DataFrame:
    """
    Будує коротку таблицю по кластерах:
    topic_id, size, share, example_comment_ids, example_quotes
    """
    n = len(df)
    rows = []
    # гарантуємо нормалізацію (для косоїдної схожості dot працює)
    # X вже нормалізовано у embeddings.embed_texts(normalize=True)

    for lab in sorted([l for l in np.unique(labels) if l != -1], key=int):
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        # центральність = схожість з центроїдом
        centroid = X[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = X[idx] @ centroid
        order = idx[np.argsort(-sims)]  # найближчі до центру

        ex_rows = df.iloc[order[:topk_quotes]]
        rows.append({
            "topic_id": int(lab),
            "size": int(len(idx)),
            "share": round(len(idx) / n, 4),
            "example_comment_ids": ex_rows["comment_id"].tolist(),
            "example_quotes": ex_rows["text_clean"].tolist(),
        })

    out = pd.DataFrame(rows).sort_values(["size","topic_id"], ascending=[False, True]).reset_index(drop=True)
    return out

def attach_cluster_columns(
    df: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Додає в df колонки: cluster, centrality (для сортування прикладів).
    """
    out = df.copy()
    out["cluster"] = labels
    # centrality: схожість із центроїдом власного кластера; для -1 — 0
    cent = np.zeros(len(out), dtype=float)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if lab == -1 or len(idx) == 0:
            continue
        centroid = X[idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        cent[idx] = (X[idx] @ centroid)
    out["centrality"] = cent
    return out
