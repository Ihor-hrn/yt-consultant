# app/tools/embeddings.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import List, Sequence
import numpy as np

from sentence_transformers import SentenceTransformer

# простий кеш моделі (в межах процесу)
_MODEL = None
_MODEL_NAME = None

def _maybe_instruction_wrap(texts: Sequence[str], model_name: str) -> List[str]:
    """
    Для e5 бажано додати префікс 'passage: '. Для bge-m3 можна без префікса.
    """
    name = model_name.lower()
    if "e5" in name:
        return [f"passage: {t}" for t in texts]
    return list(texts)

def get_embedder(model_name: str | None = None) -> SentenceTransformer:
    global _MODEL, _MODEL_NAME
    model_name = model_name or os.getenv("EMB_MODEL", "BAAI/bge-m3")
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    _MODEL = SentenceTransformer(model_name)
    _MODEL_NAME = model_name
    return _MODEL

def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    model = get_embedder(model_name)
    wrapped = _maybe_instruction_wrap(texts, _MODEL_NAME or (model_name or ""))
    vecs = model.encode(
        wrapped,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    return vecs
