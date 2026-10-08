"""Sentence embeddings (MiniLM) with cpu default and optional mps."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .common import aspect_names, ontology

_DEVICE = {"requested": "cpu", "actual": None, "reason": None}


def configure(device: str = "cpu") -> None:
    _DEVICE["requested"] = device


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    dev = _DEVICE["requested"]
    if dev == "mps":
        import torch
        if not torch.backends.mps.is_available():
            dev, _DEVICE["reason"] = "cpu", "mps unavailable"
    _DEVICE["actual"] = dev
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)


def device_info() -> dict:
    return dict(_DEVICE)


def encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return model().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def aspect_matrix() -> np.ndarray:
    specs = ontology()["aspects"]
    return encode([f"{a.replace('_', ' ')}: {specs[a]['description']}" for a in aspect_names()])
