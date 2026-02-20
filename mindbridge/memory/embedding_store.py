import math
import os
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY


EMBEDDING_MODEL = "text-embedding-3-small"
_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def embed_text(text: str) -> list[float]:
    payload = text.strip()
    if not payload:
        return []

    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=payload,
    )
    vector: Any = response.data[0].embedding
    return [float(value) for value in vector]


def compute_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) != len(vec2):
        return 0.0

    dot = sum(left * right for left, right in zip(vec1, vec2))
    norm1 = math.sqrt(sum(value * value for value in vec1))
    norm2 = math.sqrt(sum(value * value for value in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)
