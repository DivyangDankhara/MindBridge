import json
import tempfile
from pathlib import Path
from typing import Any

from memory.embedding_store import compute_cosine_similarity, embed_text
from memory.procedural_store import PROCEDURAL_FILE, ensure_procedural_file, load_procedural_strategies
from memory.schema import MissionExperience
from memory.semantic_store import SEMANTIC_FILE, ensure_semantic_file, load_semantic_rules
from memory.store import EPISODIC_FILE, ensure_episodic_file


def _normalize_embedding(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None

    vector: list[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            vector.append(float(item))
            continue
        return None
    return vector if vector else None


def _rewrite_jsonl_safely(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
        temp_path = Path(handle.name)

    temp_path.replace(path)


def _semantic_embedding_text(rule: dict[str, Any]) -> str:
    return f"{rule.get('rule', '')} {rule.get('context', '')}".strip()


def _procedural_embedding_text(strategy: dict[str, Any]) -> str:
    steps = strategy.get("steps_template", [])
    steps_text = ""
    if isinstance(steps, list):
        steps_text = " ".join(step for step in steps if isinstance(step, str))
    return (
        f"{strategy.get('strategy_name', '')} "
        f"{strategy.get('applicable_context', '')} "
        f"{steps_text}"
    ).strip()


def _episodic_embedding_text(experience: dict[str, Any]) -> str:
    return (
        f"task: {experience.get('intent_task', '')} "
        f"goal: {experience.get('intent_goal', '')} "
        f"plan: {experience.get('final_plan_summary', '')} "
        f"result: {experience.get('result_summary', '')}"
    ).strip()


def _ensure_embedding(entry: dict[str, Any], text: str) -> tuple[list[float], bool]:
    current_embedding = _normalize_embedding(entry.get("embedding"))
    if current_embedding is not None:
        return current_embedding, False

    try:
        embedding = embed_text(text)
    except Exception:
        return [], False
    if embedding:
        entry["embedding"] = embedding
    return embedding, True


def _load_episodic_records() -> list[dict[str, Any]]:
    ensure_episodic_file()
    records: list[dict[str, Any]] = []
    with EPISODIC_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
                MissionExperience.model_validate(parsed)
            except Exception:
                continue

            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def _semantic_candidates(query_embedding: list[float]) -> list[dict[str, Any]]:
    ensure_semantic_file()
    rules = load_semantic_rules()
    changed = False

    candidates: list[dict[str, Any]] = []
    for rule in rules:
        text = _semantic_embedding_text(rule)
        embedding, updated = _ensure_embedding(rule, text)
        if updated:
            changed = True

        confidence = float(rule.get("confidence", 0.0))
        derived_from = int(rule.get("derived_from", 0))
        similarity = compute_cosine_similarity(query_embedding, embedding)

        candidates.append(
            {
                "memory_type": "semantic",
                "confidence": confidence,
                "derived_from": max(0, derived_from),
                "similarity_score": similarity,
                "summary": text,
                "data": rule,
            }
        )

    if changed:
        _rewrite_jsonl_safely(SEMANTIC_FILE, rules)

    return candidates


def _procedural_candidates(query_embedding: list[float]) -> list[dict[str, Any]]:
    ensure_procedural_file()
    strategies = load_procedural_strategies()
    changed = False

    candidates: list[dict[str, Any]] = []
    for strategy in strategies:
        text = _procedural_embedding_text(strategy)
        embedding, updated = _ensure_embedding(strategy, text)
        if updated:
            changed = True

        confidence = float(strategy.get("confidence", 0.0))
        derived_from = int(strategy.get("derived_from", 0))
        similarity = compute_cosine_similarity(query_embedding, embedding)

        candidates.append(
            {
                "memory_type": "procedural",
                "confidence": confidence,
                "derived_from": max(0, derived_from),
                "similarity_score": similarity,
                "summary": text,
                "data": strategy,
            }
        )

    if changed:
        _rewrite_jsonl_safely(PROCEDURAL_FILE, strategies)

    return candidates


def _episodic_candidates(query_embedding: list[float]) -> list[dict[str, Any]]:
    records = _load_episodic_records()
    changed = False

    candidates: list[dict[str, Any]] = []
    for record in records:
        text = _episodic_embedding_text(record)
        embedding, updated = _ensure_embedding(record, text)
        if updated:
            changed = True

        attempts = int(record.get("attempts", 0)) if isinstance(record.get("attempts"), int) else 0
        success = bool(record.get("success", False))
        confidence = 0.9 if success else 0.4
        similarity = compute_cosine_similarity(query_embedding, embedding)

        candidates.append(
            {
                "memory_type": "episodic",
                "confidence": confidence,
                "derived_from": max(0, attempts),
                "similarity_score": similarity,
                "summary": text,
                "data": record,
            }
        )

    if changed:
        _rewrite_jsonl_safely(EPISODIC_FILE, records)

    return candidates


def retrieve_relevant_memory(task: str, goal: str, top_k: int = 5) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []

    query_text = f"{task.strip()} {goal.strip()}".strip()
    if not query_text:
        return []

    try:
        query_embedding = embed_text(query_text)
    except Exception:
        return []
    if not query_embedding:
        return []

    combined = [
        *_semantic_candidates(query_embedding),
        *_procedural_candidates(query_embedding),
        *_episodic_candidates(query_embedding),
    ]
    if not combined:
        return []

    max_derived = max((int(item.get("derived_from", 0)) for item in combined), default=1)
    if max_derived <= 0:
        max_derived = 1

    ranked: list[dict[str, Any]] = []
    for item in combined:
        similarity = float(item.get("similarity_score", 0.0))
        confidence = float(item.get("confidence", 0.0))
        derived_from = int(item.get("derived_from", 0))
        derived_normalized = float(derived_from) / float(max_derived)

        rank_score = (similarity * 0.7) + (confidence * 0.2) + (derived_normalized * 0.1)

        ranked.append(
            {
                **item,
                "rank_score": rank_score,
                "derived_from_normalized": derived_normalized,
            }
        )

    ranked.sort(key=lambda item: item.get("rank_score", 0.0), reverse=True)
    return ranked[:top_k]
