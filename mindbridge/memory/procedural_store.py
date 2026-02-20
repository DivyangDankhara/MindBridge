import json
import re
import tempfile
from pathlib import Path
from typing import Any


PROCEDURAL_FILE = Path(__file__).resolve().parent / "procedural.jsonl"
SIMILARITY_THRESHOLD = 0.6
CONFIDENCE_INCREMENT = 0.1
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def ensure_procedural_file() -> Path:
    PROCEDURAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCEDURAL_FILE.touch(exist_ok=True)
    return PROCEDURAL_FILE


def _normalize_strategy(strategy: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(strategy, dict):
        return None

    strategy_name = strategy.get("strategy_name")
    applicable_context = strategy.get("applicable_context")
    steps_template = strategy.get("steps_template")
    confidence = strategy.get("confidence")
    derived_from = strategy.get("derived_from")

    if not isinstance(strategy_name, str) or not strategy_name.strip():
        return None
    if not isinstance(applicable_context, str):
        applicable_context = ""
    if not isinstance(steps_template, list):
        return None

    normalized_steps: list[str] = []
    for step in steps_template:
        if isinstance(step, str) and step.strip():
            normalized_steps.append(step.strip())
    if not normalized_steps:
        return None

    try:
        confidence_value = float(confidence)
    except Exception:
        return None
    try:
        derived_from_value = int(derived_from)
    except Exception:
        return None

    return {
        "strategy_name": strategy_name.strip(),
        "applicable_context": applicable_context.strip(),
        "steps_template": normalized_steps,
        "confidence": max(0.0, min(1.0, confidence_value)),
        "derived_from": max(0, derived_from_value),
    }


def _strategy_text(strategy: dict[str, Any]) -> str:
    steps_template = strategy.get("steps_template", [])
    steps_text = ""
    if isinstance(steps_template, list):
        steps_text = " ".join(step for step in steps_template if isinstance(step, str))
    return (
        f"{strategy.get('strategy_name', '')} "
        f"{strategy.get('applicable_context', '')} "
        f"{steps_text}"
    ).strip().lower()


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text))


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / float(max(len(left), len(right)))


def _is_similar_strategy(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_text = _strategy_text(left)
    right_text = _strategy_text(right)

    if left_text and left_text == right_text:
        return True

    overlap = _token_overlap_ratio(_token_set(left_text), _token_set(right_text))
    return overlap >= SIMILARITY_THRESHOLD


def _merge_strategy(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged["confidence"] = min(1.0, float(existing.get("confidence", 0.0)) + CONFIDENCE_INCREMENT)
    merged["derived_from"] = int(existing.get("derived_from", 0)) + 1

    existing_context = str(existing.get("applicable_context", "")).strip()
    incoming_context = str(incoming.get("applicable_context", "")).strip()
    if not existing_context and incoming_context:
        merged["applicable_context"] = incoming_context

    existing_steps = existing.get("steps_template", [])
    incoming_steps = incoming.get("steps_template", [])
    if isinstance(existing_steps, list) and isinstance(incoming_steps, list):
        step_index = {step.lower(): step for step in existing_steps if isinstance(step, str)}
        merged_steps = [step for step in existing_steps if isinstance(step, str)]
        for step in incoming_steps:
            if not isinstance(step, str):
                continue
            step_text = step.strip()
            if not step_text:
                continue
            if step_text.lower() in step_index:
                continue
            merged_steps.append(step_text)
            step_index[step_text.lower()] = step_text
        if merged_steps:
            merged["steps_template"] = merged_steps

    return merged


def _consolidate_strategies(strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    consolidated: list[dict[str, Any]] = []
    for strategy in strategies:
        normalized = _normalize_strategy(strategy)
        if normalized is None:
            continue

        merged = False
        for index, existing in enumerate(consolidated):
            if _is_similar_strategy(existing, normalized):
                consolidated[index] = _merge_strategy(existing, normalized)
                merged = True
                break

        if not merged:
            consolidated.append(normalized)

    return consolidated


def _rewrite_strategies_safely(strategies: list[dict[str, Any]]) -> None:
    file_path = ensure_procedural_file()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=file_path.parent,
        prefix=f"{file_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for strategy in strategies:
            handle.write(json.dumps(strategy, ensure_ascii=False))
            handle.write("\n")
        temp_path = Path(handle.name)

    temp_path.replace(file_path)


def load_procedural_strategies() -> list[dict[str, Any]]:
    file_path = ensure_procedural_file()
    loaded: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except Exception:
                continue

            normalized = _normalize_strategy(parsed)
            if normalized is not None:
                loaded.append(normalized)

    consolidated = _consolidate_strategies(loaded)
    if consolidated != loaded:
        _rewrite_strategies_safely(consolidated)

    return consolidated


def save_procedural_strategy(strategy: dict[str, Any]) -> None:
    normalized = _normalize_strategy(strategy)
    if normalized is None:
        return

    current_strategies = load_procedural_strategies()
    updated_strategies = _consolidate_strategies([*current_strategies, normalized])
    _rewrite_strategies_safely(updated_strategies)


ensure_procedural_file()
