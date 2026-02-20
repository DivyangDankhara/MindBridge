import json
from pathlib import Path
from typing import Any


SEMANTIC_FILE = Path(__file__).resolve().parent / "semantic.jsonl"


def ensure_semantic_file() -> Path:
    SEMANTIC_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEMANTIC_FILE.touch(exist_ok=True)
    return SEMANTIC_FILE


def _normalize_rule(rule_data: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(rule_data, dict):
        return None

    rule = rule_data.get("rule")
    context = rule_data.get("context")
    confidence = rule_data.get("confidence")
    derived_from = rule_data.get("derived_from")

    if not isinstance(rule, str) or not rule.strip():
        return None
    if not isinstance(context, str):
        context = ""
    try:
        confidence_value = float(confidence)
    except Exception:
        return None
    try:
        derived_from_value = int(derived_from)
    except Exception:
        return None

    confidence_value = max(0.0, min(1.0, confidence_value))
    return {
        "rule": rule.strip(),
        "context": context.strip(),
        "confidence": confidence_value,
        "derived_from": derived_from_value,
    }


def save_semantic_rule(rule_data: dict[str, Any]) -> None:
    normalized = _normalize_rule(rule_data)
    if normalized is None:
        return

    file_path = ensure_semantic_file()
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False))
        handle.write("\n")


def load_semantic_rules() -> list[dict[str, Any]]:
    file_path = ensure_semantic_file()
    rules: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except Exception:
                continue

            normalized = _normalize_rule(parsed)
            if normalized is not None:
                rules.append(normalized)

    return rules


ensure_semantic_file()
