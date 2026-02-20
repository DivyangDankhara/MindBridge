import json
import re
import tempfile
from pathlib import Path
from typing import Any

from llm.base import LLMProvider


SEMANTIC_FILE = Path(__file__).resolve().parent / "semantic.jsonl"
SIMILARITY_THRESHOLD = 0.6
CONFIDENCE_INCREMENT = 0.1
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
        "derived_from": max(0, derived_from_value),
    }


def _rule_text(rule_data: dict[str, Any]) -> str:
    return f"{rule_data.get('rule', '')} {rule_data.get('context', '')}".strip().lower()


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text))


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / float(max(len(left), len(right)))


def _is_similar_rule(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_text = _rule_text(left)
    right_text = _rule_text(right)

    if left_text and left_text == right_text:
        return True

    overlap = _token_overlap_ratio(_token_set(left_text), _token_set(right_text))
    return overlap >= SIMILARITY_THRESHOLD


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _build_similarity_prompt(existing: dict[str, Any], incoming: dict[str, Any]) -> str:
    return (
        "Are these two semantic rules semantically equivalent or highly similar?\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "similar": true,\n'
        '  "canonical_rule": "best merged wording",\n'
        '  "canonical_context": "best merged context"\n'
        "}\n"
        "If they are not similar, set similar to false and canonical fields to empty strings.\n"
        "No markdown.\n"
        "No explanation.\n\n"
        f"Rule A: {existing.get('rule', '')}\n"
        f"Context A: {existing.get('context', '')}\n"
        f"Rule B: {incoming.get('rule', '')}\n"
        f"Context B: {incoming.get('context', '')}\n"
    )


def _parse_similarity_response(response: str) -> tuple[bool, str, str]:
    try:
        payload = json.loads(_extract_json_text(response))
    except Exception:
        return False, "", ""

    similar_raw = payload.get("similar") if isinstance(payload, dict) else False
    similar = False
    if isinstance(similar_raw, bool):
        similar = similar_raw
    elif isinstance(similar_raw, str):
        similar = similar_raw.strip().lower() in {"true", "yes", "y"}

    if not similar:
        return False, "", ""

    canonical_rule = payload.get("canonical_rule") if isinstance(payload, dict) else ""
    canonical_context = payload.get("canonical_context") if isinstance(payload, dict) else ""
    if not isinstance(canonical_rule, str):
        canonical_rule = ""
    if not isinstance(canonical_context, str):
        canonical_context = ""

    return True, canonical_rule.strip(), canonical_context.strip()


def _llm_similarity_decision(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    llm: LLMProvider | None,
) -> tuple[bool, str, str]:
    if llm is None:
        return _is_similar_rule(existing, incoming), "", ""

    prompt = _build_similarity_prompt(existing, incoming)
    try:
        response = llm.generate(prompt)
    except Exception:
        return _is_similar_rule(existing, incoming), "", ""

    parsed = _parse_similarity_response(response)
    if parsed[0]:
        return parsed

    return _is_similar_rule(existing, incoming), "", ""


def _merge_rule(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    canonical_rule: str = "",
    canonical_context: str = "",
) -> dict[str, Any]:
    existing_rule = str(existing.get("rule", "")).strip()
    incoming_rule = str(incoming.get("rule", "")).strip()
    merged_rule = canonical_rule.strip() if canonical_rule.strip() else (existing_rule or incoming_rule)
    if not merged_rule:
        merged_rule = incoming_rule or existing_rule

    existing_context = str(existing.get("context", "")).strip()
    incoming_context = str(incoming.get("context", "")).strip()
    merged_context = canonical_context.strip() if canonical_context.strip() else (existing_context or incoming_context)

    base_confidence = max(float(existing.get("confidence", 0.0)), float(incoming.get("confidence", 0.0)))
    base_derived_from = max(int(existing.get("derived_from", 0)), int(incoming.get("derived_from", 0)))

    merged = {
        "rule": merged_rule,
        "context": merged_context,
        "confidence": min(1.0, base_confidence + CONFIDENCE_INCREMENT),
        "derived_from": base_derived_from + 1,
    }
    normalized = _normalize_rule(merged)
    return normalized if normalized is not None else existing


def _rewrite_rules_safely(rules: list[dict[str, Any]]) -> None:
    file_path = ensure_semantic_file()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=file_path.parent,
        prefix=f"{file_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for rule in rules:
            handle.write(json.dumps(rule, ensure_ascii=False))
            handle.write("\n")
        temp_path = Path(handle.name)

    temp_path.replace(file_path)


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


def _consolidate_with_llm(
    rules: list[dict[str, Any]],
    llm: LLMProvider | None,
) -> list[dict[str, Any]]:
    consolidated: list[dict[str, Any]] = []
    for rule in rules:
        normalized = _normalize_rule(rule)
        if normalized is None:
            continue

        merged = False
        for index, existing in enumerate(consolidated):
            similar, canonical_rule, canonical_context = _llm_similarity_decision(existing, normalized, llm=llm)
            if not similar:
                continue

            consolidated[index] = _merge_rule(
                existing,
                normalized,
                canonical_rule=canonical_rule,
                canonical_context=canonical_context,
            )
            merged = True
            break

        if not merged:
            consolidated.append(normalized)

    return consolidated


def _merge_incoming_with_existing(
    existing_rules: list[dict[str, Any]],
    incoming_rule: dict[str, Any],
    llm: LLMProvider | None,
) -> list[dict[str, Any]]:
    merged_rule = incoming_rule
    matched_indices: list[int] = []

    for index, existing in enumerate(existing_rules):
        similar, canonical_rule, canonical_context = _llm_similarity_decision(existing, merged_rule, llm=llm)
        if not similar:
            continue

        merged_rule = _merge_rule(
            existing,
            merged_rule,
            canonical_rule=canonical_rule,
            canonical_context=canonical_context,
        )
        matched_indices.append(index)

    if not matched_indices:
        return [*existing_rules, incoming_rule]

    first_match = matched_indices[0]
    matched_lookup = set(matched_indices)
    updated_rules: list[dict[str, Any]] = []
    for index, existing in enumerate(existing_rules):
        if index in matched_lookup:
            if index == first_match:
                updated_rules.append(merged_rule)
            continue
        updated_rules.append(existing)

    return updated_rules


def save_semantic_rule(rule_data: dict[str, Any], llm: LLMProvider | None = None) -> None:
    normalized = _normalize_rule(rule_data)
    if normalized is None:
        return

    current_rules = _consolidate_with_llm(load_semantic_rules(), llm=llm)
    updated_rules = _merge_incoming_with_existing(current_rules, normalized, llm=llm)
    final_rules = _consolidate_with_llm(updated_rules, llm=llm)
    _rewrite_rules_safely(final_rules)


ensure_semantic_file()
