import json
from typing import Any

from llm.base import LLMProvider
from memory.schema import MissionExperience


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


def _build_semantic_prompt(mission_experience: MissionExperience) -> str:
    failure_text = ", ".join(mission_experience.failure_reasons) if mission_experience.failure_reasons else "none"
    tools_text = ", ".join(mission_experience.tools_used) if mission_experience.tools_used else "none"
    return (
        "Extract reusable semantic rules from a completed mission.\n"
        "Focus on:\n"
        "1) what worked\n"
        "2) what failed\n"
        "3) general reusable lessons\n\n"
        "Return JSON only with this format:\n"
        "{\n"
        '  "rules": [\n'
        "    {\n"
        '      "rule": "short general principle",\n'
        '      "context": "when this applies",\n'
        '      "confidence": 0.0,\n'
        f'      "derived_from": {mission_experience.attempts}\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules must be short and generalizable.\n"
        "No markdown.\n"
        "No explanation outside JSON.\n\n"
        f"Mission success: {mission_experience.success}\n"
        f"Attempts: {mission_experience.attempts}\n"
        f"Task: {mission_experience.intent_task}\n"
        f"Goal: {mission_experience.intent_goal}\n"
        f"Final plan summary: {mission_experience.final_plan_summary}\n"
        f"Failure reasons: {failure_text}\n"
        f"Tools used: {tools_text}\n"
        f"Result summary: {mission_experience.result_summary}\n"
    )


def _normalize_rule(rule_data: Any, derived_from_default: int) -> dict[str, Any] | None:
    if not isinstance(rule_data, dict):
        return None

    rule = rule_data.get("rule")
    context = rule_data.get("context")
    confidence = rule_data.get("confidence")
    derived_from = rule_data.get("derived_from", derived_from_default)

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
        derived_from_value = derived_from_default

    rule_text = rule.strip()
    if len(rule_text) > 200:
        rule_text = rule_text[:200].rstrip()

    return {
        "rule": rule_text,
        "context": context.strip(),
        "confidence": max(0.0, min(1.0, confidence_value)),
        "derived_from": derived_from_value,
    }


def extract_semantic_rules(mission_experience: MissionExperience, llm: LLMProvider) -> list[dict[str, Any]]:
    prompt = _build_semantic_prompt(mission_experience)
    try:
        response = llm.generate(prompt)
        parsed = json.loads(_extract_json_text(response))
    except Exception:
        return []

    raw_rules: list[Any]
    if isinstance(parsed, dict) and isinstance(parsed.get("rules"), list):
        raw_rules = parsed["rules"]
    elif isinstance(parsed, list):
        raw_rules = parsed
    else:
        return []

    normalized_rules: list[dict[str, Any]] = []
    for item in raw_rules:
        normalized = _normalize_rule(item, mission_experience.attempts)
        if normalized is not None:
            normalized_rules.append(normalized)

    return normalized_rules
