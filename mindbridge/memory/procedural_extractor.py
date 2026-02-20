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


def _build_procedural_prompt(mission_experience: MissionExperience) -> str:
    failure_text = ", ".join(mission_experience.failure_reasons) if mission_experience.failure_reasons else "none"
    tools_text = ", ".join(mission_experience.tools_used) if mission_experience.tools_used else "none"
    return (
        "Extract reusable procedural strategies from this successful mission.\n"
        "Strategies must be abstract and reusable workflows.\n"
        "Do not include mission-specific identifiers.\n\n"
        "Return JSON only with this format:\n"
        "{\n"
        '  "strategies": [\n'
        "    {\n"
        '      "strategy_name": "short strategy name",\n'
        '      "applicable_context": "when to use it",\n'
        '      "steps_template": ["step 1", "step 2"],\n'
        '      "confidence": 0.0,\n'
        f'      "derived_from": {mission_experience.attempts}\n'
        "    }\n"
        "  ]\n"
        "}\n"
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


def _normalize_strategy(strategy_data: Any, derived_from_default: int) -> dict[str, Any] | None:
    if not isinstance(strategy_data, dict):
        return None

    strategy_name = strategy_data.get("strategy_name")
    applicable_context = strategy_data.get("applicable_context")
    steps_template = strategy_data.get("steps_template")
    confidence = strategy_data.get("confidence")
    derived_from = strategy_data.get("derived_from", derived_from_default)

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
        derived_from_value = derived_from_default

    name_text = strategy_name.strip()
    if len(name_text) > 120:
        name_text = name_text[:120].rstrip()

    return {
        "strategy_name": name_text,
        "applicable_context": applicable_context.strip(),
        "steps_template": normalized_steps,
        "confidence": max(0.0, min(1.0, confidence_value)),
        "derived_from": derived_from_value,
    }


def extract_procedural_strategies(
    mission_experience: MissionExperience,
    llm: LLMProvider,
) -> list[dict[str, Any]]:
    if not mission_experience.success:
        return []

    prompt = _build_procedural_prompt(mission_experience)
    try:
        response = llm.generate(prompt)
        parsed = json.loads(_extract_json_text(response))
    except Exception:
        return []

    raw_strategies: list[Any]
    if isinstance(parsed, dict) and isinstance(parsed.get("strategies"), list):
        raw_strategies = parsed["strategies"]
    elif isinstance(parsed, list):
        raw_strategies = parsed
    else:
        return []

    normalized_strategies: list[dict[str, Any]] = []
    for item in raw_strategies:
        normalized = _normalize_strategy(item, mission_experience.attempts)
        if normalized is not None:
            normalized_strategies.append(normalized)

    return normalized_strategies
