import ast
import json
from typing import Any

from executor.tool_registry import TOOL_REGISTRY
from intent.schema import Intent
from llm.base import LLMProvider


def _summarize_failure_history(mission_history: list[dict[str, Any]] | None) -> str:
    if not mission_history:
        return "No previous attempts."

    failed_summaries: list[str] = []
    for index, attempt in enumerate(mission_history, start=1):
        if not isinstance(attempt, dict):
            continue

        evaluation = attempt.get("evaluation")
        goal_satisfied = evaluation.get("goal_satisfied") if isinstance(evaluation, dict) else False
        if goal_satisfied is True:
            continue

        reason = "No evaluation reason."
        if isinstance(evaluation, dict) and isinstance(evaluation.get("reason"), str):
            reason = evaluation["reason"].strip()

        execution_results = attempt.get("execution_results")
        error_count = 0
        if isinstance(execution_results, list):
            for item in execution_results:
                if isinstance(item, dict) and item.get("status") in {"error", "skipped"}:
                    error_count += 1

        failed_summaries.append(
            f"Attempt {index}: reason={reason}; execution_errors={error_count}"
        )

    if not failed_summaries:
        return "No previous failed attempts."

    return " | ".join(failed_summaries[-3:])


def _build_planning_prompt(
    intent: Intent,
    allowed_tools: list[str],
    mission_history: list[dict[str, Any]] | None = None,
) -> str:
    tools_text = ", ".join(allowed_tools) if allowed_tools else "none"
    failure_history_summary = _summarize_failure_history(mission_history)
    return (
        "You are an execution planner.\n"
        "Return ONLY valid JSON.\n"
        "No markdown.\n"
        "No explanation.\n"
        "Output schema:\n"
        "{\n"
        '  "steps": [\n'
        "    {\n"
        '      "description": "...",\n'
        '      "tool": "...",\n'
        '      "code": "..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Allowed tools from registry: {tools_text}\n"
        "Choose tools only from the allowed list.\n"
        "When using python tooling, code must be executable Python.\n"
        "Return valid JSON only.\n\n"
        f"Task: {intent.task}\n"
        f"Goal: {intent.goal}\n"
        f"Constraints: {intent.constraints or 'None'}\n"
        f"Expected Output: {intent.output or 'None'}\n"
        f"Previous Failures Summary: {failure_history_summary}"
    )


def _is_executable_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _normalize_plan(payload: Any, allowed_tools: set[str]) -> dict[str, list[dict[str, str]]] | None:
    if not isinstance(payload, dict):
        return None

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return None

    normalized_steps: list[dict[str, str]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue

        description = raw_step.get("description")
        tool = raw_step.get("tool")
        code = raw_step.get("code")

        if not isinstance(description, str) or not isinstance(tool, str) or not isinstance(code, str):
            continue

        if tool not in allowed_tools:
            continue

        if tool == "python_exec" and not _is_executable_python(code):
            continue

        normalized_steps.append(
            {
                "description": description.strip(),
                "tool": tool.strip(),
                "code": code.strip(),
            }
        )

    return {"steps": normalized_steps}


def create_plan(
    intent: Intent,
    llm: LLMProvider,
    mission_history: list[dict[str, Any]] | None = None,
) -> Any:
    allowed_tools = sorted(TOOL_REGISTRY.keys())
    prompt = _build_planning_prompt(intent, allowed_tools, mission_history=mission_history)
    raw_plan = llm.generate(prompt)

    try:
        payload = json.loads(raw_plan)
    except json.JSONDecodeError:
        return raw_plan.strip()

    normalized = _normalize_plan(payload, set(allowed_tools))
    if normalized is not None:
        return normalized

    return payload
