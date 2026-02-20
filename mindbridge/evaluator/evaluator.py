import json
from typing import Any

from intent.schema import Intent
from llm.base import LLMProvider


def _build_evaluation_prompt(intent: Intent, execution_results: Any) -> str:
    serialized_results = json.dumps(execution_results, ensure_ascii=True, default=str)
    return (
        "You are evaluating whether an execution satisfied the requested goal.\n"
        "Return JSON only.\n"
        "No markdown.\n"
        "No explanation outside JSON.\n"
        "Required format:\n"
        '{\n  "goal_satisfied": true,\n  "reason": "short explanation"\n}\n\n'
        f"Original task: {intent.task}\n"
        f"Goal description: {intent.goal}\n"
        f"Expected output: {intent.output or 'None'}\n"
        f"Execution results: {serialized_results}\n"
    )


def evaluate_goal(intent: Intent, execution_results: Any, llm: LLMProvider) -> dict[str, Any]:
    prompt = _build_evaluation_prompt(intent, execution_results)
    try:
        response = llm.generate(prompt)
        payload = json.loads(response)
    except Exception:
        return {
            "goal_satisfied": False,
            "reason": "Failed to parse evaluation JSON.",
        }

    goal_satisfied = payload.get("goal_satisfied")
    reason = payload.get("reason")
    if isinstance(goal_satisfied, bool) and isinstance(reason, str):
        return {
            "goal_satisfied": goal_satisfied,
            "reason": reason.strip(),
        }

    return {
        "goal_satisfied": False,
        "reason": "Invalid evaluation payload format.",
    }
