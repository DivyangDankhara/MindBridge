from intent.schema import Intent
from llm.base import LLMProvider


def _build_planning_prompt(intent: Intent) -> str:
    return (
        "You are an execution planner. Return only a concise numbered list of steps.\n\n"
        f"Task: {intent.task}\n"
        f"Goal: {intent.goal}\n"
        f"Constraints: {intent.constraints or 'None'}\n"
        f"Expected Output: {intent.output or 'None'}"
    )


def create_plan(intent: Intent, llm: LLMProvider) -> list[str]:
    prompt = _build_planning_prompt(intent)
    raw_plan = llm.generate(prompt)

    steps: list[str] = []
    for line in raw_plan.splitlines():
        candidate = line.strip()
        if not candidate:
            continue

        candidate = candidate.lstrip("-*")
        candidate = candidate.lstrip("0123456789.")
        candidate = candidate.strip(" )")
        candidate = candidate.strip()

        if candidate:
            steps.append(candidate)

    return steps or [raw_plan.strip()]
