from pathlib import Path

from config import DEFAULT_MODEL
from evaluator.evaluator import evaluate_goal
from executor.tool_registry import execute_plan
from intent.parser import parse_intent_file
from llm.openai_provider import OpenAIProvider
from planner.planner import create_plan


def main() -> None:
    project_root = Path(__file__).parent
    intent_path = project_root / "examples" / "test.intent"

    intent = parse_intent_file(intent_path)
    llm = OpenAIProvider(model=DEFAULT_MODEL)
    plan = create_plan(intent, llm)

    if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
        execution_results = execute_plan(plan, llm=llm)
        print("\nExecution results:")
        for item in execution_results:
            print(item)

        evaluation = evaluate_goal(intent, execution_results, llm)
        print("\nGoal evaluation:")
        print(evaluation)
        if not evaluation.get("goal_satisfied", False):
            print("Goal not satisfied - replanning required")
        return

    if isinstance(plan, str):
        print(f"Planner message:\n{plan}")
        return

    print(f"Planner returned unsupported output type: {type(plan).__name__}")


if __name__ == "__main__":
    main()
