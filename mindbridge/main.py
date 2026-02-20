from pathlib import Path

from config import DEFAULT_MODEL
from evaluator.evaluator import evaluate_goal
from executor.tool_registry import execute_plan
from intent.parser import parse_intent_file
from llm.openai_provider import OpenAIProvider
from planner.planner import create_plan


MAX_MISSION_ATTEMPTS = 5


def main() -> None:
    project_root = Path(__file__).parent
    intent_path = project_root / "examples" / "test.intent"

    intent = parse_intent_file(intent_path)
    llm = OpenAIProvider(model=DEFAULT_MODEL)
    mission_history: list[dict[str, object]] = []
    mission_accomplished = False

    for attempt in range(1, MAX_MISSION_ATTEMPTS + 1):
        print(f"\nMission attempt {attempt}/{MAX_MISSION_ATTEMPTS}")
        plan = create_plan(intent, llm, mission_history=mission_history)

        execution_results: list[dict[str, object]] = []
        if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
            execution_results = execute_plan(plan, llm=llm)
            print("\nExecution results:")
            for item in execution_results:
                print(item)
        elif isinstance(plan, str):
            print(f"Planner message:\n{plan}")
        else:
            print(f"Planner returned unsupported output type: {type(plan).__name__}")

        evaluation: dict[str, object]
        if execution_results:
            evaluation = evaluate_goal(intent, execution_results, llm)
        else:
            evaluation = {
                "goal_satisfied": False,
                "reason": "No executable plan/results produced.",
            }

        print("\nGoal evaluation:")
        print(evaluation)

        mission_history.append(
            {
                "attempt": attempt,
                "plan": plan,
                "execution_results": execution_results,
                "evaluation": evaluation,
            }
        )

        if evaluation.get("goal_satisfied") is True:
            print("Mission accomplished")
            mission_accomplished = True
            break

        print("Goal not satisfied. Replanning...")

    if not mission_accomplished:
        print("Mission failed after maximum attempts")


if __name__ == "__main__":
    main()
