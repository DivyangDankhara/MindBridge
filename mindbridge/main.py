from pathlib import Path

from config import DEFAULT_MODEL
from executor.tool_registry import execute_plan
from intent.parser import parse_intent_file
from llm.openai_provider import OpenAIProvider
from planner.planner import create_plan


def main() -> None:
    project_root = Path(__file__).parent
    intent_path = project_root / "examples" / "sales.intent"

    intent = parse_intent_file(intent_path)
    llm = OpenAIProvider(model=DEFAULT_MODEL)
    plan = create_plan(intent, llm)
    execute_plan(plan)


if __name__ == "__main__":
    main()
