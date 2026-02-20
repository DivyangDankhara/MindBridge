from tools.python_exec import safe_python_exec


TOOL_REGISTRY = {
    "python_exec": safe_python_exec,
}


def execute_plan(plan: list[str]) -> None:
    print("Execution plan:")
    for index, step in enumerate(plan, start=1):
        print(f"{index}. {step}")
