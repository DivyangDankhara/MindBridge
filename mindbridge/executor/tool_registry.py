from typing import Any, Callable

from tools.python_exec import safe_python_exec


TOOL_REGISTRY: dict[str, Callable[[str], Any]] = {
    "python_exec": safe_python_exec,
    "python": safe_python_exec,
}


def execute_plan(plan: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        print("No executable steps found in plan.")
        return results

    for index, step in enumerate(plan["steps"], start=1):
        if not isinstance(step, dict):
            error_message = "Invalid step format (expected object)."
            print(f"Step {index}: {error_message}")
            results.append(
                {
                    "step": index,
                    "status": "error",
                    "error": error_message,
                }
            )
            continue

        description = step.get("description")
        tool_name = step.get("tool")
        code = step.get("code")

        print(f"Step {index}: {description or 'No description'}")

        if not isinstance(tool_name, str) or not isinstance(code, str):
            error_message = "Invalid step fields (tool/code must be strings)."
            print(f"Result: {error_message}")
            results.append(
                {
                    "step": index,
                    "description": description,
                    "tool": tool_name,
                    "status": "error",
                    "error": error_message,
                }
            )
            continue

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            error_message = f"Tool '{tool_name}' is not registered. Skipping step."
            print(f"Result: {error_message}")
            results.append(
                {
                    "step": index,
                    "description": description,
                    "tool": tool_name,
                    "status": "skipped",
                    "error": error_message,
                }
            )
            continue

        try:
            result = tool_fn(code)
            print(f"Result: {result}")
            results.append(
                {
                    "step": index,
                    "description": description,
                    "tool": tool_name,
                    "status": "success",
                    "result": result,
                }
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"Result: {error_message}")
            results.append(
                {
                    "step": index,
                    "description": description,
                    "tool": tool_name,
                    "status": "error",
                    "error": error_message,
                }
            )

    return results
