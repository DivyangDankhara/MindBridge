import json
from typing import Any, Callable

from llm.base import LLMProvider
from tools.python_exec import EXECUTION_CONTEXT, reset_context, run_python


MAX_RETRIES = 3


TOOL_REGISTRY: dict[str, Callable[[str], Any]] = {
    "python_exec": run_python,
    "python": run_python,
}


def _available_context_variables() -> dict[str, str]:
    variables: dict[str, str] = {}
    for key, value in EXECUTION_CONTEXT.items():
        if key.startswith("__"):
            continue
        try:
            variables[key] = repr(value)
        except Exception:
            variables[key] = "<unrepresentable>"
    return variables


def _build_reflection_prompt(
    description: str,
    failed_code: str,
    error_message: str,
    available_variables: dict[str, str],
) -> str:
    variables_json = json.dumps(available_variables, ensure_ascii=True, sort_keys=True)
    return (
        "You are fixing a failed Python execution step.\n"
        "Return corrected Python code only.\n"
        "No markdown.\n"
        "No explanation.\n\n"
        f"Step description: {description}\n"
        "Failed code:\n"
        f"{failed_code}\n\n"
        f"Error message: {error_message}\n"
        f"Available variables in execution context (name -> repr): {variables_json}\n\n"
        "Return executable Python code only."
    )


def _extract_code_only(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def execute_plan(plan: Any, llm: LLMProvider | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    reset_context()

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

        current_code = code
        retry_attempts: list[dict[str, Any]] = []

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = tool_fn(current_code)
            except Exception as exc:
                result = f"{type(exc).__name__}: {exc}"

            if not isinstance(result, str):
                print(f"Result: {result}")
                step_result: dict[str, Any] = {
                    "step": index,
                    "description": description,
                    "tool": tool_name,
                    "status": "success",
                    "result": result,
                }
                if retry_attempts:
                    step_result["retry_attempts"] = retry_attempts
                results.append(step_result)
                break

            error_message = result

            if llm is None or attempt == MAX_RETRIES:
                print(f"Result: {error_message}")
                step_result = {
                    "step": index,
                    "description": description,
                    "tool": tool_name,
                    "status": "error",
                    "error": error_message,
                }
                if retry_attempts:
                    step_result["retry_attempts"] = retry_attempts
                results.append(step_result)
                break

            description_text = description if isinstance(description, str) else "No description"
            prompt = _build_reflection_prompt(
                description=description_text,
                failed_code=current_code,
                error_message=error_message,
                available_variables=_available_context_variables(),
            )
            retry_number = attempt + 1
            print(f"Retry {retry_number}/{MAX_RETRIES} for step {index} after error: {error_message}")
            try:
                corrected_code_raw = llm.generate(prompt)
                corrected_code = _extract_code_only(corrected_code_raw)
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                print(f"Result: LLM correction failed: {llm_error}")
                retry_attempts.append(
                    {
                        "attempt": retry_number,
                        "error": error_message,
                        "llm_error": llm_error,
                    }
                )
                results.append(
                    {
                        "step": index,
                        "description": description,
                        "tool": tool_name,
                        "status": "error",
                        "error": f"LLM correction failed: {llm_error}",
                        "retry_attempts": retry_attempts,
                    }
                )
                break

            retry_attempts.append(
                {
                    "attempt": retry_number,
                    "error": error_message,
                    "corrected_code": corrected_code,
                }
            )
            current_code = corrected_code

    return results
