import copy
from typing import Any


EXECUTION_CONTEXT: dict[str, Any] = {}


def reset_context() -> None:
    EXECUTION_CONTEXT.clear()


def _snapshot_context() -> dict[str, tuple[str, Any]]:
    snapshot: dict[str, tuple[str, Any]] = {}
    for key, value in EXECUTION_CONTEXT.items():
        if key.startswith("__"):
            continue
        try:
            snapshot[key] = ("value", copy.deepcopy(value))
        except Exception:
            snapshot[key] = ("repr", repr(value))
    return snapshot


def _is_changed(previous: tuple[str, Any], current: Any) -> bool:
    mode, old_value = previous
    if mode == "value":
        try:
            return old_value != current
        except Exception:
            return repr(old_value) != repr(current)
    return old_value != repr(current)


def run_python(code: str) -> dict[str, Any] | str:
    before = _snapshot_context()

    try:
        exec(code, EXECUTION_CONTEXT)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    changed: dict[str, Any] = {}
    for key, value in EXECUTION_CONTEXT.items():
        if key.startswith("__"):
            continue

        if key not in before:
            changed[key] = value
            continue

        if _is_changed(before[key], value):
            changed[key] = value

    return changed
