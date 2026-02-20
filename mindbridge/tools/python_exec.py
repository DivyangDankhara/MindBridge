import ast
import copy
from typing import Any


_ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
)

_FORBIDDEN_NAMES = {
    "__import__",
    "__builtins__",
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "os",
    "sys",
    "subprocess",
}


EXECUTION_CONTEXT: dict[str, Any] = {"__builtins__": _ALLOWED_BUILTINS}


def reset_execution_context() -> None:
    EXECUTION_CONTEXT.clear()
    EXECUTION_CONTEXT["__builtins__"] = _ALLOWED_BUILTINS


def _snapshot_public_context() -> dict[str, tuple[str, Any]]:
    snapshot: dict[str, tuple[str, Any]] = {}
    for key, value in EXECUTION_CONTEXT.items():
        if key.startswith("__"):
            continue
        try:
            snapshot[key] = ("value", copy.deepcopy(value))
        except Exception:
            snapshot[key] = ("repr", repr(value))
    return snapshot


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return left == right
    except Exception:
        return False


def safe_python_exec(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code, mode="exec")

        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_NODES):
                raise ValueError(f"Disallowed syntax: {type(node).__name__}")

            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("Dunder attribute access is not allowed")

            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                raise ValueError(f"Disallowed name: {node.id}")

        EXECUTION_CONTEXT["__builtins__"] = _ALLOWED_BUILTINS
        before_state = _snapshot_public_context()
        exec(code, EXECUTION_CONTEXT)

        changed: dict[str, Any] = {}
        for key, value in EXECUTION_CONTEXT.items():
            if key.startswith("__"):
                continue

            state = before_state.get(key)
            if state is None:
                changed[key] = value
                continue

            mode, previous = state
            if mode == "value":
                if not _values_equal(previous, value):
                    changed[key] = value
                continue

            if previous != repr(value):
                changed[key] = value

        return changed
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
