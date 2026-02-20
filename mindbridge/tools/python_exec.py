import ast
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


def safe_python_exec(code: str) -> dict[str, Any]:
    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed")

        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Disallowed name: {node.id}")

    local_env: dict[str, Any] = {}
    exec(compile(tree, filename="<safe_python_exec>", mode="exec"), {"__builtins__": _ALLOWED_BUILTINS}, local_env)
    return local_env
