"""Restricted, time-limited subprocess evaluation for generated Python code."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from typing import Any

MAX_SOURCE_LENGTH = 50_000
MAX_AST_NODES = 5_000
SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bool", "enumerate", "float", "int", "len", "max",
    "min", "range", "round", "sorted", "str", "sum", "zip",
}
_ALLOWED_NODE_TYPES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Expr,
    ast.Pass, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name,
    ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice, ast.IfExp, ast.ListComp, ast.comprehension,
    ast.And, ast.Or, ast.Not, ast.Invert, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.USub, ast.UAdd, ast.Attribute,
}
_SAFE_STRING_METHODS = {"lower", "casefold", "upper"}
_CHILD_SCRIPT = r'''
import json
import sys

safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "enumerate": enumerate,
    "float": float, "int": int, "len": len, "max": max, "min": min,
    "range": range, "round": round, "sorted": sorted, "str": str,
    "sum": sum, "zip": zip,
}
payload = json.load(sys.stdin)
namespace = {"__builtins__": safe_builtins}
try:
    exec(compile(payload["source"], "<benchmark-answer>", "exec"), namespace)
    function = namespace.get(payload["function_name"])
    if not callable(function):
        raise ValueError("expected function was not defined")
    passed = [function(*case["args"]) == case["expected"] for case in payload["cases"]]
    print(json.dumps({"passed": passed}))
except BaseException as exc:
    print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)}))
'''


def extract_python_source(response: str) -> str:
    """Extract the first fenced code block, preferring a Python-labelled block."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        labelled = re.findall(r"```(?:python|py)\s*\n(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
        return (labelled or blocks)[0].strip()
    return response.strip()


def evaluate_python_response(
    response: str,
    test_spec: dict[str, Any],
    timeout_seconds: float = 2.0,
) -> tuple[bool, str | None]:
    """Run allowlisted source in a child interpreter; return pass/failure and detail."""
    source = extract_python_source(response)
    error = _validate_source(source, test_spec["function_name"])
    if error:
        return False, error

    payload = {
        "source": source,
        "function_name": test_spec["function_name"],
        "cases": test_spec["cases"],
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _CHILD_SCRIPT],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Execution exceeded {timeout_seconds:g} second timeout"
    except OSError as exc:
        return False, f"Could not start evaluation subprocess: {exc}"

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = completed.stderr.strip() or f"Evaluator exited with code {completed.returncode}"
        return False, detail
    if "error" in result:
        return False, result["error"]
    passed = result.get("passed")
    if not isinstance(passed, list) or len(passed) != len(test_spec["cases"]):
        return False, "Evaluator returned malformed test results"
    return all(passed), None if all(passed) else "One or more deterministic test cases failed"


def _validate_source(source: str, function_name: str) -> str | None:
    if not source:
        return "No Python source was returned"
    if len(source) > MAX_SOURCE_LENGTH:
        return f"Python source exceeds {MAX_SOURCE_LENGTH} characters"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"Invalid Python syntax: {exc.msg}"
    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        return f"Python source exceeds {MAX_AST_NODES} AST nodes"
    if not tree.body or any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        return "Only top-level function definitions are allowed"

    function_names = {node.name for node in tree.body}
    if function_name not in function_names:
        return f"Expected function '{function_name}' was not defined"
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODE_TYPES:
            return f"Disallowed Python syntax: {type(node).__name__}"
        if isinstance(node, ast.FunctionDef) and (
            node.decorator_list or node.args.defaults or node.args.kw_defaults
            or node.args.vararg or node.args.kwarg or node.args.kwonlyargs
            or node.returns is not None
        ):
            return "Function decorators, defaults, annotations, and variadic arguments are disallowed"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "Names beginning with '__' are disallowed"
        if isinstance(node, ast.Attribute) and node.attr not in _SAFE_STRING_METHODS:
            return f"Disallowed attribute access: {node.attr}"
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value.bit_length() > 128:
            return "Integer literals larger than 128 bits are disallowed"
        if isinstance(node, ast.Call):
            safe_name_call = (
                isinstance(node.func, ast.Name)
                and node.func.id in SAFE_BUILTIN_NAMES | function_names
            )
            safe_string_call = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _SAFE_STRING_METHODS
            )
            if not (safe_name_call or safe_string_call) or node.keywords:
                return "Only direct calls to the solution, approved built-ins, or safe string methods are allowed"
    return None