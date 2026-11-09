"""Improved restricted, time-limited subprocess evaluation for generated Python code.

Changes from code_eval.py:
1. Improved code extraction: handles Python fenced blocks, unlabeled blocks, and plain responses
2. Improved AST validation: accepts valid function definitions with top-level example/demo code
3. Granular error categories: distinguishes syntax, missing function, wrong name, unsupported syntax, etc.
4. Clean execution: extracts only permitted function definitions for safe execution
5. More detailed diagnostic messages preserved in error strings

Error taxonomy:
- syntax_error: Invalid Python syntax (parse failure)
- missing_function: No function definitions found
- function_name_mismatch: Expected function name not present
- unsupported_syntax: Disallowed AST nodes (imports, decorators, dunder names, etc.)
- execution_error: Runtime errors during test execution
- timeout: Execution exceeded time limit
- wrong_answer: Tests executed but failed
- infrastructure_error: Evaluation framework issues

Security note: subprocess timeout and restricted built-ins provide limited defense-in-depth
but are NOT a complete security sandbox. Do not treat arbitrary model-generated Python as
fully isolated. The subprocess can still consume CPU, memory, and has access to the Python
interpreter's capabilities within the restricted built-in namespace.
"""

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
    "min", "range", "round", "sorted", "str", "sum", "zip", "list", "dict",
    "set", "tuple",
}

# Permitted AST node types for function definitions
_ALLOWED_NODE_TYPES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Expr,
    ast.Pass, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Name,
    ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice, ast.IfExp, ast.ListComp, ast.DictComp, ast.SetComp,
    ast.comprehension, ast.And, ast.Or, ast.Not, ast.Invert, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.USub, ast.UAdd, ast.Attribute,
    ast.Is, ast.IsNot, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift,
}

_SAFE_STRING_METHODS = {"lower", "casefold", "upper", "strip", "lstrip", "rstrip", "split", "join", "replace", "count", "find", "startswith", "endswith"}
_SAFE_LIST_METHODS = {"append", "extend", "pop", "remove", "reverse", "sort", "copy", "count", "index"}

_CHILD_SCRIPT = r'''
import json
import sys

safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "enumerate": enumerate,
    "float": float, "int": int, "len": len, "max": max, "min": min,
    "range": range, "round": round, "sorted": sorted, "str": str,
    "sum": sum, "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
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
    """Extract Python source from model response.

    Preference order:
    1. Python-labeled fenced code block (```python or ```py)
    2. Unlabeled fenced code block (```)
    3. Plain response (if no fenced blocks found)
    """
    # Try Python-labeled blocks first
    python_blocks = re.findall(
        r"```(?:python|py)\s*\n(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL
    )
    if python_blocks:
        return python_blocks[0].strip()

    # Try any fenced blocks
    any_blocks = re.findall(
        r"```(?:[a-z]*)\s*\n(.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL
    )
    if any_blocks:
        return any_blocks[0].strip()

    # Fall back to treating entire response as code
    return response.strip()


def evaluate_python_response(
    response: str,
    test_spec: dict[str, Any],
    timeout_seconds: float = 2.0,
) -> tuple[bool, str | None]:
    """Run allowlisted source in a child interpreter; return pass/failure and detail.

    Returns:
        (True, None) if all tests pass
        (False, error_message) if tests fail or evaluation cannot proceed

    Error message categories (see module docstring for taxonomy):
    - "syntax_error: ..."
    - "missing_function: ..."
    - "function_name_mismatch: ..."
    - "unsupported_syntax: ..."
    - "execution_error: ..."
    - "timeout: ..."
    - "wrong_answer: ..."
    - "infrastructure_error: ..."
    """
    source = extract_python_source(response)
    validated_source, error = _validate_and_extract_source(source, test_spec["function_name"])
    if error:
        return False, error

    payload = {
        "source": validated_source,
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
        return False, f"timeout: Execution exceeded {timeout_seconds:g} second timeout"
    except OSError as exc:
        return False, f"infrastructure_error: Could not start evaluation subprocess: {exc}"

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        detail = completed.stderr.strip() or f"Evaluator exited with code {completed.returncode}"
        return False, f"execution_error: {detail}"

    if "error" in result:
        return False, f"execution_error: {result['error']}"

    passed = result.get("passed")
    if not isinstance(passed, list) or len(passed) != len(test_spec["cases"]):
        return False, "infrastructure_error: Evaluator returned malformed test results"

    if all(passed):
        return True, None
    else:
        failed_count = sum(1 for p in passed if not p)
        return False, f"wrong_answer: {failed_count} of {len(passed)} test case(s) failed"


def _validate_and_extract_source(source: str, expected_function_name: str) -> tuple[str, str | None]:
    """Validate source and extract only permitted function definitions for execution.

    Returns:
        (cleaned_source, None) if validation passes
        (empty_string, error_message) if validation fails
    """
    if not source:
        return "", "missing_function: No Python source was returned"

    if len(source) > MAX_SOURCE_LENGTH:
        return "", f"unsupported_syntax: Python source exceeds {MAX_SOURCE_LENGTH} characters"

    # Parse the source
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return "", f"syntax_error: {exc.msg} at line {exc.lineno}"

    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        return "", f"unsupported_syntax: Python source exceeds {MAX_AST_NODES} AST nodes"

    if not tree.body:
        return "", "missing_function: Response contains no Python statements"

    # Separate function definitions from other top-level nodes
    function_defs = []
    non_function_nodes = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)
        else:
            # Track disallowed top-level nodes for error reporting
            non_function_nodes.append(type(node).__name__)

    if not function_defs:
        # No functions found - check if there are imports or other disallowed top-level statements
        if non_function_nodes:
            return "", f"unsupported_syntax: Found disallowed top-level statements: {', '.join(set(non_function_nodes))}. Only function definitions are allowed."
        return "", "missing_function: No function definitions found in response"

    function_names = {func.name for func in function_defs}

    # Check if the expected function exists
    if expected_function_name not in function_names:
        available = ", ".join(sorted(function_names))
        return "", f"function_name_mismatch: Expected function '{expected_function_name}' not found (available: {available})"

    # Validate each function definition's AST
    for func_def in function_defs:
        error = _validate_function_def(func_def, function_names)
        if error:
            return "", error

    # Build clean execution source with only function definitions
    # Top-level example calls and imports are discarded
    clean_module = ast.Module(body=function_defs, type_ignores=[])
    try:
        clean_source = ast.unparse(clean_module)
    except Exception as exc:
        return "", f"infrastructure_error: Failed to reconstruct source: {exc}"

    return clean_source, None


def _validate_function_def(func_def: ast.FunctionDef, all_function_names: set[str]) -> str | None:
    """Validate a function definition's AST for security and compatibility.

    Returns None if valid, error message string if invalid.
    """
    # Check function-level restrictions
    if func_def.decorator_list:
        return f"unsupported_syntax: Function decorators are not allowed (function: {func_def.name})"

    if func_def.args.defaults or func_def.args.kw_defaults:
        return f"unsupported_syntax: Function default arguments are not allowed (function: {func_def.name})"

    if func_def.args.vararg or func_def.args.kwarg or func_def.args.kwonlyargs:
        return f"unsupported_syntax: Variadic arguments (*args, **kwargs) are not allowed (function: {func_def.name})"

    if func_def.returns is not None:
        return f"unsupported_syntax: Function type annotations are not allowed (function: {func_def.name})"

    # Validate all nodes in the function body
    for node in ast.walk(func_def):
        if type(node) not in _ALLOWED_NODE_TYPES:
            return f"unsupported_syntax: Disallowed AST node {type(node).__name__} in function {func_def.name}"

        # Check for dunder names
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"unsupported_syntax: Names beginning with '__' are not allowed (in function: {func_def.name})"

        # Check attribute access
        if isinstance(node, ast.Attribute):
            if node.attr not in _SAFE_STRING_METHODS | _SAFE_LIST_METHODS:
                return f"unsupported_syntax: Attribute access '{node.attr}' is not in the approved list (function: {func_def.name})"

        # Check for large integer literals
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value.bit_length() > 128:
            return f"unsupported_syntax: Integer literals larger than 128 bits are not allowed (function: {func_def.name})"

        # Validate function calls
        if isinstance(node, ast.Call):
            # Check for keyword arguments
            if node.keywords:
                return f"unsupported_syntax: Keyword arguments in function calls are not allowed (function: {func_def.name})"

            # Check if it's a safe call
            safe_name_call = (
                isinstance(node.func, ast.Name)
                and node.func.id in SAFE_BUILTIN_NAMES | all_function_names
            )
            safe_method_call = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _SAFE_STRING_METHODS | _SAFE_LIST_METHODS
            )

            if not (safe_name_call or safe_method_call):
                if isinstance(node.func, ast.Name):
                    return f"unsupported_syntax: Function call to '{node.func.id}' is not in the approved list (function: {func_def.name})"
                else:
                    return f"unsupported_syntax: Only calls to approved built-ins, solution functions, or safe methods are allowed (function: {func_def.name})"

    return None
