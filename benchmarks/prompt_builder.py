"""Build improved prompts for coding tasks that explicitly specify function names and signatures.

This module enhances MBPP and other coding task prompts to include the exact function name
and signature that the evaluator expects, reducing function name mismatches.
"""

from __future__ import annotations

import re
from typing import Any


def build_coding_prompt(question: dict[str, Any]) -> str:
    """Build an improved prompt for coding tasks that includes function name and signature.

    For tasks with test_spec, extracts function name and infers signature from test cases,
    then constructs a prompt that explicitly tells the model what to implement.

    The original task wording is always preserved. If the exact expected function signature
    is not already present, it is appended with explicit instructions.

    Args:
        question: Question dict with 'prompt' and optionally 'test_spec'

    Returns:
        Enhanced prompt string that includes function name and signature
    """
    base_prompt = question.get("prompt", "").strip()

    # Only enhance if test_spec is available
    test_spec = question.get("test_spec")
    if not test_spec or not isinstance(test_spec, dict):
        return base_prompt

    function_name = test_spec.get("function_name")
    cases = test_spec.get("cases", [])

    if not function_name or not cases:
        return base_prompt

    # Infer parameter signature from test cases
    param_signature = _infer_signature(function_name, cases)

    # Check if the exact expected function signature is already explicitly provided
    if _has_explicit_signature(base_prompt, function_name):
        return base_prompt

    # Original task wording is preserved; append signature and instructions
    enhanced_parts = [base_prompt]
    enhanced_parts.append(f"\nImplement a Python function with this exact signature:")
    enhanced_parts.append(f"def {param_signature}")
    enhanced_parts.append(f"\nIMPORTANT: The function must be named '{function_name}' exactly.")
    enhanced_parts.append("Return only the function code.")

    return "\n".join(enhanced_parts)


def _has_explicit_signature(prompt: str, function_name: str) -> bool:
    """Check if the prompt already contains an explicit function signature.

    An explicit signature is one of:
    - def function_name(...): with the correct function name and opening parenthesis
    - "function named function_name(...)" with opening parenthesis showing it specifies parameters

    Args:
        prompt: The prompt text
        function_name: Expected function name

    Returns:
        True if an explicit signature is found, False otherwise
    """
    # Pattern 1: def function_name(
    # This indicates the signature is explicitly provided
    def_pattern = rf"\bdef\s+{re.escape(function_name)}\s*\("
    if re.search(def_pattern, prompt):
        return True

    # Pattern 2: function named function_name(
    # This indicates parameters are being specified
    func_named_pattern = rf"\bfunction\s+named\s+{re.escape(function_name)}\s*\("
    if re.search(func_named_pattern, prompt, re.IGNORECASE):
        return True

    return False


def _infer_signature(function_name: str, cases: list[dict[str, Any]]) -> str:
    """Infer function signature from test cases.

    Args:
        function_name: Name of the function
        cases: List of test cases with 'args' field

    Returns:
        Function signature string like "func_name(a, b, c)"
    """
    if not cases:
        return f"{function_name}()"

    # Get argument count from first test case
    first_case = cases[0]
    args = first_case.get("args", [])

    if not args:
        return f"{function_name}()"

    # Use generic parameter names based on argument types
    param_names = _generate_param_names(len(args), args)
    params = ", ".join(param_names)

    return f"{function_name}({params})"


def _generate_param_names(count: int, sample_args: list[Any]) -> list[str]:
    """Generate sensible parameter names based on argument types and count.

    Args:
        count: Number of parameters
        sample_args: Sample arguments from first test case

    Returns:
        List of parameter name strings
    """
    names = []

    for i, arg in enumerate(sample_args):
        # Try to infer meaningful names from types
        if isinstance(arg, str):
            names.append("text" if i == 0 else f"text{i+1}")
        elif isinstance(arg, list):
            names.append("items" if i == 0 else f"items{i+1}")
        elif isinstance(arg, (int, float)):
            if i == 0:
                names.append("n")
            elif i == 1:
                names.append("m")
            else:
                names.append(chr(ord('a') + i) if i < 26 else f"arg{i+1}")
        else:
            names.append(chr(ord('a') + i) if i < 26 else f"arg{i+1}")

    return names


def extract_function_name_from_prompt(prompt: str) -> str | None:
    """Try to extract function name from prompt text.

    Looks for patterns like:
    - "function named func_name"
    - "def func_name("
    - "func_name(args)"

    Args:
        prompt: Prompt text

    Returns:
        Function name if found, None otherwise
    """
    # Pattern: "function named func_name"
    match = re.search(r'function\s+named\s+([a-z_][a-z0-9_]*)', prompt, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern: "def func_name("
    match = re.search(r'def\s+([a-z_][a-z0-9_]*)\s*\(', prompt, re.IGNORECASE)
    if match:
        return match.group(1)

    return None
