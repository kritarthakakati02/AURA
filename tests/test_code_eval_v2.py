"""Comprehensive tests for improved code evaluation pipeline.

Tests cover:
1. Correct function in Python code fence
2. Correct function followed by example calls
3. Correct function surrounded by explanatory prose
4. Correct function with helper functions
5. Missing expected function
6. Function with incorrect name
7. Syntax error
8. Incorrect output
9. Unsupported AST syntax (imports reported as unsupported_syntax)
10. Timeout behavior
11. Prohibited operations
12. Valid solution with only permitted features
"""

from __future__ import annotations

import pytest

from benchmarks.code_eval_v2 import evaluate_python_response, extract_python_source


def test_correct_function_in_python_fence() -> None:
    """Test case 1: Valid function in Python-labeled code fence."""
    response = """Here's the solution:

```python
def add_numbers(a, b):
    return a + b
```

This function adds two numbers."""

    test_spec = {
        "function_name": "add_numbers",
        "cases": [
            {"args": [2, 3], "expected": 5},
            {"args": [-4, 9], "expected": 5},
            {"args": [0, 0], "expected": 0},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_correct_function_with_example_calls() -> None:
    """Test case 2: Valid function followed by example/demo code."""
    response = """```python
def is_even(number):
    return number % 2 == 0

# Example usage:
print(is_even(4))
print(is_even(5))
result = is_even(10)
```"""

    test_spec = {
        "function_name": "is_even",
        "cases": [
            {"args": [4], "expected": True},
            {"args": [-3], "expected": False},
            {"args": [0], "expected": True},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_correct_function_with_prose() -> None:
    """Test case 3: Valid function surrounded by explanatory text."""
    response = """The solution uses the modulo operator to check divisibility.

```python
def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
```

This iterative approach is efficient for small values of n."""

    test_spec = {
        "function_name": "factorial",
        "cases": [
            {"args": [0], "expected": 1},
            {"args": [1], "expected": 1},
            {"args": [5], "expected": 120},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_correct_function_with_helper() -> None:
    """Test case 4: Valid function with helper function."""
    response = """```python
def helper(x):
    return x * x

def sum_squares(numbers):
    total = 0
    for num in numbers:
        total = total + helper(num)
    return total
```"""

    test_spec = {
        "function_name": "sum_squares",
        "cases": [
            {"args": [[1, 2, 3]], "expected": 14},
            {"args": [[]], "expected": 0},
            {"args": [[5]], "expected": 25},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_missing_expected_function() -> None:
    """Test case 5: Required function not defined."""
    response = """```python
def wrong_name(a, b):
    return a + b
```"""

    test_spec = {
        "function_name": "add_numbers",
        "cases": [{"args": [2, 3], "expected": 5}]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is False
    assert error is not None
    assert error.startswith("function_name_mismatch:")
    assert "add_numbers" in error


def test_function_with_wrong_name() -> None:
    """Test case 6: Function has incorrect name."""
    response = """```python
def my_sum(a, b):
    return a + b
```"""

    test_spec = {
        "function_name": "add_numbers",
        "cases": [{"args": [2, 3], "expected": 5}]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is False
    assert error is not None
    assert "function_name_mismatch" in error
    assert "my_sum" in error


def test_syntax_error() -> None:
    """Test case 7: Invalid Python syntax."""
    response = """```python
def broken(n)
    return n + 1
```"""

    test_spec = {
        "function_name": "broken",
        "cases": [{"args": [1], "expected": 2}]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is False
    assert error is not None
    assert error.startswith("syntax_error:")


def test_incorrect_output() -> None:
    """Test case 8: Function runs but produces wrong answer."""
    response = """```python
def add_numbers(a, b):
    return a - b  # Wrong operation
```"""

    test_spec = {
        "function_name": "add_numbers",
        "cases": [
            {"args": [2, 3], "expected": 5},
            {"args": [10, 5], "expected": 15},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is False
    assert error is not None
    assert error.startswith("wrong_answer:")
    assert "test case" in error


def test_unsupported_import() -> None:
    """Test case 9: Disallowed import statement.

    The evaluator extracts only function definitions, discarding the import.
    When the function tries to use os.getenv, it's detected as an unsupported call.
    This correctly prevents imports by rejecting any calls to non-approved names.
    """
    response = """```python
import os

def get_env(key):
    return os.getenv(key)
```"""

    test_spec = {
        "function_name": "get_env",
        "cases": [{"args": ["PATH"], "expected": ""}]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is False
    assert error is not None
    # The import is discarded, but os.getenv call is detected as unsupported
    assert "unsupported_syntax" in error
    assert "call" in error.lower() or "approved" in error.lower()


def test_unsupported_attribute_access() -> None:
    """Test case 11: Prohibited attribute access."""
    response = """```python
def dangerous(obj):
    return obj.__class__
```"""

    test_spec = {
        "function_name": "dangerous",
        "cases": [{"args": [42], "expected": int}]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is False
    assert error is not None
    assert "unsupported_syntax" in error
    assert "__" in error


def test_timeout() -> None:
    """Test case 10: Execution timeout."""
    response = """```python
def never_finishes(value):
    while True:
        pass
    return value
```"""

    test_spec = {
        "function_name": "never_finishes",
        "cases": [{"args": [0], "expected": 0}]
    }

    passed, error = evaluate_python_response(response, test_spec, timeout_seconds=0.2)
    assert passed is False
    assert error is not None
    assert "timeout" in error.lower()


def test_valid_solution_with_permitted_features() -> None:
    """Test case 12: Complex valid solution using only permitted features."""
    response = """```python
def remove_duplicates(items):
    seen = []
    result = []
    for item in items:
        if item not in seen:
            seen.append(item)
            result.append(item)
    return result
```"""

    test_spec = {
        "function_name": "remove_duplicates",
        "cases": [
            {"args": [[1, 2, 1, 3, 2]], "expected": [1, 2, 3]},
            {"args": [[]], "expected": []},
            {"args": [[4, 4, 4]], "expected": [4]},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_string_methods_allowed() -> None:
    """Test that safe string methods are permitted."""
    response = """```python
def normalize(text):
    return text.strip().lower()
```"""

    test_spec = {
        "function_name": "normalize",
        "cases": [
            {"args": ["  Hello  "], "expected": "hello"},
            {"args": ["WORLD"], "expected": "world"},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_list_comprehension_allowed() -> None:
    """Test that list comprehensions are permitted."""
    response = """```python
def squares(numbers):
    return [n * n for n in numbers]
```"""

    test_spec = {
        "function_name": "squares",
        "cases": [
            {"args": [[1, 2, 3]], "expected": [1, 4, 9]},
            {"args": [[]], "expected": []},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_extraction_prefers_python_labeled_blocks() -> None:
    """Test that extraction prefers Python-labeled code blocks."""
    response = """```
def wrong():
    pass
```

```python
def correct():
    return 42
```"""

    extracted = extract_python_source(response)
    assert "def correct():" in extracted
    assert "def wrong():" not in extracted


def test_extraction_handles_unlabeled_blocks() -> None:
    """Test extraction handles unlabeled code blocks."""
    response = """```
def my_func():
    return 1
```"""

    extracted = extract_python_source(response)
    assert "def my_func():" in extracted


def test_extraction_handles_plain_code() -> None:
    """Test extraction handles plain code without fences."""
    response = """def plain_func():
    return 100"""

    extracted = extract_python_source(response)
    assert "def plain_func():" in extracted


def test_list_methods_allowed() -> None:
    """Test that safe list methods are permitted."""
    response = """```python
def process_list(items):
    result = items.copy()
    result.sort()
    return result
```"""

    test_spec = {
        "function_name": "process_list",
        "cases": [
            {"args": [[3, 1, 2]], "expected": [1, 2, 3]},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    assert passed is True
    assert error is None


def test_realistic_mbpp_response() -> None:
    """Test realistic MBPP-style response with explanation and example."""
    response = """Certainly! To find the largest subset of a list where each pair of elements is divisible, you can use a combination of sorting and a greedy algorithm. Here's a Python function that implements this approach:

```python
def largest_divisible_subset(nums):
    if not nums:
        return []

    nums.sort()
    answer = [nums[0]]
    current_subset = [nums[0]]

    for i in range(1, len(nums)):
        if nums[i] % answer[-1] == 0:
            current_subset.append(nums[i])
            answer.append(nums[i])

    return answer

# Example usage:
nums = [1, 2, 3, 4, 5, 6]
print(largest_divisible_subset(nums))
```

### Explanation:
1. **Sorting**: The list is first sorted to ensure that we can easily check divisibility.
2. **Initialization**: We initialize the `answer` list with the first element.
"""

    test_spec = {
        "function_name": "largest_divisible_subset",
        "cases": [
            {"args": [[1, 2, 4]], "expected": [1, 2, 4]},
        ]
    }

    passed, error = evaluate_python_response(response, test_spec)
    # This should pass now - example calls are stripped during validation
    assert passed is True
    assert error is None
