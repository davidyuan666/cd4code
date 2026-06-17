"""CD4Code: Baseline Generators (Raw + Self-Debug)."""
import ast
import re
import subprocess
import tempfile
import os
from suppressor import CD4CodeSuppressor


class RawBaseline:
    """Pure LLM generation with no error suppression of any kind.

    Generates code, extracts the function, runs tests once, records pass/fail.
    No retries, no filtering, no self-repair.
    """

    def __init__(self):
        self.suppressor = CD4CodeSuppressor()
        self.passed = 0
        self.total = 0
        self.failures = []

    def evaluate(self, client, generate_fn, prompt, test_code, entry_point, token_tracker=None):
        self.total += 1
        codes = generate_fn(prompt, temperature=0.7, top_p=0.95, n=1, client=client,
                           token_tracker=token_tracker)
        if not codes or not codes[0] or not codes[0].strip():
            self.failures.append({"problem": self.total, "reason": "empty_response"})
            return False, ""

        code = self.suppressor._extract_function_code(codes[0])
        try:
            ast.parse(code)
        except SyntaxError:
            self.failures.append({"problem": self.total, "reason": "syntax_error"})
            return False, code

        passed, _ = self.suppressor._run_tests(code, test_code, entry_point)
        if not passed:
            self.failures.append({"problem": self.total, "reason": "test_failure"})
            return False, code

        self.passed += 1
        return True, code

    def get_stats(self):
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.passed / max(self.total, 1),
            "failures": len(self.failures),
        }


class SelfDebugBaseline:
    """Standard LLM self-repair: generate, test, fix, retest (up to 3 rounds).

    No biological framework, no tier structure, no global monitoring.
    Just the raw model trying to fix its own errors.
    """

    def __init__(self, max_rounds=3):
        self.max_rounds = max_rounds
        self.suppressor = CD4CodeSuppressor()
        self.passed = 0
        self.total = 0
        self.fixed_count = 0
        self.failures = []

    def evaluate(self, client, generate_fn, prompt, test_code, entry_point, token_tracker=None):
        self.total += 1
        codes = generate_fn(prompt, temperature=0.7, top_p=0.95, n=1, client=client,
                           token_tracker=token_tracker)
        if not codes or not codes[0] or not codes[0].strip():
            self.failures.append({"problem": self.total, "reason": "empty_response"})
            return False, ""

        code = self.suppressor._extract_function_code(codes[0])

        for round_idx in range(self.max_rounds):
            try:
                ast.parse(code)
            except SyntaxError:
                if round_idx < self.max_rounds - 1:
                    fix_prompt = (
                        f"The following Python code has a syntax error. "
                        f"Fix it and return ONLY the corrected code.\n\n"
                        f"```python\n{code}\n```\n"
                    )
                    new_codes = generate_fn(fix_prompt, temperature=0.5, top_p=0.9,
                                            n=1, client=client,
                                            token_tracker=token_tracker)
                    if new_codes and new_codes[0]:
                        code = self.suppressor._extract_function_code(new_codes[0])
                        continue
                self.failures.append({"problem": self.total, "reason": "syntax_error"})
                return False, code

            passed, error_msg = self.suppressor._run_tests(code, test_code, entry_point)
            if passed:
                if round_idx > 0:
                    self.fixed_count += 1
                self.passed += 1
                return True, code

            if round_idx < self.max_rounds - 1:
                fix_prompt = (
                    f"The following Python code failed tests:\n\n"
                    f"```python\n{code}\n```\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Write the corrected version. Return ONLY the code."
                )
                new_codes = generate_fn(fix_prompt, temperature=0.5, top_p=0.9,
                                        n=1, client=client,
                                        token_tracker=token_tracker)
                if new_codes and new_codes[0]:
                    code = self.suppressor._extract_function_code(new_codes[0])

        self.failures.append({"problem": self.total, "reason": "test_failure_after_retries"})
        return False, code

    def get_stats(self):
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.passed / max(self.total, 1),
            "fixed_count": self.fixed_count,
            "failures": len(self.failures),
        }
