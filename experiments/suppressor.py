"""CD4Code: Four-Tier Error Suppression Framework."""
import ast
import subprocess
import tempfile
import os
import re
from config import (
    TIER1_CONFIDENCE_THRESHOLD, TIER2_LINTER, TIER2_TYPE_CHECKER,
    TIER3_MAX_RETRIES, TIER4_DEFECT_THRESHOLD,
    TIER4_CONSERVATIVE_TEMP, TIER4_CONSERVATIVE_TOPP
)


class CD4CodeSuppressor:
    def __init__(self, disabled_tiers=None, tier4_threshold_override=None,
                 t1_min_length_override=None, t1_rep_window_override=None):
        self.tier1_threshold = TIER1_CONFIDENCE_THRESHOLD
        self.tier4_threshold = tier4_threshold_override if tier4_threshold_override is not None else TIER4_DEFECT_THRESHOLD
        self.t1_min_length = t1_min_length_override if t1_min_length_override is not None else 5
        self.t1_rep_window = t1_rep_window_override if t1_rep_window_override is not None else 10
        self.failure_count = 0
        self.total_count = 0
        self.tier_stats = {"t1_filtered": 0, "t2_filtered": 0,
                          "t3_degraded": 0, "t4_activated": 0}
        self.t4_activation_history = []
        self.t4_activation_timeseries = []
        self.disabled_tiers = set(disabled_tiers or [])
        self.per_problem_tier_path = {}

    def tier1_proofreading(self, code):
        """Token-level confidence filtering (DNA Polymerase Proofreading analog).

        Rejects code with obviously malformed patterns that low-confidence
        token sampling would produce: extreme repetition, empty blocks, etc.
        """
        if not code or len(code.strip()) < self.t1_min_length:
            self.tier_stats["t1_filtered"] += 1
            self.record_failure()
            return None

        lines = code.strip().split('\n')

        for i in range(len(lines) - self.t1_rep_window):
            window = lines[i:i+self.t1_rep_window]
            if len(set(window)) == 1 and window[0].strip():
                self.tier_stats["t1_filtered"] += 1
                self.record_failure()
                return None

        return code

    def tier2_mismatch_repair(self, code):
        """Structural static analysis (Mismatch Repair analog).

        Checks syntax validity via AST parsing.
        """
        code = self._extract_function_code(code)
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.tier_stats["t2_filtered"] += 1
            self.record_failure()
            return None, str(e)

        return code, None

    def _extract_function_code(self, code, entry_point=None):
        code = code.replace('\r\n', '\n').strip()

        fence_start = re.search(r'^```(?:python\w*)?\s*$', code, re.MULTILINE)
        if fence_start:
            after_open = code[fence_start.end():]
            fence_end = re.search(r'^```\s*$', after_open, re.MULTILINE)
            if fence_end:
                code = after_open[:fence_end.start()]
            else:
                code = after_open
            code = code.strip()

        # If still has preamble text, try to find actual code start
        if not re.match(r'^\s*(def\s|\w+\s*=\s*|import\s|\bfrom\s|class\s)', code):
            match = re.search(
                r'(?:^|\n)(def\s+\w+|import\s+\w+|from\s+\w+\s+import|\bclass\s+\w+)', code)
            if match:
                code = code[match.start():].lstrip('\n')

        return code

    def tier3_test_degradation(self, code, test_code, entry_point, client=None, generate_fn=None, token_tracker=None):
        """Test-driven discard and regeneration (Ubiquitin-Proteasome analog)."""
        code = self._extract_function_code(code)

        for attempt in range(TIER3_MAX_RETRIES):
            passed, error_msg = self._run_tests(code, test_code, entry_point)
            if passed:
                return code, True

            if attempt < TIER3_MAX_RETRIES - 1 and client and generate_fn:
                fix_prompt = (
                    f"The following Python code failed tests:\n\n```python\n{code}\n```\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Write the corrected version of the code. Return ONLY the code."
                )
                new_codes = generate_fn(fix_prompt, client=client, token_tracker=token_tracker)
                if new_codes and new_codes[0]:
                    code = self._extract_function_code(new_codes[0])

        self.tier_stats["t3_degraded"] += 1
        return None, False

    def _run_tests(self, code, test_code, entry_point):
        full_code = f"{code}\n\n{test_code}\n\n"
        if entry_point:
            full_code += f"if __name__ == '__main__':\n"
            if 'def check(' in test_code:
                full_code += f"    check({entry_point})\n"
            else:
                full_code += "    pass\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                         encoding='utf-8') as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ['python', tmp_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return True, None
            return False, result.stderr[:500] if result.stderr else result.stdout[:500]
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def tier4_global_monitor(self):
        """Global defect density monitoring (ER Stress Response analog).

        Returns True if conservative mode should be activated.
        """
        ratio = self.failure_count / max(self.total_count, 1)
        if ratio >= self.tier4_threshold and self.total_count > 10:
            self.tier_stats["t4_activated"] += 1
            return True
        return False

    def record_failure(self):
        self.failure_count += 1

    def process(self, code, test_code=None, entry_point=None,
                client=None, generate_fn=None, token_tracker=None,
                problem_id=None):
        """Apply CD4Code tiers to a generated code sample.

        Tiers listed in self.disabled_tiers are bypassed (code passes through).
        Use disabled_tiers=[1,2,3,4] for ablation studies.
        """
        result = {
            "original_code": code,
            "passed_t1": False,
            "passed_t2": False,
            "passed_t3": False,
            "t4_conservative": False,
            "final_code": None,
            "success": False,
            "tier_path": {"t1": "bypass", "t2": "bypass", "t3": "bypass", "t4": "bypass"},
        }

        self.total_count += 1
        pid = problem_id if problem_id is not None else self.total_count

        code = self._extract_function_code(code)

        if 4 not in self.disabled_tiers:
            result["t4_conservative"] = self.tier4_global_monitor()
            self.t4_activation_history.append(result["t4_conservative"])
            result["tier_path"]["t4"] = "active" if result["t4_conservative"] else "inactive"
            if result["t4_conservative"]:
                self.t4_activation_timeseries.append({
                    "problem_id": pid,
                    "defect_ratio": round(self.failure_count / max(self.total_count, 1), 4),
                    "total_count": self.total_count,
                })
        else:
            result["tier_path"]["t4"] = "bypass"

        if 1 not in self.disabled_tiers:
            code = self.tier1_proofreading(code)
            if code is None:
                result["tier_path"]["t1"] = "filtered"
                self.per_problem_tier_path[pid] = result["tier_path"]
                return result
            result["tier_path"]["t1"] = "passed"
            result["passed_t1"] = True
        else:
            result["passed_t1"] = True

        if 2 not in self.disabled_tiers:
            code, syntax_error = self.tier2_mismatch_repair(code)
            if code is None:
                result["tier_path"]["t2"] = "filtered"
                self.per_problem_tier_path[pid] = result["tier_path"]
                return result
            result["tier_path"]["t2"] = "passed"
            result["passed_t2"] = True
        else:
            code = self._extract_function_code(code)
            result["passed_t2"] = True

        if test_code and 3 not in self.disabled_tiers:
            code, passed = self.tier3_test_degradation(
                code, test_code, entry_point, client, generate_fn, token_tracker
            )
            if not passed:
                result["tier_path"]["t3"] = "degraded"
                self.per_problem_tier_path[pid] = result["tier_path"]
                self.record_failure()
                return result
            result["tier_path"]["t3"] = "passed"
            result["passed_t3"] = True
        elif test_code:
            passed, _ = self._run_tests(code, test_code, entry_point)
            if not passed:
                result["tier_path"]["t3"] = "failed"
                self.per_problem_tier_path[pid] = result["tier_path"]
                self.record_failure()
                return result
            result["tier_path"]["t3"] = "passed"
            result["passed_t3"] = True

        result["final_code"] = code
        result["success"] = True
        self.per_problem_tier_path[pid] = result["tier_path"]
        return result

    def get_stats(self):
        return self.tier_stats.copy()

    def get_defect_ratio(self):
        return self.failure_count / max(self.total_count, 1)

    def get_activation_history(self):
        return list(self.t4_activation_history)

    def get_activation_timeseries(self):
        return list(self.t4_activation_timeseries)

    def get_per_problem_tier_path(self):
        return dict(self.per_problem_tier_path)
