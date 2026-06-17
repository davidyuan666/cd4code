"""CD4Code: Evaluation and Metrics."""
import json
import os
import random
from datetime import datetime
from config import BOOTSTRAP_SAMPLES, BOOTSTRAP_CONFIDENCE


def compute_pass_at_k(n, c, k):
    """Unbiased estimator of Pass@k (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - (float(_comb(n - c, k)) / float(_comb(n, k)))


def _comb(n, k):
    if k > n or k < 0:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def bootstrap_ci(samples, n_bootstrap=BOOTSTRAP_SAMPLES, ci=BOOTSTRAP_CONFIDENCE):
    """Compute bootstrap confidence interval for a metric (e.g. scalar Pass@1).

    samples: list of (n_passed, n_total) tuples, one per problem.
    Returns (lower, upper, estimate).
    """
    if not samples or len(samples) < 2:
        return 0.0, 0.0, sum(p for p, _ in samples) / max(sum(t for _, t in samples), 1)

    random.seed(42)
    n = len(samples)
    boot_estimates = []
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        passed = sum(samples[i][0] for i in indices)
        total = sum(samples[i][1] for i in indices)
        boot_estimates.append(passed / max(total, 1))
    boot_estimates.sort()
    alpha = (1.0 - ci) / 2.0
    lower = boot_estimates[int(alpha * n_bootstrap)]
    upper = boot_estimates[int((1.0 - alpha) * n_bootstrap)]
    estimate = sum(p for p, _ in samples) / max(sum(t for _, t in samples), 1)
    return round(lower, 4), round(upper, 4), round(estimate, 4)


def bootstrap_paired_test(results_a, results_b, n_bootstrap=BOOTSTRAP_SAMPLES):
    """Bootstrap paired test for Pass@1 difference.

    results_a, results_b: lists of (passed_bool, passed_bool) per problem.
    Returns (p_value, mean_diff, ci_lower, ci_upper).
    """
    if not results_a or len(results_a) < 2:
        return 1.0, 0.0, 0.0, 0.0

    random.seed(42)
    n = len(results_a)
    diffs = [1.0 * results_a[i][0] - 1.0 * results_b[i][0] for i in range(n)]
    obs_diff = sum(diffs) / n

    boot_diffs = []
    for _ in range(n_bootstrap):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        boot_diffs.append(sum(diffs[i] for i in indices) / n)
    boot_diffs.sort()
    ci = BOOTSTRAP_CONFIDENCE
    alpha = (1.0 - ci) / 2.0
    ci_lower = boot_diffs[int(alpha * n_bootstrap)]
    ci_upper = boot_diffs[int((1.0 - alpha) * n_bootstrap)]

    # Two-sided p-value from bootstrap null distribution
    centered = [d - obs_diff for d in boot_diffs]
    p_value = sum(1 for d in centered if abs(d) >= abs(obs_diff)) / n_bootstrap

    return round(p_value, 4), round(obs_diff, 4), round(ci_lower, 4), round(ci_upper, 4)


class ExperimentEvaluator:
    def __init__(self, n_samples=1):
        self.n_samples = n_samples
        self.results = []
        self.token_stats = None

    def add_result(self, problem_id, samples, passed):
        self.results.append({
            "problem_id": problem_id,
            "n_samples": len(samples),
            "n_passed": sum(1 for p in passed if p),
            "passed_list": passed,
        })

    def set_token_stats(self, stats):
        self.token_stats = stats

    def compute_metrics(self):
        total = len(self.results)
        correct_at_1 = sum(1 for r in self.results if r["passed_list"][0]) if self.results else 0

        pass_at_1 = correct_at_1 / max(total, 1)

        all_n = [r["n_samples"] for r in self.results]
        all_c = [r["n_passed"] for r in self.results]
        pass_at_5_vals = []
        for n, c in zip(all_n, all_c):
            if n >= 5:
                val = compute_pass_at_k(n, c, 5)
            else:
                val = 1.0 if c > 0 else 0.0
            pass_at_5_vals.append(val)
        pass_at_5 = sum(pass_at_5_vals) / max(len(pass_at_5_vals), 1)

        defect_density = 1.0 - (sum(r["n_passed"] for r in self.results) /
                                max(sum(r["n_samples"] for r in self.results), 1))

        samples = [(r["n_passed"], r["n_samples"]) for r in self.results]
        pass_at_1_lower, pass_at_1_upper, _ = bootstrap_ci(samples)
        fdd_lower, fdd_upper, _ = bootstrap_ci(
            [(r["n_samples"] - r["n_passed"], r["n_samples"]) for r in self.results]
        )

        return {
            "total_problems": total,
            "pass_at_1": round(pass_at_1, 4),
            "pass_at_1_95ci": [pass_at_1_lower, pass_at_1_upper],
            "pass_at_5": round(pass_at_5, 4),
            "functional_defect_density": round(defect_density, 4),
            "fdd_95ci": [fdd_lower, fdd_upper],
        }

    def get_paired_for_test(self):
        return [(1.0 if r["passed_list"][0] else 0.0, r["n_samples"]) for r in self.results]

    def get_per_problem_results(self):
        return {r["problem_id"]: {"passed": r["passed_list"][0], "n_samples": r["n_samples"]}
                for r in self.results}

    def save_results(self, filepath):
        metrics = self.compute_metrics()
        output = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "details": self.results,
        }
        if self.token_stats:
            output["token_stats"] = self.token_stats
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return metrics
