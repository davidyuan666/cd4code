"""CD4Code: Per-Problem Transition Logging (Experiment 3)."""


class TransitionTracker:
    def __init__(self):
        self.problems = {}

    def record_raw(self, problem_id, passed):
        self._ensure(problem_id)
        self.problems[problem_id]["Raw"] = {"passed": passed}

    def record_selfdebug(self, problem_id, passed, rounds_used=0):
        self._ensure(problem_id)
        self.problems[problem_id]["SelfDebug"] = {
            "passed": passed, "rounds": rounds_used
        }

    def record_cd4code_mode(self, problem_id, mode_name, passed, tier_path=None):
        self._ensure(problem_id)
        entry = {"passed": passed}
        if tier_path:
            entry["tier_path"] = tier_path
        self.problems[problem_id][mode_name] = entry

    def _ensure(self, problem_id):
        if problem_id not in self.problems:
            self.problems[problem_id] = {"problem_id": problem_id}

    def compute_repaired_regressed(self, baseline_key="Raw", target_key="CD4Code"):
        repaired = []
        regressed = []
        unchanged_pass = []
        unchanged_fail = []

        for pid, data in self.problems.items():
            base = data.get(baseline_key, {}).get("passed")
            target = data.get(target_key, {}).get("passed")
            if base is None or target is None:
                continue
            if not base and target:
                repaired.append(pid)
            elif base and not target:
                regressed.append(pid)
            elif base:
                unchanged_pass.append(pid)
            else:
                unchanged_fail.append(pid)

        return {
            "repaired": repaired,
            "regressed": regressed,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
            "summary": {
                "n_repaired": len(repaired),
                "n_regressed": len(regressed),
                "n_unchanged_pass": len(unchanged_pass),
                "n_unchanged_fail": len(unchanged_fail),
            },
        }

    def compute_cross_mode_matrix(self, mode_keys):
        matrix = {}
        for pid, data in self.problems.items():
            row = {}
            for mk in mode_keys:
                row[mk] = data.get(mk, {}).get("passed")
            matrix[pid] = row
        return matrix

    def to_dict(self):
        return {
            "problems": self.problems,
            "mode_count": len(self.problems),
        }
