#!/usr/bin/env python
"""CD4Code: Full Experiment Pipeline with Baselines and Ablation.

Usage:
    python run_all.py                                   # Run all experiments
    python run_all.py --dataset humaneval               # HumanEval only
    python run_all.py --dataset mbpp                     # MBPP only
    python run_all.py --dataset mbpp --mbpp-full         # All 974 MBPP problems
    python run_all.py --mode raw,cd4code                 # Specific modes
    python run_all.py --max-problems 20                  # Quick test
    python run_all.py --t4stress                         # All Tier4 stress tests
    python run_all.py --threshold-sweep                  # Threshold sensitivity
    python run_all.py --mode cd4code --t4-threshold 0.25 # Override threshold

Modes: raw, selfdebug, t3only, t123, cd4code,
       t4stress_hard, t4stress_heat, t4stress_perturb, t4stress_combined
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    TIER4_CONSERVATIVE_TEMP, TIER4_CONSERVATIVE_TOPP,
    STRESS_TEMPERATURE, STRESS_TOP_P, STRESS_HARD_PROBLEM_COUNT,
    HUMANEVAL_PATH, MBPP_PATH, RESULTS_DIR, FIGURES_DIR,
    THRESHOLD_SWEEP_VALUES, MBPP_DEFAULT_SAMPLES, MBPP_RANDOM_SEED,
    STRESS_PERTURB_RATE, STRESS_WEAKER_TEMP, STRESS_WEAKER_TOPP,
    STRESS_WEAKER_MAX_TOKENS,
)
from generate import create_client, generate_code, load_humaneval, load_mbpp, TokenTracker, perturb_prompt
from suppressor import CD4CodeSuppressor
from baselines import RawBaseline, SelfDebugBaseline
from evaluate import ExperimentEvaluator, bootstrap_paired_test
from transitions import TransitionTracker
from plot import (
    plot_defect_density_curve, plot_tier_survival, plot_ablation_comparison,
    plot_t4stress_comparison, plot_cost_comparison,
    plot_t4_activation_timeline, plot_transition_matrix,
    plot_threshold_sensitivity,
)


def _get_prompt_humaneval(problem):
    return problem.get("prompt", "")

def _get_prompt_mbpp(problem):
    text = problem.get("text", "")
    test_list = problem.get("test_list", [])
    prompt = (
        f"Write a Python function to solve the following task:\n\n"
        f"{text}\n\n"
        f"The function should pass these test cases:\n"
    )
    for t in test_list[:3]:
        prompt += f"  {t}\n"
    prompt += "\nReturn ONLY the Python code."
    return prompt

def _get_test_humaneval(problem):
    return problem.get("test", ""), problem.get("entry_point", "")

def _get_test_mbpp(problem):
    tests = problem.get("test_list", [])
    test_func_name = problem.get("test_setup_code", "")
    entry = problem.get("entry_point", problem.get("task_id", ""))
    code = ""
    if test_func_name:
        code += f"{test_func_name}\n\n"
    code += f"\n".join(tests)
    return code, entry


def _select_hardest(problems, test_fn, entry_fn, client, n=30, prompt_fn=None):
    """Select n hardest problems by raw pass rate (quick evaluation)."""
    if prompt_fn is None:
        prompt_fn = _get_prompt_humaneval
    scores = []
    for idx, problem in enumerate(problems):
        prompt = prompt_fn(problem)
        test_code, entry = test_fn(problem), entry_fn(problem)
        codes = generate_code(prompt, temperature=0.7, top_p=0.95, n=1, client=client)
        if not codes or not codes[0] or not codes[0].strip():
            scores.append((idx, True))
            continue
        sup = CD4CodeSuppressor()
        code = sup._extract_function_code(codes[0])
        import ast
        try:
            ast.parse(code)
        except SyntaxError:
            scores.append((idx, True))
            continue
        passed, _ = sup._run_tests(code, test_code, entry)
        scores.append((idx, not passed))
        if (idx + 1) % 10 == 0:
            print(f"  Difficulty scan: {idx + 1}/{len(problems)}")
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [problems[i] for i, _ in scores[:n]]
    print(f"  Selected {len(selected)} hardest problems")
    return selected


def _run_cd4code_mode(problems, prompt_fn, test_fn, entry_fn, client,
                      disabled_tiers=None, use_conservative_mode=False,
                      token_tracker=None, stress_temp=None, stress_top_p=None,
                      stress_max_tokens=None,
                      tier4_threshold_override=None,
                      t1_min_length_override=None,
                      t1_rep_window_override=None,
                      transition_tracker=None, mode_label="CD4Code"):
    """Run CD4Code with optional tier disabling for ablation."""
    mode_name = "CD4Code" if disabled_tiers is None else \
        ("T1+T2+T3" if disabled_tiers == [4] else
         ("T3-Only" if set(disabled_tiers) == {1, 2, 4} else
          f"Ablation:{disabled_tiers}"))
    print(f"\n--- {mode_name} ---")

    suppressor = CD4CodeSuppressor(
        disabled_tiers=disabled_tiers,
        tier4_threshold_override=tier4_threshold_override,
        t1_min_length_override=t1_min_length_override,
        t1_rep_window_override=t1_rep_window_override,
    )
    evaluator = ExperimentEvaluator()
    defect_history = []
    temp = stress_temp if stress_temp is not None else DEFAULT_TEMPERATURE
    top_p = stress_top_p if stress_top_p is not None else DEFAULT_TOP_P
    max_tk = stress_max_tokens if stress_max_tokens is not None else None

    for idx, problem in enumerate(problems):
        pid = problem.get("task_id", problem.get("entry_point", str(idx)))
        prompt = prompt_fn(problem)
        test_code, entry = test_fn(problem), entry_fn(problem)

        gen_kwargs = dict(temperature=temp, top_p=top_p, n=1, client=client,
                          token_tracker=token_tracker)
        if max_tk is not None:
            gen_kwargs["max_tokens"] = max_tk
        code = generate_code(prompt, **gen_kwargs)
        if not code or not code[0] or not code[0].strip():
            suppressor.total_count += 1
            suppressor.record_failure()
            evaluator.add_result(idx, [""], [False])
            defect_history.append(suppressor.get_defect_ratio())
            if transition_tracker:
                transition_tracker.record_cd4code_mode(
                    pid, mode_label, False, {"t1": "empty", "t2": "empty", "t3": "empty", "t4": "bypass"})
            continue
        code = code[0]

        result = suppressor.process(
            code, test_code=test_code, entry_point=entry,
            client=client, generate_fn=generate_code,
            token_tracker=token_tracker,
            problem_id=pid,
        )
        passed = result["success"]
        if passed:
            code = result["final_code"]
        defect_history.append(suppressor.get_defect_ratio())

        if use_conservative_mode and result["t4_conservative"]:
            temp = TIER4_CONSERVATIVE_TEMP
            top_p = TIER4_CONSERVATIVE_TOPP

        evaluator.add_result(idx, [code], [passed])
        if transition_tracker:
            transition_tracker.record_cd4code_mode(
                pid, mode_label, passed, result.get("tier_path"))

        if (idx + 1) % 5 == 0:
            metrics = evaluator.compute_metrics()
            t4_act = sum(1 for a in suppressor.t4_activation_history if a)
            print(f"  [{idx + 1}/{len(problems)}] Pass@1: {metrics['pass_at_1']:.3f}, "
                  f"FDD: {metrics['functional_defect_density']:.3f}, "
                  f"T4 activated: {t4_act}")

    metrics = evaluator.compute_metrics()
    stats = suppressor.get_stats()
    stats["total"] = suppressor.total_count
    stats["t4_activated_total"] = sum(1 for a in suppressor.t4_activation_history if a)
    stats["t4_activation_history"] = suppressor.get_activation_history()
    stats["t4_activation_timeseries"] = suppressor.get_activation_timeseries()
    stats["per_problem_tier_path"] = suppressor.get_per_problem_tier_path()
    print(f"  Final: Pass@1={metrics['pass_at_1']:.3f}, "
          f"Pass@5={metrics['pass_at_5']:.3f}, "
          f"FDD={metrics['functional_defect_density']:.3f}")
    print(f"  Pass@1 95%CI: {metrics['pass_at_1_95ci']}")
    print(f"  Tier stats: {stats}")

    return {
        "config": mode_name,
        "metrics": metrics,
        "stats": stats,
        "defect_history": defect_history,
        "evaluator": evaluator,
    }


def _run_raw_mode(problems, prompt_fn, test_fn, entry_fn, client, mode_name="Raw",
                  token_tracker=None, transition_tracker=None):
    print(f"\n--- {mode_name} ---")
    baseline = RawBaseline()
    evaluator = ExperimentEvaluator()
    defect_history = []

    for idx, problem in enumerate(problems):
        pid = problem.get("task_id", problem.get("entry_point", str(idx)))
        prompt = prompt_fn(problem)
        test_code, entry = test_fn(problem), entry_fn(problem)
        passed, _ = baseline.evaluate(
            client, generate_code, prompt, test_code, entry, token_tracker=token_tracker)
        evaluator.add_result(idx, [""], [passed])
        defect_history.append(1.0 - (baseline.passed / max(baseline.total, 1)))
        if transition_tracker:
            transition_tracker.record_raw(pid, passed)

        if (idx + 1) % 5 == 0:
            m = evaluator.compute_metrics()
            print(f"  [{idx + 1}/{len(problems)}] Pass@1: {m['pass_at_1']:.3f}, "
                  f"FDD: {m['functional_defect_density']:.3f}")

    metrics = evaluator.compute_metrics()
    stats = baseline.get_stats()
    print(f"  Final: Pass@1={metrics['pass_at_1']:.3f}, "
          f"Pass@5={metrics['pass_at_5']:.3f}, "
          f"FDD={metrics['functional_defect_density']:.3f}")
    print(f"  Pass@1 95%CI: {metrics['pass_at_1_95ci']}")
    print(f"  Stats: {stats}")
    return {
        "config": mode_name,
        "metrics": metrics,
        "stats": stats,
        "defect_history": defect_history,
        "evaluator": evaluator,
    }


def _run_selfdebug_mode(problems, prompt_fn, test_fn, entry_fn, client,
                        token_tracker=None, transition_tracker=None):
    print(f"\n--- Self-Debug ---")
    baseline = SelfDebugBaseline(max_rounds=3)
    evaluator = ExperimentEvaluator()
    defect_history = []

    for idx, problem in enumerate(problems):
        pid = problem.get("task_id", problem.get("entry_point", str(idx)))
        prompt = prompt_fn(problem)
        test_code, entry = test_fn(problem), entry_fn(problem)
        passed, _ = baseline.evaluate(
            client, generate_code, prompt, test_code, entry, token_tracker=token_tracker)
        evaluator.add_result(idx, [""], [passed])
        defect_history.append(1.0 - (baseline.passed / max(baseline.total, 1)))
        if transition_tracker:
            transition_tracker.record_selfdebug(pid, passed)

        if (idx + 1) % 5 == 0:
            m = evaluator.compute_metrics()
            print(f"  [{idx + 1}/{len(problems)}] Pass@1: {m['pass_at_1']:.3f}, "
                  f"FDD: {m['functional_defect_density']:.3f}")

    metrics = evaluator.compute_metrics()
    stats = baseline.get_stats()
    print(f"  Final: Pass@1={metrics['pass_at_1']:.3f}, "
          f"Pass@5={metrics['pass_at_5']:.3f}, "
          f"FDD={metrics['functional_defect_density']:.3f}")
    print(f"  Pass@1 95%CI: {metrics['pass_at_1_95ci']}")
    print(f"  Stats: {stats}")
    return {
        "config": "Self-Debug",
        "metrics": metrics,
        "stats": stats,
        "defect_history": defect_history,
        "evaluator": evaluator,
    }


def _run_threshold_sweep(problems, prompt_fn, test_fn, entry_fn, client,
                         ds_name, threshold_values, token_tracker=None):
    print(f"\n{'=' * 50}")
    print(f"Threshold Sensitivity Sweep ({ds_name})")
    print(f"Thresholds: {threshold_values}")
    print(f"{'=' * 50}")

    sweep_results = {}
    for tval in threshold_values:
        print(f"\n  --- T4 Threshold = {tval} ---")
        result = _run_cd4code_mode(
            problems, prompt_fn, test_fn, entry_fn, client,
            disabled_tiers=None, use_conservative_mode=True,
            token_tracker=token_tracker,
            tier4_threshold_override=tval,
            mode_label=f"CD4Code_T{tval}",
        )
        result["threshold"] = tval
        sweep_results[f"CD4Code_T{tval:.1f}"] = result

    print(f"\n  Threshold Sweep Summary:")
    print(f"  {'Threshold':<12} {'Pass@1':<10} {'FDD':<10} {'T4 Act':<10} {'Filtered':<10}")
    print(f"  {'-' * 52}")
    for tval in threshold_values:
        key = f"CD4Code_T{tval:.1f}"
        r = sweep_results[key]
        m = r["metrics"]
        s = r["stats"]
        filtered = s.get("t1_filtered", 0) + s.get("t2_filtered", 0)
        print(f"  {tval:<12.1f} {m['pass_at_1']:<10.3f} {m['functional_defect_density']:<10.3f} "
              f"{s.get('t4_activated_total', 0):<10} {filtered:<10}")

    return sweep_results


def _get_perturbed_prompt_fn(original_prompt_fn, perturb_rate=STRESS_PERTURB_RATE):
    def perturbed_fn(problem):
        return perturb_prompt(original_prompt_fn(problem), perturb_rate)
    return perturbed_fn


def run_experiments(args):
    client = create_client()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_modes = ["raw", "selfdebug", "t3only", "t123", "cd4code",
                 "t4stress_hard", "t4stress_heat", "t4stress_combined",
                 "t4stress_perturb"]
    if args.mode:
        modes = [m.strip() for m in args.mode.split(",")]
    else:
        modes = all_modes

    threshold_sweep_mode = args.threshold_sweep or "threshold_sweep" in (args.mode or "")

    datasets = []
    mbpp_sample_info = None
    if not args.dataset or args.dataset == "humaneval":
        datasets.append(("HumanEval", load_humaneval(HUMANEVAL_PATH),
                         _get_prompt_humaneval, _get_test_humaneval))
    if not args.dataset or args.dataset == "mbpp":
        n_samples = None
        if args.mbpp_full:
            n_samples = 9999
        elif args.mbpp_samples:
            n_samples = args.mbpp_samples
        mbpp_problems, mbpp_sample_info = load_mbpp(MBPP_PATH, n_samples=n_samples)
        datasets.append(("MBPP", mbpp_problems, _get_prompt_mbpp, _get_test_mbpp))

    if args.max_problems:
        for i in range(len(datasets)):
            name, probs, fn, tf = datasets[i]
            datasets[i] = (name, probs[:args.max_problems], fn, tf)

    all_results = {}
    all_token_trackers = {}

    for ds_name, problems, prompt_fn, test_fn in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} ({len(problems)} problems)")
        print(f"{'=' * 60}")

        if ds_name == "HumanEval":
            te_fn, en_fn = lambda p: _get_test_humaneval(p)[0], \
                           lambda p: _get_test_humaneval(p)[1]
        else:
            te_fn = lambda p: _get_test_mbpp(p)[0]
            en_fn = lambda p: _get_test_mbpp(p)[1]

        trans_tracker = TransitionTracker()

        for mode in modes:
            tt = TokenTracker()
            key = None
            result = None

            if mode == "raw":
                result = _run_raw_mode(problems, prompt_fn, te_fn, en_fn,
                                       client, "Raw", token_tracker=tt,
                                       transition_tracker=trans_tracker)
                key = f"{ds_name}_Raw"

            elif mode == "selfdebug":
                result = _run_selfdebug_mode(problems, prompt_fn, te_fn,
                                             en_fn, client, token_tracker=tt,
                                             transition_tracker=trans_tracker)
                key = f"{ds_name}_SelfDebug"

            elif mode == "t3only":
                result = _run_cd4code_mode(
                    problems, prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=[1, 2, 4], use_conservative_mode=False,
                    token_tracker=tt, transition_tracker=trans_tracker,
                    mode_label="T3Only")
                key = f"{ds_name}_T3Only"

            elif mode == "t123":
                result = _run_cd4code_mode(
                    problems, prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=[4], use_conservative_mode=False,
                    token_tracker=tt, transition_tracker=trans_tracker,
                    mode_label="T123")
                key = f"{ds_name}_T123"

            elif mode == "cd4code":
                result = _run_cd4code_mode(
                    problems, prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=None, use_conservative_mode=True,
                    token_tracker=tt, transition_tracker=trans_tracker,
                    tier4_threshold_override=args.t4_threshold,
                    mode_label="CD4Code")
                key = f"{ds_name}_CD4Code"

            elif mode == "t4stress_hard":
                print("\n  [T4Stress-Hard] Selecting hardest problems...")
                hard_problems = _select_hardest(
                    problems, te_fn, en_fn, client, n=STRESS_HARD_PROBLEM_COUNT,
                    prompt_fn=prompt_fn)
                print(f"  [T4Stress-Hard] Running on {len(hard_problems)} "
                      f"hard problems with default temperature...")
                result = _run_cd4code_mode(
                    hard_problems, prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=None, use_conservative_mode=True,
                    token_tracker=tt, mode_label=f"{ds_name}_T4Stress_Hard")
                key = f"{ds_name}_T4Stress_Hard"

            elif mode == "t4stress_heat":
                print("\n  [T4Stress-Heat] Running with high temperature "
                      f"(T={STRESS_TEMPERATURE}, top_p={STRESS_TOP_P})...")
                result = _run_cd4code_mode(
                    problems, prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=None, use_conservative_mode=True,
                    token_tracker=tt,
                    stress_temp=STRESS_TEMPERATURE,
                    stress_top_p=STRESS_TOP_P,
                    mode_label=f"{ds_name}_T4Stress_Heat")
                key = f"{ds_name}_T4Stress_Heat"

            elif mode == "t4stress_perturb":
                print("\n  [T4Stress-Perturb] Running with perturbed prompts "
                      f"(rate={STRESS_PERTURB_RATE})...")
                perturbed_prompt_fn = _get_perturbed_prompt_fn(
                    prompt_fn, perturb_rate=STRESS_PERTURB_RATE)
                result = _run_cd4code_mode(
                    problems, perturbed_prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=None, use_conservative_mode=True,
                    token_tracker=tt,
                    mode_label=f"{ds_name}_T4Stress_Perturb")
                key = f"{ds_name}_T4Stress_Perturb"

            elif mode == "t4stress_combined":
                print("\n  [T4Stress-Combined] Hardest problems + high "
                      f"temperature (T={STRESS_TEMPERATURE}, top_p={STRESS_TOP_P})...")
                hard_problems = _select_hardest(
                    problems, te_fn, en_fn, client, n=STRESS_HARD_PROBLEM_COUNT,
                    prompt_fn=prompt_fn)
                print(f"  [T4Stress-Combined] Running on {len(hard_problems)} "
                      "hard problems with high temperature...")
                result = _run_cd4code_mode(
                    hard_problems, prompt_fn, te_fn, en_fn, client,
                    disabled_tiers=None, use_conservative_mode=True,
                    token_tracker=tt,
                    stress_temp=STRESS_TEMPERATURE,
                    stress_top_p=STRESS_TOP_P,
                    mode_label=f"{ds_name}_T4Stress_Combined")
                key = f"{ds_name}_T4Stress_Combined"

            if key is None or result is None:
                continue

            result["evaluator"].set_token_stats(tt.snapshot())
            all_results[key] = result
            all_token_trackers[key] = tt
            result_path = os.path.join(RESULTS_DIR, f"{key}.json")
            result["evaluator"].save_results(result_path)

        if threshold_sweep_mode:
            sweep_vals = args.t4_threshold_sweep_vals if args.t4_threshold_sweep_vals else THRESHOLD_SWEEP_VALUES
            sweep_results = _run_threshold_sweep(
                problems, prompt_fn, te_fn, en_fn, client,
                ds_name, sweep_vals)
            for sweep_key, sweep_result in sweep_results.items():
                sweep_result["evaluator"].set_token_stats(TokenTracker().snapshot())
                all_results[f"{ds_name}_{sweep_key}"] = sweep_result
                all_token_trackers[f"{ds_name}_{sweep_key}"] = TokenTracker()

        transition_path = os.path.join(RESULTS_DIR, f"{ds_name}_transition_log.json")
        repaired_regressed = trans_tracker.compute_repaired_regressed()
        transition_data = trans_tracker.to_dict()
        transition_data["repaired_regressed"] = repaired_regressed
        if mbpp_sample_info and ds_name == "MBPP":
            transition_data["sample_info"] = mbpp_sample_info
        with open(transition_path, 'w', encoding='utf-8') as f:
            json.dump(transition_data, f, indent=2, ensure_ascii=False)
        print(f"\nTransition log saved: {transition_path}")
        rr = repaired_regressed["summary"]
        print(f"  Repaired (fail->pass): {rr['n_repaired']}")
        print(f"  Regressed (pass->fail): {rr['n_regressed']}")
        print(f"  Unchanged (pass): {rr['n_unchanged_pass']}")
        print(f"  Unchanged (fail): {rr['n_unchanged_fail']}")

    # Token cost summary
    print("\n" + "=" * 60)
    print("API Token & Cost Summary")
    print("=" * 60)
    print(f"{'Mode':<28} {'Calls':>6} {'Input Tok':>12} {'Output Tok':>12} {'Cost (USD)':>12}")
    print("-" * 70)
    token_records = []
    for key, tt in all_token_trackers.items():
        s = tt.snapshot()
        token_records.append((key, s))
        print(f"{key:<28} {s['api_calls']:>6} {s['input_tokens']:>12,} "
              f"{s['output_tokens']:>12,} ${s['cost_usd']:>10.4f}")
    if len(token_records) > 1:
        records_sorted = sorted(token_records, key=lambda x: x[1]['cost_usd'])
        cheapest = records_sorted[0]
        costliest = records_sorted[-1]
        savings = costliest[1]['cost_usd'] - cheapest[1]['cost_usd']
        if savings > 0:
            pct = savings / costliest[1]['cost_usd'] * 100
            print(f"\n  Cost range: ${cheapest[1]['cost_usd']:.4f} ("
                  f"{cheapest[0]}) to ${costliest[1]['cost_usd']:.4f} ("
                  f"{costliest[0]}), spread ${savings:.4f} ({pct:.1f}%)")

    # Statistical significance report
    print("\n" + "=" * 60)
    print("Statistical Significance (Bootstrap Paired Test, 95% CI)")
    print("=" * 60)
    for ds_name, _, _, _ in datasets:
        raw_key = f"{ds_name}_Raw"
        cd4code_key = f"{ds_name}_CD4Code"
        if raw_key in all_results and cd4code_key in all_results:
            raw_paired = all_results[raw_key]["evaluator"].get_paired_for_test()
            cd4code_paired = all_results[cd4code_key]["evaluator"].get_paired_for_test()
            p_val, mean_diff, ci_lower, ci_upper = bootstrap_paired_test(
                cd4code_paired, raw_paired)
            print(f"  {ds_name}: CD4Code vs Raw")
            print(f"    Pass@1 mean difference: {mean_diff:+.4f}")
            print(f"    95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
            print(f"    p-value (bootstrap): {p_val:.4f}"
                  f"{' *' if p_val < 0.05 else ''}"
                  f"{' **' if p_val < 0.01 else ''}"
                  f"{' ***' if p_val < 0.001 else ''}")
            raw_metrics = all_results[raw_key]["metrics"]
            cd4_metrics = all_results[cd4code_key]["metrics"]
            print(f"    Raw     Pass@1={raw_metrics['pass_at_1']:.4f} "
                  f"95%CI {raw_metrics['pass_at_1_95ci']}")
            print(f"    CD4Code Pass@1={cd4_metrics['pass_at_1']:.4f} "
                  f"95%CI {cd4_metrics['pass_at_1_95ci']}")

    # T4 stress activation summary
    print("\n" + "=" * 60)
    print("Tier4 Stress Test Activation Summary")
    print("=" * 60)
    for key in sorted(all_results.keys()):
        if "T4Stress" in key:
            stats = all_results[key]["stats"]
            activated = stats.get("t4_activated_total", 0)
            total = stats.get("total", 0)
            fdd = all_results[key]["metrics"]["functional_defect_density"]
            print(f"  {key}: T4 activated {activated}/{total} times, "
                  f"FDD={fdd:.4f}")

    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)

        for ds_name, _, _, _ in datasets:
            raw_key = f"{ds_name}_Raw"
            cd4code_key = f"{ds_name}_CD4Code"
            if raw_key in all_results and cd4code_key in all_results:
                fig_path = os.path.join(FIGURES_DIR, f"{ds_name}_defect_density_curve.pdf")
                plot_defect_density_curve(
                    all_results[raw_key]["defect_history"],
                    all_results[cd4code_key]["defect_history"],
                    fig_path
                )

        cd4code_key = f"{datasets[0][0]}_CD4Code"
        if cd4code_key in all_results:
            fig_path = os.path.join(FIGURES_DIR, "tier_survival.pdf")
            plot_tier_survival(all_results[cd4code_key]["stats"], fig_path)

        ablation_data = {v["config"]: v["metrics"] for k, v in all_results.items()
                        if k.startswith(datasets[0][0])}
        if ablation_data:
            fig_path = os.path.join(FIGURES_DIR, "ablation_comparison.pdf")
            plot_ablation_comparison(ablation_data, fig_path)

        t4stress_data = {}
        for k, v in all_results.items():
            if "T4Stress" in k:
                label = k.split("_", 1)[1].replace("_", " ")
                t4stress_data[label] = {
                    "metrics": v["metrics"],
                    "stats": v["stats"],
                    "defect_history": v["defect_history"],
                }
        if t4stress_data:
            fig_path = os.path.join(FIGURES_DIR, "t4stress_comparison.pdf")
            plot_t4stress_comparison(t4stress_data, fig_path)

        token_data = {}
        for k, v in all_token_trackers.items():
            if "T4Stress" not in k:
                token_data[k] = v.snapshot()
        if len(token_data) > 1:
            fig_path = os.path.join(FIGURES_DIR, "cost_comparison.pdf")
            plot_cost_comparison(token_data, fig_path)

        cd4code_key = f"{datasets[0][0]}_CD4Code"
        if cd4code_key in all_results:
            ts = all_results[cd4code_key]["stats"].get("t4_activation_timeseries", [])
            if ts:
                fig_path = os.path.join(FIGURES_DIR, "t4_activation_timeline.pdf")
                plot_t4_activation_timeline(ts, fig_path)

        for ds_name, _, _, _ in datasets:
            transition_path = os.path.join(RESULTS_DIR, f"{ds_name}_transition_log.json")
            if os.path.exists(transition_path):
                with open(transition_path, 'r', encoding='utf-8') as f:
                    trans_data = json.load(f)
                rr = trans_data.get("repaired_regressed", {})
                if rr.get("summary", {}).get("n_repaired", 0) > 0 or rr.get("summary", {}).get("n_regressed", 0) > 0:
                    fig_path = os.path.join(FIGURES_DIR, f"{ds_name}_transition_matrix.pdf")
                    plot_transition_matrix(rr, fig_path)

        sweep_keys = sorted([k for k in all_results if "CD4Code_T" in k])
        if sweep_keys:
            sweep_data = {}
            for k in sweep_keys:
                entry = all_results[k]
                threshold = entry.get("threshold")
                if threshold is not None:
                    sweep_data[f"T{threshold:.1f}"] = entry
            if sweep_data:
                fig_path = os.path.join(FIGURES_DIR, "threshold_sensitivity.pdf")
                plot_threshold_sensitivity(sweep_data, fig_path)

        from framework_diagram import draw_framework_diagram
        draw_framework_diagram(os.path.join(FIGURES_DIR, "framework_diagram.pdf"))

        print("\nFigures generated in:", FIGURES_DIR)
    except Exception as e:
        import traceback
        print(f"Warning: figure generation failed: {e}")
        traceback.print_exc()

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    summary = {}
    if mbpp_sample_info:
        summary["_mbpp_sample_info"] = mbpp_sample_info
    for k, v in all_results.items():
        entry = {"config": v["config"], "metrics": v["metrics"], "stats": v["stats"]}
        if k in all_token_trackers:
            entry["token_stats"] = all_token_trackers[k].snapshot()
        summary[k] = entry
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CD4Code Experiment Pipeline")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"],
                       help="Run specific dataset only")
    parser.add_argument("--mode", type=str,
                       help="Comma-separated modes: raw,selfdebug,t3only,t123,cd4code,"
                            "t4stress_hard,t4stress_heat,t4stress_perturb,t4stress_combined")
    parser.add_argument("--max-problems", type=int,
                       help="Limit number of problems")
    parser.add_argument("--t4stress", action="store_true",
                       help="Include all Tier4 stress tests")
    parser.add_argument("--mbpp-full", action="store_true",
                       help="Run MBPP on all 974 problems (not 50-sample)")
    parser.add_argument("--mbpp-samples", type=int,
                       help="Number of MBPP problems to sample")
    parser.add_argument("--t4-threshold", type=float,
                       help="Override Tier4 defect density threshold")
    parser.add_argument("--threshold-sweep", action="store_true",
                       help="Run threshold sensitivity analysis")
    parser.add_argument("--t4-threshold-sweep-vals", type=str,
                       help="Comma-separated threshold values for sweep "
                            "(e.g. 0.1,0.2,0.3,0.4,0.5)")
    args = parser.parse_args()
    if args.t4stress:
        stress_modes = "t4stress_hard,t4stress_heat,t4stress_perturb,t4stress_combined"
        args.mode = stress_modes if not args.mode else args.mode + "," + stress_modes
    if args.t4_threshold_sweep_vals:
        args.t4_threshold_sweep_vals = [
            float(v.strip()) for v in args.t4_threshold_sweep_vals.split(",")]
    run_experiments(args)


if __name__ == "__main__":
    main()
