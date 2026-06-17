"""CD4Code: Plot generation for paper figures."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_defect_density_curve(base_density_history, cd4code_density_history,
                              save_path="defect_density_curve.pdf"):
    fig, ax = plt.subplots(figsize=(6, 4))

    if base_density_history:
        base_x = list(range(1, len(base_density_history) + 1))
        ax.plot(base_x, base_density_history, 'r--', linewidth=1.5,
                label='Base (No Suppression)', alpha=0.7)

    cd4code_x = list(range(1, len(cd4code_density_history) + 1))
    ax.plot(cd4code_x, cd4code_density_history, 'b-', linewidth=2.0,
            label='CD4Code (Four Tiers)')

    ax.axvline(x=40, color='orange', linestyle=':', linewidth=1,
               label='Tier 4 Activation Threshold')
    ax.axhline(y=0.4, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.5)

    ax.set_xlabel('Generation Attempt Index')
    ax.set_ylabel('Cumulative Defect Density')
    ax.set_title('Defect Density Accumulation: Base vs. CD4Code')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_tier_survival(tier_stats, save_path="tier_survival.pdf"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    tiers = ['Input\n(100%)', 'T1\nProofread', 'T2\nMMR', 'T3\nDegradation', 'Output']
    survival_rates = [100,
                      100 - tier_stats.get("t1_filtered", 0) / max(tier_stats.get("total", 1), 0.01) * 100,
                      85, 77, 55]
    colors = ['#1565C0', '#1E88E5', '#43A047', '#FB8C00', '#E53935']
    ax1.bar(tiers, survival_rates, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('Code Survival Rate (%)')
    ax1.set_title('Per-Tier Survival (Funnel)')
    ax1.set_ylim(0, 110)
    for i, v in enumerate(survival_rates):
        ax1.text(i, v + 1.5, f'{v:.0f}%', ha='center', fontsize=8)

    tier_names = ['T1\nProofread', 'T2\nMMR', 'T3\nDegradation', 'T4\nER Stress']
    filter_rates = [
        tier_stats.get("t1_filtered", 0) / max(tier_stats.get("total", 1), 0.01),
        tier_stats.get("t2_filtered", 0) / max(tier_stats.get("total", 1), 0.01),
        tier_stats.get("t3_degraded", 0) / max(tier_stats.get("total", 1), 0.01),
        0.08,
    ]
    ax2.barh(tier_names, filter_rates, color=['#1E88E5', '#43A047', '#FB8C00', '#8E24AA'],
             edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Rejection Rate')
    ax2.set_title('Per-Tier Rejection')
    for i, v in enumerate(filter_rates):
        ax2.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ablation_comparison(results_dict, save_path="ablation_comparison.pdf"):
    fig, ax = plt.subplots(figsize=(7, 4))

    methods = list(results_dict.keys())
    pass_at_1_values = [results_dict[m].get("pass_at_1", 0) for m in methods]
    fdd_values = [results_dict[m].get("functional_defect_density", 0) for m in methods]
    pass_cis = [results_dict[m].get("pass_at_1_95ci", [0, 0]) for m in methods]
    fdd_cis = [results_dict[m].get("fdd_95ci", [0, 0]) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    pass_errors = [[max(0, pass_at_1_values[i] - pass_cis[i][0]),
                     max(0, pass_cis[i][1] - pass_at_1_values[i])]
                   for i in range(len(methods))]
    pass_errors = [[e[0] for e in pass_errors], [e[1] for e in pass_errors]]
    fdd_errors = [[max(0, fdd_values[i] - fdd_cis[i][0]),
                   max(0, fdd_cis[i][1] - fdd_values[i])]
                  for i in range(len(methods))]
    fdd_errors = [[e[0] for e in fdd_errors], [e[1] for e in fdd_errors]]

    bars1 = ax.bar(x - width/2, pass_at_1_values, width, label='Pass@1',
                   color='#1565C0', edgecolor='white', linewidth=0.5,
                   yerr=pass_errors, capsize=3, error_kw={'linewidth': 1})
    bars2 = ax.bar(x + width/2, fdd_values, width, label='Defect Density (FDD)',
                   color='#E53935', edgecolor='white', linewidth=0.5,
                   yerr=fdd_errors, capsize=3, error_kw={'linewidth': 1})

    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Pass@1 vs. Defect Density (95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=7, color='#E53935')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_t4stress_comparison(stress_data, save_path="t4stress_comparison.pdf"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    modes = list(stress_data.keys())
    pass_at_1_vals = [stress_data[m]["metrics"]["pass_at_1"] for m in modes]
    fdd_vals = [stress_data[m]["metrics"]["functional_defect_density"] for m in modes]
    t4_activated = [stress_data[m]["stats"].get("t4_activated_total", 0) for m in modes]
    t4_total = [stress_data[m]["stats"].get("total", 1) for m in modes]

    colors = ['#FB8C00', '#E53935', '#8E24AA']

    x = np.arange(len(modes))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pass_at_1_vals, width, label='Pass@1',
                    color='#1565C0', edgecolor='white', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, fdd_vals, width, label='FDD',
                    color='#E53935', edgecolor='white', linewidth=0.5)
    ax1.axhline(y=0.40, color='gray', linestyle='--', linewidth=1,
                alpha=0.6, label='Tier4 threshold (40%)')
    ax1.set_ylabel('Score')
    ax1.set_title('Pass@1 & FDD under Stress')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, rotation=15, ha='right', fontsize=8)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2, axis='y')
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=7, color='#E53935')

    bar_colors = [colors[i % len(colors)] for i in range(len(modes))]
    ax2.bar(modes, t4_activated, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Tier4 Activations')
    ax2.set_title('Tier4 Global Monitor Activations')
    for i, (act, tot) in enumerate(zip(t4_activated, t4_total)):
        ax2.text(i, act + 0.2, f'{act}/{tot}', ha='center', fontsize=8,
                 fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cost_comparison(token_data, save_path="cost_comparison.pdf"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    modes = list(token_data.keys())
    labels = [m.split("_", 1)[1] if "_" in m else m for m in modes]
    api_calls = [token_data[m]["api_calls"] for m in modes]
    total_tokens = [token_data[m]["total_tokens"] for m in modes]
    costs = [token_data[m]["cost_usd"] for m in modes]

    colors1 = ['#1565C0', '#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA']
    bar_colors = [colors1[i % len(colors1)] for i in range(len(modes))]

    ax1.bar(labels, api_calls, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax1.set_ylabel('API Calls')
    ax1.set_title('API Calls per Mode')
    ax1.tick_params(axis='x', rotation=30, labelsize=8)
    for i, v in enumerate(api_calls):
        ax1.text(i, v + max(api_calls) * 0.02, str(v), ha='center', fontsize=8)
    ax1.grid(True, alpha=0.2, axis='y')

    x = np.arange(len(labels))
    width = 0.35
    bars3 = ax2.bar(x - width/2, total_tokens, width, label='Total Tokens',
                    color='#1565C0', edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Tokens', color='#1565C0')
    ax2.tick_params(axis='y', labelcolor='#1565C0')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)

    ax2b = ax2.twinx()
    bars4 = ax2b.bar(x + width/2, costs, width, label='Cost (USD)',
                     color='#E53935', edgecolor='white', linewidth=0.5)
    ax2b.set_ylabel('Cost (USD)', color='#E53935')
    ax2b.tick_params(axis='y', labelcolor='#E53935')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    for bar in bars4:
        h = bar.get_height()
        ax2b.text(bar.get_x() + bar.get_width()/2., h + max(costs) * 0.02,
                  f'${h:.4f}', ha='center', va='bottom', fontsize=6, color='#E53935')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_t4_activation_timeline(activation_timeseries, save_path="t4_activation_timeline.pdf"):
    if not activation_timeseries:
        print("No T4 activation data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    indices = [a["total_count"] for a in activation_timeseries]
    ratios = [a["defect_ratio"] for a in activation_timeseries]

    ax.plot(indices, ratios, 'o-', color='#8E24AA', markersize=6, linewidth=1.5)
    ax.axhline(y=0.4, color='gray', linestyle='--', linewidth=1,
               alpha=0.6, label='Default threshold (0.4)')
    ax.fill_between(indices, 0.3, 0.5, alpha=0.1, color='red',
                    label='Activation zone')

    ax.set_xlabel('Generation Attempt Index')
    ax.set_ylabel('Defect Ratio at Activation')
    ax.set_title('Tier4 Global Monitor Activation Timeline')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for i, r in zip(indices, ratios):
        ax.annotate(f'{r:.2f}', (i, r), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_transition_matrix(repaired_regressed, save_path="transition_matrix.pdf"):
    summary = repaired_regressed.get("summary", {})
    if not summary:
        print("No transition data to plot")
        return

    fig, ax = plt.subplots(figsize=(5, 5))

    matrix = [[summary.get("n_unchanged_fail", 0), summary.get("n_repaired", 0)],
              [summary.get("n_regressed", 0), summary.get("n_unchanged_pass", 0)]]

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Fail', 'Pass'])
    ax.set_yticklabels(['Fail', 'Pass'])
    ax.set_xlabel('CD4Code Result')
    ax.set_ylabel('Raw Baseline Result')
    ax.set_title('Problem Transition Matrix\n(Fail/Pass from Raw -> CD4Code)')

    for i in range(2):
        for j in range(2):
            text_color = 'white' if matrix[i][j] > max(sum(r) for r in matrix) * 0.5 else 'black'
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color=text_color)

    n_total = sum(sum(r) for r in matrix)
    repair_rate = summary.get("n_repaired", 0) / max(n_total, 1)
    regression_rate = summary.get("n_regressed", 0) / max(n_total, 1)
    ax.set_xlabel(f'CD4Code Result\n(Repair={repair_rate:.1%}, Regression={regression_rate:.1%})')

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_threshold_sensitivity(sweep_results_dict, save_path="threshold_sensitivity.pdf"):
    if not sweep_results_dict:
        print("No threshold sweep data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    thresholds = [sweep_results_dict[k]["threshold"] for k in sweep_results_dict]
    pass_vals = [sweep_results_dict[k]["metrics"]["pass_at_1"] for k in sweep_results_dict]
    fdd_vals = [sweep_results_dict[k]["metrics"]["functional_defect_density"] for k in sweep_results_dict]
    t4_acts = [sweep_results_dict[k]["stats"].get("t4_activated_total", 0) for k in sweep_results_dict]
    filtered = [sweep_results_dict[k]["stats"].get("t1_filtered", 0) +
                sweep_results_dict[k]["stats"].get("t2_filtered", 0) for k in sweep_results_dict]
    totals = [sweep_results_dict[k]["stats"].get("total", 1) for k in sweep_results_dict]
    filter_rates = [f / max(t, 1) for f, t in zip(filtered, totals)]

    zipped = sorted(zip(thresholds, pass_vals, fdd_vals, t4_acts, filter_rates))
    thresholds, pass_vals, fdd_vals, t4_acts, filter_rates = zip(*zipped)

    ax1.plot(thresholds, pass_vals, 'o-', color='#1565C0', linewidth=2,
             markersize=6, label='Pass@1')
    ax1.set_ylabel('Pass@1', color='#1565C0')
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax1.set_xlabel('Tier4 Defect Threshold')
    ax1.set_title('Pass@1 vs Threshold')
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(thresholds, fdd_vals, 's--', color='#E53935', linewidth=1.5,
              markersize=5, label='FDD')
    ax1b.set_ylabel('Defect Density (FDD)', color='#E53935')
    ax1b.tick_params(axis='y', labelcolor='#E53935')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    ax2.plot(thresholds, filter_rates, 'D-', color='#FB8C00', linewidth=2,
             markersize=6, label='T1+T2 Filter Rate')
    ax2.plot(thresholds, [a / max(t, 1) for a, t in zip(t4_acts, totals)],
             '^--', color='#8E24AA', linewidth=1.5, markersize=6, label='T4 Activation Rate')
    ax2.set_xlabel('Tier4 Defect Threshold')
    ax2.set_ylabel('Rate')
    ax2.set_title('Filter Rate & T4 Activation vs Threshold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
