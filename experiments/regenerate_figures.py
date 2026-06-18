"""Regenerate all paper figures from saved result JSONs using full 164-problem data."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from plot import (
    plot_defect_density_curve,
    plot_tier_survival,
    plot_ablation_comparison,
    plot_cost_comparison,
    plot_t4_activation_timeline,
    plot_transition_matrix,
    plot_threshold_sensitivity,
)
from framework_diagram import draw_framework_diagram

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_FIGURES = os.path.join(SCRIPT_DIR, '..', '..', '..',
                              '论文学术', 'manuscripts', 'SCI',
                              'Automated_Software_Engineering', 'figures')


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def reconstruct_defect_history(details):
    history = []
    cumulative_passed = 0
    cumulative_total = 0
    for r in details:
        cumulative_passed += r['n_passed']
        cumulative_total += r['n_samples']
        ratio = cumulative_passed / max(cumulative_total, 1)
        history.append(1.0 - ratio)
    return history


def main():
    os.makedirs(PAPER_FIGURES, exist_ok=True)

    raw_data = load_json('HumanEval_Raw.json')
    mgc_data = load_json('HumanEval_MultiGuardCode.json')

    if raw_data and mgc_data:
        raw_defect = reconstruct_defect_history(raw_data['details'])
        mgc_defect = reconstruct_defect_history(mgc_data['details'])

        plot_defect_density_curve(
            raw_defect, mgc_defect,
            os.path.join(PAPER_FIGURES, 'HumanEval_defect_density_curve.pdf')
        )

    ablation_data = {}
    for fname in ['HumanEval_Raw.json', 'HumanEval_SelfDebug.json',
                  'HumanEval_T3Only.json', 'HumanEval_T123.json',
                  'HumanEval_MultiGuardCode.json']:
        if os.path.exists(os.path.join(RESULTS_DIR, fname)):
            d = load_json(fname)
            config = fname.replace('HumanEval_', '').replace('.json', '')
            config = {'Raw': 'Raw', 'SelfDebug': 'Self-Debug',
                      'T3Only': 'T3-Only', 'T123': 'T1+T2+T3',
                      'MultiGuardCode': 'MultiGuardCode'}.get(config, config)
            if d:
                ablation_data[config] = d['metrics']

    if ablation_data:
        plot_ablation_comparison(
            ablation_data,
            os.path.join(PAPER_FIGURES, 'ablation_comparison.pdf')
        )

    if mgc_data:
        plot_tier_survival(
            mgc_data.get('stats', {}),
            os.path.join(PAPER_FIGURES, 'tier_survival.pdf')
        )

    token_data = {}
    for fname in ['HumanEval_Raw.json', 'HumanEval_SelfDebug.json',
                  'HumanEval_T3Only.json', 'HumanEval_T123.json',
                  'HumanEval_MultiGuardCode.json']:
        if os.path.exists(os.path.join(RESULTS_DIR, fname)):
            d = load_json(fname)
            if d:
                ts = d.get('token_stats', {})
                if ts:
                    token_data[fname.replace('.json', '')] = ts

    if len(token_data) > 1:
        plot_cost_comparison(
            token_data,
            os.path.join(PAPER_FIGURES, 'cost_comparison.pdf')
        )

    draw_framework_diagram(
        os.path.join(PAPER_FIGURES, 'framework_diagram.pdf')
    )

    transition_path = os.path.join(RESULTS_DIR, 'HumanEval_transition_log.json')
    if os.path.exists(transition_path):
        trans = load_json(transition_path)
        if trans:
            rr = trans.get('repaired_regressed', {})
            if rr:
                plot_transition_matrix(
                    rr,
                    os.path.join(PAPER_FIGURES, 'transition_matrix.pdf')
                )

    print('All figures regenerated in:', PAPER_FIGURES)


if __name__ == '__main__':
    main()
