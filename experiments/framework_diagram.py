"""MultiGuardCode: Framework Diagram Generation."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def draw_framework_diagram(save_path="framework_diagram.pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.axis('off')

    color_source = '#1565C0'
    color_source_light = '#E3F2FD'
    color_tier = '#2E7D32'
    color_tier_light = '#E8F5E9'
    color_error = '#C62828'

    box_width = 5.5
    box_height = 1.1
    x_center = 4.0

    stages = [
        {"y": 8.2, "label": "LLM Generation", "desc": "Prompt-based code synthesis via DeepSeek V4 Pro"},
        {"y": 6.8, "label": "T1: Output Filter", "desc": "Length check + repetition detection"},
        {"y": 5.4, "label": "T2: AST Validate", "desc": "Code extraction + syntax validation"},
        {"y": 4.0, "label": "T3: Test Repair", "desc": "Test-driven iterative regeneration (3 retries)"},
        {"y": 2.6, "label": "T4: Defect Monitor", "desc": "Global defect rate monitoring + adaptive params"},
    ]

    for i, stage in enumerate(stages):
        color = color_source_light if i == 0 else color_tier_light
        edge = color_source if i == 0 else color_tier
        y = stage["y"]

        rect = FancyBboxPatch(
            (x_center - box_width / 2, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.2",
            facecolor=color,
            edgecolor=edge,
            linewidth=3
        )
        ax.add_patch(rect)
        ax.text(x_center, y + 0.12, stage["label"],
                ha='center', va='center', fontsize=16, fontweight='bold',
                color=color_source if i == 0 else color_tier)
        ax.text(x_center, y - 0.28, stage["desc"],
                ha='center', va='center', fontsize=10,
                color='#424242')

    for i in range(len(stages) - 1):
        y_start = stages[i]["y"] - box_height / 2
        y_end = stages[i + 1]["y"] + box_height / 2
        ax.annotate('', xy=(x_center, y_end + 0.05),
                    xytext=(x_center, y_start - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#424242',
                                    lw=3, connectionstyle='arc3,rad=0'))
        mid_y = (y_start + y_end) / 2
        ax.text(x_center + 0.5, mid_y, f'S{i+2}',
                ha='center', va='center', fontsize=9,
                color='#757575', fontstyle='italic')

    y_repair = stages[3]["y"]
    ax.annotate('', xy=(x_center + box_width / 2 + 0.7, y_repair + 0.5),
                xytext=(x_center + box_width / 2 + 0.7, y_repair - 0.5),
                arrowprops=dict(arrowstyle='->', color=color_error,
                                lw=2, connectionstyle='arc3,rad=-0.3'))
    ax.annotate('', xy=(x_center + box_width / 2 + 0.2, y_repair + 0.5),
                xytext=(x_center + box_width / 2 + 0.2, y_repair - 0.5),
                arrowprops=dict(arrowstyle='->', color=color_error,
                                lw=2, connectionstyle='arc3,rad=0.3'))
    ax.text(x_center + box_width / 2 + 0.85, y_repair,
            'Retry\nLoop', ha='center', va='center', fontsize=9,
            color=color_error, fontweight='bold')

    ax.text(x_center, 1.0, 'Validated Reliable Code Output',
            ha='center', va='center', fontsize=12,
            color=color_tier, fontweight='bold')
    ax.text(x_center, 8.95, 'Raw LLM Output (may contain errors)',
            ha='center', va='center', fontsize=11,
            color=color_error)

    ax.text(x_center, 9.6, 'MultiGuardCode: Multi-Tier Error Suppression Framework',
            ha='center', va='center', fontsize=22, fontweight='bold',
            color='#212121')

    ax.set_aspect('equal')
    fig.tight_layout(rect=[0, 0.01, 1, 0.98])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Framework diagram saved: {save_path}")


if __name__ == "__main__":
    draw_framework_diagram("../paper/figures/framework_diagram.pdf")
