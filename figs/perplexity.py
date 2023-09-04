import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
models = ['opt-350m', 'opt-1.6b', 'llama-7b', 'llama-13b', 'llama-30b']
QuAC = [3.255, 2.96, 2.1, 1.88, 1.67]
BoolQ = [3.05, 2.85, 2.24, 1.82, 1.68]
SQuAD = [3.5, 2.6, 2.5, 2.0, 1.85]
Wikitext = [3.2, 2.908, 1.82, 1.7, 1.44]
NewWiki = [3.49, 3.2, 2.42, 2.7, 2.66]
LatestEval = [3.44, 3.02, 2.66, 2.35, 2.33]

metrics_data = {
    'QuAC': (QuAC, 's', 'violet'),
    'BoolQ': (BoolQ, '+', 'violet'),
    'SQuAD': (SQuAD, 'x', 'violet'),
    'Wikitext': (Wikitext, 'D', 'navy'),
    'NewWiki': (NewWiki, '^', 'navy'),
    'LatestEval': (LatestEval, 'o', 'gold')
}

fig, ax = plt.subplots(figsize=(10, 3.3), dpi=150)

# Create a horizontal scatter plot for each metric
for metric_name, (values, marker_style, color) in metrics_data.items():
    plt.scatter(values, models, label=metric_name, s=20, marker=marker_style, color=color)

# Adjust plot
plt.ylabel('Models', fontweight='bold')
plt.xlabel('Perplexity', fontweight='bold')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(left = 1.3)  # Adjusting xlim to be slightly more than the max value for better visualization

# plt.gca().xaxis.tick_top()
# plt.gca().xaxis.set_label_position('top')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.show()