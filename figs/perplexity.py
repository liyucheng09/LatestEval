import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
df = pd.read_csv('figs/perplexity.tsv', sep='\t').set_index('Model')
df.drop('LatestEval', axis = 1, inplace=True)
df.drop(['opt-350m', 'opt-1.6b'], axis=0, inplace=True)

data_dict = df.to_dict()

metrics_data = {
    'QuAC': ('s', 'violet'),
    'BoolQ': ('+', 'violet'),
    'SQuAD': ('x', 'violet'),
    'Wikitext': ('D', 'navy'),
    'NewWiki': ('^', 'navy'),
    # 'LatestEval': ('o', 'gold')
}

fig, ax = plt.subplots(figsize=(8, 2.8), dpi=150)

# Create a horizontal scatter plot for each metric
for benchmark, numbers in data_dict.items():
    marker_style, color = metrics_data[benchmark]
    models, perplexities = list(numbers.keys()), list(numbers.values())
    # perplexities = np.exp(perplexities)  # assuming metrics are in log scale
    plt.scatter(perplexities, models, label=benchmark, s=20, marker=marker_style, color=color)

# Adjust plot
plt.ylabel('Models', fontweight='bold')
plt.xlabel('Perplexity', fontweight='bold')
plt.legend( loc='upper right', bbox_to_anchor=(1.05, 1.0), ncol=1, fontsize=8)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlim(left = 1.3)  # Adjusting xlim to be slightly more than the max value for better visualization

# plt.gca().xaxis.tick_top()
# plt.gca().xaxis.set_label_position('top')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.show()