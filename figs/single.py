import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
df = pd.read_csv('figs/perplexity.tsv', sep='\t').set_index('Model')
df.drop('LatestEval', axis = 1, inplace=True)
df.drop(['opt-350m', 'opt-1.6b', 'gpt-3', 'llama-7b', 'llama-30b'], axis=0, inplace=True)

data_dict = df.to_dict()

metrics_data = {
    'QuAC': ('s', 'violet'),
    'BoolQ': ('+', 'violet'),
    'SQuAD': ('x', 'violet'),
    'memorised': ('D', 'navy'),
    'clean': ('^', 'navy'),
    # 'LatestEval': ('o', 'gold')
}

fig, ax = plt.subplots(figsize=(4, 1), dpi=200)

# Create a horizontal scatter plot for each metric
for benchmark, numbers in data_dict.items():
    marker_style, color = metrics_data[benchmark]
    models, perplexities = list(numbers.keys()), list(numbers.values())
    # perplexities = np.exp(perplexities)  # assuming metrics are in log scale
    plt.scatter(perplexities, ['perplexity'], label=benchmark, s=20, marker=marker_style, color=color)
    ax.annotate(benchmark, xy=(perplexities[-1], 'perplexity'), xytext=(0, 2), textcoords='offset points', va='bottom', fontsize=7, rotation=45)

# Adjust plot
plt.grid(True, linestyle='--', linewidth=0.5, axis='y')
plt.xlim(left = 1.6)  # Adjusting xlim to be slightly more than the max value for better visualization

ax.xaxis.set_visible(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()

plt.show()