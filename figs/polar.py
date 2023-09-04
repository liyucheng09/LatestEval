import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_score = pd.read_csv("figs/single_answer_score.tsv", sep='\t')

# Calculate the number of categories
categories = df_score['category'].unique()
N = len(categories)

# Calculate angle for each category
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
theta = np.append(theta, theta[0])
# Set up the polar axis
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6), dpi=150)
ax.set_facecolor("#f5f5f5")

markers = {
    'gpt-3.5-turbo': 'o',
    'gpt-4': '+',
    'llama-13b': 'x',
    'llama-30b': 's',
    'vicuna-13b': 'd',
}

# Loop through each model and plot on the polar axis
for model in df_score['model'].unique():
    values = df_score[df_score['model'] == model]['score'].values
    # Ensure the plot is closed by repeating the first value
    values = np.append(values, values[0])
    ax.plot(theta, values, label=model, marker=markers[model], alpha=0.8, markersize=5)

# Fill the area under the plot for better visualization (optional)
# ax.fill(theta, values, 'b', alpha=0.1)

# Set the y-ticks (radii) and x-ticks (categories)
ax.set_xticks(theta[:-1])
ax.set_xticklabels(categories, fontsize=14)  # Label x-ticks with categories

ax.set_yticks([0, 2, 4, 6, 8, 10])

# Customize the grid and title
ax.grid(True)

# Display a legend
ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.1), fontsize=14)

# Save the figure
fig.tight_layout()
fig.savefig("fig.png", dpi=150)

# Show the plot
# plt.show()
