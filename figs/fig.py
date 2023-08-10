import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
models = ['babbage', 'curie', 'davinci', 'llama-13b', 'falcon-7b']
QuAD = [0.22 * 1000, 0.29 * 1000, 0.39 * 1000, 0.25 * 1000, 0.24 * 1000]
BoolQ = [0 * 1000, 0.03 * 1000, 0.05 * 1000, 0.09 * 1000, 0.04 * 1000]
SQuAD = [0.08 * 1000, 0.09 * 1000, 0.15 * 1000, 0.16 * 1000, 0.11 * 1000]
LatestEval = [0 * 1000, 0.002 * 1000, 0 * 1000, 0.005 * 1000, 0.005 * 1000]

metrics_data = {
    'QuAD': (QuAD, 's'),
    'BoolQ': (BoolQ, '+'),
    'SQuAD': (SQuAD, 'x'),
    'LatestEval': (LatestEval, 'D')
}

# Create a scatter plot for each metric
for metric_name, (values, marker_style) in metrics_data.items():
    plt.scatter(models, values, label=metric_name, s=20, marker=marker_style)

# Adjust plot
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Num of examples', fontweight='bold')
plt.title('Dot Plot Visualization of Model Scores')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.ylim(0, 500)  # Adjusting ylim to be slightly more than the max value for better visualization
plt.show()
