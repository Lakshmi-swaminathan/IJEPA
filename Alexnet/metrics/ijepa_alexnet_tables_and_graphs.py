# -*- coding: utf-8 -*-
"""IJEPA Alexnet Tables and Graphs.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hLzzrJGzUjYMMUiS1co-8PE_MTavWu0u
"""

import pandas as pd

# Sample DataFrames with style applied

# Downstream Task Performance
data_performance = {
    "Method": ["I-JEPA Alexnet", "Standard Alexnet"],
    "Top-1 Accuracy": [0.85, 0.82],
    "Top-5 Accuracy": [0.93, 0.90]
}
df_performance = pd.DataFrame(data_performance)
print(""" ### Downstream Task Performance """)
display(df_performance)

# Training Efficiency
data_efficiency = {
    "Method": ["I-JEPA Alexnet", "Standard Alexnet"],
    "Time per Epoch (s)": [120, 140],
    "GPU Memory Usage (GB)": [4.2, 5.1]
}
df_efficiency = pd.DataFrame(data_efficiency)
print("""\n### Training Efficiency""")
display(df_efficiency)

# Representation Quality
data_embedding_quality = {
    "Method": ["I-JEPA Alexnet", "Standard Alexnet"],
    "Cosine Similarity": [0.98, 0.92],
    "Mean Squared Error": [0.05, 0.10]
}
df_embedding_quality = pd.DataFrame(data_embedding_quality)
print("""\n### Representation Quality""")
display(df_embedding_quality)

# Robustness to Corruptions
data_robustness = {
    "Method": ["I-JEPA Alexnet", "Standard Alexnet"],
    "Accuracy on Clean Data": [0.85, 0.82],
    "Accuracy on Noisy Data": [0.75, 0.65],
    "Accuracy on Occluded Data": [0.72, 0.60]
}
df_robustness = pd.DataFrame(data_robustness)
print("""\n### Robustness to Corruptions""")
display(df_robustness)

import matplotlib.pyplot as plt
import seaborn as sns


# Plot settings
fig, axes = plt.subplots(2, 2, figsize=(14, 6))
fig.suptitle("I-JEPA Alexnet vs Standard Alexnet: Performance Comparison")

# Downstream Task Performance
sns.barplot(data=df_performance.melt(id_vars="Method"), x="variable", y="value", hue="Method", ax=axes[0, 0])
axes[0, 0].set_title("Downstream Task Performance")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_xlabel("")

# Training Efficiency
sns.barplot(data=df_efficiency.melt(id_vars="Method"), x="variable", y="value", hue="Method", ax=axes[0, 1])
axes[0, 1].set_title("Training Efficiency")
axes[0, 1].set_ylabel("Value")
axes[0, 1].set_xlabel("")

# Representation Quality
sns.barplot(data=df_embedding_quality.melt(id_vars="Method"), x="variable", y="value", hue="Method", ax=axes[1, 0])
axes[1, 0].set_title("Representation Quality")
axes[1, 0].set_ylabel("Score")
axes[1, 0].set_xlabel("")

# Robustness to Corruptions
sns.barplot(data=df_robustness.melt(id_vars="Method"), x="variable", y="value", hue="Method", ax=axes[1, 1])
axes[1, 1].set_title("Robustness on Corrupted Images")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].set_xlabel("")

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to fit the main title
plt.show()