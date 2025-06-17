import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

results_df = pd.read_csv("results/results.csv")
similarity_df = pd.read_csv("results/similarity_vs_loss.csv")

classification_datasets = ["iris", "wine", "breast_cancer"]
regression_datasets = ["diabetes", "california_housing"]

similarity_df["task_type"] = similarity_df["dataset"].apply(
    lambda x: "classification" if x in classification_datasets else "regression"
)

os.makedirs("plots", exist_ok=True)

for task_type in ["classification", "regression"]:
    task_df = similarity_df[similarity_df["task_type"] == task_type]
    for method in task_df["method"].unique():
        plt.figure(figsize=(8, 6))
        subset = task_df[task_df["method"] == method]
        sns.scatterplot(
            data=subset,
            x="similarity_score",
            y="test_loss_diff",
            hue="dataset",
            alpha=0.7
        )
        plt.title(f"{task_type.title()} – {method} Similarity vs Loss Difference")
        plt.xlabel("Similarity Score")
        plt.ylabel("Absolute Test Loss Difference")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{task_type}_{method}_similarity_vs_loss.png")
        plt.close()

        corr, _ = spearmanr(subset["similarity_score"], subset["test_loss_diff"])
        print(f"{task_type} – {method} Spearman correlation: {corr:.2f}")

