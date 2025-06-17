import pandas as pd
from benchmarking.architecture_similarity import ArchitectureComparator
from models import make_classification_models, make_regression_models


input_dims = {
    "iris": 4,
    "wine": 13,
    "breast_cancer": 30,
    "diabetes": 10,
    "california_housing": 8
}
output_dims = {
    "iris": 3,
    "wine": 3,
    "breast_cancer": 2,
    "diabetes": 1,
    "california_housing": 1
}

df = pd.read_csv("results/results.csv")

all_models = {}
for dataset in df["dataset"].unique():
    in_dim = input_dims[dataset]
    out_dim = output_dims[dataset]
    if dataset in ["diabetes", "california_housing"]:
        models = make_regression_models(in_dim)
    else:
        models = make_classification_models(in_dim, out_dim)
    all_models.update(models)

rows = []
for dataset, group in df.groupby("dataset"):
    for comparison_method in ["graph_structural", "graph_structural_with_params", "string"]:
        for i, row_i in group.iterrows():
            for j, row_j in group.iterrows():
                if j <= i:
                    continue
                model_a = all_models[row_i["model_name"]]
                model_b = all_models[row_j["model_name"]]
                comparator = ArchitectureComparator(model_a, model_b, method=comparison_method)

                similarity = comparator.compute_similarity()
                test_loss_diff = abs(row_i["test_loss"] - row_j["test_loss"])

                rows.append({
                    "method": comparison_method,
                    "dataset": dataset,
                    "model_a": row_i["model_name"],
                    "model_b": row_j["model_name"],
                    "similarity_score": similarity,
                    "test_loss_diff": test_loss_diff
                })

similarities_df = pd.DataFrame(rows)
similarities_df.to_csv("results/similarity_vs_loss.csv", index=False)
