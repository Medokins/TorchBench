import sys
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from benchmarking.benchmark import Benchmark
from datasets.dataset_loader import load_dataset
from models import make_classification_models, make_regression_models

all_results = []

dataset_info = {
    "iris": {"is_regression": False, "input_dim": 4, "output_dim": 3},
    "wine": {"is_regression": False, "input_dim": 13, "output_dim": 3},
    "breast_cancer": {"is_regression": False, "input_dim": 30, "output_dim": 2},
    "diabetes": {"is_regression": True, "input_dim": 10, "output_dim": 1},
    "california_housing": {"is_regression": True, "input_dim": 8, "output_dim": 1}
}

for dataset_name, info in dataset_info.items():
    print(f"\n=== Loading dataset: {dataset_name} ===")
    train_loader, test_loader, _ = load_dataset(dataset_name)

    input_dim = info["input_dim"]
    output_dim = info["output_dim"]
    is_regression = info["is_regression"]

    models = (
        make_regression_models(input_dim)
        if is_regression else
        make_classification_models(input_dim, output_dim)
    )

    for model_name, model in models.items():
        print(f"\n=== Benchmarking {model_name} on {dataset_name} ===")

        criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        benchmark = Benchmark(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=10,
            flatten_inputs=True
        )

        results = benchmark.benchmark_model()

        all_results.append({
            "model_name": model_name,
            "dataset": dataset_name,
            "train_losses": results["train_losses"],
            "avg_train_loss": results["avg_train_loss"],
            "final_train_loss": results["final_train_loss"],
            "test_loss": results["test_loss"],
            "model_architecture": results["model_architecture"]
        })

df = pd.DataFrame(all_results)
df.to_csv("results/results.csv", index=False)
print("\nAll benchmarks completed and saved to results/results.csv")
