import time
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from benchmarking.architecture_similarity import ArchitectureComparator


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def benchmark_model(hidden_dims, dataset_name, X_train, y_train, input_dim, output_dim, method_list, epochs=20):
    result = {
        "model_size": str(hidden_dims),
        "dataset": dataset_name,
    }

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    model = Net(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss() if output_dim > 1 else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    start_train = time.time()
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train
    result["train_time"] = train_time

    model_ref = Net(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)

    for method in method_list:
        start_cmp = time.time()
        comparator = ArchitectureComparator(model, model_ref, method=method)
        similarity = comparator.compute_similarity()
        cmp_time = time.time() - start_cmp
        result[f"{method}_time"] = cmp_time
        result[f"{method}_similarity"] = similarity

    return result


# Prepare datasets
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train_iris, _, y_train_iris, _ = train_test_split(X, y, stratify=y, random_state=42)
X_train_iris = torch.tensor(X_train_iris, dtype=torch.float32)
y_train_iris = torch.tensor(y_train_iris, dtype=torch.long)

X_large, y_large = make_regression(n_samples=5000, n_features=512, noise=0.1)
X_large = StandardScaler().fit_transform(X_large)
y_large = (y_large - y_large.mean()) / y_large.std()
X_train_large, _, y_train_large, _ = train_test_split(X_large, y_large, random_state=42)
X_train_large = torch.tensor(X_train_large, dtype=torch.float32)
y_train_large = torch.tensor(y_train_large, dtype=torch.float32).view(-1, 1)

# Define configs
model_sizes = [
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512],
    [1024, 1024],
    [1024, 512, 256]
]
methods = ["string", "graph_structural", "graph_structural_with_params"]

results = []
for size in model_sizes:
    results.append(benchmark_model(size, "iris", X_train_iris, y_train_iris, input_dim=4, output_dim=3, method_list=methods))
    results.append(benchmark_model(size, "synthetic", X_train_large, y_train_large, input_dim=512, output_dim=1, method_list=methods))

df = pd.DataFrame(results)
df.to_csv("benchmark_scalability_results.csv", index=False)