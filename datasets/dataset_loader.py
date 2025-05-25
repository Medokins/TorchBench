import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import hashlib


class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Ensure classification labels are long, regression labels are 2D float
        if len(self.y.shape) == 1:
            if y.dtype == int or y.dtype == torch.int64:
                self.y = self.y.long()  # Classification
            else:
                self.y = self.y.unsqueeze(1)  # Regression (makes shape [batch_size, 1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def compute_dataset_hash(X, y):
    combined = np.hstack((X, y.reshape(-1, 1) if len(y.shape) == 1 else y))
    return hashlib.md5(combined.tobytes()).hexdigest()

def load_dataset(name, batch_size=32, test_size=0.2):
    datasets = {
        "california_housing": fetch_california_housing,
        "diabetes": load_diabetes,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "iris": load_iris
    }

    if name not in datasets:
        raise ValueError(f"Dataset '{name}' not supported. Available: {list(datasets.keys())}")

    data = datasets[name]()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    train_dataset = TorchDataset(X_train, y_train)
    test_dataset = TorchDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    full_hash = compute_dataset_hash(X, y)

    return train_loader, test_loader, full_hash
