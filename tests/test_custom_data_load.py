import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from benchmarking.benchmark import Benchmark


class TestBenchmark(unittest.TestCase):
    def test_benchmark_model_classification(self):
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X = torch.randn(200, 4)
        y = torch.randint(0, 3, (200,))

        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        benchmark = Benchmark(model, train_loader, test_loader, criterion, optimizer, epochs=5)
        results = benchmark.benchmark_model()
        
        self.assertIsInstance(results, dict)
        print(results)
        self.assertIn("avg_train_loss", results)
        self.assertIn("test_loss", results)
        self.assertIn("train_losses", results)
        self.assertGreater(results["final_train_loss"], 0)
        self.assertGreater(results["test_loss"], 0)