import unittest
import torch
import torch.nn as nn
from datasets.dataset_loader import load_dataset
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

        train_loader, test_loader = load_dataset("iris")
        benchmark = Benchmark(model, train_loader, test_loader, criterion, optimizer, epochs=5)
        results = benchmark.benchmark_model()
        
        self.assertIsInstance(results, dict)
        print(results)
        self.assertIn("avg_train_loss", results)
        self.assertIn("test_loss", results)
        self.assertIn("train_losses", results)
        self.assertGreater(results["final_train_loss"], 0)
        self.assertGreater(results["test_loss"], 0)

    def test_benchmark_model_regression(self):
        model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        train_loader, test_loader = load_dataset("california_housing")
        benchmark = Benchmark(model, train_loader, test_loader, criterion, optimizer, epochs=5)
        results = benchmark.benchmark_model()

        self.assertIsInstance(results, dict)
        print(results)
        self.assertIn("avg_train_loss", results)
        self.assertIn("test_loss", results)
        self.assertIn("train_losses", results)
        self.assertGreater(results["final_train_loss"], 0)
        self.assertGreater(results["test_loss"], 0)
