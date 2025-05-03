import unittest
import torch.nn as nn
import torch.optim as optim
import os
import tempfile
from benchmarking.benchmark import Benchmark
from datasets.dataset_loader import load_dataset
from database.database_handler import DatabaseHandler
from utils.helpers import stringify_criterion, stringify_optimizer


class TestEndToEndFlow(unittest.TestCase):
    def setUp(self):
        self.temp_db_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db_file.close()
        self.db_handler = DatabaseHandler(db_path=self.temp_db_file.name)

        self.model_exact = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        self.model_similar = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model_exact.parameters(), lr=0.01)
        self.criterion_str = stringify_criterion(self.criterion)
        self.optimizer_str = stringify_optimizer(self.optimizer)

        self.dataset_name = "iris"

        train_loader, test_loader = load_dataset(self.dataset_name)
        benchmark = Benchmark(self.model_exact, train_loader, test_loader, self.criterion, self.optimizer, epochs=5)
        results = benchmark.benchmark_model()
        self.db_handler.save_result(str(self.model_exact), self.dataset_name, self.criterion_str, self.optimizer_str, results)

    def tearDown(self):
        os.unlink(self.temp_db_file.name)

    def test_exact_match_found(self):
        result = self.db_handler.search_result(
            self.model_exact,
            self.dataset_name,
            self.criterion_str,
            self.optimizer_str,
            match_mode="exact"
        )
        self.assertIsNotNone(result)
        self.assertIn("avg_train_loss", result)

    def test_similar_match_graph_structural(self):
        result = self.db_handler.search_result(
            self.model_similar,
            self.dataset_name,
            match_mode="similar",
            similarity_threshold=0.9, 
            similarity_method="graph_structural"
        )
        self.assertIsNotNone(result)

    def test_similar_match_graph_with_params_high(self):
        result = self.db_handler.search_result(
            self.model_similar,
            self.dataset_name,
            match_mode="similar",
            similarity_threshold=0.9, 
            similarity_method="graph_structural_with_params"
        )
        self.assertIsNone(result)

    def test_similar_match_graph_with_params_low(self):
        result = self.db_handler.search_result(
            self.model_similar,
            self.dataset_name,
            match_mode="similar",
            similarity_threshold=0.6, 
            similarity_method="graph_structural_with_params"
        )
        self.assertIsNotNone(result)

    def test_similar_match_string(self):
        result = self.db_handler.search_result(
            self.model_similar,
            self.dataset_name,
            match_mode="similar",
            similarity_threshold=0.9, 
            similarity_method="string"
        )
        self.assertIsNotNone(result)

    def test_fallback_to_benchmark(self):
        new_model = nn.Sequential(
            nn.Linear(4, 32),
            nn.Sigmoid(),
            nn.Linear(32, 3)
        )

        result = self.db_handler.search_result(
            new_model,
            self.dataset_name,
            match_mode="exact"
        )
        self.assertIsNone(result)

        result = self.db_handler.search_result(
            new_model,
            self.dataset_name,
            match_mode="similar",
            similarity_threshold=0.9, 
            similarity_method="graph_structural_with_params"
        )
        self.assertIsNone(result)

        train_loader, test_loader = load_dataset(self.dataset_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(new_model.parameters(), lr=0.01)
        benchmark = Benchmark(new_model, train_loader, test_loader, criterion, optimizer, epochs=5)
        results = benchmark.benchmark_model()

        self.assertIn("avg_train_loss", results)
        crietrion_str = stringify_criterion(criterion)
        optimizer_str = stringify_optimizer(optimizer),

        self.db_handler.save_result(str(new_model), self.dataset_name, crietrion_str, optimizer_str, results)

        saved = self.db_handler.search_result(
            new_model,
            self.dataset_name,
            crietrion_str,
            optimizer_str,
            match_mode="exact"
        )
        self.assertIsNotNone(saved)
