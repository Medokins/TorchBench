import unittest
import torch.nn as nn
from benchmarking.architecture_similarity import ArchitectureComparator


class TestModelHandler(unittest.TestCase):
    def test_model_similarity_graph_structural(self):
        model1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        comparator = ArchitectureComparator(model1, model2, method="graph_structural")
        similarity = comparator.compute_similarity()
        print(similarity)

    def test_model_similarity_graph_structural_with_params(self):
        model1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        model2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

        comparator = ArchitectureComparator(model1, model2, method="graph_structural_with_params")
        similarity = comparator.compute_similarity()
        print(similarity)
