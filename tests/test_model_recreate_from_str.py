import unittest
import torch.nn as nn
import json
from benchmarking.recreate_model_from_str import recreate_model_from_string
from database.database_handler import DatabaseHandler


class TestModelRecreation(unittest.TestCase):
    def test_recreate_model_from_str(self):
        db = DatabaseHandler()
        conn = db._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT result_json FROM benchmarks ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "No rows returned from database.")

        result_json = json.loads(row[0])
        model_str = result_json.get("model_architecture")
        self.assertIsNotNone(model_str, "'model_architecture' missing from result_json")

        model = recreate_model_from_string(model_str)
        self.assertIsInstance(model, nn.Sequential, "Recreated model is not nn.Sequential")

        if hasattr(model[0], "in_features"):
            import torch
            dummy_input = torch.randn(1, model[0].in_features)
            try:
                output = model(dummy_input)
                self.assertIsNotNone(output, "Model forward pass failed.")
                print("Model forward pass succeded")
            except Exception as e:
                self.fail(f"Forward pass raised exception: {e}")
