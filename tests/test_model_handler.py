import unittest
import torch
import torch.nn as nn
import torchvision.models as models
from benchmarking.model_handler import ModelHandler


class TestModelHandler(unittest.TestCase):
    def test_model_operations(self):
        model = models.resnet18(weights=None)
        handler = ModelHandler(model)
        
        summary_before = handler.get_model_summary()
        self.assertIsInstance(summary_before, str)
        
        handler.add_layer(nn.Linear(512, 10))
        summary_after_add = handler.get_model_summary()
        self.assertNotEqual(summary_before, summary_after_add)
        
        handler.remove_layer(1)
        summary_after_remove = handler.get_model_summary()
        self.assertNotEqual(summary_after_add, summary_after_remove)
        
        handler.save_model("test_model.pth")
        handler.load_model("test_model.pth")
        self.assertIsInstance(handler.model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
