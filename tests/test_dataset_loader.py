import unittest
from datasets.dataset_loader import load_dataset


class TestDatasetLoader(unittest.TestCase):
    def test_load_all_datasets(self):
        datasets = ["california_housing", "diabetes", "wine", "breast_cancer"]
        
        for dataset_name in datasets:
            with self.subTest(dataset=dataset_name):
                train_loader, test_loader = load_dataset(dataset_name)
            
                train_iter = iter(train_loader)
                test_iter = iter(test_loader)

                train_batch = next(train_iter, None)
                test_batch = next(test_iter, None)

                self.assertIsNotNone(train_batch)
                self.assertIsNotNone(test_batch)

    def test_load_unsupported_dataset(self):
        with self.assertRaises(ValueError) as context:
            load_dataset("iris")

        assert("Dataset 'iris' not supported" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
