import torch.nn as nn
import torch.optim as optim
from benchmarking.benchmark import Benchmark
from database.database_handler import DatabaseHandler
from datasets.dataset_loader import load_dataset
from utils.helpers import stringify_criterion, stringify_optimizer


def main():
    model = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
    dataset_name = "iris"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion_str = stringify_criterion(criterion)
    optimizer_str = stringify_optimizer(optimizer)

    db_handler = DatabaseHandler()
    existing_result = db_handler.search_result(
        model,
        dataset_name,
        criterion_str,
        optimizer_str,
        match_mode="exact"
    )
    
    if existing_result:
        print("Exact match found in database. Returning saved result.")
        return existing_result
    else:
        print("No Exact match found in database. Trying to find similar...")

    similar_result = db_handler.search_result(
        model,
        dataset_name, 
        match_mode="similar",
        similarity_threshold=0.9, 
        similarity_method="string"
    )

    if similar_result:
        print("Similar model found in database (based on string comparison).")
        return similar_result
    else:
        print("No similar match found in database. Running benchmark")

    train_loader, test_loader = load_dataset(dataset_name)
    benchmark = Benchmark(model, train_loader, test_loader, criterion, optimizer, epochs=5)
    results = benchmark.benchmark_model()

    db_handler.save_result(str(model), dataset_name, criterion_str, optimizer_str, results)
    print("Benchmark complete. Results saved to database.")
    return results


if __name__ == "__main__":
    results = main()
    print(results)
