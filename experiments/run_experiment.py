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
        nn.Linear(10, 10)
    )
    dataset_name = "breast_cancer"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion_str = stringify_criterion(criterion)
    optimizer_str = stringify_optimizer(optimizer)

    train_loader, test_loader, dataset_hash = load_dataset(dataset_name)

    db_handler = DatabaseHandler()
    db_model, existing_result = db_handler.search_result(
        model,
        dataset_hash,
        criterion_str,
        optimizer_str,
        match_mode="exact"
    )

    if existing_result:
        print("Exact match found in database. Returning saved result.")
        return db_model, existing_result
    else:
        print("No Exact match found in database. Trying to find similar...")

    db_model, similar_result = db_handler.search_result(
        model,
        dataset_hash,
        match_mode="similar",
        similarity_threshold=0.9,
        similarity_method="string"
    )

    if similar_result:
        print("Similar model found in database (based on string comparison).")
        return db_model, similar_result
    else:
        print("No similar match found in database. Running benchmark")

    benchmark = Benchmark(model, train_loader, test_loader, criterion, optimizer, epochs=5)
    results = benchmark.benchmark_model()

    db_handler.save_result(model, dataset_name, dataset_hash, criterion_str, optimizer_str, results)
    print("Benchmark complete. Results saved to database.")
    return model, results


if __name__ == "__main__":
    returned_model, returned_results = main()
    benchmark = Benchmark(returned_model, None, load_dataset("iris")[1], nn.CrossEntropyLoss(), optim.Adam(returned_model.parameters(), lr=0.01), epochs=5)
    avg_loss = benchmark.evaluate_model()
    print(f"Saved results: {returned_results}")
    print(f"Model results: {avg_loss}")
