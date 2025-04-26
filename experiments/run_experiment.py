import torch.nn as nn
import torch.optim as optim
from benchmarking.benchmark import Benchmark
from database.database_handler import DatabaseHandler
from datasets.dataset_loader import load_dataset


def main():
    model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    dataset_name = "iris"
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    db_handler = DatabaseHandler()
    architecture_str = str(model)
    existing_result = db_handler.search_result(architecture_str, dataset_name, match_mode="exact")
    
    if existing_result:
        print("Exact match found in database. Returning saved result.")
        return existing_result
    else:
        print("No Exact match found in database. Trying to find similar...")


    similar_result = db_handler.search_result(
        architecture_str, 
        dataset_name, 
        match_mode="similar", 
        similarity_threshold=0.9, 
        similarity_method="structure"
    )

    if similar_result:
        print("Similar model found in database (based on structure comparison).")
        return similar_result
    else:
        print("No similar match found in database. Running benchmark")

    train_loader, test_loader = load_dataset(dataset_name)
    benchmark = Benchmark(model, train_loader, test_loader, criterion, optimizer, epochs=5)
    results = benchmark.benchmark_model()

    db_handler.save_result(architecture_str, dataset_name, results)
    print("Benchmark complete. Results saved to database.")
    print(results)
    return results


if __name__ == "__main__":
    results = main()
    print(results)
