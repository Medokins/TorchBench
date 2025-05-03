import sqlite3
import json

def display_database_contents(db_path="database\\benchmark_results.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, model_architecture, dataset_name, result_json FROM benchmarks")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        print(f"ID: {row[0]}")
        print(f"Model Architecture:\n{row[1]}")
        print(f"Dataset Name: {row[2]}")
        print("Result:")
        try:
            result_dict = json.loads(row[3])
            for key, value in result_dict.items():
                if key != "model_architecture":
                    print(f"  {key}: {value}")
        except json.JSONDecodeError:
            print("  [Invalid JSON]")
        print("-" * 40)

if __name__ == "__main__":
    display_database_contents("database\\benchmark_results.db")
