import sqlite3
import json
from benchmarking.architecture_similarity import ArchitectureComparator


class DatabaseHandler:
    def __init__(self, db_path="database/benchmark_results.db"):
        self.db_path = db_path
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_architecture TEXT,
                dataset_name TEXT,
                result_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_result(self, model_architecture, dataset_name, result_dict):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO benchmarks (model_architecture, dataset_name, result_json)
            VALUES (?, ?, ?)
        ''', (model_architecture, dataset_name, json.dumps(result_dict)))
        conn.commit()
        conn.close()

    def search_result(self, 
                      model, 
                      dataset_name, 
                      match_mode="exact", 
                      similarity_threshold=0.9, 
                      similarity_method="structure"):
        
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT model_architecture, dataset_name, result_json FROM benchmarks
            WHERE dataset_name = ?
        ''', (dataset_name,))

        entries = cursor.fetchall()
        conn.close()

        if match_mode == "exact":
            for db_model, dataset, result_json in entries:
                if db_model == model:
                    return json.loads(result_json)


        elif match_mode == "similar":
            for db_model, dataset, result_json in entries:
                comparator = ArchitectureComparator(model, db_model, method=similarity_method)
                similarity = comparator.compute_similarity()
                if similarity >= similarity_threshold:
                    return json.loads(result_json)

        return None
