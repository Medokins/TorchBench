import sqlite3
import json
from benchmarking.architecture_similarity import ArchitectureComparator
from benchmarking.recreate_model_from_str import recreate_model_from_string


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
                criterion TEXT,
                optimizer TEXT,
                result_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_result(self, model_architecture, dataset_name, criterion, optimizer, result_dict):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO benchmarks (model_architecture, dataset_name, criterion, optimizer, result_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            model_architecture,
            dataset_name,
            str(criterion),
            str(optimizer),
            json.dumps(result_dict)
        ))
        conn.commit()
        conn.close()

    def search_result(self, 
                      model, 
                      dataset_name, 
                      criterion=None,
                      optimizer=None,
                      match_mode="exact", 
                      similarity_threshold=0.9, 
                      similarity_method="graph_structural"):
                
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT model_architecture, dataset_name, criterion, optimizer, result_json FROM benchmarks
            WHERE dataset_name = ?
        ''', (dataset_name,))

        entries = cursor.fetchall()
        conn.close()

        if match_mode == "exact":
            model = str(model)
            for db_model, db_dataset, db_criterion, db_optimizer, result_json in entries:
                if db_model == model and db_optimizer == str(optimizer) and db_criterion == str(criterion):
                    return json.loads(result_json)

        elif match_mode == "similar":
            for db_model_str, db_dataset, db_criterion, db_optimizer, result_json in entries:
                db_model = recreate_model_from_string(db_model_str)
                comparator = ArchitectureComparator(model, db_model, method=similarity_method)
                similarity = comparator.compute_similarity()
                if similarity >= similarity_threshold:
                    return json.loads(result_json)

        return None
