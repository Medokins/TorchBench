import sqlite3
import json
import io
import torch
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
                dataset_name TEXT,
                dataset_hash TEXT,
                criterion TEXT,
                optimizer TEXT,
                result_json TEXT,
                model_blob BLOB
            )
        ''')
        conn.commit()
        conn.close()

    def save_result(self, model, dataset_name, dataset_hash, criterion, optimizer, result_dict):
        conn = self._connect()
        cursor = conn.cursor()

        buffer = io.BytesIO()
        torch.save(model, buffer)
        model_blob = buffer.getvalue()

        cursor.execute('''
            INSERT INTO benchmarks (dataset_name, dataset_hash, criterion, optimizer, result_json, model_blob)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            dataset_name,
            dataset_hash,
            str(criterion),
            str(optimizer),
            json.dumps(result_dict),
            model_blob
        ))
        conn.commit()
        conn.close()

    def search_result(self, 
                      model, 
                      dataset_hash, 
                      criterion=None,
                      optimizer=None,
                      match_mode="exact", 
                      similarity_threshold=0.9, 
                      similarity_method="graph_structural"):

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT model_blob, result_json, criterion, optimizer FROM benchmarks
            WHERE dataset_hash = ?
        ''', (dataset_hash,))

        entries = cursor.fetchall()
        conn.close()

        if match_mode == "exact":
            for model_blob, result_json, db_criterion, db_optimizer in entries:
                try:
                    buffer = io.BytesIO(model_blob)
                    candidate_model = torch.load(buffer, weights_only=False)
                    candidate_model.eval()

                    comparator = ArchitectureComparator(model, candidate_model, method="graph_structural_with_params")
                    similarity = comparator.compute_similarity()

                    if (
                        similarity == 1.0 and
                        db_criterion == str(criterion) and
                        db_optimizer == str(optimizer)
                    ):
                        return candidate_model, json.loads(result_json)

                except Exception as e:
                    print(f"Error loading exact model: {e}")
                    continue

        elif match_mode == "similar":
            for model_blob, result_json, db_criterion, db_optimizer in entries:
                try:
                    buffer = io.BytesIO(model_blob)
                    candidate_model = torch.load(buffer, weights_only=False)
                    candidate_model.eval()
                except Exception as e:
                    print(f"Skipping entry due to error: {e}")
                    continue

                comparator = ArchitectureComparator(model, candidate_model, method=similarity_method)
                similarity = comparator.compute_similarity()
                if similarity >= similarity_threshold:
                    return candidate_model, json.loads(result_json)

        return None, None
    
    def get_all_by_dataset_hash(self, dataset_hash):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT model_blob, result_json, criterion, optimizer FROM benchmarks
            WHERE dataset_hash = ?
        ''', (dataset_hash,))
        results = cursor.fetchall()
        conn.close()
        return results
