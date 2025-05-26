import io
import json
import requests
import torch
from benchmarking.architecture_similarity import ArchitectureComparator


class APIDatabaseHandler:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def save_result(self, model, dataset_name, dataset_hash, criterion, optimizer, result_dict):
        buffer = io.BytesIO()
        torch.save(model, buffer)
        model_blob = buffer.getvalue()

        files = {
            "model_blob": ("model.pt", model_blob, "application/octet-stream")
        }

        data = {
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "criterion": str(criterion),
            "optimizer": str(optimizer),
            "result_dict": json.dumps(result_dict)
        }

        response = requests.post(f"{self.base_url}/save", data=data, files=files)
        if response.status_code != 200:
            raise Exception(f"Failed to save result: {response.text}")

    def search_result(self, 
                      model, 
                      dataset_hash, 
                      criterion=None,
                      optimizer=None,
                      match_mode="exact", 
                      similarity_threshold=0.9, 
                      similarity_method="graph_structural"):

        response = requests.post(f"{self.base_url}/search", json={"dataset_hash": dataset_hash})
        if response.status_code != 200:
            raise Exception(f"Failed to query results: {response.text}")

        entries = response.json()

        for entry in entries:
            try:
                model_blob = bytes.fromhex(entry["model_blob"])
                buffer = io.BytesIO(model_blob)
                candidate_model = torch.load(buffer, weights_only=False)
                candidate_model.eval()

                if match_mode == "exact":
                    comparator = ArchitectureComparator(model, candidate_model, method="graph_structural_with_params")
                    similarity = comparator.compute_similarity()
                    if (
                        similarity == 1.0 and
                        entry["criterion"] == str(criterion) and
                        entry["optimizer"] == str(optimizer)
                    ):
                        return candidate_model, json.loads(entry["result_json"])

                elif match_mode == "similar":
                    comparator = ArchitectureComparator(model, candidate_model, method=similarity_method)
                    similarity = comparator.compute_similarity()
                    if similarity >= similarity_threshold:
                        return candidate_model, json.loads(entry["result_json"])

            except Exception as e:
                print(f"Skipping entry due to error: {e}")
                continue

        return None, None
