from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import torch
import io
import json
from database.database_handler import DatabaseHandler

app = FastAPI()
db_handler = DatabaseHandler()

class SearchRequest(BaseModel):
    dataset_hash: str

@app.post("/save")
async def save_model(
    model_blob: UploadFile = File(...),
    dataset_name: str = Form(...),
    dataset_hash: str = Form(...),
    criterion: str = Form(...),
    optimizer: str = Form(...),
    result_dict: str = Form(...)
):
    model_data = await model_blob.read()
    buffer = io.BytesIO(model_data)
    model = torch.load(buffer, weights_only=False)
    result = json.loads(result_dict)

    db_handler.save_result(model, dataset_name, dataset_hash, criterion, optimizer, result)
    return {"status": "saved"}

@app.post("/search")
def search_model(data: SearchRequest):
    entries = db_handler.get_all_by_dataset_hash(data.dataset_hash)

    response = []
    for model_blob, result_json, criterion, optimizer in entries:
        response.append({
            "model_blob": model_blob.hex(),
            "result_json": result_json,
            "criterion": criterion,
            "optimizer": optimizer
        })

    return response
