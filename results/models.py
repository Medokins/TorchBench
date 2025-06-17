import torch.nn as nn

def make_classification_models(input_dim, output_dim):
    return {
        "C1": nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, output_dim)),
        "C2": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, output_dim)),
        "C3": nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, output_dim)),
        "C4": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim)),
        "C5": nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)),
        "C6": nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, output_dim)),
        "C7": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim)),
        "C8": nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)),
        "C9": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, output_dim)),
        "C10": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, output_dim))
    }

def make_regression_models(input_dim):
    return {
        "R1": nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1)),
        "R2": nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
        "R3": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1)),
        "R4": nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)),
        "R5": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)),
        "R6": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)),
        "R7": nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)),
        "R8": nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)),
        "R9": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)),
        "R10": nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))
    }
