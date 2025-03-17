import torch
import torch.nn as nn


class ModelHandler:
    def __init__(self, model: nn.Module):
        self.model = model
    
    def get_model_summary(self):
        """Returns a summary of the model architecture."""
        return str(self.model)
    
    def add_layer(self, layer: nn.Module, position: int = -1):
        """Adds a layer to the model at a specified position."""
        layers = list(self.model.children())
        if position == -1:
            layers.append(layer)
        else:
            layers.insert(position, layer)
        self.model = nn.Sequential(*layers)
    
    def remove_layer(self, position: int):
        """Removes a layer at a specified position."""
        layers = list(self.model.children())
        if 0 <= position < len(layers):
            layers.pop(position)
            self.model = nn.Sequential(*layers)
    
    def replace_layer(self, position: int, new_layer: nn.Module):
        """Replaces a layer at a specified position."""
        layers = list(self.model.children())
        if 0 <= position < len(layers):
            layers[position] = new_layer
            self.model = nn.Sequential(*layers)
    
    def save_model(self, path: str):
        """Saves the model to a given path."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Loads the model from a given path."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
