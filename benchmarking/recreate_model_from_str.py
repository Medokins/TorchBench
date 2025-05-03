import torch.nn as nn
import re


def recreate_model_from_string(model_str):
    pattern = r'\((\d+)\): (.+)'
    lines = model_str.splitlines()
    layers = []

    for line in lines:
        match = re.search(pattern, line.strip())
        if not match:
            continue
        layer_str = match.group(2)

        if layer_str.startswith("Linear"):
            args_str = layer_str[len("Linear("):-1]
            args = {k: eval(v) for k, v in [arg.strip().split("=") for arg in args_str.split(",")]}
            layers.append(nn.Linear(**args))

        elif layer_str.startswith("ReLU"):
            layers.append(nn.ReLU())

        elif layer_str.startswith("Sigmoid"):
            layers.append(nn.Sigmoid())

        elif layer_str.startswith("Tanh"):
            layers.append(nn.Tanh())

        elif layer_str.startswith("LeakyReLU"):
            args_str = layer_str[len("LeakyReLU("):-1]
            if args_str:
                args = {k: eval(v) for k, v in [arg.strip().split("=") for arg in args_str.split(",")]}
                layers.append(nn.LeakyReLU(**args))
            else:
                layers.append(nn.LeakyReLU())

        elif layer_str.startswith("Dropout"):
            args_str = layer_str[len("Dropout("):-1]
            p = float(args_str.split('=')[1]) if args_str else 0.5
            layers.append(nn.Dropout(p))

        elif layer_str.startswith("BatchNorm1d"):
            args_str = layer_str[len("BatchNorm1d("):-1]
            num_features = int(args_str.split('=')[1]) if '=' in args_str else int(args_str)
            layers.append(nn.BatchNorm1d(num_features))

        elif layer_str.startswith("Flatten"):
            layers.append(nn.Flatten())

        elif layer_str.startswith("Conv2d"):
            args_str = layer_str[len("Conv2d("):-1]
            args = {k: eval(v) for k, v in [arg.strip().split("=") for arg in args_str.split(",")]}
            layers.append(nn.Conv2d(**args))

        elif layer_str.startswith("MaxPool2d"):
            args_str = layer_str[len("MaxPool2d("):-1]
            args = {k: eval(v) for k, v in [arg.strip().split("=") for arg in args_str.split(",")]}
            layers.append(nn.MaxPool2d(**args))

        elif layer_str.startswith("AvgPool2d"):
            args_str = layer_str[len("AvgPool2d("):-1]
            args = {k: eval(v) for k, v in [arg.strip().split("=") for arg in args_str.split(",")]}
            layers.append(nn.AvgPool2d(**args))

    return nn.Sequential(*layers)