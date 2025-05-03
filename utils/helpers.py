def stringify_criterion(criterion):
    return criterion.__class__.__name__

def stringify_optimizer(optimizer):
    opt_class = optimizer.__class__.__name__
    opt_params = optimizer.defaults
    return f"{opt_class}({opt_params})"
