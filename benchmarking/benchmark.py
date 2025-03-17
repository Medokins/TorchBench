import torch


def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)  # Average loss for this epoch
        epoch_losses.append(epoch_loss)  # Store epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

    return epoch_losses

def evaluate_model(model, test_loader, criterion, epochs=10):
    model.eval()
    epoch_losses = []

    with torch.no_grad():
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

            avg_loss = total_loss / len(test_loader)  # Average test loss
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_loss:.4f}")

    return epoch_losses

def benchmark_model(model, train_loader, test_loader, criterion, optimizer, epochs=10):
    print("Starting benchmarking...")
    
    print("Training model...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs)
    
    print("Evaluating model...")
    test_losses = evaluate_model(model, test_loader, criterion, epochs)
    
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")

    return {
        "model_architecture": str(model),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "avg_train_loss":  sum(train_losses) / len(train_losses),
        "avg_test_loss": sum(test_losses) / len(test_losses),
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1]
    }
