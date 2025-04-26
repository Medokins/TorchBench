import torch


class Benchmark:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, epochs=10, flatten_inputs=True):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.flatten_inputs = flatten_inputs

    def train_model(self):
        self.model.train()
        epoch_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for inputs, targets in self.train_loader:
                if self.flatten_inputs:
                    inputs = inputs.view(inputs.size(0), -1)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(self.train_loader)
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {epoch_loss:.4f}")

        return epoch_losses

    def evaluate_model(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                if self.flatten_inputs:
                    inputs = inputs.view(inputs.size(0), -1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def benchmark_model(self):
        print("Starting benchmarking...")

        print("Training model...")
        train_losses = self.train_model()

        print("Evaluating model...")
        test_loss = self.evaluate_model()

        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        return {
            "model_architecture": str(self.model),
            "train_losses": train_losses,
            "test_losses": test_loss,
            "avg_train_loss": sum(train_losses) / len(train_losses),
            "final_train_loss": train_losses[-1],
            "test_loss": test_loss
        }
