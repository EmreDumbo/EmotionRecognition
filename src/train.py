from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import os

class ModelTrainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, epochs=10, log_file="training_log.txt"):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.train_losses = []
        self.train_accuracies = []
        self.log_file = log_file
        self.best_accuracy = 0

        if not os.path.exists(self.log_file) or os.stat(self.log_file).st_size == 0:
            with open(self.log_file, "w") as f:
                f.write("Epoch\tLoss\tAccuracy\n")
                f.flush()

    def train(self):
        self.model.train()  
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", total=len(self.train_loader), ncols=100, leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1) 
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)

            if epoch_acc >= self.best_accuracy:
                self.best_accuracy = epoch_acc
            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1}\t{epoch_loss:.4f}\t{epoch_acc:.4f}\n")
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        return self.train_losses, self.train_accuracies
    
    def plot_training_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label = "Training loss", marker='o')
        plt.plot(range(1, len(self.train_accuracies) + 1), self.train_accuracies, label="Training Accuracy", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.legend()
        plt.grid(True)
        plt.show()