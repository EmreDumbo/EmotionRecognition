import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ModelEvaluator:
    def __init__(self, model, test_loader, criterion, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.class_names = class_names
        self.losses = []

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

        test_loss = running_loss / len(self.test_loader)
        test_acc = correct / total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        for i, class_name in enumerate(self.class_names):
            print(f"Precision for class '{class_name}': {precision[i]:.4f}")
            print(f"Recall for class '{class_name}': {recall[i]:.4f}")
            print(f"F1 Score for class '{class_name}': {f1[i]:.4f}")

        self.losses.append(test_loss)
        print("Test loss being added:", test_loss)
        self.plot_metrics(precision, recall, f1)

        return test_loss, test_acc, precision, recall, f1

    def plot_metrics(self, precision, recall, f1):
        x = np.arrange(len(self.class_names))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision, width, label='Precision', color='skyblue')
        plt.bar(x, recall, width, label='Recall', color='lightgreen')
        plt.bar(x + width, f1, width, label='F1 Score', color='salmon')

        plt.xlabel('Classes')
        plt.ylabel('Scores')
        plt.title('Evaluation Metrics per Class')
        plt.xticks(x, self.class_names, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    def plot_confusion_matrix(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.class_names)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.show()

    def compute_class_wise_accuracy(self):
        self.model.eval()
        correct = [0] * len(self.class_names)
        total = [0] * len(self.class_names)

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                
                for label in labels:
                    if label.item() >= len(self.class_names):
                        print(f"Warning: Invalid label {label.item()} found in the dataset.")
                    else:
                        total[label.item()] += 1

                
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        correct[label.item()] += 1

        for i, class_name in enumerate(self.class_names):
            accuracy = 100 * correct[i] / total[i] if total[i] > 0 else 0
            print(f"Accuracy for class '{class_name}': {accuracy:.2f}%")

    def visualize_predictions(self):
        self.model.eval()

        total_images = len(self.test_loader.dataset)
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)

                inputs = inputs.cpu()
                predicted = predicted.cpu()
                labels = labels.cpu()

                for i in range(inputs.size(0)):
                    image = inputs[i].numpy().transpose((1, 2, 0))
                    image = np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

                    plt.imshow(image)
                    plt.title(f"Predicted: {self.class_names[predicted[i]]}\n True:{self.class_names[labels[i]]}")
                    plt.axis('off')
                    plt.show()

                    if (i + 1) >= total_images:
                        return
                    input("Press Enter to show another image or Ctrl+C to stop...")