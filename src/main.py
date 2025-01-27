import torch
import torch.optim as optim
import torch.nn as nn
from dataloader.data import get_dataloaders
from model import ModelCreator
from evaluate import ModelEvaluator
from train import ModelTrainer



class EmotionRecognitionPipeline:
    def __init__(self, train_csv, test_csv, batch_size=16, epochs=10, learning_rate=0.001, device=None, log_file="src/models/resnet50/training_log_resnet50.txt"):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.log_file = log_file
        self.train_loader, self.test_loader, self.train_dataset = self.get_dataloaders()
        self.class_names = self.train_dataset.img_labels['label'].unique()
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_losses = []
        self.train_accuracies = []


    def get_dataloaders(self):
        return get_dataloaders(self.train_csv, self.test_csv, self.batch_size)
    
    def _create_model(self):
        model_creator = ModelCreator(model_name='resnet50', num_classes=len(self.class_names))
        model = model_creator._create_model()
        return model.to(self.device)
    
    def train(self):
        trainer = ModelTrainer(self.model, self.train_loader, self.criterion, self.optimizer, self.device, self.epochs, log_file=self.log_file )
        self.train_losses, self.train_accuracies = trainer.train()        
        trainer.plot_training_metrics()

    def evaluate(self):
        evaluator = ModelEvaluator(self.model, self.test_loader, self.criterion, self.device, self.class_names)
        evaluator.evaluate()
        evaluator.plot_metrics()
        evaluator.plot_confusion_matrix()
        evaluator.compute_class_wise_accuracy()
        evaluator.visualize_predictions()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def run(self, save_filename="models/resnet50/emotion_recognition_resnet50.pth"):
        self.train()
        self.save_model(save_filename)
        self.evaluate()

if __name__ == "__main__":
    train_csv = "dataset/train_dataset.csv"
    test_csv = "dataset/test_dataset.csv"


    pipeline = EmotionRecognitionPipeline(train_csv=train_csv, test_csv=test_csv, batch_size=16, epochs=10, learning_rate=0.001)
    pipeline.run(save_filename="models/resnet50/emotion_recognition_resnet50.pth")     