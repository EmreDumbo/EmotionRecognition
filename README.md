# Emotion Recognition from Facial Expressions

This repository contains the code for training and evaluating a deep learning model for emotion recognition from facial expressions using the FER-2013 dataset.

## Project Structure

```plaintext
emotion-recognition/
│
├── src/
│   ├── dataloader/
│   │   ├── __init__.py
│   │   ├── transformations.py
│   │   └── data.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet50 
│   │
│   │── train.py
│   │── evaluate.py
│   │── model.py
│   └── main.py
│   
│
├── dataset/
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
├── requirements.txt
└── README.md
```

## Requirements

To set up the environment, you can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```plaintext
torch==2.1.0
torchvision==0.15.0
pandas==1.5.3
matplotlib==3.7.0
scikit-learn==1.2.2
Pillow==9.4.0
tqdm==4.64.1
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/emotion-recognition.git
   cd emotion-recognition
   ```

2. Ensure that you have the correct dependencies installed (using `pip install -r requirements.txt`).

3. Prepare the `train_dataset.csv` and `test_dataset.csv` files in the `dataset/` folder. The CSV should contain the image filenames and their corresponding labels.

4. Run the training and evaluation pipeline:

   ```bash
   python src/main.py
   ```

This will start the training, evaluate the model, and display results such as training loss, accuracy, precision, recall, F1-score, confusion matrix, and misclassified images.

## Code Structure

### `src/main.py`
This script initializes the emotion recognition pipeline, runs the training, and saves the trained model.

### `src/models/model.py`
Defines the `ModelCreator` class which creates various model architectures (e.g., ResNet50, VGG16, VGG19, InceptionV3) for emotion recognition.

### `src/models/train.py`
Contains the `ModelTrainer` class responsible for training the model, calculating losses, and saving training logs.

### `src/models/evaluate.py`
Contains the `ModelEvaluator` class to evaluate the trained model on the test dataset, compute metrics like precision, recall, F1-score, and visualize results.

### `src/dataloader/data.py`
Handles data loading and preparation using PyTorch's `DataLoader`. It uses a custom `CustomImageDataset` class for reading image data from the CSV annotations and applying transformations.

### `src/dataloader/transformations.py`
Applies data augmentation and normalization techniques on the images to enhance model generalization during training.

## Model Architectures

You can experiment with different model architectures, including:
- **ResNet50**
- **VGG16**
- **VGG19**
- **InceptionV3**

The models are pretrained on ImageNet and fine-tuned for emotion recognition.

## Training and Evaluation

The training process involves the following steps:

1. **Model Training**: 
   - The model is trained using the `ModelTrainer` class.
   - Training logs (loss and accuracy per epoch) are saved to a file (`training_log.txt`).

2. **Model Evaluation**:
   - The trained model is evaluated on the test dataset using the `ModelEvaluator` class.
   - Metrics like precision, recall, F1-score, and confusion matrix are calculated.
   - Misclassified images are displayed for further inspection.

3. **Results Visualization**:
   - Metrics are visualized using Matplotlib.
   - Confusion matrix and class-wise accuracies are shown.

## Example Output

During training, you will see logs like this:

```
Epoch 1/10, Loss: 1.2785, Accuracy: 0.2894
Epoch 2/10, Loss: 1.0835, Accuracy: 0.3893
...
```

After training, the evaluation will output results such as:

```
Test Loss: 0.7458, Test Accuracy: 0.5980
Precision for class 'Happy': 0.76
Recall for class 'Anger': 0.89
F1 Score for class 'Anger': 0.82
...
```

The confusion matrix and misclassified images will be displayed as well.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER-2013 dataset
- PyTorch and torchvision for model implementations
- Scikit-learn for metrics and evaluation
- Matplotlib for visualization
