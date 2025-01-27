from dataloader.transformations import Transformations
from torch.utils.data import DataLoader
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        print(img_path) 
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(train_csv, test_csv, batch_size=16):
    transformations = Transformations()

    train_dataset = CustomImageDataset(annotations_file=train_csv, img_dir = '/Users/emre/Desktop/emotion/dataset/train', transform=transformations)
    test_dataset = CustomImageDataset(annotations_file=test_csv, img_dir='/Users/emre/Desktop/emotion/dataset/test', transform=transformations)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset
