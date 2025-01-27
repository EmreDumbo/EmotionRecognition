import torch.nn as nn
import torchvision.models as models

class ModelCreator(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=7):
        super(ModelCreator, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._create_model()

    def _create_model(self):
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_classes)
        elif self.model_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_classes)
        elif self.model_name == 'inception_v3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
        
        return model

    def forward(self, x):
        return self.model(x)   

    