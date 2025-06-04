import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Classifier(nn.Module):
    """
    ResNet50 personalizado para PyTorch con fine-tuning.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        self.base_model = models.resnet50(pretrained=pretrained)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        # Congelar solo las primeras 15 capas (parámetros)
        params = list(self.base_model.parameters())
        for i, param in enumerate(params):
            if i < 15:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)