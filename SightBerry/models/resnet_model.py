import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Classifier(nn.Module):
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
        # Congelar capas excepto layer4 y fc
        for name, param in self.base_model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.base_model(x)