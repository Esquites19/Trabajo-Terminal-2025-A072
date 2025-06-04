import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MobileNetV2Classifier, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)