import torch.nn as nn
from torchvision import models
from transformers import AutoModel
import logging

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256)
        )
        for param in self.model.parameters():
            param.requires_grad = False
        logging.info("Loaded MobileNetV2 model.")

    def forward(self, x):
        return self.model(x)

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
        self.fc = nn.Linear(768, 256)
        for param in self.model.parameters():
            param.requires_grad = False
        logging.info("Loaded RoBERTa model.")

    def forward(self, x):
        output = self.model(x).last_hidden_state[:, 0, :]
        return self.fc(output)

class MultiModalModel(nn.Module):
    def __init__(self, alpha=0.5):
        super(MultiModalModel, self).__init__()
        self.image_model = ImageModel()
        self.text_model = TextModel()
        self.alpha = alpha
        self.fc = nn.Linear(256, 1)
        logging.info("Loaded MobileNet + RoBERTa model.")

    def forward(self, img, text):
        img_feat = self.image_model(img)
        text_feat = self.text_model(text)
        combined = self.alpha * img_feat + (1 - self.alpha) * text_feat
        return self.fc(combined).squeeze(1)
