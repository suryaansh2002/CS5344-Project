import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import logging

class MultiModalModel(nn.Module):
    def __init__(self, tokenizer, hidden_dim=128):
        super(MultiModalModel, self).__init__()

        # Image model
        self.cnn = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        logging.info("Loaded EfficientNet-B0 model.")
        self.cnn.classifier = nn.Identity()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.img_fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Text model
        self.embedding = nn.Embedding(tokenizer.vocab_size, 128, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        logging.info("Loaded biLSTM model.")

        # Combined
        self.fc = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        logging.info('Entire model loaded.')

    def forward(self, img, text):
        img_feat = self.img_fc(self.cnn(img))
        text_emb = self.embedding(text)
        _, (hidden, _) = self.lstm(text_emb)
        text_feat = self.text_fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined).squeeze(1)
