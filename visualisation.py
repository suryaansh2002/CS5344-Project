import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import nltk
from collections import Counter
from tqdm import tqdm

nltk.download('punkt')

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logging.info(f"Using device: {device}")

df = pd.read_csv("balanced_dataset.csv")

def most_common(df):
    hate_texts = df[df['Binary_label_str'] == "Hate"]
    nonhate_texts = df[df['Binary_label_str'] == "Hate"]

    hate_words_tokens = []
    nonhate_words_tokens = []
    for text in hate_texts:
        hate_words_tokens.extend(nltk.word_tokenize(text))
    for text in nonhate_texts:
        nonhate_words_tokens.extend(nltk.word_tokenize(text))

    hate_word_count = Counter(hate_words_tokens)
    nonhate_word_count = Counter(nonhate_words_tokens)

    distinctive_hate_words = {}
    for word, count in hate_word_count.items():
        delta = count - nonhate_word_count.get(word, 0)
        if delta > 100:
            distinctive_hate_words[word] = delta

    distinctive_hate_words = dict(sorted(distinctive_hate_words.items(), key=lambda x: x[1], reverse=True))

    most_common = hate_word_count.most_common(50)
    return most_common

class MultiModalDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        # return 100
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        try:
            image = Image.open(image_path)
        except:
            image = torch.zeros(3, 224, 224)

        text = row['cleaned_text']
        label = torch.tensor(row['binary_label'], dtype=torch.float32)
        return image, text, label