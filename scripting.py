import json
import os
import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Script started!")

# Load annotations
with open('DATA/MMHS150K_GT.json', 'r') as f:
    annotations = json.load(f)

# Convert JSON dict to DataFrame
def parse_annotations(annotations):
    data = []
    for tweet_id, info in annotations.items():
        data.append({
            'tweet_id': tweet_id,
            'tweet_text': info['tweet_text'],
            'labels': info['labels'],
            'labels_str': info['labels_str']
        })
    return pd.DataFrame(data)

df = parse_annotations(annotations)
logging.info("Loaded dataset:")
logging.info(df.head())

# Define paths
image_folder = './multimodal-hate-speech/img_resized'
df['image_path'] = df['tweet_id'].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

# Define majority vote function
def majority_vote(labels):
    label_count = Counter(labels)
    return label_count.most_common(1)[0][0]

df['majority_label'] = df['labels'].apply(majority_vote)

# Map labels
label_mapping = {
    0: "NotHate",
    1: "Racist",
    2: "Sexist",
    3: "Homophobe",
    4: "Religion",
    5: "OtherHate"
}
df['majority_label_str'] = df['majority_label'].map(label_mapping)
df['binary_label'] = df['majority_label'].apply(lambda x: 1 if x in [2, 3, 4, 5] else x)
df['binary_label_str'] = df['majority_label_str'].apply(lambda x: 'Hate' if x in ['Religion', 'OtherHate', 'Racist', 'Sexist', 'Homophobe'] else x)

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_text'] = df['tweet_text'].apply(preprocess_text)
df.drop(['labels', 'labels_str', 'majority_label', 'majority_label_str', 'tweet_text'], axis=1, inplace=True)

logging.info("Label distribution:")
logging.info(df['binary_label_str'].value_counts())
logging.info("Sample cleaned text:")
logging.info(df['cleaned_text'].head())

# Load and preprocess images
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    if not os.path.exists(img_path):
        return np.zeros((target_size[0], target_size[1], 3))
    try:
        img = load_img(img_path, target_size=target_size)
        return img_to_array(img) / 255.0
    except:
        return np.zeros((target_size[0], target_size[1], 3))

# Split data into 5 chunks to process in batches
chunk_size = 30000
num_chunks = (len(df) // chunk_size) + 1

data_chunks = [df[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

for i, chunk in enumerate(data_chunks):
    logging.info(f"Processing chunk {i+1}...")
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(chunk['cleaned_text'])
    text_data = pad_sequences(tokenizer.texts_to_sequences(chunk['cleaned_text']), maxlen=100)
    image_data = np.array([load_and_preprocess_image(path) for path in chunk['image_path']])
    logging.info(f"Chunk {i+1} processed successfully!")

logging.info("All chunks processed successfully!")
