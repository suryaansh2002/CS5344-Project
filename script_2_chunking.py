import pandas as pd
import logging
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Script started!")

df = pd.read_csv("balanced_dataset.csv")
logging.info("Loaded dataset:")
logging.info(df.info())
             
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