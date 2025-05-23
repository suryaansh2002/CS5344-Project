import json
import os
import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import random
from torchvision import transforms
import nltk
from nltk.corpus import wordnet


# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Script started!")

# annotations
with open('DATA/MMHS150K_GT.json', 'r') as f:
    annotations = json.load(f)

# JSON dict to DataFrame
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

IMAGE_DIR = 'DATA/img_resized'
AUGMENTED_IMAGE_DIR = "DATA/img_augmented"
os.makedirs(AUGMENTED_IMAGE_DIR, exist_ok=True)

# add image paths based on tweet_id to dataframe
image_folder = 'DATA/img_resized'
df['image_path'] = df['tweet_id'].apply(lambda x: os.path.join(image_folder, f"{x}.jpg"))

logging.info("Loaded dataset:")
logging.info(df.head())

# majority vote function
def majority_vote(labels):
    label_count = Counter(labels)
    return label_count.most_common(1)[0][0]

df['majority_label'] = df['labels'].apply(majority_vote)

# map labels
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

# preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['cleaned_text'] = df['tweet_text'].apply(preprocess_text)
df.drop(['labels', 'labels_str', 'majority_label', 'majority_label_str', 'tweet_text','tweet_id'], axis=1, inplace=True)

logging.info('Current datframe:')
logging.info(df.info())
logging.info("Label distribution:")
logging.info(df['binary_label_str'].value_counts())
logging.info("Sample cleaned text:")
logging.info(df['cleaned_text'].head())

# begin balancing

def augment_image(image_path):
    img = Image.open(image_path).convert("RGB")

    augmentations = [
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
    ]
    
    transform = random.choice(augmentations)
    img = transform(img)
    new_image_name = f"aug_{random.randint(10000, 99999)}.jpg"
    new_image_path = os.path.join(AUGMENTED_IMAGE_DIR, new_image_name)
    img.save(new_image_path)

    return new_image_path

# Separate classes
df_nothate = df[df["binary_label"] == 0]
df_hate = df[df["binary_label"] == 1]

df_nothate_sampled = df_nothate.sample(n=75000, random_state=42)

# augmenters
def synonym_replacement(text, n=2):
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_words = set()
            for syn in synonyms:
                for lemma in syn.lemmas():
                    synonym_words.add(lemma.name().replace("_", " "))
            synonym_words.discard(word)
            if synonym_words:
                new_words = [w if w != word else random.choice(list(synonym_words)) for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
    return ' '.join(new_words)

def random_deletion(text, p=0.1):
    words = nltk.word_tokenize(text)
    if len(words) == 1:
        return text
    new_words = [word for word in words if random.uniform(0, 1) > p]
    if len(new_words) == 0:
        return random.choice(words)  # if all words deleted, return one
    return ' '.join(new_words)

def random_swap(text, n=2):
    words = nltk.word_tokenize(text)
    length = len(words)
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return ' '.join(words)

def augment_text(text, num_augments=2):
    methods = [synonym_replacement, random_deletion, random_swap]
    augmented = []

    for _ in range(num_augments):
        method = random.choice(methods)
        try:
            aug_text = method(text)
            augmented.append(aug_text)
        except:
            augmented.append(text)  # fallback to original

    return augmented



# generate
augmented_data = []
for _, row in df_hate.iterrows():
    augmented_texts = augment_text(row["cleaned_text"])
    
    for aug_text in augmented_texts:
        new_image_path = augment_image(row["image_path"])
        augmented_data.append((new_image_path, row["binary_label"], row["binary_label_str"], aug_text))

# to DataFrame
df_hate_augmented = pd.DataFrame(augmented_data, columns=["image_path", "binary_label", "binary_label_str", "cleaned_text"])

required_augmented_samples = 75000 - len(df_hate)
df_hate_augmented = df_hate_augmented.sample(n=min(len(df_hate_augmented), required_augmented_samples), random_state=42)

df_hate_balanced = pd.concat([df_hate, df_hate_augmented])

df_balanced = pd.concat([df_nothate_sampled, df_hate_balanced])

df_balanced.to_csv("balanced_dataset.csv", index=False)

logging.info(df_balanced["binary_label"].value_counts())  # should be 75K for both classes
