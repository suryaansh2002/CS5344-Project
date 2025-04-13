import pandas as pd
import numpy as np
import re
import string
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score

import matplotlib.pyplot as plt
import re
import joblib

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

#######################################
# 1. Define Custom Preprocessing Tools
#######################################

# Text cleaning function
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom transformer for VADER sentiment scores
class SentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            scores = self.analyzer.polarity_scores(text)
            # Order: negative, neutral, positive, compound
            features.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
        return np.array(features)

# 2. Build the Feature Extraction Pipeline
#######################################

# Create a pipeline for sentiment features that scales them to non-negative values.
sentiment_pipeline = Pipeline([
    ('sentiment', SentimentTransformer()),
    ('scaler', MinMaxScaler())  # This scales each feature to the range [0, 1]
])

# The FeatureUnion combines TF-IDF features with sentiment scores.
text_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(preprocessor=clean_text, stop_words='english', max_features=5000)),
    ('sentiment', sentiment_pipeline)
])

#######################################
# 3. Load and Split the Dataset
#######################################

# Read the dataset and select relevant columns
df = pd.read_csv('DATA/balanced_dataset.csv')
df = df[['cleaned_text', 'binary_label']]

# Separate features and target
X = df['cleaned_text']
y = df['binary_label']

# Split into train (70%), validation (15%), and test (15%) sets (stratified for balance)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Fit the feature union on the training data and transform all splits
X_train_transformed = text_features.fit_transform(X_train)
X_val_transformed   = text_features.transform(X_val)
X_test_transformed  = text_features.transform(X_test)

#######################################
# 4. Define a Custom F1 Score Function
#######################################

def f1_score_custom(y_true, y_pred):
    # Compute true positives, false positives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    # Avoid division by zero
    if tp + fp == 0 or tp + fn == 0:
        return 0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

#######################################
# 5. Setup Logging
#######################################

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger

logger = setup_logger('training_logger', 'logs/training_log_text_only.txt')



# List of classes (needed for partial_fit)
classes = np.unique(y_train)

### 6.1 Model 1: SGDClassifier (Logistic Regression via SGD)
logger.info("Starting training for SGDClassifier (Logistic Regression with SGD)...")
model1 = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=42)
epochs = 12

for epoch in range(epochs):
    # Shuffle the training data each epoch
    indices = np.arange(X_train_transformed.shape[0])
    np.random.shuffle(indices)
    X_train_epoch = X_train_transformed[indices]
    y_train_epoch = y_train.values[indices]
    
    # Use partial_fit to simulate epoch-based training
    if epoch == 0:
        model1.partial_fit(X_train_epoch, y_train_epoch, classes=classes)
    else:
        model1.partial_fit(X_train_epoch, y_train_epoch)
    
    # Calculate training metrics
    y_train_pred = model1.predict(X_train_transformed)
    train_acc = accuracy_score(y_train, y_train_pred)
    try:
        y_train_proba = model1.predict_proba(X_train_transformed)
        train_loss = log_loss(y_train, y_train_proba)
    except Exception:
        train_loss = float('nan')
    train_f1 = f1_score_custom(y_train.values, y_train_pred)
    
    # Calculate validation metrics
    y_val_pred = model1.predict(X_val_transformed)
    val_acc = accuracy_score(y_val, y_val_pred)
    try:
        y_val_proba = model1.predict_proba(X_val_transformed)
        val_loss = log_loss(y_val, y_val_proba)
    except Exception:
        val_loss = float('nan')
    val_f1 = f1_score_custom(y_val.values, y_val_pred)
    
    logger.info(f"SGDClassifier Epoch {epoch+1}: "
                f"Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | "
                f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")



#######################################
# 7. Testing and Final Results
#######################################

print("\n----- FINAL TEST RESULTS -----\n")

# Testing SGDClassifier
y_test_pred_sgd = model1.predict(X_test_transformed)
test_acc_sgd = accuracy_score(y_test, y_test_pred_sgd)
try:
    y_test_proba_sgd = model1.predict_proba(X_test_transformed)
    test_loss_sgd = log_loss(y_test, y_test_proba_sgd)
except Exception:
    test_loss_sgd = float('nan')
test_f1_sgd = f1_score_custom(y_test.values, y_test_pred_sgd)
print("SGDClassifier:")
print(f"  Accuracy: {test_acc_sgd:.4f}")
print(f"  Log Loss: {test_loss_sgd:.4f}")
print(f"  F1 Score: {test_f1_sgd:.4f}\n")

model_filename = 'checkpoints/sgd_classifier_model.pt'
joblib.dump(model1, model_filename)

# Replace with your actual filename
log_file = "logs/training_log_text_only.txt"
# Replace with your desired output file path
output_file = "plots/training_validation_metrics_plot_text_only.png"

# Initialize metric containers
epochs = []
train_acc = []
val_acc = []
train_f1 = []
val_f1 = []
train_precision = []
val_precision = []
train_recall = []
val_recall = []

# Define regex to extract all needed metrics (including optional precision/recall)
pattern = re.compile(
    r"Epoch (\d+): Train Acc: ([\d.]+), Train Loss: [\d.]+, Train F1: ([\d.]+)"
    r"(?:, Train Precision: ([\d.]+), Train Recall: ([\d.]+))? \| "
    r"Val Acc: ([\d.]+), Val Loss: [\d.]+, Val F1: ([\d.]+)"
    r"(?:, Val Precision: ([\d.]+), Val Recall: ([\d.]+))?"
)

# Read and extract values
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            (
                epoch, tr_acc, tr_f1, tr_prec, tr_rec,
                va_acc, va_f1, va_prec, va_rec
            ) = match.groups()
            epochs.append(int(epoch))
            train_acc.append(float(tr_acc))
            train_f1.append(float(tr_f1))
            val_acc.append(float(va_acc))
            val_f1.append(float(va_f1))

            # Handle optional precision/recall
            train_precision.append(float(tr_prec) if tr_prec else None)
            train_recall.append(float(tr_rec) if tr_rec else None)
            val_precision.append(float(va_prec) if va_prec else None)
            val_recall.append(float(va_rec) if va_rec else None)

# Plotting
plt.figure(figsize=(14, 7))

# Accuracy
plt.plot(epochs, train_acc, label='train_accuracy', linestyle='--', marker='o')
plt.plot(epochs, val_acc, label='val_accuracy', linestyle='--', marker='*')

# F1
plt.plot(epochs, train_f1, label='train_f1', linestyle='--', marker='o')
plt.plot(epochs, val_f1, label='val_f1', linestyle='--', marker='*')

# Precision (if present)
if all(x is not None for x in train_precision):
    plt.plot(epochs, train_precision, label='train_precision', linestyle='--', marker='o')
if all(x is not None for x in val_precision):
    plt.plot(epochs, val_precision, label='val_precision', linestyle='--', marker='*')

# Recall (if present)
if all(x is not None for x in train_recall):
    plt.plot(epochs, train_recall, label='train_recall', linestyle='--', marker='o')
if all(x is not None for x in val_recall):
    plt.plot(epochs, val_recall, label='val_recall', linestyle='--', marker='*')

# Labels and title
plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Training and Validation Metrics (SGD Classifier using only Text)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to the specified file
plt.savefig(output_file)

# Display the plot
plt.show()

