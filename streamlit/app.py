import re
import os
import sys
import torch
import string
import joblib
import logging
import numpy as np
from PIL import Image
import streamlit as st
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientnet_bilstm import MultiModalModel

# ------------------ Config ------------------
IMAGE_SIZE = 224
LOAD_PATH_MULTI = "checkpoints/bilstm_efficientnet/best_model.pt"
LOAD_PATH_TEXT = "checkpoints/text_only/sgd_classifier_model.pkl"
LOG_DIR = "logs/streamlit_app/"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting script...")
logging.info(f"Using device: {DEVICE}")

# ------------------ TOKENIZER AND TRANSFORMS ------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def text_pipeline(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

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
    

sentiment_pipeline = Pipeline([
    ('sentiment', SentimentTransformer()),
    ('scaler', MinMaxScaler())  # This scales each feature to the range [0, 1]
])

text_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(preprocessor=clean_text, stop_words='english', max_features=5000)),
    ('sentiment', sentiment_pipeline)
])

logging.info("Tokenizer and image transform initialized.")

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Hate Speech Classifier", layout="centered", page_icon="🚫")
st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 1rem;
        }
            
        div[data-testid="stForm"] {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

with st.form("classification_form"):
    st.markdown("""
        <h1 style='text-align: center; padding-top: 1px; padding-bottom: 1px;'>Social Media Hate Detector</h1>
        <p style='text-align: center; font-size: 18px;'>
            Detect hate speech from a social media post using <strong>text</strong> and an <strong>associated image</strong>.
        </p>
    """, unsafe_allow_html=True)

    st.subheader("Enter Text")
    text_input = st.text_area("Text (e.g., tweet):", placeholder="Type your message here...", height=100)

    st.subheader("Upload Image")
    image_input = st.file_uploader("Choose an image (optional)", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("Classify")

if submitted:
    if text_input and image_input:
        try:
            with st.spinner("Classifying... please wait"):
                # Load model
                multi_model = MultiModalModel(tokenizer = tokenizer)
                multi_model.load_state_dict(torch.load(LOAD_PATH_MULTI, map_location = DEVICE))
                multi_model.to(DEVICE)
                multi_model.eval()

                # Preprocess
                image = Image.open(image_input).convert("RGB")
                image = image_transform(image).unsqueeze(0).to(DEVICE)
                text_dict = text_pipeline(text_input)
                text_input_ids = text_dict["input_ids"].to(DEVICE)

                # Predict
                with torch.no_grad():
                    output = multi_model(image, text_input_ids)
                    prob = torch.sigmoid(output).item()
                    label = "Hate Speech Detected" if prob > 0.5 else "No Hate Speech"
                    confidence_color = "red" if prob > 0.5 else "green"

                st.success("Classification complete.")
                st.markdown(f"### Prediction Result: **:{confidence_color}[{label}]**")
                st.markdown(f"**Confidence Score:** :{confidence_color}[{prob:.2f}]")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            logging.info(f"Error during text+image prediction: {str(e)}")
    elif text_input:
        try:
            # Load model
            text_model = joblib.load(LOAD_PATH_TEXT)

            # Predict
            with torch.no_grad():
                output = text_model.predict_proba([text_input])
                prob = output[0][1]
                label = "Hate Speech Detected" if prob > 0.5 else "No Hate Speech"
                confidence_color = "red" if prob > 0.5 else "green"

            st.success("Classification complete.")
            st.markdown(f"### Prediction Result: **:{confidence_color}[{label}]**")
            st.markdown(f"**Confidence Score:** :{confidence_color}[{prob:.2f}]")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            logging.info(f"Error during text prediction: {str(e)}")
    else:
        st.warning("Please provide input")
# Set initial state

st.markdown("""---""")
st.caption("Hate Detection · CS5344 · Team 17")