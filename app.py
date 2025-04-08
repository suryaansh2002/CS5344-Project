import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Text preprocessing
def text_pipeline(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")["input_ids"].squeeze(0)

# Multimodal model
class MultiModalModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(MultiModalModel, self).__init__()
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.img_fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.embedding = nn.Embedding(tokenizer.vocab_size, 128, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, img, text):
        img_feat = self.img_fc(self.cnn(img))
        text_emb = self.embedding(text)
        _, (hidden, _) = self.lstm(text_emb)
        text_feat = self.text_fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined).squeeze(1)
    
class TextModel():
    def init(self):
        return 0

# Streamlit UI
st.title("Social Media Hate Speech Detector")
st.write("Enter some text and upload an image to classify.")

text_input = st.text_area("Enter Text:", "")
image_input = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if st.button("Classify"):
    if text_input and image_input:
        try:
            # Load model
            multi_model = MultiModalModel()
            multi_model.load_state_dict(torch.load("best_model.pth", map_location = device))
            multi_model.to(device)
            multi_model.eval()

            # Preprocess
            image = Image.open(image_input).convert("RGB")
            image = image_transform(image).unsqueeze(0).to(device)
            text = text_pipeline(text_input).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = multi_model(image, text)
                prob = torch.sigmoid(output).item()
                label = "Hate Speech Detected" if prob > 0.5 else "No Hate Speech"

            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"Confidence: `{prob:.2f}`")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    elif text_input:
        try:
            # Load model
            text_model = TextModel()
            text_model.load_state_dict(torch.load("best_model.pth", map_location = device))
            text_model.to(device)
            text_model.eval()

            # Preprocess
            text = text_pipeline(text_input).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = text_model(text)
                prob = torch.sigmoid(output).item()
                label = "Hate Speech Detected" if prob > 0.5 else "No Hate Speech"

            st.markdown(f"### Prediction: **{label}**")
            st.markdown(f"Confidence: `{prob:.2f}`")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please provide input")
