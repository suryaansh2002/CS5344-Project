import pandas as pd
import logging
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Embedding, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0

# --- CONFIGURATION ---
# Set logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Script started!")

# --- LOAD AND PREPROCESS DATA ---
# Load the dataset
df = pd.read_csv("balanced_dataset.csv")
logging.info("Loaded dataset:")
logging.info(df.info())

# Log label distribution and a sample of the cleaned text
logging.info("Label distribution:")
logging.info(df['binary_label_str'].value_counts())
logging.info("Sample cleaned text:")
logging.info(df['cleaned_text'].head())

# --- DATA SPLIT ---
# Split data into train, validation, and test sets
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the sizes of train, validation, and test sets
logging.info(f"Training data size: {train_data.shape}")
logging.info(f"Validation data size: {val_data.shape}")
logging.info(f"Testing data size: {test_data.shape}")

# --- IMAGE PREPROCESSING FUNCTION ---
def load_and_preprocess_image(img_path, target_size=(96, 96)):
    """Loads and preprocesses images."""
    try:
        if not os.path.exists(img_path):
            return np.zeros((target_size[0], target_size[1], 3))
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img) / 255.0
        return img
    except Exception as e:
        logging.error(f"Error loading image {img_path}: {e}")
        return np.zeros((target_size[0], target_size[1], 3))

# --- TEXT PREPROCESSING ---
# Initialize tokenizer
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['cleaned_text'])

# Convert text to padded sequences
X_train_text = pad_sequences(tokenizer.texts_to_sequences(train_data['cleaned_text']), maxlen=100)
X_val_text = pad_sequences(tokenizer.texts_to_sequences(val_data['cleaned_text']), maxlen=100)
X_test_text = pad_sequences(tokenizer.texts_to_sequences(test_data['cleaned_text']), maxlen=100)

# --- IMAGE PREPROCESSING ---
# Preprocess images for each dataset split
X_train_image = np.array([load_and_preprocess_image(path) for path in train_data['image_path']])
X_val_image = np.array([load_and_preprocess_image(path) for path in val_data['image_path']])
X_test_image = np.array([load_and_preprocess_image(path) for path in test_data['image_path']])

# --- LABELS ---
# Convert labels to numpy arrays
y_train = np.array(train_data['binary_label'])
y_val = np.array(val_data['binary_label'])
y_test = np.array(test_data['binary_label'])

# --- MODEL PARAMETERS ---
input_shape = (96, 96, 3)

# --- IMAGE MODEL (EfficientNetB0) ---
# Load EfficientNetB0 base model with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Define image model input and top layers
image_input = Input(shape=input_shape)
x_image = base_model(image_input, training=False)
x_image = GlobalAveragePooling2D()(x_image)
x_image = Dense(256, activation='relu')(x_image)
x_image = Dropout(0.5)(x_image)


# Explanation of x_image and x_text:

# x_image and x_text are intermediate layers within the model, not the entire models themselves.

# 1. Image Model (x_image):
#    The x_image variable is the result of applying the EfficientNetB0 base model to the image input (image_input).
#    The EfficientNetB0 model is a pre-trained image model that extracts feature representations from the input images.
#    After passing through EfficientNetB0, you apply:
#       - Global Average Pooling (GlobalAveragePooling2D) to reduce the spatial dimensions.
#       - A Dense layer with ReLU activation, followed by a Dropout layer to prevent overfitting.
#    So, x_image is the final feature representation for the image input after these transformations.
#
# 2. Text Model (x_text):
#    Similarly, x_text represents the result of processing the text input (text_input) through several layers:
#       - Embedding layer to convert words into word embeddings.
#       - LSTM layer to capture the sequence patterns and context in the text.
#       - Dense layer with ReLU activation, followed by a Dropout layer for regularization.
#    So, x_text is the final feature representation for the text input after these transformations.


# --- TEXT MODEL (LSTM) ---
# Define text model input and layers
text_input = Input(shape=(100,))
x_text = Embedding(input_dim=20000, output_dim=128, input_length=100)(text_input)
x_text = LSTM(128, return_sequences=False)(x_text)
x_text = Dense(128, activation='relu')(x_text)
x_text = Dropout(0.5)(x_text)

# --- COMBINE IMAGE AND TEXT MODELS ---
# Concatenate image and text features
combined = Concatenate()([x_image, x_text])

# This combines both image and text features into a single vector representation.
# This combined representation is then passed through additional layers
# (Dense, Dropout, etc.) to produce the final output.

x_combined = Dense(128, activation='relu')(combined)
x_combined = Dropout(0.5)(x_combined)

# Output layer (classification task)
output = Dense(1, activation='sigmoid')(x_combined)  # For binary classification

# --- BUILD AND COMPILE MODEL ---
multimodal_model = Model(inputs=[image_input, text_input], outputs=output)
multimodal_model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
logging.info("Model Summary:")
multimodal_model.summary()

# --- TRAIN THE MODEL ---
history = multimodal_model.fit(
    [X_train_image, X_train_text], y_train,
    validation_data=([X_val_image, X_val_text], y_val),
    epochs=10,
    batch_size=128,
    verbose=1
)

# --- SAVE THE MODEL ---
multimodal_model.save('multimodal_model_updated.h5') 

# --- EVALUATE THE MODEL ---
loss, accuracy = multimodal_model.evaluate([X_test_image, X_test_text], y_test, verbose=1)
logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
