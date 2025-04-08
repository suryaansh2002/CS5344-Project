import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class MultiModalDataset(Dataset):
    def __init__(self, dataframe, text_pipeline, image_transform, image_size=224, tokenizer=None):
        self.dataframe = dataframe
        self.text_pipeline = text_pipeline
        self.image_transform = image_transform
        self.image_size = image_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.image_transform(image)
        except:
            image = torch.zeros(3, *self.image_transform.transforms[0].size)  # e.g., (3, 224, 224) or (3, 96, 96)

        text_tensor = self.tokenizer(row['cleaned_text'], padding="max_length", truncation=True,
                                     max_length=100, return_tensors="pt")["input_ids"].squeeze(0)

        label = torch.tensor(row['binary_label'], dtype=torch.float32)
        return image, text_tensor, label

def collate_fn(batch, pad_token_id):
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_token_id)
    return images, texts_padded, labels
