import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BigBirdModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os
from tqdm import tqdm
import re
import csv

print("loading csv...")
df = pd.read_csv('/data/train_data.csv')
df = df.sample(n=350000, random_state=42)
go_terms = pd.read_csv('/data/GOIDs.csv')

print("converting embeddings to list...")
df['RoundedEmbedding'] = df['RoundedEmbedding'].apply(json.loads)

def parse_go_terms(go_terms_str):
    clean_str = go_terms_str.strip('[]').strip()
    go_terms = np.array([x for x in re.split(r'\s+', clean_str.replace(',', '').replace('\'', ''))])
    return go_terms

print('parsing GO terms')
df['GOTerms'] = df['GOTerms'].apply(parse_go_terms)

print('getting unique GO terms')
all_go_terms = set()
with open('../../data/GOIDs.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        go_id = row[0]
        all_go_terms.add(go_id)

all_go_terms = sorted(list(all_go_terms))
print('length of all go terms:', len(all_go_terms))

train_go_terms = df['GOTerms'].tolist()

print("converting labels...")
mlb = MultiLabelBinarizer(classes=all_go_terms)
labels = mlb.fit_transform(train_go_terms)
labels = torch.tensor(labels, dtype=torch.float32)

num_classes = len(mlb.classes_)

print("converting embeddings...")
embeddings = torch.tensor(np.stack(df['RoundedEmbedding'])).float()

print("creating dataset...")
dataset = TensorDataset(embeddings, labels)

class CustomBigBirdModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomBigBirdModel, self).__init__()
        self.embedding_proj = nn.Linear(1280, 768)  # Adjust input embedding dimension if necessary
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base')
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, embeddings):
        embeddings = self.embedding_proj(embeddings)
        embeddings = embeddings.unsqueeze(1)  # add sequence_length dimension
        outputs = self.bigbird(inputs_embeds=embeddings)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

def train_model(model, train_dataloader, optimizer, criterion, scheduler, num_epochs, device, save_dir):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_dataloader = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in train_dataloader:
            batch_embeddings, batch_labels = batch
            batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_dataloader.set_postfix(loss=loss.item())
        
        # Update learning rate
        scheduler.step(epoch_loss)
        print(f'Epoch {epoch+1} loss: {epoch_loss}')
        
        # Save the model after each epoch
        model_path = os.path.join(save_dir, f'model_8_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path} after epoch {epoch+1}')

    return model

# Hyperparameters for different models
hyperparameters = [
    {'learning_rate': 1e-5, 'batch_size': 8, 'num_epochs': 5}
]

models = []
save_dir = '/model/'
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

print('Training models with different hyperparameters...')
for i, params in enumerate(hyperparameters):
    model = CustomBigBirdModel(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor=0.5,
        patience=2,
        verbose=True,
        min_lr=1e-7
    )
    
    criterion = nn.BCEWithLogitsLoss()
    train_dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    print(f'Training model {i+1} with learning rate {params["learning_rate"]} and batch size {params["batch_size"]}')
    trained_model = train_model(model, train_dataloader, optimizer, criterion, scheduler, params['num_epochs'], device, save_dir)
    
    models.append(trained_model)

print('All models trained and saved.')

