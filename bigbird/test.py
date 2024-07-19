import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BigBirdModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
import json
import os
import re
import csv

print("loading csv...")
df = pd.read_csv('/data/test_data.csv')
df = df.sample(n=500, random_state=42)

go_terms = pd.read_csv('/data/GOIDs.csv')

print("converting embeddings to list...")
df['RoundedEmbedding'] = df['RoundedEmbedding'].apply(json.loads)

def truncate_embeddings(embeddings, target_length):
        truncated_embeddings = [emb[:target_length] for emb in embeddings]
        return truncated_embeddings

target_length = 1280
print("truncating embeddings to target length...")
df['TruncatedEmbedding'] = truncate_embeddings(df['RoundedEmbedding'].tolist(), target_length)


def parse_go_terms(go_terms_str):
    clean_str = go_terms_str.strip('[]').strip()
    go_terms = np.array([x for x in re.split(r'\s+', clean_str.replace(',', '').replace('\'', ''))])
    return go_terms

def parse_go_terms_regular(go_terms_str):
    go_terms = go_terms_str.split(',')
    categories = ['CC', 'MF', 'BP']
    filtered_list = [item for item in go_terms if item not in categories]
    return filtered_list

print('parsing GO terms...')
df['GOTerms'] = df['GOTerms'].apply(parse_go_terms)

print('getting unique GO terms...')
all_go_terms = set()
with open('/data/GOIDs.csv', mode = 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        go_id = row[0]
        all_go_terms.add(go_id)

all_go_terms = sorted(list(all_go_terms))

train_go_terms = df['GOTerms'].tolist()

print("converting labels...")
mlb = MultiLabelBinarizer(classes=all_go_terms)
labels = mlb.fit_transform(train_go_terms)
labels = torch.tensor(labels, dtype=torch.float32)

num_classes = len(mlb.classes_)

print("converting embeddings...")
embeddings = torch.tensor(np.stack(df['TruncatedEmbedding'])).float()

print("creating dataset...")
dataset = TensorDataset(embeddings, labels)

class CustomBigBirdModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomBigBirdModel, self).__init__()
        self.embedding_proj = nn.Linear(1280, 768)
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base', ignore_mismatched_sizes=True)
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, embeddings):
        embeddings = self.embedding_proj(embeddings)
        embeddings = embeddings.unsqueeze(1)
        outputs = self.bigbird(inputs_embeds=embeddings)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


save_dir = '/model/'
model_name = 'model_6.pth'
model_path = os.path.join(save_dir, model_name)

print('Initializing model...')
model = CustomBigBirdModel(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

all_predictions = []
all_labels = []

print('Evaluating model...')
with torch.no_grad():
    for batch in test_dataloader:
        batch_embeddings, batch_labels = batch
        batch_embeddings, batch_labels = batch_embeddings.to(device), batch_labels.to(device)
        outputs = model(batch_embeddings)
        predictions = torch.sigmoid(outputs)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch_labels.cpu())

all_predictions = torch.cat(all_predictions)
all_labels = torch.cat(all_labels)

threshold = 0.69
predicted_labels = (all_predictions > threshold).float()

accuracy_per_label = (predicted_labels == all_labels).float().mean(dim=0)
print(f'Accuracy per label: {accuracy_per_label}')

final_output = []

final_output.append("AUTHOR\tAlphaAnalyzers")
final_output.append("MODEL\tProteinext")
final_output.append("KEYWORDS\tde novo prediction, machine learning.")

for idx, prot_id in enumerate(df['ProteinID']):
    for go_term_idx, go_term in enumerate(mlb.classes_):
        if predicted_labels[idx, go_term_idx] == 1:
            label_accuracy = round(accuracy_per_label[go_term_idx].item(), 2)

            final_output.append(f"{prot_id}\t{go_term}\t{label_accuracy}")

print('saving predictions...')
with open('/result/model6_predicted.csv', 'w') as f:
    for line in final_output:
        f.write(f"{line}\n")

final_output_actual = []

final_output_actual.append("AUTHOR\tAlphaAnalyzers")
final_output_actual.append("MODEL\tProteinext")
final_output_actual.append("KEYWORDS\tde novo prediction, machine learning.")

for idx, prot_id in enumerate(df['ProteinID']):
    for go_term_idx, go_term in enumerate(mlb.classes_):
        if all_labels[idx, go_term_idx] == 1:
            final_output_actual.append(f"{prot_id}\t{go_term}")

print('saving actual data...')
with open('/result/model6_actual.csv', 'w') as f:
    for line in final_output_actual:
        f.write(f"{line}\n")

