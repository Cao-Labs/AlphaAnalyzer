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
from sklearn.metrics import precision_score, recall_score, f1_score

print("loading csv...")
# Load the CSV data
df = pd.read_csv('../scripts/test_data.csv')
#df = pd.read_csv('../data/test_data.csv')
#df = pd.read_csv('../cafa_test.csv')
df = df.sample(n=500, random_state=42)

go_terms = pd.read_csv('../../data/GOIDs.csv')

print("converting embeddings to list...")
# Convert embeddings from string to list
df['RoundedEmbedding'] = df['RoundedEmbedding'].apply(json.loads)


def truncate_embeddings(embeddings, target_length):
        truncated_embeddings = [emb[:target_length] for emb in embeddings]
        return truncated_embeddings

# Determine the target length for the embeddings (1280 as per the error message)
target_length = 1280

# Truncate the embeddings
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
go_term_mapping = { 'GO:0000986':'GO:0000987', 'GO:0001105':'GO:0003713', 'GO:0001131':'GO:0003700', 'GO:0001948':'GO:0005515', 'GO:0002740':'GO:0002719', 'GO:0004003':'GO:0003678', 'GO:0004004':'GO:0003724', 'GO:0005887':'GO:0005886', 'GO:0005913':'GO:0005912', 'GO:0006461':'GO:0065003', 'GO:0006464':'GO:0036211',
        'GO:0007050':'GO:0051726', 'GO:0007067':'GO:0000278', 'GO:0009870':'GO:0002758', 'GO:0015197':'GO:1904680', 'GO:0015238':'GO:0042910', 'GO:0015758':'GO:1904659', 'GO:0016021':'GO:0016020', 'GO:0016023':'GO:0031410', 'GO:0016534':'GO:0061575', 'GO:0017137':'GO:0031267',
        'GO:0019012':'GO:0044423', 'GO:0020012':'GO:0042783', 'GO:0022891':'GO:0022857', 'GO:0030529':'GO:1990904', 'GO:0030898':'GO:0000146', 'GO:0031988':'GO:0031982', 'GO:0032270':'GO:0051247', 'GO:0032403':'GO:0044877', 'GO:0032947':'GO:0060090', 'GO:0033317':'GO:0015940', 'GO:0033680':'GO:0033677','GO:0034613':'GO:0008104', 'GO:0035690':'GO:0071466', 'GO:0042493':'GO:0009410', 
        'GO:0042517':'GO:0042531', 'GO:0042623':'GO:0016887', 'GO:0043234':'GO:0032991', 'GO:0044117':'GO:0051701', 'GO:0044119':'GO:0051701', 'GO:0044121':'GO:0051701', 'GO:0044212':'GO:0000976', 'GO:0044267':'GO:0019538', 'GO:0044662':'GO:0051673', 'GO:0044822':'GO:0003723', 'GO:0045416':'GO:0032757', 'GO:0050690':'GO:0019049', 'GO:0050715':'GO:0001819', 'GO:0051023':'GO:0002637', 'GO:0052059':'GO:0052164', 'GO:0052060':'GO:0052163', 'GO:0052556':'GO:0052559', 'GO:0061489':'GO:0098710', 'GO:0070627':'GO:0033212', 'GO:0072643':'GO:0032609', 'GO:0090004':'GO:1903078', 'GO:1903991':'GO:1904440', 'GO:1904469':'GO:0032760', 'GO:2000778':'GO:0032755' }
def fix_alt_IDs(go_term_list):
    updated_go_terms = [go_term_mapping.get(term, term) for term in go_term_list]
    return updated_go_terms

#print('parsing GO terms')
df['GOTerms'] = df['GOTerms'].apply(parse_go_terms)
df['GOTerms'] = df['GOTerms'].apply(fix_alt_IDs)

print('getting unique GO terms')
all_go_terms = set()
with open('../../data/GOIDs.csv', mode = 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        go_id = row[0]
        all_go_terms.add(go_id)

all_go_terms = sorted(list(all_go_terms))
print('length of all go terms')
print(len(all_go_terms))

train_go_terms = df['GOTerms'].tolist()
#print('len of go terms')
#print(len(train_go_terms))

print("converting labels...")
mlb = MultiLabelBinarizer(classes=all_go_terms)
labels = mlb.fit_transform(train_go_terms)
labels = torch.tensor(labels, dtype=torch.float32)

# Compute the number of classes
num_classes = len(mlb.classes_)
#print("numclasses" + str(num_classes))

print("converting embeddings...")
# Convert embeddings to tensor format
embeddings = torch.tensor(np.stack(df['TruncatedEmbedding'])).float()


print("creating dataset...")
# Create a TensorDataset
dataset = TensorDataset(embeddings, labels)

# Compute the number of classes
num_classes = len(mlb.classes_)

class CustomBigBirdModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomBigBirdModel, self).__init__()
        self.embedding_proj = nn.Linear(1280, 768)  # Adjust input embedding dimension if necessary
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base', ignore_mismatched_sizes=True)
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, embeddings):
        embeddings = self.embedding_proj(embeddings)
        embeddings = embeddings.unsqueeze(1)  # add sequence_length dimension
        outputs = self.bigbird(inputs_embeds=embeddings)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# Load the saved model
save_dir = '../saved_models/'
model_name = 'model_6.pth'
model_path = os.path.join(save_dir, model_name)

print('Initializing model...')
model = CustomBigBirdModel(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create DataLoader for test data
test_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Make predictions and evaluate
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

# Example evaluation: using a threshold for binary classification
threshold = 0.3
predicted_labels = (all_predictions > threshold).float()

print(predicted_labels.shape)  # Should match the shape of all_labels
print(predicted_labels[:10]) 

accuracy_per_label = (predicted_labels == all_labels).float().mean(dim=0)
print(f'Accuracy per label: {accuracy_per_label}')

# Create an empty list to store the final output
final_output = []

# Headers for the output file
final_output.append("AUTHOR\tCaoLab3")
final_output.append("MODEL\t3")
final_output.append("KEYWORDS\tde novo prediction, machine learning.")

# Iterate over each protein ID and its associated predictions
for idx, prot_id in enumerate(df['ProteinID']):
    for go_term_idx, go_term in enumerate(mlb.classes_):
        if predicted_labels[idx, go_term_idx] == 1:
            # Calculate accuracy for the current label
            label_accuracy = round(accuracy_per_label[go_term_idx].item(), 2)

            # Append the formatted string to the final output
            final_output.append(f"{prot_id}\t{go_term}\t{label_accuracy}")

print('Saving predictions...')
# Save final_output to a file
with open('../result/model6/model6_predicted2.csv', 'w') as f:
    for line in final_output:
        f.write(f"{line}\n")


# Create an empty list to store the final output for actual data
final_output_actual = []

# Headers for the actual data file
final_output_actual.append("AUTHOR\tCaoLab3")
final_output_actual.append("MODEL\t3")
final_output_actual.append("KEYWORDS\tde novo prediction, machine learning.")

# Iterate over each protein ID and its associated actual labels
for idx, prot_id in enumerate(df['ProteinID']):
    for go_term_idx, go_term in enumerate(mlb.classes_):
        if all_labels[idx, go_term_idx] == 1:
            # Append the formatted string to the final output
            final_output_actual.append(f"{prot_id}\t{go_term}")

print('Saving actual data...')
# Save final_output_actual to a file
with open('../result/model6/model6_actual2.csv', 'w') as f:
    for line in final_output_actual:
        f.write(f"{line}\n")

