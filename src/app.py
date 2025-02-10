from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BigBirdModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
import json
import re
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # More secure CORS policy

# Define Custom BigBird Model
class CustomBigBirdModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomBigBirdModel, self).__init__()
        self.embedding_proj = nn.Linear(1280, 768)
        self.bigbird = BigBirdModel.from_pretrained('google/bigbird-roberta-base', ignore_mismatched_sizes=True)
        self.classifier = nn.Linear(self.bigbird.config.hidden_size, num_classes)

    def forward(self, embeddings):
        embeddings = self.embedding_proj(embeddings).unsqueeze(1)
        outputs = self.bigbird(inputs_embeds=embeddings)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

@app.route('/predict', methods=['POST'])
def predict():
    print("\n=== Request received ===")
    print("Request files:", request.files)

    if 'file' not in request.files:
        print("ERROR: No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    print("File received:", file.filename)

    # Read CSV file
    try:
        df = pd.read_csv(file)
        print("CSV read successfully. Columns:", df.columns.tolist())
    except Exception as e:
        print("ERROR reading CSV file:", str(e))
        return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400

    # Ensure required columns exist
    required_columns = {'RoundedEmbedding', 'GOTerms', 'ProteinID'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400

    # Load GO terms
    try:
        go_terms = pd.read_csv('/Users/boen_liu/CaoProject/bigbird/data/GOIDs.csv')
        all_go_terms = sorted(set(go_terms.iloc[:, 0]))
        print(f"Loaded {len(all_go_terms)} GO terms.")
    except Exception as e:
        print("ERROR loading GO terms:", str(e))
        return jsonify({'error': f'Error loading GO terms: {str(e)}'}), 500

    # Process embeddings
    try:
        df['RoundedEmbedding'] = df['RoundedEmbedding'].apply(json.loads)
        df['TruncatedEmbedding'] = df['RoundedEmbedding'].apply(lambda emb: emb[:1280])
        print("Embeddings processed successfully.")
    except Exception as e:
        print("ERROR processing embeddings:", str(e))
        return jsonify({'error': f'Error processing embeddings: {str(e)}'}), 400

    # Parse GO terms
    try:
        df['GOTerms'] = df['GOTerms'].apply(lambda x: np.array(re.split(r'\s+', x.strip('[]').replace(',', '').replace('\'', ''))))
        print("GO terms parsed successfully.")
    except Exception as e:
        print("ERROR parsing GO terms:", str(e))
        return jsonify({'error': f'Error parsing GO terms: {str(e)}'}), 400

    # Prepare dataset
    try:
        mlb = MultiLabelBinarizer(classes=all_go_terms)
        labels = torch.tensor(mlb.fit_transform(df['GOTerms'].tolist()), dtype=torch.float32)
        embeddings = torch.tensor(np.stack(df['TruncatedEmbedding'])).float()
        dataset = TensorDataset(embeddings, labels)
        print("Dataset prepared successfully.")
    except Exception as e:
        print("ERROR preparing dataset:", str(e))
        return jsonify({'error': f'Error preparing dataset: {str(e)}'}), 400

    # Load model
    model = CustomBigBirdModel(len(mlb.classes_))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model.load_state_dict(torch.load('model_6.pth', map_location=torch.device('cpu')))
        print("Model loaded successfully.")
    except Exception as e:
        print("ERROR loading model:", str(e))
        return jsonify({'error': f'Error loading model: {str(e)}'}), 500

    model.to(device)
    model.eval()

    test_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Make predictions
    all_predictions = []
    try:
        with torch.no_grad():
            for batch_embeddings, _ in test_dataloader:
                batch_embeddings = batch_embeddings.to(device)
                outputs = model(batch_embeddings)
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions.cpu())

        all_predictions = torch.cat(all_predictions)
        predicted_labels = (all_predictions > 0.69).float()
        print("Predictions computed successfully.")
    except Exception as e:
        print("ERROR during prediction:", str(e))
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

    # Format results
    try:
        results = [
            {"ProteinID": df['ProteinID'][idx], "GO_Term": go_term}
            for idx in range(len(df))
            for go_term_idx, go_term in enumerate(mlb.classes_)
            if predicted_labels[idx, go_term_idx] == 1
        ]
        print(f"Results formatted successfully. {len(results)} predictions made.")
    except Exception as e:
        print("ERROR formatting results:", str(e))
        return jsonify({'error': f'Error formatting results: {str(e)}'}), 500

    return jsonify({"predictions": results})

if __name__ == '__main__':
    app.run(debug=True)
