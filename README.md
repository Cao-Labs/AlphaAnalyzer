# AlphaAnalyzer

## Proteinext: Protein Function Prediction with Sequence Embeddings and NLP

Proteins are essential molecules driving critical biological processes, but predicting their functions from sequence alone remains a major challenge. Proteinext introduces an efficient and scalable method for protein function prediction by combining advanced protein sequence embeddings with state-of-the-art natural language processing (NLP) models.

This repository provides code, models, and instructions for running Proteinext.

## Abstract

Proteins support vital processes in the body such as muscle development, cell growth, tissue repair, and immune defense. Their complex structures and diverse functions, however, make them difficult to fully understand. While structural prediction has advanced rapidly, function prediction lags behind due to computational demands, inefficiency, and difficulty handling highly specific proteins.

To address this, Proteinext leverages:

Meta’s 15B-parameter ESM (Evolutionary Scale Modeling) model to generate high-quality protein sequence embeddings.

A fine-tuned BigBird transformer for refining embeddings and performing function classification.

Our model was trained on 372,683 protein sequences annotated with Gene Ontology (GO) and UniProtKB labels, enabling broad and accurate function prediction.

Proteinext represents a step forward in bridging protein sequence data and functional understanding.

Proteinext/
│
├── bigbird/               # Scripts to run protein function prediction
│   ├── train.py            # Training script for BigBird model
│   ├── predict.py          # Function prediction script
│   ├── utils.py            # Helper functions
│   └── ...
│
├── data/                  # Dataset preparation and loaders
├── models/                # Model definitions
├── notebooks/             # Example Jupyter notebooks for usage
└── README.md


## Installation

Clone the repository and install dependencies:

git clone https://github.com/Cao-Labs/Proteinext.git
cd Proteinext
pip install -r requirements.txt


Dependencies include:

transformers (for BigBird & ESM embeddings)

torch

pandas, numpy, scikit-learn

tqdm

## Usage
1. Generate ESM Embeddings

Run the embedding script on your protein FASTA sequences:

python bigbird/generate_embeddings.py --input proteins.fasta --output embeddings.pt

2. Train the BigBird Model

Train Proteinext with embeddings:

python bigbird/train.py --data data/train.csv --embeddings embeddings.pt --output models/proteinext.pt

3. Predict Protein Functions

Use the fine-tuned model for function prediction:

python bigbird/predict.py --model models/proteinext.pt --input embeddings.pt --output predictions.csv

## Dataset

372,683 protein sequences from:

Gene Ontology (GO) annotations

UniProtKB dataset

Preprocessed and split into training/validation/test sets.


Citation

If you use Proteinext in your research, please cite:

@article{cao2025proteinext,
  title={Proteinext: Protein Function Prediction with Sequence Embeddings and Natural Language Processing},
  author={Cao, ...},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
