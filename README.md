# Fraud Classifier Demo

This repository contains two short Python scripts that demonstrate how to work with the [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) dataset for bitcoin transaction classification.

1. **`elliptic_extract_balanced.py`** – takes the original Elliptic CSV files and extracts a small balanced subset of 9,090 transactions. Each transaction keeps only the first 100 feature columns. The script writes three new CSV files:
   - `elliptic_balanced_classes.csv`
   - `elliptic_balanced_edgelist.csv`
   - `elliptic_balanced_features.csv`

2. **`elliptic_classification_mlp.py`** – trains a simple multilayer perceptron (MLP) classifier using the balanced subset created above. It reports accuracy and a classification report on a held‑out test split.

## Requirements

- Python 3
- `pandas`
- `scikit-learn`

Install the requirements with:

```bash
pip install pandas scikit-learn
```

## Usage

1. Download the original Elliptic dataset CSV files from Kaggle and place them in the repository directory.
2. Run the extraction script:

```bash
python elliptic_extract_balanced.py
```

This will create the balanced CSV files listed above.

3. Train the classifier:

```bash
python elliptic_classification_mlp.py
```

The script prints training progress, overall accuracy, and a detailed classification report.

The dataset files are not included in this repository due to licensing restrictions.
