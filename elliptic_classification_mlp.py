#!/usr/bin/env python3

"""Simple tutorial for classifying bitcoin addresses using an MLP.

This script expects three CSV files that contain a **small balanced subset**
of the Elliptic dataset.  The subset has a total of 9,090 transactions: 4,545
licit (label ``0``) and 4,545 illicit (label ``1``).  Each transaction is
represented by only the first 100 features (out of the original 166) from the
full dataset.

See original Elliptic dataset here: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the preprocessed subset of the Elliptic dataset
#    - ``elliptic_balanced_features.csv`` contains ``txId`` plus the first 100
#      feature columns from the original 166
#    - ``elliptic_balanced_classes.csv`` provides the label for each
#      transaction (``0`` = licit, ``1`` = illicit)
features_df = pd.read_csv("elliptic_balanced_features.csv")
classes_df  = pd.read_csv("elliptic_balanced_classes.csv")
# edges_df    = pd.read_csv("filtered_edgelist.csv")  # not used for MLP but included if graph info is needed

# 2. Merge features with labels on txId
df = features_df.merge(classes_df, on="txId")

# 3. Prepare X and y
X = df.drop(columns=["txId", "label"])
y = df["label"]

# 4. Train/test split (stratified 70% train / 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# 5. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 6. Build and train the MLP
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=100,
    verbose=True,        # ← turn on progress messages
    random_state=42
)
clf.fit(X_train, y_train)

# 7. Evaluate on the test set
y_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
