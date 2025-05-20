"""Script to simplify and extract a balanced subset from the original Elliptic dataset.

This script expects the original three CSVs from the elliptic dataset, namely: 
"elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv", and 
"elliptic_txs_features.csv"

It outputs three CSV files that contain a **small balanced subset**
of the Elliptic dataset.  The subset has a total of 9,090 transactions: 4,545
licit (label ``0``) and 4,545 illicit (label ``1``).  Each transaction is
represented by only the first 100 features (out of the original 166) from the
full dataset.

One reason to work with this smaller balanced subset is that the resulting features 
file is a lot smaller than the original, which can be helpful in working with various
cloud platforms with limited storage/upload limits. Another is that it's a lot faster
to train and evaluate, which is good for illustrative purposes. 

"""

#!/usr/bin/env python3
import pandas as pd

# 1) Load the original CSVs
classes_df  = pd.read_csv("elliptic_txs_classes.csv", dtype=str)
edges_df    = pd.read_csv("elliptic_txs_edgelist.csv", dtype=str)
features_df = pd.read_csv("elliptic_txs_features.csv", dtype=str)

# 2) Ensure the first column of features_df is named 'txId'
if features_df.columns[0] != "txId":
    features_df.rename(columns={features_df.columns[0]: "txId"}, inplace=True)

# 3) Strip whitespace from all column names
for df in (classes_df, edges_df, features_df):
    df.columns = df.columns.str.strip()

# 4) Filter classes to keep only '1' (illicit) and '2' (licit), then map → label 1/0
mask12 = classes_df["class"].isin(["1", "2"])
classes_df = classes_df.loc[mask12].copy()
classes_df["label"] = classes_df["class"].map({"1": 1, "2": 0}).astype(int)

# 5) Balance: take all illicit (label==1) and sample equal number of legit (label==0)
illicit_df       = classes_df[classes_df["label"] == 1]
legit_df         = classes_df[classes_df["label"] == 0]
legit_sample_df  = legit_df.sample(n=len(illicit_df), random_state=42)
balanced_classes = pd.concat([illicit_df, legit_sample_df], ignore_index=True)

# 6) Build set of valid txIds
valid_tx = set(balanced_classes["txId"])

# 7) Filter edges to keep only those between valid transactions
edges_df = edges_df.loc[
    edges_df["txId1"].isin(valid_tx) &
    edges_df["txId2"].isin(valid_tx)
].copy()

# 8) Filter features to keep only valid txIds
features_df = features_df.loc[
    features_df["txId"].isin(valid_tx)
].copy()

# 9) Drop the last 66 feature columns (keep first 100 features)
#    (Assumes original features_df had 166 feature columns + txId)
feat_cols = [c for c in features_df.columns if c != "txId"]
keep_cols = feat_cols[: len(feat_cols) - 66]   # now length == 100
features_df = features_df[["txId"] + keep_cols]

# 9b) Rename the 100 feature columns to f1…f100
features_df.columns = ["txId"] + [f"f{i}" for i in range(1, len(keep_cols) + 1)]

# 10) Save out the balanced, reduced datasets
balanced_classes[["txId", "label"]].to_csv("elliptic_balanced_classes.csv", index=False)
edges_df.to_csv("elliptic_balanced_edgelist.csv", index=False)
features_df.to_csv("elliptic_balanced_features.csv", index=False)

# 11) Sanity check
print(f"Transactions kept: {len(balanced_classes)} "
      f"(illicit={len(illicit_df)}, legit={len(illicit_df)})")
print(f"Edges kept:        {len(edges_df)}")
print(f"Feature rows kept: {len(features_df)}, features per tx: {len(keep_cols)}")
print("Header row for features.csv:", features_df.columns.tolist())
