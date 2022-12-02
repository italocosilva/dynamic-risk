import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


# Function for training the model
def train_model():
    # Load data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = df.drop(["corporation", "exited"], axis=1)
    y = df["exited"]

    # Use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # Fit the logistic regression to your data
    model.fit(X, y)

    # Write the trained model in a file called trainedmodel.pkl
    with open(os.path.join(model_path, "trainedmodel.pkl"), "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    train_model()
