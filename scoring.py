import pandas as pd
import pickle
import os
from sklearn import metrics
import json


# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])


# Function for model scoring
def score_model():
    # Load model
    with open(os.path.join(model_path, "trainedmodel.pkl"), "rb") as file:
        model = pickle.load(file)

    # Load test data
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X = df.drop(["corporation", "exited"], axis=1)
    y = df["exited"]

    # Predict
    pred = model.predict(X)

    # Score
    score = metrics.f1_score(y, pred)

    # Save score
    with open(os.path.join(model_path, "latestscore.txt"), "w") as file:
        file.write(str(score))


if __name__ == "__main__":
    score_model()
