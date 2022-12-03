import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics


# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

test_data_path = os.path.join(config["test_data_path"])
model_path = os.path.join(config["output_model_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])


# Function for reporting
def score_model():
    # Load test data
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    y = df["exited"]

    # Calculate a confusion matrix using the test data and the deployed model
    pred = diagnostics.model_predictions()

    # Write the confusion matrix to the workspace
    fig = sns.heatmap(
        metrics.confusion_matrix(y, pred),
        cmap="viridis",
        annot=True
    )
    fig.set_xlabel("Predicted")
    fig.set_ylabel("True")
    plt.savefig(os.path.join(model_path, "confusionmatrix.png"))


if __name__ == "__main__":
    score_model()
