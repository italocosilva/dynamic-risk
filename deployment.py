import os
import json
import shutil


# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

model_path = os.path.join(config["output_model_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Function for deployment
def store_model_into_pickle(model):
    # Copy the latest pickle file into the deployment directory
    shutil.copy2(
        os.path.join(model_path, model),
        prod_deployment_path
    )

    # Copy latestscore.txt value into the deployment directory
    shutil.copy2(
        os.path.join(model_path, "latestscore.txt"),
        prod_deployment_path
    )

    # Copy ingestfiles.txt file into the deployment directory
    shutil.copy2(
        os.path.join(dataset_csv_path, "ingestedfiles.txt"),
        prod_deployment_path
    )


if __name__ == "__main__":
    store_model_into_pickle("trainedmodel.pkl")
