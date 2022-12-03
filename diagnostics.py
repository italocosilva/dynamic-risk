import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

# Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = os.path.join(config["prod_deployment_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])


# Function to get model predictions
def model_predictions():
    # Read the deployed model and a test dataset, calculate predictions
    with open(
        os.path.join(prod_deployment_path, "trainedmodel.pkl"),
        "rb"
    ) as file:
        model = pickle.load(file)

    # Load test data
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X = df.drop(["corporation", "exited"], axis=1)

    # Predict
    pred = model.predict(X)

    return pred


# Function to get summary statistics
def dataframe_summary():
    # Load data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    statistics = []
    for col in df.select_dtypes("number").columns:
        mean = np.mean(df[col])
        median = np.median(df[col])
        std_dev = np.std(df[col])

        statistics.append([col, mean, median, std_dev])

    return statistics


# Function to get missing data percentage
def missing_data():
    # Load data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    # Calculate missing percentage
    df_missing = df.isna().sum() / df.shape[0]

    return df_missing.to_dict()


# Function to get timings
def execution_time():
    # Calculate timing of training.py
    start = timeit.default_timer()
    os.system("python training.py")
    train_time = timeit.default_timer() - start

    # Calculate timing of ingestion.py
    start = timeit.default_timer()
    os.system("python ingestion.py")
    ing_time = timeit.default_timer() - start

    return [train_time, ing_time]


# Function to check dependencies
def outdated_packages_list():
    # Get latest versions
    latest_version = (
        subprocess.check_output(["pip", "list", "--outdated"])
        .decode("utf-8")
        .split("\n")
    )
    dict_latest = {}
    for line in latest_version[2:-1]:
        split = line.split()
        dict_latest[split[0]] = split[2]

    # Read requirements.txt
    with open("requirements.txt", "r") as file:
        reqs = file.readlines()

    # Create dict with infos
    dict_reqs = {"package": [], "installed_version": [], "latest_version": []}

    # Check current and latest versions
    for line in reqs:
        split = line[:-1].split("==")
        dict_reqs["package"].append(split[0])
        dict_reqs["installed_version"].append(split[1])
        try:
            dict_reqs["latest_version"].append(dict_latest[split[0]])
        except KeyError:
            dict_reqs["latest_version"].append(split[1])

    return dict_reqs


if __name__ == "__main__":
    print(model_predictions())
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())
