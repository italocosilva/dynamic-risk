from flask import Flask, jsonify, request
import pandas as pd
import pickle
import json
import os
from scoring import score_model
from diagnostics import (
    dataframe_summary,
    missing_data,
    execution_time,
    outdated_packages_list,
)


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

with open(
    os.path.join(prod_deployment_path, "trainedmodel.pkl"),
    "rb"
) as file:
    model = pickle.load(file)


# Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    file_name = request.args.get("file")
    df = pd.read_csv(file_name)
    X = df[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    pred = model.predict(X)

    return jsonify({"pred": pred.tolist()})


# Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    score = score_model()
    return jsonify(score)


# Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    stats = dataframe_summary()
    return jsonify(stats)


# Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    response = {}

    response["missing_data"] = missing_data()
    response["execution_time"] = execution_time()
    response["outdated_packages_list"] = outdated_packages_list()

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
