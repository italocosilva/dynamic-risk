import os
import requests
import json

# Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

model_path = os.path.join(config["output_model_path"])

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000/"


# Call each API endpoint and store the responses
response1 = requests.post(
    URL +
    "prediction?file=testdata/testdata.csv"
).content.decode("ascii")
response2 = requests.get(URL + "scoring").content.decode("ascii")
response3 = requests.get(URL + "summarystats").content.decode("ascii")
response4 = requests.get(URL + "diagnostics").content.decode("ascii")

# combine all API responses
responses = (
    "-----> /prediction response \n"
    + response1
    + "\n\n\n-----> /scoring response \n"
    + response2
    + "\n\n\n-----> /summarystats response \n"
    + response3
    + "\n\n\n-----> /diagnostics response \n"
    + response4
)

# write the responses to your workspace
with open(os.path.join(model_path, "apireturns.txt"), "w") as file:
    file.write(responses)
