import os
import logging
import json
import ingestion
import training
import scoring
import deployment
import reporting

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
model_path = os.path.join(config["output_model_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

# Check and read new data
# first, read ingestedfiles.txt
ingested_files = []
for line in open(
    os.path.join(output_folder_path, "ingestedfiles.txt"), "r"
).readlines():
    ingested_files.append(line[:-1])
logger.info("Read ingestedfiles.txt")

# second, determine whether the source data folder has files that aren't
# listed in ingestedfiles.txt
input_files = []
for file in os.listdir(input_folder_path):
    if file[-4:] == ".csv":
        input_files.append(file)
logger.info("Read current files")

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if set(ingested_files) == set(input_files):
    exit()
logger.info("Current files are different from ingestedfiles.txt")

ingestion.merge_multiple_dataframe()
logger.info("Create new finaldata.csv")

# Checking for model drift
# check whether the score from the deployed model is different from the score
# from the model that uses the newest ingested data
training.train_model()
scoring.score_model()
logger.info("Train and score new model")

with open(os.path.join(model_path, "latestscore.txt"), "r") as file:
    score_new = file.read()
logger.info("Load new score")


with open(os.path.join(prod_deployment_path, "latestscore.txt"), "r") as file:
    score_old = file.read()
logger.info("Load old score")

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process
# here
if float(score_new) <= float(score_old):
    exit()
logger.info("New score is better than old one")


# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle("trainedmodel.pkl")
logger.info("Deploy new model")

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
reporting.score_model()
logger.info("Report new model")
