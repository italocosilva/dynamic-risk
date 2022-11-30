import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


# Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    df = pd.DataFrame(
        columns=[
            'corporation',
            'lastmonth_activity',
            'lastyear_activity',
            'number_of_employees',
            'exited'
        ]
    )
    
    record = open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w')

    for file in os.listdir(input_folder_path):
        if file[-4:] == '.csv':
            df = df.append(
                pd.read_csv(os.path.join(input_folder_path, file)), 
                ignore_index=True
            )
            record.write(file + '\n')

    record.close()
    
    df.drop_duplicates().to_csv(
        os.path.join(output_folder_path, 'finaldata.csv'),
        index=False
    )


if __name__ == '__main__':
    merge_multiple_dataframe()
