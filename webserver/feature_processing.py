import os
from datetime import datetime
import pandas as pd
import re 
import requests

def get_recent_file(base_dir, pattern):
    """
    Use a regex to search for the most recent openface output file
    Openface creates file names using YYYY-MM-DD-HH-MM so searching for csv
    files down to the minute may cause issues
    """
    regex = re.compile(pattern)
    
    most_recent_time = None
    most_recent_file = None
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(base_dir):
        for filename in filenames:
            # Check if the filename matches the OpenFace output pattern
            if regex.match(filename):
                # Construct full filepath
                filepath = os.path.join(dirpath, filename)
                file_time = os.path.getmtime(filepath)
                if most_recent_time is None or file_time > most_recent_time:
                    most_recent_time = file_time
                    most_recent_file = filepath
    
    return most_recent_file

def csv_to_json(filename):
    """
    Reads a CSV file and converts it to JSON format.
    """
    # Read the CSV file into a DataFrame
    data = pd.read_csv(filename)
    
    # Convert the DataFrame to JSON
    json_str = data.to_json(orient='records', lines=False)
    
    return json_str

def send_data(json_data, url):
    """
    Sends JSON data to webserver using a POST request.
    """
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, data=json_data, headers=headers)
        if response.status_code == 200:
            print("Data successfully sent to the server.")
            return response
        else:
            print(f"Failed to send data. Server responded with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending data: {e}")
    return None

# Define the base directory and regex pattern for OpenFace output files
base_dir = '../../openface/'
pattern = r'webcam_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.csv'

# Webserver url
server_url = 'http://199.98.27.216:5000'

most_recent_file = get_recent_file(base_dir, pattern)

if most_recent_file:
    json_data = csv_to_json(most_recent_file)
    print(json_data)
    send_data(json_data, server_url + "/process_data")
else:
    print("No matching files found")
