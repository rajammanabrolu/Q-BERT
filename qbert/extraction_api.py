import requests
import json
import random
import time

def call_askbert(sentence, threshold=0.2, attribute=True):
    url = "http://localhost:5000/"

    response = requests.request("POST", url, data={"state": sentence, "threshold": threshold, "attribute": attribute})
    response = json.JSONDecoder().decode(response.text)

    return response

def set_batch_mode(batch_size):
    url = "http://localhost:8081/models"

    querystring = {"url": "albert.mar", "batch_size": batch_size, "max_batch_delay": "100", "initial_workers": "1"}

    response = requests.request("POST", url, params=querystring)

    print(response.text)
