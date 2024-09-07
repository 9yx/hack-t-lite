import logging

import requests, os
import json

BASE_CORE_URL = os.environ["BOT_BASE_URL"]

def send_bot(user_id,dialog):
    try:
        response = requests.post(BASE_CORE_URL + "/send_answer/" + str(user_id), data=json.dumps({"txt": dialog}), timeout=90)
        response.raise_for_status()
        return response.json()
    except Exception as err:
         logging.error("Error from server info ",  err)