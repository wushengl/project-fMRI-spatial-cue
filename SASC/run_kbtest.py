import json
from functions import func_kbtest

config_file = 'config/config.json'

# read config file
with open(config_file, 'r') as file:
    config = json.load(file)

# related parameters
response_key_1 = config['keys']['response_key_1']
response_key_2 = config['keys']['response_key_2'] 
enter_key = config['keys']['enter_key']

# run task
test_keys = [ord(response_key_1), ord(response_key_2), ord(enter_key)]
key_labels = ["Blue (Right)", "Yellow (Left)", "Enter (TODO)"]

func_kbtest.test_keyboard_input(keys=test_keys, labels=key_labels)