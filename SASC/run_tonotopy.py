import json
import psylab
import medussa as m
import pandas as pd
import os
from functions import func_soundtest
from functions import utils


test_location = 'booth3'  # 'booth3' or 'scanner' to switch audio devices
subject = utils.ask_subject_id()
ses_num = utils.ask_session_num()
run_num = utils.ask_run_num()