import json
import psylab
import medussa as m
import pandas as pd
import os
from functions import func_tonotopy
from functions import utils

test_location = 'booth3'  # 'booth3' or 'scanner' to switch audio devices
task_name = 'tonotopy'
task_mode = utils.ask_task_mode()
subject = utils.ask_subject_id()
ses_num = utils.ask_session_num()
tone_type = utils.ask_tone_type()
start_run_num = utils.ask_start_run_num() 

# TODO: need to save a generated run order, and if restart, should restart 
# from paused place to make sure conditions are balanced

#---------------------------------------
#  load configurations
#---------------------------------------

config_file = 'config/config.json'
config = utils.get_config(config_file)

data_folder = config['path']['data_folder']
save_folder = data_folder + subject + '/'

# related parameters
key_1 = config['keys']['response_key_1'] # TODO: make sure these keys are correctly matched
key_2 = config['keys']['response_key_2'] 
key_enter = config['keys']['enter_key'] 
accept_keys = [key_1, key_2, key_enter]

dev_name = config['audiodev'][test_location]['dev_name']
dev_ch = config['audiodev'][test_location]['dev_ch']

fs = config['sound']['fs']
ref_tone = config['sound']['ref_tone']

frequency_cycle = config['tonotopy']['cycles'][tone_type]
total_run_num = config['run-setting'][task_mode][tone_type]['run_num']
cycle_per_run = config['run-setting'][task_mode][tone_type]['cycle_per_run']

freq_step_direction = config['tonotopy']['freq_step_direction'] # this controls whether go from low to high or vice versa

do_eyetracker = config[task_mode]['do_eyetracker']

#---------------------------------------
#  load soundtest and loudness match
#---------------------------------------

soundtest_file_path = save_folder + subject + '_soundtest_ses0' + ses_num + '.csv'
ref_rms, probe_ild = utils.load_soundtest(soundtest_file_path)

matched_levels_ave = utils.get_matched_levels(subject,save_folder)

#---------------------------------------
#  running the task
#---------------------------------------

# check if sound device available 

dev_id, out_id = utils.find_dev_id(dev_name=dev_name, dev_ch=dev_ch)

if dev_id:
    pass
else:
    raise Exception(f"The audio device {dev_name} was not found")

# initialize logger and open log in powershell

logger = utils.init_logger(subject, task_name, save_folder)
logger.info("--------------------------------------------------------------")
logger.info("Now start tonotopy task...")
logger.info("Tone type: %s"%tone_type)
logger.info("Frequency cycle: %s"%str(frequency_cycle))
logger.info("Cycles per run: %d"%cycle_per_run)
logger.info("Total run number:%d"%total_run_num)

# run all tonotopy runs for 

for current_run_num in range(start_run_num, total_run_num+1): 

    logger.info("---------------------------")
    logger.info("Now running run"+str(current_run_num))

    this_cycle = frequency_cycle.copy()
    do_switch_step = freq_step_direction[current_run_num-1]
    if do_switch_step:
        this_cycle.reverse()
        
    func_tonotopy.run_tonotopy_task(this_cycle, dev_id[0], exp, do_adjust_level,matched_levels_ave[:-2], cycle_per_run=cycle_per_run,round_idx=i)

    # TODO: exp variables 
