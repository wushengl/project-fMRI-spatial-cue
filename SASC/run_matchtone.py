import json
import psylab
import medussa as m
import pandas as pd
import os
from functions import func_matchtone
from functions import utils
import numpy as np

test_location = 'booth3'  # 'booth3' or 'scanner' to switch audio devices
task_name = 'matchtone'
subject = utils.ask_subject_id()
ses_num = utils.ask_session_num()

# TODO: need to test this script

#---------------------------------------
#  load configurations
#---------------------------------------

config_file = 'config/config.json'
config = utils.get_config(config_file)

data_folder = config['path']['data_folder']
save_folder = data_folder + subject + '/'

# related parameters
key_down = config['keys']['response_key_1'] # TODO: make sure these keys are correctly matched
key_up = config['keys']['response_key_2'] 
key_enter = config['keys']['enter_key'] 
accept_keys = [key_down, key_up, key_enter]

dev_name = config['audiodev'][test_location]['dev_name']
dev_ch = config['audiodev'][test_location]['dev_ch']

fs = config['sound']['fs']
ref_tone = config['sound']['ref_tone']

puretone_pool = config['tonotopy']['puretone_pool']
complextone_pool = config['tonotopy']['complextone_pool']
matching_pool = puretone_pool + complextone_pool
matching_pool = [cf for cf in matching_pool if cf != ref_tone] # remove ref_tone from matching_pool

minimum_match_time = config['matchtone']['minimum_match']
extreme_threshold = config['matchtone']['extreme_threshold']

#---------------------------------------
#  load soundtest
#---------------------------------------

soundtest_file_path = save_folder + subject + '_soundtest_ses0' + ses_num + '.csv'
ref_rms, probe_ild = utils.load_soundtest(soundtest_file_path)

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
logger.info("Now start loudness matching...")
logger.info("Matching pool: "+str(matching_pool))


# First, run 2 baseline match 

all_matched_levels = np.empty((0,len(matching_pool)))

for i in range(minimum_match_time): 
    logger.info("Now start matching round "+str(i+1))

    # matched_levels is a list of matched level difference, without 2016Hz

    matched_levels = func_matchtone.get_loudness_match(ref_tone,matching_pool,dev_id[0],tone_level_start=ref_rms,round_idx=i, key_up=ord(key_up), key_dn=ord(key_down), key_enter=key_enter)
    all_matched_levels = np.concatenate((all_matched_levels,np.array(matched_levels).reshape(1,-1)),axis=0)
    logger.info(matched_levels)

# remove extremely large value anyway, or it might converge even variance is large and final averaged data would be not reliable
all_matched_levels[abs(all_matched_levels) >= extreme_threshold] = np.nan

# find the convergence boundaries to determine if next matched values converge
lower_bound, upper_bound, last_mean = func_matchtone.find_convergence_bounds(all_matched_levels)


# Then, start adaptive matching process

logger.info("Finished baseline matching, now start adaptive matching...")
conv_check = np.zeros(all_matched_levels.shape(1)).astype(int)
this_matching_pool = np.array(matching_pool.copy(),dtype=object)


while np.sum(conv_check)<len(conv_check): # stop when conv_check = len(conv_check), which means all 1

    logger.info("----------------------- new adaptive matching -------------------------")
    logger.info("Current convergence status: " + str(conv_check.astype(bool)))
    logger.info("This matching pool: " + str(this_matching_pool))

    # update matching pool, only leave those haven't converged yet
    this_matching_pool = np.array(matching_pool)[(1-conv_check).astype(bool)]

    # get new matching
    i+=1
    this_matched_levels = func_matchtone.get_loudness_match(ref_tone, list(this_matching_pool), dev_id[0],tone_level_start=ref_rms, round_idx=i,key_up=ord(key_up), key_dn=ord(key_down), key_enter=key_enter)

    # add new matching to all matched levels
    matched_levels = func_matchtone.fill_matched_levels(this_matched_levels,conv_check,last_mean)
    all_matched_levels = np.concatenate((all_matched_levels, np.array(matched_levels).reshape(1, -1)), axis=0)

    # check if new sample within boundary
    conv_check = (matched_levels>=lower_bound).astype(int) * (matched_levels<=upper_bound).astype(int)

    # log matching results 
    logger.info("New matched levels: " + str(np.round(this_matched_levels,3)))
    logger.info("Upper bounds: " + str(upper_bound))
    logger.info("Lower bounds: " + str(lower_bound))
    logger.info("Updated convergence status: " + str(conv_check.astype(bool)))

    # update boundary
    lower_bound, upper_bound, last_mean = func_matchtone.find_convergence_bounds(all_matched_levels)


matched_levels_ave = np.nanmean(all_matched_levels,axis=0)

ref_index = puretone_pool.index(ref_tone)
matched_levels_ave = np.insert(matched_levels_ave, ref_index, 0)  # inset level 0 for reference frequency

logger.info("--------------------------------------------------------------")
logger.info("Matching finished!")
logger.info("Averaged matched levels: " + str(matched_levels_ave))
logger.info("--------------------------------------------------------------")


#---------------------------------------
#  save results
#---------------------------------------

save_path = save_folder + subject + '_matchtone.csv'
np.savetxt(save_path,all_matched_levels,delimiter=',')
   


