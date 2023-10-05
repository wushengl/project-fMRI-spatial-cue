import json
import psylab
import medussa as m
import pandas as pd
import os
import pylink
from functions import func_zigzagtask
from functions import func_eyetracker
from functions import utils
import numpy as np
from gustav.forms import rt as theForm

test_location = 'booth3'  # 'booth3' or 'scanner' to switch audio devices
task_name = 'zigzagtask'
task_mode = 'debug' # utils.ask_task_mode()
subject = 'test' # utils.ask_subject_id()
ses_num = '1' # utils.ask_session_num()
start_run_num = 1 # int(utils.ask_start_run_num())

# TODO: generate a run order and save it, so that when restart from middle, the whole study will still be balanced 

#---------------------------------------
#  load configurations
#---------------------------------------

config_file = 'config/config.json'
config = utils.get_config()

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

do_eyetracker = config['run-setting'][task_mode]['do_eyetracker']
total_run_num = config['run-setting'][task_mode][task_name]['run_num']
trial_per_run = config['run-setting'][task_mode][task_name]['trial_per_run']

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

# initialize eyetracker 

if do_eyetracker:
    el_tracker = func_eyetracker.init_eyetracker()
    SCN_WIDTH, SCN_HEIGHT = func_eyetracker.init_eyetracker_graphics()
    func_eyetracker.send_initial_info(el_tracker, SCN_WIDTH, SCN_HEIGHT)

    el_tracker.doTrackerSetup()
    pylink.closeGraphics()


# initialize logger and open log in powershell

logger = utils.init_logger(subject, task_name, save_folder)
logger.info("--------------------------------------------------------------")
logger.info("Now start zigzag task...")


# run all zigzag task runs 

if start_run_num == 1:
    # start from beginning, need to generate condition sequence 
    cond_seq = utils.generate_cond_sequence(task_mode, save_folder)
else: 
    # load existing sequence
    cond_seq_path = save_folder + 'cond_sequence.csv'
    cond_seq = np.loadtxt(cond_seq_path, delimiter=',')


# generate all sequences needed 
low_pitch_seqs, high_pitch_seqs = func_zigzagtask.create_miniseq(ref_rms, matched_levels_ave)
low_pitch_seqs_ILD, low_pitch_seqs_ITD, high_pitch_seqs_ILD, high_pitch_seqs_ITD = func_zigzagtask.spatialize_miniseq(low_pitch_seqs, high_pitch_seqs)
seqs = {
    "low_pitch_seqs_ILD": low_pitch_seqs_ILD,
    "low_pitch_seqs_ITD": low_pitch_seqs_ITD,
    "high_pitch_seqs_ILD": high_pitch_seqs_ILD,
    "high_pitch_seqs_ITD": high_pitch_seqs_ITD
}


# initialize interface and data file (pre exp)
utils.init_logger(subject,task_name,save_folder)
file_name = save_folder + subject + task_name + '.csv'
if not os.path.isfile(file_name):
    fid = open(file_name, 'a')
    word_line = f"SubjectID,Trial,SpatialCond,isTargetLeft,isLowLeft,TargetNum,TargetTime,ResponseTime,TrialStartTime" 
    fid.write(word_line + "\n")
else:
    fid = open(file_name, 'a')
    fid.write("\n\n")


# initialize interface 
interface = theForm.Interface()
interface.update_Title_Center(task_name)
interface.update_Title_Right("S %s"%subject, redraw=False)
interface.update_Prompt("Hit a key to begin", show=True, redraw=True)
ret = interface.get_resp()


for current_run_num in range(start_run_num, total_run_num+1): 

    logger.info("---------------------------")
    logger.info("Now running run 0"+str(current_run_num))

    this_run_seq = cond_seq[cond_seq['Block']==current_run_num] 
    func_zigzagtask.run_block(current_run_num, this_run_seq, probe_ild, trial_per_run, save_folder, logger, task_mode, seqs, file_name, interface, dev_id)

