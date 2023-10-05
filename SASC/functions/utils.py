import medussa as m
import psylab
import json
from tkinter import messagebox
from tkinter import simpledialog
import pandas as pd
from datetime import datetime
import os
import logging
import subprocess
import numpy as np


def get_config():
    '''
    This function loads config file into dictionary. 
    '''
    config_file = 'config/config.json'
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


def computeRMS(sig):
    return np.sqrt(np.mean(sig**2))


def attenuate_db(sig,db):
    '''attenuate sig by db'''
    out = sig * np.exp(np.float32(-db)/8.6860)
    return out

def apply_probe_ild(mix_mat,probe_ild):
    if probe_ild > 0:
        # move right by attenuating left
        mix_mat[0,0] = psylab.signal.atten(1, probe_ild)
        mix_mat[1,1] = 1
    else:
        # move left by attenuating right 
        mix_mat[1,1] = psylab.signal.atten(1, -probe_ild)
        mix_mat[0,0] = 1
    return mix_mat

def ask_task_mode():
    task_mode = simpledialog.askstring("SASC-fMRI", "Enter task mode (task / debug): ")
    return task_mode

def ask_subject_id():
    subject = simpledialog.askstring("SASC-fMRI", "Enter subject ID: ")
    return subject

def ask_session_num():
    ses_num = simpledialog.askstring("SASC-fMRI", "Enter session number: ")
    return ses_num

def ask_start_run_num():
    run_num = simpledialog.askstring("SASC-fMRI", "Enter run number to start: \n(1 if not skipping runs)")
    return run_num

def ask_tone_type():
    tone_type = simpledialog.askstring("SASC-fMRI", "Enter tonotopy tone type (pure / complex): ")
    return tone_type


def find_dev_id(dev_name, dev_ch):
    '''
    For medussa, to open an audio device, you need the device id, which is the index of the device in 
    all available devices. And then you can open the device with 
    
    audiodev = m.open_device(id, id, ch)

    Sometimes he has dev_id = id, id, ch. So note which is actually being passed. 
    '''
    devs = m.get_available_devices()
    dev_id = None
    for i,di in enumerate(devs):
        name = psylab.string.as_str(di.name)
        ch = di.maxOutputChannels
        if name.startswith(dev_name) and ch == dev_ch: 
            dev_id = i,i,ch
            out_id = i

            return dev_id, out_id
        
def wait_for_subject(interface):
    '''
    This function is used for waiting for button input from subject, avoid being affected by trigger 't'
    '''
    config = get_config()
    key_1 = config['keys']['response_key_1'] 
    key_2 = config['keys']['response_key_2'] 
    key_enter = config['keys']['enter_key']

    accept_keys = [key_1, key_2, key_enter]

    listen_moveon = True
    while listen_moveon:
        ret = interface.get_resp()
        if ret in accept_keys:
            listen_moveon = False

def suggest_sys_volume():
    '''
    Running this function will pop up a dialog suggesting system volume to start with, as a reminder. 
    '''
    config = get_config()
    suggest_sys_volume = config['sound']['start_sys_volume']

    messagebox.showinfo("Volume adjust reminder", "Make sure the system volume is set to a safe level! \n(suggested: %d)"%suggest_sys_volume)

def load_soundtest(soundtest_file_path):
    '''
    This function is used for loading existing soundtest files given the file path. 
    When the files does not exist, if file path is for session 1, then ask them to run soundtest first,
    if the file path is for session 2, ask them if they want to load soundtest file for session 1. 
    '''

    # TODO: test this function

    try:

        df = pd.read_csv(soundtest_file_path)

    except:

        # parse session number from soundtest file path
        ses_num = soundtest_file_path.split('ses0')[1][0]

        # if no soundtest file for session 1, ask to do soundtest first 
        # TODO: is there any possibility that we run session 2 first then session 1? 
        if ses_num == 1:
            messagebox.showinfo("SASC-fMRI", "Soundtest data not exist for this subject.\nDo run_soundtest first.")
            raise ValueError("No soundtest data saved for session 1!")
        
        # if no soundtest file for session 2, ask if use soundtest from session 1 (testing or running 2 sessions on same day)
        if ses_num == 2:
            try_load_ses01 = messagebox.askyesno("SASC-fMRI", "Soundtest data not exist for this session, load data from session 1?")

            if try_load_ses01:
                try: 
                    soundtest_file_path_ses01 = soundtest_file_path.replace("ses02", "ses01")
                    df = pd.read_csv(soundtest_file_path_ses01)
                except:
                    raise ValueError("No soundtest data saved for session 1!")

    ref_rms = df['ref_rms'][0]
    probe_ild = df['probe_ild'][0]

    return ref_rms, probe_ild


def get_matched_levels(subject, save_folder): 
    '''
    This function is used for loading loudness matched levels, return an array with ave matched values.
    '''
    config = get_config()
    extreme_threshold = config['matchtone']['extreme_threshold']
    ref_tone = config['sound']['ref_tone']
    puretont_pool = config['tonotopy']['puretone_pool']

    save_path = save_folder + subject + '_matchtone.csv'
    matched_levels_all = np.loadtxt(save_path, delimiter=',')

    # don't include extreme values when computing average matched levels 
    matched_levels_all[abs(matched_levels_all)>=extreme_threshold] = np.nan
    matched_levels_ave = np.nanmean(matched_levels_all, axis=0)

    # inset level 0 for reference frequency
    ref_index = puretont_pool.index(ref_tone)
    matched_levels_ave = np.insert(matched_levels_ave, ref_index, 0)  
    
    return matched_levels_ave


def init_logger(subject,task_name,save_folder):
    '''
    This function will initialize a logger and return the logger.
    '''
    log_folder = save_folder + 'logs/'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M")

    log_file_name = 'log_'+subject+'_'+task_name+'_'+time_str+'.log'
    log_file_path = os.path.join(log_folder,log_file_name)
    logger = logging.getLogger('logger_'+task_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)

    command = f'start powershell.exe -Command "Get-Content -Path "{log_file_path}" -Wait'
    subprocess.Popen(command, shell=True)

    return logger


def generate_cond_sequence(task_mode, save_folder):
    '''
    This function generates a sequence of conditions, including
    - spaCond: itd / ild
    - isTargetLeft: True / False
    - isLowLeft: True / False 

    Given task mode, 
    - if task mode: return 80 balanced trials 
    - if debug mode: return random trials 

    Returned sequences will be list of length trial_num, each element is list of 3, including value for 3 factors.
    Also save the condition sequence.
    '''

    config = get_config()
    run_num = config['run-setting'][task_mode]['zigzagtask']['run_num']
    trial_per_run = config['run-setting'][task_mode]['zigzagtask']['trial_per_run']

    spaConds = ['ild','itd']
    isTargetLefts = [True, False]
    isLowLefts = [True, False]

    # create a pandas frame with balanced trials
    col_block = np.repeat(np.arange(1,run_num+1),trial_per_run).reshape(-1,1)
    col_trial = np.tile(np.arange(1,trial_per_run+1),run_num).reshape(-1,1)

    if task_mode == 'task':
        col_spaCond = np.tile(np.repeat(spaConds, int(trial_per_run/2)),run_num).reshape(-1,1)
        col_isTargetLeft = np.tile(np.repeat(isTargetLefts, int(trial_per_run/4)),2*run_num).reshape(-1,1)
        col_isLowLeft = np.tile(np.repeat(isLowLefts, int(trial_per_run/8)),4*run_num).reshape(-1,1)
    else:
        col_spaCond = np.random.choice(spaConds, size=trial_per_run*run_num).reshape(-1,1)
        col_isTargetLeft = np.random.choice(isTargetLefts, size=trial_per_run*run_num).reshape(-1,1)
        col_isLowLeft = np.random.choice(isLowLefts, size=trial_per_run*run_num).reshape(-1,1)
    
    cond_seqs = pd.DataFrame({
        'Block': col_block,
        'Trial': col_trial,
        'spaCond': col_spaCond,
        'isTargetLeft': col_isTargetLeft,
        'isLowLeft': col_isLowLeft
    })

    # randomize rows 
    cond_seqs = cond_seqs.groupby('Block').apply(lambda x: x.sample(frac=1)).reset_index(drop=True)

    # save dataframe 
    save_path = save_folder + 'cond_sequence.csv'
    cond_seqs.to_csv(save_path, index=False)

    return cond_seqs


if __name__ == '__main__':

    # test popup window
    # suggest_sys_volume()

    # test init logger
    # init_logger('test','testtask','../data/test')

    # test condition sequenc
    cond_seqs = generate_cond_sequence('task')

