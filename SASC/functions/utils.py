import medussa as m
import psylab
import json
from tkinter import messagebox
from tkinter import simpledialog
import pandas as pd
import datetime
import os
import logging
import subprocess
import numpy as np

# global variables
config_file = 'config/config.json'

def computeRMS(sig):
    return np.sqrt(np.mean(sig**2))


def attenuate_db(sig,db):
    '''attenuate sig by db'''
    out = sig * np.exp(np.float32(-db)/8.6860)
    return out


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

def get_config(config_file):
    '''
    This function loads config file into dictionary. 
    '''
    config_file = 'config/config.json'
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def find_dev_id(dev_name, dev_ch):
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
    config = get_config(config_file)
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
    config = get_config(config_file)
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
    config = get_config(config_file)
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

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M")

    log_file_name = 'log_'+subject+'_'+task_name+'_'+time_str+'.log'
    log_file_path = os.path.join(save_folder,log_file_name)
    logger = logging.getLogger('logger_'+task_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)

    command = f'start powershell.exe -Command "Get-Content -Path "{log_file_path}" -Wait'
    subprocess.Popen(command, shell=True)

    return logger


if __name__ == '__main__':

    # test codes
    suggest_sys_volume()