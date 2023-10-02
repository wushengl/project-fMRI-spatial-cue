import medussa as m
import psylab
import json
from tkinter import messagebox
from tkinter import simpledialog

# global variables
config_file = 'config/config.json'

def ask_subject_id():
    subject = simpledialog.askstring("SASC-fMRI", "Enter subject ID: ")
    return subject

def ask_session_num():
    ses_num = simpledialog.askstring("SASC-fMRI", "Enter session number: ")
    return ses_num

def get_config(config_file):
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

    config = get_config(config_file)
    suggest_sys_volume = config['sound']['start_sys_volume']

    messagebox.showinfo("Volume adjust reminder", "Make sure the system volume is set to a safe level! \n(suggested: %d)"%suggest_sys_volume)


if __name__ == '__main__':

    # test codes
    suggest_sys_volume()