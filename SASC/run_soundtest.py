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
key_r = config['keys']['response_key_1'] 
key_l = config['keys']['response_key_2'] 
key_enter = config['keys']['enter_key'] 
accept_keys = [key_down, key_up, key_enter]

dev_name = config['audiodev'][test_location]['dev_name']
dev_ch = config['audiodev'][test_location]['dev_ch']

fs = config['sound']['fs']
ref_tone = config['sound']['ref_tone']

dev_id, out_id = utils.find_dev_id(dev_name=dev_name, dev_ch=dev_ch)


#---------------------------------------
#  running the task
#---------------------------------------

if dev_id:
    audiodev = m.open_device(*dev_id)

    # --------------- get_comfortable_level -------------------
    # The returned value is the amount of attenuation, in dB, 
    # that should be applied for the signal level to be comfortable to the subject.
    # When start with tone amp = 1, the resultant ref_rms is our desired rms. 
    # But need to make sure the system volume is high enough, as it starts with 50dB attenuation. 

    ret = func_soundtest.get_comfortable_level(\
        ref_tone, \
        out_id, \
        fs=fs, \
        tone_dur_s=1, \
        tone_level_start=1, \
        atten_start=50, \
        ear='both', \
        key_up=ord(key_up), \
        key_dn=ord(key_down), \
        key_enter=key_enter)
    
    ref_rms = psylab.signal.atten(.707, float(ret)) 


    # ---------------- get_centered_image --------------------
    # The interaural difference in dB that resulted in a centered image. Negative 
    # values indicate that attenuation should be applied to the left ear (ie., the 
    # original image was to the left, thus the left ear 
    # should be attenuated by that amount). 

    # TODO: check that mix mat thing 
    probe_ild = func_soundtest.get_centered_image(\
        ref_tone, \
        out_id, \
        tone_level_start=ref_rms, \
        adj_step=0.5, \
        key_l=ord(key_l), \
        key_r=ord(key_r), \
        key_enter=key_enter)

else:
    raise Exception(f"The audio device {dev_name} was not found")


#---------------------------------------
#  save useful data
#---------------------------------------

soundtest_df = pd.DataFrame(data={"ref_rms":[ref_rms], "probe_ild":[probe_ild]})

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

soundtest_file_path = save_folder + subject + '_soundtest_ses0' + ses_num + '.csv'
soundtest_df.to_csv(soundtest_file_path)

# TODO: saved files with index 0, see if that will cause a trouble
