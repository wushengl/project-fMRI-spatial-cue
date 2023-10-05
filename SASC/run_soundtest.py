import json
import psylab
import medussa as m
import pandas as pd
import os
from functions import func_soundtest
from functions import utils

# FIXME: 
# get_centered_imag:
# When testing at booth3, the amplitude shown on fireface mixer looks correct
# but the sound came from headphone only presenting left channel
# (both channels increases when press left, decreases when press right)

subject = utils.ask_subject_id()
ses_num = utils.ask_session_num()

#---------------------------------------
#  load configurations
#---------------------------------------

config_file = 'config/config.json'
config = utils.get_config()

test_location = config['run-setting']['location']
data_folder = config['path']['data_folder']
save_folder = data_folder + subject + '/'

# related parameters
key_down = config['keys']['response_key_1']     # b down 
key_up = config['keys']['response_key_2']       # y up
key_r = config['keys']['response_key_1']        # b right
key_l = config['keys']['response_key_2']        # y left
key_enter = config['keys']['enter_key']         # g enter
accept_keys = [key_down, key_up, key_enter]

dev_name = config['audiodev'][test_location]['dev_name']
dev_ch = config['audiodev'][test_location]['dev_ch']

fs = config['sound']['fs']
ref_tone = config['sound']['ref_tone']

# dev_id is tuple of (id, id, ch), out_id is device index itself 
dev_id, out_id = utils.find_dev_id(dev_name=dev_name, dev_ch=dev_ch)


#---------------------------------------
#  running the task
#---------------------------------------

if dev_id:

    # --------------- get_comfortable_level -------------------
    # The returned value is the amount of attenuation, in dB, 
    # that should be applied for the signal level to be comfortable to the subject.
    # When start with tone amp = 1, the resultant ref_rms is our desired rms. 

    ret = func_soundtest.get_comfortable_level(\
        ref_tone, \
        dev_id, \
        fs=fs, \
        tone_dur_s=1, \
        tone_level_start=1, \
        atten_start=20, \
        key_up=key_up, \
        key_dn=key_down, \
        key_enter=key_enter)
    
    ref_rms = psylab.signal.atten(.707, float(ret)) 


    # ---------------- get_centered_image --------------------
    # The interaural difference in dB that resulted in a centered image. 
    # Negative values indicate the image should be shifted to left, 
    # the attenuation should be applied to the right ear.

    probe_ild = func_soundtest.get_centered_image(\
        ref_tone, \
        dev_id, \
        tone_level_start=ref_rms, \
        adj_step=0.5, \
        key_l=key_l, \
        key_r=key_r, \
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

