'''
This script contains all functions needed for running a task, and it's using udpated task, 
which is looking for reversal (zigzag) pattern. 
'''

import soundfile as sf
#import sounddevice as sd
#import pdb



####### utils.py 

import numpy as np
import os

import random
import time
from scipy.signal import windows


def computeRMS(sig):
    return np.sqrt(np.mean(sig**2))


def attenuate_db(sig,db):
    out = sig * np.exp(np.float32(-db)/8.6860)
    return out


def generate_tone(f_l,f_h,duration,ramp,volume,fs):

    sample_len = int(fs * duration)

    # create samples
    samples_low = (np.sin(2 * np.pi * np.arange(sample_len) * f_l / fs)).astype(np.float32)
    if f_h:
        samples_high =  (np.sin(2 * np.pi * np.arange(sample_len) * f_h / fs)).astype(np.float32)
        samples = samples_low + samples_high
    else:
        samples = samples_low

    # adjust rms
    samples = samples*volume/computeRMS(samples) # np.max(samples)
    
    # add linear ramp
    ramp_len = int(fs * ramp/2)
    #ramp_on = np.arange(ramp_len)/ramp_len
    #ramp_off = np.flip(ramp_on)
    
    # raised cosine ramp
    ramp_on = windows.cosine(int(2*ramp_len))[:int(ramp_len)]
    ramp_off = windows.cosine(int(2*ramp_len))[-int(ramp_len):]
    ramp_samples = np.concatenate((ramp_on,np.ones((sample_len-2*ramp_len)),ramp_off))
    samples = samples * ramp_samples

    #pdb.set_trace()

    return samples

def parse_trial_info(trial_info):

    tone_dur_str = '{}d{}'.format(*str(trial_info["tone_dur"]).split(".")) 
    seq_per_trial_str = str(trial_info["seq_per_trial"]) + "seq"
    tarN_T_str = str(trial_info["tarN_T"]) + "tarT"
    tarN_D_str = str(trial_info["tarN_D"]) + "tarD"
    isLowLeft_str = "lowLeft" if trial_info["isLowLeft"] else "lowRight"
    isTargetLeft_str = "targetLeft" if trial_info["isTargetLeft"] else "targetRight"
    isTargetPresent_str = "targetTrue" if trial_info["isTargetPresent"] else "targetFalse"
    repeat_loc_T_str = "Trepeat" + ('').join(list(trial_info["target_index"].astype(str)))
    repeat_loc_D_str = "Drepeat" + ('').join(list(trial_info["distractor_index"].astype(str)))

    if trial_info["isTargetPresent"]:
        trial_info_str = '-'.join([trial_info["spa_cond"],tone_dur_str,seq_per_trial_str,tarN_T_str,tarN_D_str,isLowLeft_str,isTargetLeft_str,isTargetPresent_str,repeat_loc_T_str,repeat_loc_D_str]) 
    else:
        trial_info_str = '-'.join([trial_info["spa_cond"],tone_dur_str,seq_per_trial_str,tarN_D_str,isLowLeft_str,isTargetPresent_str,repeat_loc_D_str]) 

    return trial_info_str


def parse_trial_info_ptask(trial_info):

    tone_dur_str = '{}d{}'.format(*str(trial_info["tone_dur"]).split(".")) 
    seq_per_trial_str = str(trial_info["seq_per_trial"]) + "seq"
    tarN_T_str = str(trial_info["tarN_T"]) + "tarT"
    tarN_D_str = str(trial_info["tarN_D"]) + "tarD"
    isLowLeft_str = "lowLeft" if trial_info["isLowLeft"] else "lowRight"
    isTargetLeft_str = "targetLeft" if trial_info["isTargetLeft"] else "targetRight"
    isTargetPresent_str = "targetTrue" if trial_info["isTargetPresent"] else "targetFalse"
    tar_loc_str = "tarlocs" + ('').join(list(np.array(trial_info["target_index"]).astype(str)))

    if trial_info["isTargetPresent"]:
        trial_info_str = '-'.join([trial_info["spa_cond"],tone_dur_str,seq_per_trial_str,tarN_T_str,isLowLeft_str,isTargetLeft_str,isTargetPresent_str,tar_loc_str]) 
    else:
        trial_info_str = '-'.join([trial_info["spa_cond"],tone_dur_str,seq_per_trial_str,isLowLeft_str,isTargetPresent_str]) 

    return trial_info_str


def get_unrepeated_filename(trial_info_str,save_prefix):

    filename = save_prefix + trial_info_str + '.wav'

    if os.path.exists(filename):
        index = 1
        while os.path.exists(f"{filename}_{index}"):
            index += 1
        filename = f"{filename}_{index}"

    return filename


def get_repeat_idxs(pool,tarN):
    '''
    return an array of repeat start index. 
    The input pool has removed the last element, so can choose randomly from the entire pool.
    After each sample, the index itself is removed from the pool to avoid repeat.
    The index before it is removed, so that next repeat onset before it is at least 1 element away. 
    The index after it is also removed, so that next repeat onset after it is at least 1 element away.  
    '''

    repeat_idxs = []
    indicator = np.ones(len(pool))

    for i in range(tarN):

        idx_i = np.random.choice(pool[indicator.astype(bool)])
        repeat_idxs.append(idx_i)

        indicator[idx_i] = 0
        if idx_i-1 >= 0:
            indicator[idx_i-1] = 0 
        if idx_i+1 <= len(pool)-1:
            indicator[idx_i+1] = 0

    return np.array(repeat_idxs)


def get_partial_seqpool(semitones):

    semitones.sort()
    
    if semitones == [0,1]:
        seq_pool = ['up_seq_1','up_seq_3','down_seq_6','down_seq_7','zigzag_seq_1','zigzag_seq_10']
    elif semitones == [0,2]:
        seq_pool = ['up_seq_2','up_seq_5','down_seq_1','down_seq_4','zigzag_seq_4','zigzag_seq_6']
    elif semitones == [1,2]:
        seq_pool = ['up_seq_6','up_seq_7','down_seq_3','down_seq_5','zigzag_seq_5','zigzag_seq_9']

    return seq_pool

def get_trial_with_noise(trial):
    scanner_noise, fs = sf.read('../stimuli/scanner_Minn_HCP_2.2mm_S3_TR2000.wav')
    clip_onset = int(2*fs)
    clip_offset = clip_onset + trial.shape[0]
    scanner_noise_clip = scanner_noise[clip_onset:clip_offset]
    scanner_noise_clip = np.tile(scanner_noise_clip.reshape(-1,1),(1,2))

    noise_rms = computeRMS(scanner_noise_clip)
    trial_rms = computeRMS(trial)
    scanner_noise_clip = scanner_noise_clip.copy()*trial_rms/noise_rms

    trial_with_noise = trial + scanner_noise_clip
    trial_with_noise = trial_with_noise * 0.5/np.max(trial_with_noise)
    
    return trial_with_noise


########## func_stimuli.py



def generate_miniseq(cf,step,cf_ratio,interval,duration,ramp,volume,fs):
    '''
    return a dictionary with all types of mini-sequences
    - up 7 conditions 
    - down 7 conditions 
    - zigzag up 5 conditions 
    - zigzag down 5 conditions 
    '''
    
    tone_1_low = cf / step
    tone_2_low = cf
    tone_3_low = cf * step

    tone_1_high = tone_1_low * cf_ratio
    tone_2_high = tone_2_low * cf_ratio
    tone_3_high = tone_3_low * cf_ratio

    tone_1 = generate_tone(tone_1_low,tone_1_high,duration,ramp,volume,fs)
    tone_2 = generate_tone(tone_2_low,tone_2_high,duration,ramp,volume,fs)
    tone_3 = generate_tone(tone_3_low,tone_3_high,duration,ramp,volume,fs)
    
    interval_samps = np.zeros((int(fs * interval)))

    up_seq_1 = np.concatenate((tone_1,interval_samps,tone_1,interval_samps,tone_2))
    up_seq_2 = np.concatenate((tone_1,interval_samps,tone_1,interval_samps,tone_3))
    up_seq_3 = np.concatenate((tone_1,interval_samps,tone_2,interval_samps,tone_2))
    up_seq_4 = np.concatenate((tone_1,interval_samps,tone_2,interval_samps,tone_3))
    up_seq_5 = np.concatenate((tone_1,interval_samps,tone_3,interval_samps,tone_3))
    up_seq_6 = np.concatenate((tone_2,interval_samps,tone_2,interval_samps,tone_3))
    up_seq_7 = np.concatenate((tone_2,interval_samps,tone_3,interval_samps,tone_3))

    down_seq_1 = np.concatenate((tone_3,interval_samps,tone_1,interval_samps,tone_1))
    down_seq_2 = np.concatenate((tone_3,interval_samps,tone_2,interval_samps,tone_1))
    down_seq_3 = np.concatenate((tone_3,interval_samps,tone_2,interval_samps,tone_2))
    down_seq_4 = np.concatenate((tone_3,interval_samps,tone_3,interval_samps,tone_1))
    down_seq_5 = np.concatenate((tone_3,interval_samps,tone_3,interval_samps,tone_2))
    down_seq_6 = np.concatenate((tone_2,interval_samps,tone_1,interval_samps,tone_1))
    down_seq_7 = np.concatenate((tone_2,interval_samps,tone_2,interval_samps,tone_1))

    zigzag_seq_1 = np.concatenate((tone_2,interval_samps,tone_1,interval_samps,tone_2))
    zigzag_seq_2 = np.concatenate((tone_2,interval_samps,tone_1,interval_samps,tone_3))
    zigzag_seq_3 = np.concatenate((tone_3,interval_samps,tone_1,interval_samps,tone_2))
    zigzag_seq_4 = np.concatenate((tone_3,interval_samps,tone_1,interval_samps,tone_3))
    zigzag_seq_5 = np.concatenate((tone_3,interval_samps,tone_2,interval_samps,tone_3))

    zigzag_seq_6 = np.concatenate((tone_1,interval_samps,tone_3,interval_samps,tone_1))
    zigzag_seq_7 = np.concatenate((tone_1,interval_samps,tone_3,interval_samps,tone_2))
    zigzag_seq_8 = np.concatenate((tone_2,interval_samps,tone_3,interval_samps,tone_1))
    zigzag_seq_9 = np.concatenate((tone_2,interval_samps,tone_3,interval_samps,tone_2))
    zigzag_seq_10 = np.concatenate((tone_1,interval_samps,tone_2,interval_samps,tone_1))

    seq_dict = {
        "up_seq_1": up_seq_1, "up_seq_2":up_seq_2, "up_seq_3": up_seq_3, "up_seq_4": up_seq_4,
        "up_seq_5": up_seq_5, "up_seq_6": up_seq_6, "up_seq_7": up_seq_7,
        "down_seq_1": down_seq_1, "down_seq_2": down_seq_2, "down_seq_3": down_seq_3, "down_seq_4": down_seq_4, 
        "down_seq_5": down_seq_5, "down_seq_6": down_seq_6, "down_seq_7": down_seq_7,
        "zigzag_seq_1": zigzag_seq_1, "zigzag_seq_2": zigzag_seq_2, "zigzag_seq_3": zigzag_seq_3, "zigzag_seq_4": zigzag_seq_4, "zigzag_seq_5":zigzag_seq_5,
        "zigzag_seq_6": zigzag_seq_6, "zigzag_seq_7": zigzag_seq_7, "zigzag_seq_8": zigzag_seq_8, "zigzag_seq_9": zigzag_seq_9, "zigzag_seq_10": zigzag_seq_10
    }

    #pdb.set_trace()

    return seq_dict


def generate_trial_findzigzag_clean(params,low_pitch_seq_dict,high_pitch_seq_dict,isCueIncluded,cue_interval=0.5):
    '''
    This function is used for generating a task trial with task being find reversal pattern (zigzag pattern) from target direction.  
    
    Each trial contains 2 streams, a high pitch and a low pitch, a target and a distractor. 
    Which stream is target is controlled by isTargetLeft, pitch of each stream is controlled by isLowLeft. 

    The number and locations of targets (and distractors) are randomly selected. 
    The targets (and distractors) are randomly selected from zigzag pattern pools. 
    The rest mini-sequences are selected from up/down pattern pools. 

    Temporal randomization is done by randomly switching time of each pair of mini-sequences from left/right. 

    ====================
    Inputs:
    - params: a dictionary containing all parameters needed to customize a trial, except cue related variables
    - low_pitch_seq_dict: a dictionary containing all spatialized sequences made from low pitch sound, key example: "up_seq_1_l"
    - high_pitch_seq_dict: similar to low_pitch_seq_dict, but used high pitch tones for sequences, key example: "up_seq_1_l"
    - isCueIncluded: a Boolean controlling weather cue interval is included in the generated trial
    - cue_interval: length of cue (visual fixation)

    Outputs:
    - trial: a N*2 numpy array containing the trial 
    - trial_info: an dictionary include all information about one trial 
    '''

    # -------------- preparation ----------------

    # read parameters from params

    spaCond_str = params["spatial_condition"]
    tone_duration = params["tone_duration"]
    tone_interval = params["tone_interval"]
    seq_interval = params["seq_interval"]
    seq_per_trial = params["seq_per_trial"]
    tarN_T = params["target_number_T"]
    tarN_D = params["target_number_D"]
    fs = params["fs"]
    isLowLeft = params["isLowLeft"]
    isTargetLeft = params["isTargetLeft"]
    isTargetPresent = params["isTargetPresent"]
    cue2stim_interval = params["cue2stim_interval"]

    # prepare zigzag and non-zigzag sequence pools, where each "pool" is a list containing all seq names for seq in that pool 

    seq_pool_up = np.array(['up_seq_'+str(n+1) for n in range(7)])
    seq_pool_down = np.array(['down_seq_'+str(n+1) for n in range(7)])
    seq_pool_zigzag = np.array(['zigzag_seq_'+str(n+1) for n in range(10)])
    seq_pool_nonzigzag = np.concatenate((seq_pool_up,seq_pool_down))


    # -------------- create trial without cue ----------------

    if isTargetPresent: 

        # number of zigzag patterns in each stream 
        target_num = tarN_T
        distractor_num = tarN_D

        # location of zigzag patterns in each stream 
        target_location_idxes = random.sample(range(0,seq_per_trial),target_num)
        distractor_location_idxes = random.sample(range(0,seq_per_trial),distractor_num)

        # randomly select zigzag patterns for target and distractor streams
        target_pattern_idxes = random.sample(range(len(seq_pool_zigzag)),target_num)
        distractor_pattern_idxes = random.sample(range(len(seq_pool_zigzag)),distractor_num)

        target_nonpattern_idxes = random.sample(range(len(seq_pool_nonzigzag)),int(seq_per_trial-target_num))
        distractor_nonpattern_idxes = random.sample(range(len(seq_pool_nonzigzag)),int(seq_per_trial-distractor_num))
        
        # create an array containing seq names 
        target_stream_seq_order = (99*np.ones(seq_per_trial).astype(int)).astype('U21')
        target_stream_seq_order[np.array(target_location_idxes)] = seq_pool_zigzag[target_pattern_idxes] 
        target_stream_seq_order[target_stream_seq_order == "99"] = seq_pool_nonzigzag[target_nonpattern_idxes] 

        distractor_stream_seq_order = (99*np.ones(seq_per_trial).astype(int)).astype('U21')
        distractor_stream_seq_order[np.array(distractor_location_idxes)] = seq_pool_zigzag[distractor_pattern_idxes]
        distractor_stream_seq_order[distractor_stream_seq_order == "99"] = seq_pool_nonzigzag[distractor_nonpattern_idxes] 

    else:

        # target stream nonzigzag patterns 
        target_nonpattern_idxes = random.sample(range(len(seq_pool_nonzigzag)),seq_per_trial)
        distractor_nonpattern_idxes = random.sample(range(len(seq_pool_nonzigzag)),seq_per_trial)
        
        target_stream_seq_order = seq_pool_nonzigzag(target_nonpattern_idxes)
        distractor_stream_seq_order = seq_pool_nonzigzag(distractor_nonpattern_idxes)

    # create trial with left/right being target and pitch 
    seq_interval_padding = np.zeros((int(seq_interval*fs),2)) 
    onset_diff_padding = np.zeros((int(tone_duration*fs),2))

    target_stream = np.empty((0,2))
    distractor_stream = np.empty((0,2)) 
    
    if isTargetLeft:
        target_seq_suffix = '_l'
        distractor_seq_suffix = '_r'
        if isLowLeft:
            target_seq_dict = low_pitch_seq_dict
            distractor_seq_dict = high_pitch_seq_dict
        else:
            target_seq_dict = high_pitch_seq_dict
            distractor_seq_dict = low_pitch_seq_dict
    else: 
        target_seq_suffix = '_r'
        distractor_seq_suffix = '_l'
        if isLowLeft:
            target_seq_dict = low_pitch_seq_dict
            distractor_seq_dict = high_pitch_seq_dict
        else:
            target_seq_dict = high_pitch_seq_dict
            distractor_seq_dict = low_pitch_seq_dict

    for i in range(seq_per_trial):

        this_target_key = target_stream_seq_order[i] + target_seq_suffix
        this_distractor_key = distractor_stream_seq_order[i] + distractor_seq_suffix

        # here always set target stream leading is fine, since we'll randomly switch pairs later
        target_stream = np.concatenate((target_stream,target_seq_dict[this_target_key],onset_diff_padding),axis=0)
        #print('===============debug=====================')
        #print(distractor_stream_seq_order)
        #print(distractor_stream_seq_order.shape)
        #print(distractor_stream_seq_order.dtype)
        #print('===============debug=====================')
        distractor_stream = np.concatenate((distractor_stream,onset_diff_padding,distractor_seq_dict[this_distractor_key]),axis=0)

        # add interval between mini-sequences 
        target_stream = np.concatenate((target_stream,seq_interval_padding),axis=0)
        distractor_stream = np.concatenate((distractor_stream,seq_interval_padding),axis=0)

    trial = target_stream + distractor_stream

    # -------------- randomly switch pair ----------------

    # create indicator for switch or not for each tone pair, 3 pairs per seq 
    switch_indicator = [random.randint(0, 1) for i in range(3*seq_per_trial)] 

    # create array indicating onset for each pair 
    pair_num = int(3*seq_per_trial)
    pair_sample_diff = int(tone_duration*2*fs)
    seq_interval_diff = int(seq_interval*fs)
    seq_interval_array = np.repeat(np.arange(seq_per_trial)*seq_interval_diff,3)
    pair_onsets = np.arange(0,pair_num*pair_sample_diff,pair_sample_diff) + seq_interval_array

    # an array with ones at pair onset, for sanity check 
    pair_onsets_indicator = np.zeros(trial.shape[0])
    pair_onsets_indicator[pair_onsets] = 1

    tone_samples_num = int(tone_duration*fs)

    # also initialize target time 
    target_location_idxes.sort()
    target_index = np.array(target_location_idxes)

    # target time is computed if didn't switch last pair in the miniseq 
    seq_block_time = tone_duration*6
    target_time = target_index*(seq_block_time + seq_interval) + tone_duration*4
    key_pair_idxes = target_index*3+2

    # switch pairs
    for j in range(pair_num):
        if switch_indicator[j] == 1: # do switch 
            this_pair_onset = pair_onsets[j]
            temp = trial[this_pair_onset:this_pair_onset+tone_samples_num,:].copy()

            trial[this_pair_onset:this_pair_onset+tone_samples_num,:] = trial[this_pair_onset+tone_samples_num:this_pair_onset+2*tone_samples_num,:]
            trial[this_pair_onset+tone_samples_num:this_pair_onset+2*tone_samples_num,:] = temp

            if j in key_pair_idxes:
                this_target_index = int((j-2)/3)
                this_target_order = np.where(target_index==this_target_index)[0][0]
                target_time[this_target_order] += tone_duration

    # test for target time 
    target_time_testing = np.zeros(trial.shape[0])
    target_time_testing[(target_time*fs).astype(int)] = 1
    #plt.plot(trial);plt.plot(target_time_testing);plt.show()

    # -------------- add cue ----------------

    if isCueIncluded:
        # assuming we're using visual cues, here pad extra time for cue 

        cue_pad_time = cue_interval + cue2stim_interval
        cue_pad_samples = int(cue_pad_time*fs)
        cue_padding = np.zeros((cue_pad_samples,2))
        trial = np.concatenate((cue_padding,trial),axis=0)

    if True:
        # add some extra padding at the end of trial to allow reaction time for targets happening at the end
        pad_time = 1.0
        pad_samples = int(pad_time*fs)
        trial = np.concatenate((trial,np.zeros((pad_samples,2))),axis=0)

    trial_info = {"spa_cond": spaCond_str,\
                  "tone_dur":tone_duration,\
                  "seq_per_trial":seq_per_trial,\
                  "tarN_T": tarN_T,\
                  "tarN_D": tarN_D,\
                  "isLowLeft":isLowLeft,\
                  "isTargetLeft":isTargetLeft,\
                  "isTargetPresent":isTargetPresent,\
                  "target_index":target_index,\
                  "target_time":target_time} # onset of last tone in target seq, 0 as first tone onset 
    
    return trial, trial_info


def generate_trial_1back(params,low_pitch_seq_dict,high_pitch_seq_dict,isCueIncluded,cue_interval,isOnsetDiff=True,doRandPerMiniseq=True):

    spaCond_str = params["spatial_condition"]
    tone_duration = params["tone_duration"]
    tone_interval = params["tone_interval"]
    seq_interval = params["seq_interval"]
    seq_per_trial = params["seq_per_trial"]
    tarN_T = params["target_number_T"]
    tarN_D = params["target_number_D"]
    fs = params["fs"]
    isLowLeft = params["isLowLeft"]
    isTargetLeft = params["isTargetLeft"]
    isTargetPresent = params["isTargetPresent"]
    cue2stim_interval = params["cue2stim_interval"]


    seq_pool_up_l = ['up_seq_'+str(n+1)+'_l' for n in range(7)] 
    seq_pool_up_r = ['up_seq_'+str(n+1)+'_r' for n in range(7)] 
    seq_pool_down_l = ['down_seq_'+str(n+1)+'_l' for n in range(7)] 
    seq_pool_down_r = ['down_seq_'+str(n+1)+'_r' for n in range(7)]
    seq_pool_zigzag_l = ['zigzag_seq_'+str(n+1)+'_l' for n in range(10)] 
    seq_pool_zigzag_r = ['zigzag_seq_'+str(n+1)+'_r' for n in range(10)] 


    seq_pool_l = seq_pool_up_l + seq_pool_down_l + seq_pool_zigzag_l
    seq_pool_r = seq_pool_up_r + seq_pool_down_r + seq_pool_zigzag_r 


    if isTargetPresent:

        seq_interval_list = []
        seq_leading_list = []

        # target stream
        repeat_seq_idxs_T = random.sample(range(len(seq_pool_l)),tarN_T)
        repeat_loc_idxs_T = get_repeat_idxs(np.arange(seq_per_trial-1),tarN_T)

        nonrepeat_pool_T = [n for n in range(len(seq_pool_l)) if n not in repeat_seq_idxs_T]
        nonrepeat_seq_idxs_T = random.sample(nonrepeat_pool_T,seq_per_trial-2*tarN_T)

        target_stream_order = np.ones(seq_per_trial).astype(int)*99
        for t in range(tarN_T):
            t_loc = repeat_loc_idxs_T[t]
            target_stream_order[t_loc:t_loc+2] = repeat_seq_idxs_T[t]
        
        target_stream_order[target_stream_order==99] = nonrepeat_seq_idxs_T

        # distractor stream 
        repeat_seq_idxs_D = random.sample(range(len(seq_pool_l)),tarN_D)
        repeat_loc_idxs_D = get_repeat_idxs(np.arange(seq_per_trial-1),tarN_D)

        nonrepeat_pool_D = [n for n in range(len(seq_pool_l)) if n not in repeat_seq_idxs_D]
        nonrepeat_seq_idxs_D = random.sample(nonrepeat_pool_D,seq_per_trial-2*tarN_D)

        distractor_stream_order = np.ones(seq_per_trial).astype(int)*99
        for t in range(tarN_D):
            t_loc = repeat_loc_idxs_D[t]
            distractor_stream_order[t_loc:t_loc+2] = repeat_seq_idxs_D[t]

        distractor_stream_order[distractor_stream_order==99] = nonrepeat_seq_idxs_D

        # ------ previous code for only 1 target in target stream -----
        ## generate trial with adjacent repeat
        #repeat_seq_idx = random.randint(0,len(seq_pool_l)-1)
        #repeat_loc_idx = random.randint(0,seq_per_trial-2) # first miniseq of the repeat 
        # 
        #nonrepeat_pool = [n for n in range(len(seq_pool_l)) if n != repeat_seq_idx]
        #nonrepeat_seq_idxs = random.sample(nonrepeat_pool,seq_per_trial-2) # random.sample has no repetition
        #
        #target_stream_order = np.ones(seq_per_trial).astype(int)*99
        #target_stream_order[repeat_loc_idx:repeat_loc_idx+2] = repeat_seq_idx
        #target_stream_order[target_stream_order==99] = nonrepeat_seq_idxs
        # --------------------------------------------------------------
    else:

        target_stream_order = random.sample(range(len(seq_pool_l)),seq_per_trial)
        repeat_loc_idxs_T = np.ones(tarN_T) * 99

        # distractor stream 
        repeat_seq_idxs_D = random.sample(range(len(seq_pool_l)),tarN_D)
        repeat_loc_idxs_D = get_repeat_idxs(np.arange(seq_per_trial-1),tarN_D)

        nonrepeat_pool_D = [n for n in range(len(seq_pool_l)) if n not in repeat_seq_idxs_D]
        nonrepeat_seq_idxs_D = random.sample(nonrepeat_pool_D,seq_per_trial-2*tarN_D)

        distractor_stream_order = np.ones(seq_per_trial).astype(int)*99
        for t in range(tarN_D):
            t_loc = repeat_loc_idxs_D[t]
            distractor_stream_order[t_loc:t_loc+2] = repeat_seq_idxs_D[t]
        distractor_stream_order[distractor_stream_order==99] = nonrepeat_seq_idxs_D
    
    # if want no target in distractor stream use this: 
    #distractor_stream_order = random.sample(range(len(seq_pool_l)),seq_per_trial)
    

    if isTargetLeft:
        left_stream_seqs = np.array(seq_pool_l)[target_stream_order]
        right_stream_seqs = np.array(seq_pool_r)[distractor_stream_order]
    else:
        left_stream_seqs = np.array(seq_pool_l)[distractor_stream_order] 
        right_stream_seqs = np.array(seq_pool_r)[target_stream_order]


    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

    left_stream = np.empty((0,2))
    right_stream = np.empty((0,2))

    if isOnsetDiff: # always True

        if doRandPerMiniseq:

            if isLowLeft:
                for i in range(seq_per_trial):

                    this_onset_late = 1/2*(tone_duration + tone_interval)  # always = tone_duration
                    this_onset_late_samps = np.zeros((int(fs*this_onset_late),2)) 

                    if random.randint(0,1) == 0:

                        left_stream = np.concatenate((left_stream,low_pitch_seq_dict[left_stream_seqs[i]],this_onset_late_samps),axis=0)
                        right_stream = np.concatenate((right_stream,this_onset_late_samps,high_pitch_seq_dict[right_stream_seqs[i]]),axis=0)
                        
                        # left leading 
                        if isTargetLeft:
                            this_leading = 'target'
                        else:
                            this_leading = 'distractor'
                        seq_leading_list.append(this_leading)

                    else:
                        left_stream = np.concatenate((left_stream,this_onset_late_samps,low_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                        right_stream = np.concatenate((right_stream,high_pitch_seq_dict[right_stream_seqs[i]],this_onset_late_samps),axis=0)
                        
                        #right leading
                        if isTargetLeft:
                            this_leading = 'distractor'
                        else:
                            this_leading = 'target'
                        seq_leading_list.append(this_leading)
 
                    seq_interval = random.uniform(0.65,0.85) 
                    seq_interval_list.append(seq_interval)
                    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

            else:
                for i in range(seq_per_trial):

                    this_onset_late = 1/2*(tone_duration + tone_interval)  #  random.uniform(0.2,0.4) 
                    this_onset_late_samps = np.zeros((int(fs*this_onset_late),2)) 

                    if random.randint(0,1) == 0:

                        left_stream = np.concatenate((left_stream,high_pitch_seq_dict[left_stream_seqs[i]],this_onset_late_samps),axis=0)
                        right_stream = np.concatenate((right_stream,this_onset_late_samps,low_pitch_seq_dict[right_stream_seqs[i]]),axis=0)
                        
                        # left leading 
                        if isTargetLeft:
                            this_leading = 'target'
                        else:
                            this_leading = 'distractor'
                        seq_leading_list.append(this_leading)

                    else:
                        left_stream = np.concatenate((left_stream,this_onset_late_samps,high_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                        right_stream = np.concatenate((right_stream,low_pitch_seq_dict[right_stream_seqs[i]],this_onset_late_samps),axis=0)
                        
                        #right leading
                        if isTargetLeft:
                            this_leading = 'distractor'
                        else:
                            this_leading = 'target'
                        seq_leading_list.append(this_leading)

                    seq_interval = random.uniform(0.65,0.85) 
                    seq_interval_list.append(seq_interval)
                    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

        else: # randomize leading pitch per trial instead of per minisequence (don't run this, didn't update)

            print("Warning: this setup is not updated!")

            if isLowLeft:
                for i in range(seq_per_trial):
                    left_stream = np.concatenate((left_stream,low_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,high_pitch_seq_dict[right_stream_seqs[i]]),axis=0) 
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)
            else: 
                for i in range(seq_per_trial):
                    left_stream = np.concatenate((left_stream,high_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,low_pitch_seq_dict[right_stream_seqs[i]]),axis=0) 
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

            # add random time difference between streams

            onset_late = 1/2*(tone_duration + tone_interval) #  random.uniform(0.2,0.4) 
            onset_late_samps = np.zeros((int(fs*onset_late),2)) 

            if random.randint(0,1) == 0:
                left_stream = np.concatenate((left_stream,onset_late_samps),axis=0)
                right_stream = np.concatenate((onset_late_samps,right_stream),axis=0)
            else:
                left_stream = np.concatenate((onset_late_samps,left_stream),axis=0)
                right_stream = np.concatenate((right_stream,onset_late_samps),axis=0)


    trial = left_stream + right_stream


    # add visual cue 
    if isCueIncluded:
        onset_pad_len = int((cue_interval + cue2stim_interval)*fs) 
        trial = np.concatenate((np.zeros((onset_pad_len,2)),trial),axis=0)

    if isTargetPresent:

        target_index = repeat_loc_idxs_T + 1       # return index of second item of the repeat (still in 0-based index)
        distractor_index = repeat_loc_idxs_D + 1
        target_time = []
        for t in range(tarN_T):
            this_target_index = target_index[t]
            this_target_time = this_target_index * (tone_duration*6) + np.sum(seq_interval_list[:this_target_index])
            if seq_leading_list[this_target_index]=='distractor':  # if target miniseq is lagging, add time offset
                this_target_time += tone_duration
            target_time.append(this_target_time)
        target_time = np.array(target_time)

        distractor_time = []
        for d in range(tarN_D):
            this_distractor_index = distractor_index[d]
            this_distractor_time = this_distractor_index * (tone_duration*6) + np.sum(seq_interval_list[:this_distractor_index])
            if seq_leading_list[this_target_index]=='target':  # if target miniseq is lagging, add time offset
                this_distractor_time += tone_duration
            distractor_time.append(this_distractor_time)
        distractor_time = np.array(distractor_time)

    else:
        target_index = repeat_loc_idxs_T
        distractor_index = repeat_loc_idxs_D
        target_time = []
        distractor_time = []


    trial_info = {"spa_cond": spaCond_str,\
                  "tone_dur":tone_duration,\
                  "seq_per_trial":seq_per_trial,\
                  "tarN_T": tarN_T,\
                  "tarN_D": tarN_D,\
                  "isLowLeft":isLowLeft,\
                  "isTargetLeft":isTargetLeft,\
                  "isTargetPresent":isTargetPresent,\
                  "target_index":target_index,\
                  "target_time":target_time,\
                  "distractor_index":distractor_index,\
                  "distractor_time":distractor_time}
    

    return trial, trial_info



def generate_trial_pattern(params,low_pitch_seq_dict,high_pitch_seq_dict,cue_seq_dict,isCueIncluded,isOnsetDiff=True,doRandPerMiniseq=True):

    spaCond_str = params["spatial_condition"]
    tone_duration = params["tone_duration"]
    tone_interval = params["tone_interval"]
    seq_interval = params["seq_interval"]
    seq_per_trial = params["seq_per_trial"]
    tarN_T = params["target_number_T"]
    tarN_D = params["target_number_D"]
    fs = params["fs"]
    isLowLeft = params["isLowLeft"]
    isTargetLeft = params["isTargetLeft"]
    isTargetPresent = params["isTargetPresent"]
    cue2stim_interval = params["cue2stim_interval"]

    random_semitones = random.sample(range(3),2) 
    seq_pool = get_partial_seqpool(random_semitones)
    seq_pool_l = [seq+'_l' for seq in seq_pool] 
    seq_pool_r = [seq+'_r' for seq in seq_pool] 


    if isTargetPresent:

        seq_interval_list = []
        seq_leading_list = []

        target_pattern_idx = random.sample(range(len(seq_pool)),1)
        nontarget_pattern_idxs = [n for n in range(len(seq_pool)) if n != target_pattern_idx[0]]

        target_pattern_locs = random.sample(range(seq_per_trial),tarN_T)
        target_pattern_locs.sort()

        target_stream_order = np.ones(seq_per_trial).astype(int)*99
        target_stream_order[np.array(target_pattern_locs)] = target_pattern_idx
        target_stream_order[target_stream_order==99] = random.choices(nontarget_pattern_idxs, k=seq_per_trial-tarN_T)

    else:
        target_stream_order = random.choices(range(len(seq_pool_l)),k=seq_per_trial)
        target_pattern_locs = np.ones(tarN_T) * 99
    
    # if want no target in distractor stream use this: 
    distractor_stream_order = random.choices(range(len(seq_pool_l)),k=seq_per_trial)
    

    if isTargetLeft:
        left_stream_seqs = np.array(seq_pool_l)[target_stream_order]
        right_stream_seqs = np.array(seq_pool_r)[distractor_stream_order]
    else:
        left_stream_seqs = np.array(seq_pool_l)[distractor_stream_order] 
        right_stream_seqs = np.array(seq_pool_r)[target_stream_order]


    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

    left_stream = np.empty((0,2))
    right_stream = np.empty((0,2))

    if isOnsetDiff: # always True

        if doRandPerMiniseq:

            if isLowLeft:
                for i in range(seq_per_trial):

                    this_onset_late = 1/2*(tone_duration + tone_interval)  # always = tone_duration
                    this_onset_late_samps = np.zeros((int(fs*this_onset_late),2)) 

                    if random.randint(0,1) == 0:

                        left_stream = np.concatenate((left_stream,low_pitch_seq_dict[left_stream_seqs[i]],this_onset_late_samps),axis=0)
                        right_stream = np.concatenate((right_stream,this_onset_late_samps,high_pitch_seq_dict[right_stream_seqs[i]]),axis=0)

                        if isTargetLeft:
                            this_leading = 'target'
                        else:
                            this_leading = 'distractor'
                        seq_leading_list.append(this_leading)

                    else:
                        left_stream = np.concatenate((left_stream,this_onset_late_samps,low_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                        right_stream = np.concatenate((right_stream,high_pitch_seq_dict[right_stream_seqs[i]],this_onset_late_samps),axis=0)

                        if isTargetLeft:
                            this_leading = 'distractor'
                        else:
                            this_leading = 'target'
                        seq_leading_list.append(this_leading)
 
                    seq_interval = random.uniform(0.65,0.85) 
                    seq_interval_list.append(seq_interval)
                    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

            else:
                for i in range(seq_per_trial):

                    this_onset_late = 1/2*(tone_duration + tone_interval)  #  random.uniform(0.2,0.4) 
                    this_onset_late_samps = np.zeros((int(fs*this_onset_late),2)) 

                    if random.randint(0,1) == 0:

                        left_stream = np.concatenate((left_stream,high_pitch_seq_dict[left_stream_seqs[i]],this_onset_late_samps),axis=0)
                        right_stream = np.concatenate((right_stream,this_onset_late_samps,low_pitch_seq_dict[right_stream_seqs[i]]),axis=0)

                        if isTargetLeft:
                            this_leading = 'target'
                        else:
                            this_leading = 'distractor'
                        seq_leading_list.append(this_leading)

                    else:
                        left_stream = np.concatenate((left_stream,this_onset_late_samps,high_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                        right_stream = np.concatenate((right_stream,low_pitch_seq_dict[right_stream_seqs[i]],this_onset_late_samps),axis=0)

                        if isTargetLeft:
                            this_leading = 'distractor'
                        else:
                            this_leading = 'target'
                        seq_leading_list.append(this_leading)

                    seq_interval = random.uniform(0.65,0.85) 
                    seq_interval_list.append(seq_interval)
                    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

        else: # randomize leading pitch per trial instead of per minisequence

            if isLowLeft:
                for i in range(seq_per_trial):
                    left_stream = np.concatenate((left_stream,low_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,high_pitch_seq_dict[right_stream_seqs[i]]),axis=0) 
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)
            else: 
                for i in range(seq_per_trial):
                    left_stream = np.concatenate((left_stream,high_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,low_pitch_seq_dict[right_stream_seqs[i]]),axis=0) 
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

            # add random time difference between streams

            onset_late = 1/2*(tone_duration + tone_interval) #  random.uniform(0.2,0.4) 
            onset_late_samps = np.zeros((int(fs*onset_late),2)) 

            if random.randint(0,1) == 0:
                left_stream = np.concatenate((left_stream,onset_late_samps),axis=0)
                right_stream = np.concatenate((onset_late_samps,right_stream),axis=0)
            else:
                left_stream = np.concatenate((onset_late_samps,left_stream),axis=0)
                right_stream = np.concatenate((right_stream,onset_late_samps),axis=0)


    trial = left_stream + right_stream

    # add an auditory cue 
    if isCueIncluded:

        if isTargetLeft:
            cue_seq_name = seq_pool_l[target_pattern_idx[0]] 
        else:
            cue_seq_name = seq_pool_r[target_pattern_idx[0]] 
        
        cue_seq = cue_seq_dict[cue_seq_name]

        onset_pad_len = cue_seq.shape[0] + int(cue2stim_interval*fs)
        trial_padded = np.concatenate((np.zeros((onset_pad_len,2)),trial),axis=0)

        cue_seq_padded = np.concatenate((cue_seq,np.zeros((trial_padded.shape[0]-cue_seq.shape[0],2))),axis=0)
        trial_with_cue = trial_padded + cue_seq_padded

    else:
        trial_with_cue = trial



    if isTargetPresent:

        target_index = np.array(target_pattern_locs) 

        target_time = []
        for t in range(tarN_T):
            this_target_index = target_index[t]
            this_target_time = this_target_index * (tone_duration*6) + np.sum(seq_interval_list[:this_target_index])
            if seq_leading_list[this_target_index]=='distractor':  # if target miniseq is lagging, add time offset
                this_target_time += tone_duration
            target_time.append(this_target_time)
        target_time = np.array(target_time)

    else:
        target_time = []

    trial_info = {"spa_cond": spaCond_str,\
                  "tone_dur":tone_duration,\
                  "seq_per_trial":seq_per_trial,\
                  "tarN_T": tarN_T,\
                  "tarN_D": tarN_D,\
                  "isLowLeft":isLowLeft,\
                  "isTargetLeft":isTargetLeft,\
                  "isTargetPresent":isTargetPresent,\
                  "target_index":target_index,\
                  "target_time":target_time}

    return trial_with_cue, trial_info


def generate_trial_findzigzag(params,low_pitch_seq_dict,high_pitch_seq_dict,cue_seq_dict,isCueIncluded,isOnsetDiff=True,doRandPerMiniseq=True):

    spaCond_str = params["spatial_condition"]
    tone_duration = params["tone_duration"]
    tone_interval = params["tone_interval"]
    seq_interval = params["seq_interval"]
    seq_per_trial = params["seq_per_trial"]
    tarN_T = params["target_number_T"]
    tarN_D = params["target_number_D"]
    fs = params["fs"]
    isLowLeft = params["isLowLeft"]
    isTargetLeft = params["isTargetLeft"]
    isTargetPresent = params["isTargetPresent"]
    cue2stim_interval = params["cue2stim_interval"]

    random_semitones = random.sample(range(3),2) 
    seq_pool = get_partial_seqpool(random_semitones)

    seq_pool_l = [seq+'_l' for seq in seq_pool] 
    seq_pool_r = [seq+'_r' for seq in seq_pool] 


    if isTargetPresent:

        seq_interval_list = []
        seq_leading_list = []

        # TODO
        # 1. create zigzag pool and nonzigzag pool 
        # 2. randomly choose location of targets 
        # 3. randomly choose zigzag pattern from zigzag pool 
        # 4. adjust target and non target related indexing 

        target_pattern_idx = random.sample(range(len(seq_pool)),1)
        nontarget_pattern_idxs = [n for n in range(len(seq_pool)) if n != target_pattern_idx[0]]

        target_pattern_locs = random.sample(range(seq_per_trial),tarN_T)
        target_pattern_locs.sort()

        target_stream_order = np.ones(seq_per_trial).astype(int)*99
        target_stream_order[np.array(target_pattern_locs)] = target_pattern_idx
        target_stream_order[target_stream_order==99] = random.choices(nontarget_pattern_idxs, k=seq_per_trial-tarN_T)

        pdb.set_trace()

    else:
        target_stream_order = random.choices(range(len(seq_pool_l)),k=seq_per_trial)
        target_pattern_locs = np.ones(tarN_T) * 99
    
    # if want no target in distractor stream use this: 
    distractor_stream_order = random.choices(range(len(seq_pool_l)),k=seq_per_trial)
    

    if isTargetLeft:
        left_stream_seqs = np.array(seq_pool_l)[target_stream_order]
        right_stream_seqs = np.array(seq_pool_r)[distractor_stream_order]
    else:
        left_stream_seqs = np.array(seq_pool_l)[distractor_stream_order] 
        right_stream_seqs = np.array(seq_pool_r)[target_stream_order]


    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

    left_stream = np.empty((0,2))
    right_stream = np.empty((0,2))

    left_stream_onsets = np.empty((0,1))
    right_stream_onsets = np.empty((0,1))

    miniseq_shape = low_pitch_seq_dict[left_stream_seqs[0]].shape[0]
    tone_sample_len = int(tone_duration*fs)
    onset_indicator = np.zeros((miniseq_shape,1)) # a whole miniseq 
    onset_indicator[0] = 1 # onset of mini sequence, first tone in miniseq 
    onset_indicator[tone_sample_len*2] = 1 # second tone in miniseq 
    onset_indicator[tone_sample_len*4] = 1 # third tone in miniseq 

    if isOnsetDiff: # always True

        if doRandPerMiniseq:

            if isLowLeft:
                for i in range(seq_per_trial):

                    this_onset_late = 1/2*(tone_duration + tone_interval)  # always = tone_duration
                    this_onset_late_samps = np.zeros((int(fs*this_onset_late),2)) 
                    #pdb.set_trace()

                    if random.randint(0,1) == 0:

                        left_stream = np.concatenate((left_stream,low_pitch_seq_dict[left_stream_seqs[i]],this_onset_late_samps),axis=0)
                        right_stream = np.concatenate((right_stream,this_onset_late_samps,high_pitch_seq_dict[right_stream_seqs[i]]),axis=0)
                        
                        this_onset_late_samps_mono = this_onset_late_samps[:,0].reshape(-1,1)
                        left_stream_onsets = np.concatenate((left_stream_onsets,onset_indicator*(i+1),this_onset_late_samps_mono),axis=0)
                        right_stream_onsets = np.concatenate((right_stream_onsets,this_onset_late_samps_mono,onset_indicator*(i+2)),axis=0)

                        if isTargetLeft:
                            this_leading = 'target'
                        else:
                            this_leading = 'distractor'
                        seq_leading_list.append(this_leading)

                    else:
                        left_stream = np.concatenate((left_stream,this_onset_late_samps,low_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                        right_stream = np.concatenate((right_stream,high_pitch_seq_dict[right_stream_seqs[i]],this_onset_late_samps),axis=0)
                        
                        this_onset_late_samps_mono = this_onset_late_samps[:,0].reshape(-1,1)
                        left_stream_onsets = np.concatenate((left_stream_onsets,onset_indicator*(i+1),this_onset_late_samps_mono),axis=0)
                        right_stream_onsets = np.concatenate((right_stream_onsets,this_onset_late_samps_mono,onset_indicator*(i+2)),axis=0)

                        if isTargetLeft:
                            this_leading = 'distractor'
                        else:
                            this_leading = 'target'
                        seq_leading_list.append(this_leading)
 
                    seq_interval = random.uniform(0.65,0.85) 
                    seq_interval_list.append(seq_interval)
                    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

                    seq_interval_samps_mono = np.zeros((int(fs*seq_interval),1)) 
                    left_stream_onsets = np.concatenate((left_stream_onsets,seq_interval_samps_mono),axis=0)
                    right_stream_onsets = np.concatenate((right_stream_onsets,seq_interval_samps_mono),axis=0)

            else:
                for i in range(seq_per_trial):

                    this_onset_late = 1/2*(tone_duration + tone_interval)  #  random.uniform(0.2,0.4) 
                    this_onset_late_samps = np.zeros((int(fs*this_onset_late),2)) 

                    if random.randint(0,1) == 0:

                        left_stream = np.concatenate((left_stream,high_pitch_seq_dict[left_stream_seqs[i]],this_onset_late_samps),axis=0)
                        right_stream = np.concatenate((right_stream,this_onset_late_samps,low_pitch_seq_dict[right_stream_seqs[i]]),axis=0)

                        this_onset_late_samps_mono = this_onset_late_samps[:,0].reshape(-1,1)
                        left_stream_onsets = np.concatenate((left_stream_onsets,onset_indicator*(i+1),this_onset_late_samps_mono),axis=0)
                        right_stream_onsets = np.concatenate((right_stream_onsets,this_onset_late_samps_mono,onset_indicator*(i+2)),axis=0)

                        if isTargetLeft:
                            this_leading = 'target'
                        else:
                            this_leading = 'distractor'
                        seq_leading_list.append(this_leading)

                    else:
                        left_stream = np.concatenate((left_stream,this_onset_late_samps,high_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                        right_stream = np.concatenate((right_stream,low_pitch_seq_dict[right_stream_seqs[i]],this_onset_late_samps),axis=0)

                        this_onset_late_samps_mono = this_onset_late_samps[:,0].reshape(-1,1)
                        left_stream_onsets = np.concatenate((left_stream_onsets,onset_indicator*(i+1),this_onset_late_samps_mono),axis=0)
                        right_stream_onsets = np.concatenate((right_stream_onsets,this_onset_late_samps_mono,onset_indicator*(i+2)),axis=0)

                        if isTargetLeft:
                            this_leading = 'distractor'
                        else:
                            this_leading = 'target'
                        seq_leading_list.append(this_leading)

                    seq_interval = random.uniform(0.65,0.85) 
                    seq_interval_list.append(seq_interval)
                    seq_interval_samps = np.zeros((int(fs*seq_interval),2)) 

                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

                    seq_interval_samps_mono = np.zeros((int(fs*seq_interval),1)) 
                    left_stream_onsets = np.concatenate((left_stream_onsets,seq_interval_samps_mono),axis=0)
                    right_stream_onsets = np.concatenate((right_stream_onsets,seq_interval_samps_mono),axis=0)

        else: # randomize leading pitch per trial instead of per minisequence (this function is not updated)

            if isLowLeft:
                for i in range(seq_per_trial):
                    left_stream = np.concatenate((left_stream,low_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,high_pitch_seq_dict[right_stream_seqs[i]]),axis=0) 
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)
            else: 
                for i in range(seq_per_trial):
                    left_stream = np.concatenate((left_stream,high_pitch_seq_dict[left_stream_seqs[i]]),axis=0)
                    left_stream = np.concatenate((left_stream,seq_interval_samps),axis=0)
                    right_stream = np.concatenate((right_stream,low_pitch_seq_dict[right_stream_seqs[i]]),axis=0) 
                    right_stream = np.concatenate((right_stream,seq_interval_samps),axis=0)

            # add random time difference between streams

            onset_late = 1/2*(tone_duration + tone_interval) #  random.uniform(0.2,0.4) 
            onset_late_samps = np.zeros((int(fs*onset_late),2)) 

            if random.randint(0,1) == 0:
                left_stream = np.concatenate((left_stream,onset_late_samps),axis=0)
                right_stream = np.concatenate((onset_late_samps,right_stream),axis=0)
            else:
                left_stream = np.concatenate((onset_late_samps,left_stream),axis=0)
                right_stream = np.concatenate((right_stream,onset_late_samps),axis=0)


    # add a indicator array locating onset of each mini seq 
    trial = left_stream + right_stream
    trial_onsets = left_stream_onsets + right_stream_onsets
    trial_onsets_idx = np.where(trial_onsets)[0]

    # randomly switch 2 tones in each sequence 
    switch_indicator = [random.randint(0, 1) for i in range(3*seq_per_trial)] # each indicating weather swicth for this tone pair or not, 3 pairs per seq 
    current_pair_idx = 0
    for do_switch in switch_indicator:
        if do_switch == 0: 
            pass
        elif do_switch == 1:
            # find the onset index of tone to switch in left_stream and right stream 
            switch_inx_left = trial_onsets_idx[current_pair_idx*2]
            #switch_inx_left = trial_onsets_idx[current_pair_idx*2+1]

            # switch left and right 
            switch_len = int(tone_duration*fs)
            
            swich_block_1 = trial[switch_inx_left:switch_inx_left+switch_len,:].copy()
            trial[switch_inx_left:switch_inx_left+switch_len,:] = trial[switch_inx_left+switch_len:switch_inx_left+2*switch_len,:].copy()
            trial[switch_inx_left+switch_len:switch_inx_left+2*switch_len,:] = swich_block_1

        current_pair_idx += 1

    # add an auditory cue 
    if isCueIncluded:

        if isTargetLeft:
            cue_seq_name = seq_pool_l[target_pattern_idx[0]] 
        else:
            cue_seq_name = seq_pool_r[target_pattern_idx[0]] 
        
        cue_seq = cue_seq_dict[cue_seq_name]

        onset_pad_len = cue_seq.shape[0] + int(cue2stim_interval*fs)
        trial_padded = np.concatenate((np.zeros((onset_pad_len,2)),trial),axis=0)

        cue_seq_padded = np.concatenate((cue_seq,np.zeros((trial_padded.shape[0]-cue_seq.shape[0],2))),axis=0)
        trial_with_cue = trial_padded + cue_seq_padded

    else:
        trial_with_cue = trial



    if isTargetPresent:

        target_index = np.array(target_pattern_locs) 

        target_time = [] # TODO: test target time
        for t in range(tarN_T):

            this_target_index = target_index[t]
            this_target_time = this_target_index * (tone_duration*6) + np.sum(seq_interval_list[:this_target_index]) # onset of target sequence block

            do_switch_last_pair = switch_indicator[this_target_index*3+2]

            if do_switch_last_pair == 0: # didn't switch last pair 
                if seq_leading_list[this_target_index]=='distractor':  # if target miniseq is lagging, add time offset
                    this_target_time += tone_duration*5
                else:
                    this_target_time += tone_duration*4

            if do_switch_last_pair == 0: # switched last pair
                if seq_leading_list[this_target_index]=='distractor':  # if orignal distractor leading in last pair, now target leading 
                    this_target_time += tone_duration*4
                else:
                    this_target_time += tone_duration*5

            target_time.append(this_target_time)
        target_time = np.array(target_time)

    else:
        target_time = []

    trial_info = {"spa_cond": spaCond_str,\
                  "tone_dur":tone_duration,\
                  "seq_per_trial":seq_per_trial,\
                  "tarN_T": tarN_T,\
                  "tarN_D": tarN_D,\
                  "isLowLeft":isLowLeft,\
                  "isTargetLeft":isTargetLeft,\
                  "isTargetPresent":isTargetPresent,\
                  "target_index":target_index,\
                  "target_time":target_time} # target block ontset 

    return trial_with_cue, trial_info




######## func_spatialization.py


def spatialize_seq(seq_dict,ild,itd,fs):
    '''
    This function read each minisequence in seq_dic and apply ild and itd to it and create a new dict with 
    all possible combination of minisequence and ild and itd.

    Note: here we're using broadband ild and itd. For itd, the signal power should be the same as source, 
    however for ild, we're attenuating the far ear to achieve the interaural level difference. 
    To compensate for the lower average energy for ild spatialized condition, I'm attenuating itd condition 
    to make the average rms power for the 2 channels to be the same for ild and itd stimuli. 
    Also, since I'm delaying far ear with itd (~20 samples with long itd), to make sure ild and itd stimuli 
    are of same length, I'm truncating setting the extra samples for the far ear to be 0 and used a 0.01 sec 
    linear ramp for the resulting 

    Input:
    - seq_dict: a dictionary containing all minisequences, with key being condition+idx, e.g. "up-1", "zigzag-4"
    - ild: a scalar in dB
    - itd: a scalar in miscrosec 
    - fs: sampling rate

    Output:
    - seq_dict_ild: spatialized minisequence with ild
    - seq_dict_itd: spatialized minisequence with itd
    '''
    
    seq_dict_ild = dict()
    seq_dict_itd = dict()

    for key in seq_dict:
        key_l = key + '_l'
        key_r = key + '_r'
        sig = seq_dict[key]

        # for ild, attenuate weaker channel
        seq_ild_l = np.concatenate((sig.reshape(-1,1),attenuate_db(sig,ild).reshape(-1,1)),axis=1)
        seq_ild_r = np.concatenate((attenuate_db(sig,ild).reshape(-1,1),sig.reshape(-1,1)),axis=1)

        # for itd, delay further channel
        itd_samps = int(itd * fs)
        seq_itd_l = np.concatenate((np.concatenate((sig,np.zeros(itd_samps))).reshape(-1,1),np.concatenate((np.zeros(itd_samps),sig)).reshape(-1,1)),axis=1)
        seq_itd_r = np.concatenate((np.concatenate((np.zeros(itd_samps),sig)).reshape(-1,1),np.concatenate((sig,np.zeros(itd_samps))).reshape(-1,1)),axis=1)

        # adjust mean RMS (did this before adjust length to avoid effect of extra final ramp)
        mean_rms_ild = np.mean([computeRMS(seq_ild_l[:,0]),computeRMS(seq_ild_l[:,1])])
        mean_rms_itd = np.mean([computeRMS(seq_itd_l[:,0]),computeRMS(seq_itd_l[:,1])])
        seq_itd_l = seq_itd_l*mean_rms_ild/mean_rms_itd
        seq_itd_r = seq_itd_r*mean_rms_ild/mean_rms_itd

        # adjusted length of ILD and ITD spatialized stimuli
        ramp_len = int(0.01*fs)
        trunc_func = np.ones(seq_itd_l.shape)
        trunc_func[-itd_samps:] = 0
        trunc_func[-(itd_samps+ramp_len):-itd_samps] = np.tile(np.linspace(1,0,ramp_len).reshape(-1,1),(1,2)) 
        
        seq_itd_l = seq_itd_l*trunc_func
        seq_itd_r = seq_itd_r*trunc_func
        seq_itd_l = seq_itd_l[:seq_ild_l.shape[0]]
        seq_itd_r = seq_itd_r[:seq_ild_r.shape[0]]

        # add spatialized sequences into new dicts
        seq_dict_ild[key_l] = seq_ild_l
        seq_dict_ild[key_r] = seq_ild_r
        seq_dict_itd[key_l] = seq_itd_l
        seq_dict_itd[key_r] = seq_itd_r

    return seq_dict_ild, seq_dict_itd
    


######### run_generate_stimuli.py


if __name__ == '__main__':

    ################ parameters ##################

    f0 = 220 # Hz
    fs = 44100 # Hz

    low_pitch_cf_1 = f0 # cf for center frequency
    low_pitch_cf_2 = 3*f0
    high_pitch_cf_1 = 2*f0
    high_pitch_cf_2 = 4*f0 

    low_pitch_cf_ratio = int(low_pitch_cf_2/low_pitch_cf_1)
    high_pitch_cf_ratio = int(high_pitch_cf_2/high_pitch_cf_1)

    tone_duration = 0.25 # s
    ramp_duration = 0.04 # s (this is total length for on and off ramps) # TODO weird tone with too short ramp

    tone_interval = tone_duration # this is offset to onset interval
    seq_interval = 0.8 # Note: this is not being used in generate_trial, random.uniform(0.65,0.85) is used for each miniseq

    seq_per_trial = 10
    target_number_T = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for target stream 
    target_number_D = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for distractor stream 

    # use matched ILD to ITD for all subjects 
    itd = 500e-6    # 685e-6 # s
    ild = 10        #15 # db

    semitone_step = 2**(1/12)
    volume = 0.3    # peak of tone samples could be around 1.5, set volume to be around 0.6 times the peak value you want 

    cue2stim_interval = 0.5

    ############### create minisequence ###############

    low_pitch_seqs = generate_miniseq(low_pitch_cf_1, semitone_step, low_pitch_cf_ratio, tone_interval, tone_duration, ramp_duration, volume, fs)
    high_pitch_seqs = generate_miniseq(high_pitch_cf_1, semitone_step, high_pitch_cf_ratio, tone_interval, tone_duration, ramp_duration, volume, fs)

    # sd.play(low_pitch_seqs['up_seq_1'], fs) 


    ############### spatialize minisequence ############ 

    low_pitch_seqs_ILD, low_pitch_seqs_ITD = spatialize_seq(low_pitch_seqs,ild,itd,fs)
    high_pitch_seqs_ILD, high_pitch_seqs_ITD = spatialize_seq(high_pitch_seqs,ild,itd,fs)

    # sd.play(low_pitch_seqs_ILD['up_seq_1_l'], fs) 


    ############### create trials ######################

    params = {
        "spatial_condition": "ITD500", # "ILD10" or "ITD500"
        "tone_duration": tone_duration,
        "tone_interval": tone_interval,
        "seq_interval": seq_interval,
        "seq_per_trial": seq_per_trial,
        "target_number_T": target_number_T,
        "target_number_D": target_number_D,
        "fs":fs,
        "isLowLeft": True,
        "isTargetLeft": True,
        "isTargetPresent": True,
        "cue2stim_interval": cue2stim_interval
    }

    if params["spatial_condition"] == 'ILD10':
        low_pitch_seqs_dict = low_pitch_seqs_ILD
        high_pitch_seqs_dict = high_pitch_seqs_ILD
    else:
        low_pitch_seqs_dict = low_pitch_seqs_ITD
        high_pitch_seqs_dict = high_pitch_seqs_ITD


    # ------------ find zigzag clean task -----------------
    ''''''
    #cue_pitch_seqs = generate_miniseq(330, 2**(1/12), 1.5, tone_interval, tone_duration, ramp_duration, volume, fs)
    #cue_pitch_seqs_ILD, cue_pitch_seqs_ITD = spatialize_seq(cue_pitch_seqs,ild,itd,fs)
    #if params["spatial_condition"] == 'ILD10':
    #    cue_pitch_seqs_dict = cue_pitch_seqs_ILD
    #else:
    #    cue_pitch_seqs_dict = cue_pitch_seqs_ITD

    test_trial, trial_info = generate_trial_findzigzag_clean(params,low_pitch_seqs_dict,high_pitch_seqs_dict,isCueIncluded=True) # isCueIncluded has to be True for this task
    trial_info_str = parse_trial_info_ptask(trial_info)

    save_prefix = '../stimuli/findzigzag_trial-'
    save_path = get_unrepeated_filename(trial_info_str,save_prefix)

    pdb.set_trace()

    sd.play(test_trial,fs)
    sf.write(save_path,test_trial,fs)

    test_trial_with_noise = get_trial_with_noise(test_trial)
    save_prefix_noisy = '../stimuli/noisy-findzigzag_trial-'
    save_path_noisy = get_unrepeated_filename(trial_info_str,save_prefix_noisy)

    sd.play(test_trial_with_noise,fs)
    sf.write(save_path_noisy,test_trial_with_noise,fs)

    pdb.set_trace()


    # ------------ find zigzag task -----------------
    ''''''
    cue_pitch_seqs = generate_miniseq(330, 2**(1/12), 1.5, tone_interval, tone_duration, ramp_duration, volume, fs)
    cue_pitch_seqs_ILD, cue_pitch_seqs_ITD = spatialize_seq(cue_pitch_seqs,ild,itd,fs)
    if params["spatial_condition"] == 'ILD10':
        cue_pitch_seqs_dict = cue_pitch_seqs_ILD
    else:
        cue_pitch_seqs_dict = cue_pitch_seqs_ITD

    test_trial, trial_info = generate_trial_findzigzag(params,low_pitch_seqs_dict,high_pitch_seqs_dict,cue_pitch_seqs_dict,isCueIncluded=True) # isCueIncluded has to be True for this task
    trial_info_str = parse_trial_info_ptask(trial_info)

    save_prefix = '../stimuli/findzigzag_trial-'
    save_path = get_unrepeated_filename(trial_info_str,save_prefix)

    pdb.set_trace()

    sd.play(test_trial,fs)
    sf.write(save_path,test_trial,fs)

    test_trial_with_noise = get_trial_with_noise(test_trial)
    save_prefix_noisy = '../stimuli/noisy-findzigzag_trial-'
    save_path_noisy = get_unrepeated_filename(trial_info_str,save_prefix_noisy)

    sd.play(test_trial_with_noise,fs)
    sf.write(save_path_noisy,test_trial_with_noise,fs)

    pdb.set_trace()


    # ------------ pattern task -----------------
    ''''''
    cue_pitch_seqs = generate_miniseq(330, 2**(1/12), 1.5, tone_interval, tone_duration, ramp_duration, volume, fs)
    cue_pitch_seqs_ILD, cue_pitch_seqs_ITD = spatialize_seq(cue_pitch_seqs,ild,itd,fs)
    if params["spatial_condition"] == 'ILD10':
        cue_pitch_seqs_dict = cue_pitch_seqs_ILD
    else:
        cue_pitch_seqs_dict = cue_pitch_seqs_ITD

    test_trial, trial_info = generate_trial_pattern(params,low_pitch_seqs_dict,high_pitch_seqs_dict,cue_pitch_seqs_dict,isCueIncluded=True) # isCueIncluded has to be True for this task
    trial_info_str = parse_trial_info_ptask(trial_info)

    save_prefix = '../stimuli/pattern_trial-'
    save_path = get_unrepeated_filename(trial_info_str,save_prefix)

    pdb.set_trace()

    sd.play(test_trial,fs)
    sf.write(save_path,test_trial,fs)

    test_trial_with_noise = get_trial_with_noise(test_trial)
    save_prefix_noisy = '../stimuli/noisy-pattern_trial-'
    save_path_noisy = get_unrepeated_filename(trial_info_str,save_prefix_noisy)

    sd.play(test_trial_with_noise,fs)
    sf.write(save_path_noisy,test_trial_with_noise,fs)

    pdb.set_trace()

    # ------------ 1-back task -----------------

    # low left, high right, detect repeat on right
    test_trial, trial_info = generate_trial_1back(params,low_pitch_seqs_dict,high_pitch_seqs_dict,isCueIncluded=True,cue_interval=0.5)

    trial_info_str = parse_trial_info(trial_info)
    save_prefix = '../stimuli/trial-'
    save_path = get_unrepeated_filename(trial_info_str,save_prefix)

    pdb.set_trace()

    sd.play(test_trial,fs)
    sf.write(save_path,test_trial,fs)

    test_trial_with_noise = get_trial_with_noise(test_trial)
    save_prefix_noisy = '../stimuli/noisy-trial-'
    save_path_noisy = get_unrepeated_filename(trial_info_str,save_prefix_noisy)

    sd.play(test_trial_with_noise,fs)
    sf.write(save_path_noisy,test_trial_with_noise,fs)


    #test_trial, trial_info = generate_trial_1back(tone_duration,tone_interval,low_pitch_seqs_ILD,high_pitch_seqs_ILD,seq_interval,seq_per_trial,fs,isOnsetDiff=True,doRandPerMiniseq=True,isLowLeft=True,isTargetLeft=False,isTargetPresent=True)

    #test_trial, trial_info = generate_trial_1back(tone_duration,tone_interval,low_pitch_seqs_ITD,high_pitch_seqs_ITD,seq_interval,seq_per_trial,fs,isOnsetDiff=True,doRandPerMiniseq=True,isLowLeft=True,isTargetLeft=False,isTargetPresent=True)

