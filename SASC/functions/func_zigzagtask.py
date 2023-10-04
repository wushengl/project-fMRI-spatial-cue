import utils
import func_toneseq
import medussa as m
from gustav.forms import rt as theForm
from datetime import datetime
import time
import numpy as np
import random
import psylab
import pylink
import func_eyetracker
import func_toneseq
import os


config_file = 'config/config.json'
config = utils.get_config(config_file)


def create_miniseq(ref_rms, matched_levels_ave): # gustav line 707

    f0 = config['zigzagtask']['f0']
    low_pitch_cf_1 = f0*config['zigzagtask']['low_pitch_cf_1']
    high_pitch_cf_1 = f0*config['zigzagtask']['high_pitch_cf_1']
    low_pitch_cf_ratio = config['zigzagtask']['low_pitch_cf_2'] / config['zigzagtask']['low_pitch_cf_1']
    high_pitch_cf_ratio = config['zigzagtask']['high_pitch_cf_2'] / config['zigzagtask']['high_pitch_cf_1']
    tone_interval = config['zigzagtask']['tone_interval']
    tone_duration = config['zigzagtask']['tone_duration']
    ramp_duration = config['zigzagtask']['ramp_duration']
    fs = config['sound']['fs']
    
    semitone_step = 2**(1/12)

    desired_rms_low = utils.attenuate_db(ref_rms, -1*matched_levels_ave[-2])
    desired_rms_high = utils.attenuate_db(ref_rms, -1*matched_levels_ave[-1])

    low_pitch_seqs = func_toneseq.generate_miniseq(low_pitch_cf_1, semitone_step, low_pitch_cf_ratio,tone_interval, tone_duration,ramp_duration, desired_rms_low, fs)
    high_pitch_seqs = func_toneseq.generate_miniseq(high_pitch_cf_1, semitone_step, high_pitch_cf_ratio,tone_interval, tone_duration,ramp_duration, desired_rms_high, fs)

    return low_pitch_seqs, high_pitch_seqs


def spatialize_seq_matched(seq_dict,ild,itd,fs):
    '''
    This function is mostly the same as spatialize_seq, except that this time we're adjusting levels according to matched values. 
    We're using original levels for itd condition (the complex tones are matched with 2016Hz tone perceived loudness during tonotopy scan),
    for ild condition, amplify louder ear by half ild, attenurate weaker ear by half ild.
    '''
    
    seq_dict_ild = dict()
    seq_dict_itd = dict()

    for key in seq_dict:
        key_l = key + '_l'
        key_r = key + '_r'
        sig = seq_dict[key]

        # for ild, attenuate weaker channel by 0.5 ild, amplify louder channel by 0.5 ild
        seq_ild_l = np.concatenate((utils.attenuate_db(sig,-ild/2).reshape(-1,1),utils.attenuate_db(sig,ild/2).reshape(-1,1)),axis=1)
        seq_ild_r = np.concatenate((utils.attenuate_db(sig,ild/2).reshape(-1,1),utils.attenuate_db(sig,-ild/2).reshape(-1,1)),axis=1)

        # for itd, delay further channel
        itd_samps = int(itd * fs)
        seq_itd_l = np.concatenate((np.concatenate((sig,np.zeros(itd_samps))).reshape(-1,1),np.concatenate((np.zeros(itd_samps),sig)).reshape(-1,1)),axis=1)
        seq_itd_r = np.concatenate((np.concatenate((np.zeros(itd_samps),sig)).reshape(-1,1),np.concatenate((sig,np.zeros(itd_samps))).reshape(-1,1)),axis=1)

        # adjust mean RMS (did this before adjust length to avoid effect of extra final ramp)
        #mean_rms_ild = np.mean([computeRMS(seq_ild_l[:,0]),computeRMS(seq_ild_l[:,1])])
        #mean_rms_itd = np.mean([computeRMS(seq_itd_l[:,0]),computeRMS(seq_itd_l[:,1])])
        #seq_itd_l = seq_itd_l*mean_rms_ild/mean_rms_itd
        #seq_itd_r = seq_itd_r*mean_rms_ild/mean_rms_itd

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

def spatialize_miniseq(low_pitch_seqs, high_pitch_seqs): # gustav line 719

    ild = config['zigzagtask']['ild']
    itd = config['zigzagtask']['itd']
    fs = config['sound']['fs']

    low_pitch_seqs_ILD, low_pitch_seqs_ITD = spatialize_seq_matched(low_pitch_seqs, ild, itd, fs)
    high_pitch_seqs_ILD, high_pitch_seqs_ITD = spatialize_seq_matched(high_pitch_seqs, ild, itd, fs)

    return low_pitch_seqs_ILD, low_pitch_seqs_ITD, high_pitch_seqs_ILD, high_pitch_seqs_ITD

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
        target_stream_seq_order = (99*np.ones(seq_per_trial).astype(int)).astype('U21') # astype('U21) is needed for lab computer, due to different versions 
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
        #pdb.set_trace()
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

def play_trial(stim_out, audiodev, this_cond, interface, trial_info, ref_rms, probe_ild, task_mode, file_name):
    '''
    present trial stuff: 
    - interface 
    - sound 
    - response collection 
    '''

    isTargetLeft = this_cond[1]
    rt_good_delay = config['zigzagtask']['rt_good_delay']
    fs = config['sound']['fs']

    do_eyetracker = config[task_mode]['do_eyetracker']
    LEFT_EYE = config['eyetracker']['LEFT_EYE'] 
    RIGHT_EYE = config['eyetracker']['RIGHT_EYE'] 
    BINOCULAR = config['eyetracker']['BINOCULAR'] 

    if isTargetLeft: # TODO: check if this is string or boolean
        interface.update_Prompt('<- Listen Left', show=True, redraw=True)
    else:
        interface.update_Prompt('Listen Right ->', show=True, redraw=True)
    time.sleep(2)

    interface.update_Prompt("   ██   \n   ██   \n████████\n   ██   \n   ██   ", show=True, redraw=True)

    responses = []
    valid_responses = []


    target_times = trial_info['target_time']
    target_times_end = target_times.copy() + rt_good_delay

    s = audiodev.open_array(stim_out,fs)

    # TODO: check if this is working correctly
    mix_mat = np.zeros((2, 2))
    if probe_ild > 0:
        mix_mat[0, 0] = psylab.signal.atten(1, probe_ild)
        mix_mat[1, 1] = 1
    else:
        mix_mat[1, 1] = psylab.signal.atten(1, -probe_ild)
        mix_mat[0, 0] = 1
    s.mix_mat = mix_mat

    if do_eyetracker:

        el_tracker = func_eyetracker.get_eyetracker()
        # log a message to mark the time at which the initial display came on
        el_tracker.sendMessage("SYNCTIME")

        # determine which eye(s) is/are available
        eye_used = el_tracker.eyeAvailable()
        if eye_used == RIGHT_EYE:
            el_tracker.sendMessage("EYE_USED 1 RIGHT")
        elif eye_used == LEFT_EYE or eye_used == BINOCULAR:
            el_tracker.sendMessage("EYE_USED 0 LEFT")
            eye_used = LEFT_EYE
        else:
            print("Error in getting the eye information!")
            return pylink.TRIAL_ERROR

    dur_ms = len(stim_out) / fs * 1000
    this_wait_ms = 500
    this_elapsed_ms = 0
    resp_percent = []
    s.play()
    #time.sleep(1)

    start_ms = interface.timestamp_ms()
    while s.is_playing:

        if do_eyetracker:
            error = el_tracker.isRecording()
            if error != pylink.TRIAL_OK:
                el_active = pylink.getEYELINK()
                el_active.stopRecording()
                raise RuntimeError("Recording stopped!")

        ret = interface.get_resp(timeout=this_wait_ms/1000)
        this_current_ms = interface.timestamp_ms()
        this_elapsed_ms = this_current_ms - start_ms
        this_elapsed_percent = this_elapsed_ms / dur_ms * 100
        if ret in ['b','y']: # TODO: keys
            resp = np.round(this_elapsed_ms/1000, 3)
            responses.append(str(resp))
            resp_percent.append(this_elapsed_ms / dur_ms * 100)

            # valid responses
            bool_1 = (resp > target_times)
            bool_2 = (resp <= target_times_end)
            bool_valid = bool_1 * bool_2   # same as "AND"

            if bool_valid.any():
                valid_responses.append(str(resp))
                #exp.user.valid_response_count += 1
                this_tar_idx = np.where(bool_valid)[0][0]   # index of first valid target
                target_times = np.delete(target_times,this_tar_idx)
                target_times_end = np.delete(target_times_end,this_tar_idx)

    fid = open(file_name, 'a')
    word_line = f"{','.join(responses)}"
    fid.write(word_line+"\n")


def run_trial(seqs, this_cond, file_name, subject, task_mode):
    '''
    mostly pre trial stuff and post trial stuff:
    - parameters 
    - generate trial zigzag clean 
    - eyetracker related stuff 
    '''

    # TODO: add pre trial log info 
    # TODO: clean up target time those shits and duplicate variables 

    cue = this_cond[0]
    isTargetLeft = this_cond[1]
    isLowLeft = this_cond[2]

    ### Begin generate trial, code from mri_tones.py
    target_number_T = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for target stream 
    target_number_D = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for distractor stream

    if isTargetLeft == 'True':
        current_isTargetLeft = True
    else:
        current_isTargetLeft = False

    if isLowLeft == 'True':
        current_isLowLeft = True
    else:
        current_isLowLeft = False

    f0 = config['zigzagtask']['f0']
    do_eyetracker = config[task_mode]['do_eyetracker']


    params = {
        "spatial_condition": cue, # "ILD10" or "ITD500"
        "tone_duration": config['zigzagtask']['tone_duration'],
        "tone_interval": config['zigzagtask']['tone_interval'],
        "seq_interval": config['zigzagtask']['seq_interval'],
        "seq_per_trial": config['zigzagtask']['seq_per_trial'],
        "target_number_T": target_number_T,
        "target_number_D": target_number_D,
        "fs": config['sound']['fs'],
        "isLowLeft":current_isTargetLeft,   # np.random.choice([True,False]),
        "isTargetLeft": current_isLowLeft,   # np.random.choice([True,False]),
        "isTargetPresent": True,
        "cue2stim_interval": config['zigzagtask']['cue2stim_interval']
    }

    direction = current_isTargetLeft # TODO: why so many duplicates????????
    tar_num = target_number_T
    target_number_count += target_number_T

    if params["spatial_condition"] == 'ILD10': # can use better structure, e.g. seqs_dict[tone_pitch] = seqs[tone_pitch][spaCond]
        low_pitch_seqs_dict = seqs['low_pitch_seqs_ILD'] 
        high_pitch_seqs_dict = seqs['high_pitch_seqs_ILD'] 
    else:
        low_pitch_seqs_dict = seqs['low_pitch_seqs_ITD'] 
        high_pitch_seqs_dict = seqs['high_pitch_seqs_ITD'] 

    # ------------ pattern task -----------------

    test_trial, trial_info = generate_trial_findzigzag_clean(params,low_pitch_seqs_dict,high_pitch_seqs_dict,isCueIncluded=False) # isCueIncluded has to be True for this task

    fid = open(file_name, 'a')
    word_line = f"{subject},{'zigzagtask'},{trial_info['spa_cond']},{trial_info['isTargetLeft']},{trial_info['isLowLeft']},{trial_info['tarN_T']},\
    {','.join(trial_info['target_time'].astype(str))}"

    fid.write(word_line+',')

    stim_out = test_trial
    play_trial(stim_out)
    
        
    # TODO: add post trial log info 
    pass

def run_block(current_run_num,this_cond, ref_rms, matched_levels_ave, trial_per_run, save_folder, logger, task_mode, seqs, file_name, interface):
    '''
    - loop through run_trials
    - should start eyetracker recording in this function 
    '''

    trigger_key = config['keys']['trigger_key']
    total_run_num = config['run-setting'][task_mode]['zigzagtask']['run_num']

    cue = this_cond[0]
    isTargetLeft = this_cond[1]
    isLowLeft = this_cond[2]
    subject = save_folder.split('/')[-2]

    # -------------------- eyetracker --------------------------

    do_eyetracker = config[task_mode]['do_eyetracker']
    

    if do_eyetracker:
        edf_file_name = subject + 'zz' + str(current_run_num) + '.EDF'

        if os.path.exists(edf_file_name):
            edf_file_name = edf_file_name.split('.')[0] + 'd.EDF'

        el_tracker = func_eyetracker.get_eyetracker()
        el_tracker.openDataFile(edf_file_name) 
        # TODO: not sure if it works to have file opened later than sending those commands

        logger.info("Eye tracker file opened!")

        # clear tracker display to black
        el_tracker.sendCommand("clear_screen 0")

        # switch tracker to idle mode
        el_tracker.setOfflineMode()

        error = el_tracker.startRecording(1, 1, 1, 1)
        if error:
            return error

    # -------------------- pre run --------------------------

    interface.update_Prompt("Starting run %d\n\nHit a key when you hear a zig-zag melody\n\nHit a key to move on and wait for trigger"%current_run_num, show=True, redraw=True)
    ret = interface.get_resp() # no triggers will be sent now, no need to rule out trigger impacts 

    interface.update_Prompt("Waiting for trigger\n\nHit a key when you hear a zig-zag melody", show=True, redraw=True)
    wait = True
    while wait:
        ret = interface.get_resp()
        if ret in [trigger_key]:
            trial_start_time = datetime.now() # TODO: log this time 
            wait = False

    if do_eyetracker:

        # log a message to mark the time at which the initial display came on
        el_tracker.sendMessage("SYNCTIME")

    logger.info("Trigger received at %s"%trial_start_time.strftime("%H:%M:%S.%f"))

    # -------------------- task loop --------------------------

    for trial_i in range(trial_per_run):

        logger.info("*** Now running trial %d"%(trial_i+1))
        logger.info("cue: %s"%cue)
        logger.info("isTargetLeft: %s"%str(isTargetLeft))
        logger.info("isLowLeft: %s"%str(isLowLeft))

        if do_eyetracker:
            # show some info about the current trial on the Host PC screen
            pars_to_show = ('zigzagtask', current_run_num, total_run_num, trial_i+1, trial_per_run) 
            status_message = 'Link event example, %s, Run %d/%d, Trial%d/%d' % pars_to_show
            el_tracker.sendCommand("record_status_message '%s'" % status_message)

            # log a TRIALID message to mark trial start, before starting to record.
            # EyeLink Data Viewer defines the start of a trial by the TRIALID message.
            el_tracker.sendMessage("TRIALID %d" % (trial_i+1)) # exp.run.trials_exp

            # log a message to mark the time at which the initial display came on
            el_tracker.sendMessage("SYNCTIME")

        run_trial()

        logger.info("Trial finished!")

    # TODO: end block close eyetracker 

    pass