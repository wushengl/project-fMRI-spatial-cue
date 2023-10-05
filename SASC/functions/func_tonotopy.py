import os
import medussa as m
from gustav.forms import rt as theForm
from datetime import datetime
import time
import numpy as np
import random
import pylink
from . import func_eyetracker
from . import func_toneseq
from . import utils


config_file = 'config/config.json'
config = utils.get_config()

def generate_all_seqs(freq_cycle, matched_levels, ref_rms):

    '''
    This function is used for generating all possible 4-tone minisequences, 
    it will return a dictionary storing all minisequences.
    '''

    semitone_step =  2**(1/12)
    tone_duration = config['tonotopy']['tone_duration']
    ramp_duration = config['tonotopy']['ramp_duration']
    tone_interval = config['tonotopy']['tone_interval']
    fs = config['sound']['fs']
    
    all_seqs = dict()

    for i, cf in enumerate(freq_cycle):
        this_level_adjust = matched_levels[i]
        if isinstance(cf, list):
            cf_low = cf[0]
            cf_ratio = cf[1]/cf[0]
            cf_key = str(cf_low)+'c'
        else:
            cf_ratio = None
            cf_key = str(cf)

        # apply level adjustment to reference rms to get desired rms for current frequency
        desired_rms = utils.attenuate_db(ref_rms, -this_level_adjust)

        this_cf_seqs = func_toneseq.generate_miniseq_4tone(cf,semitone_step,cf_ratio,tone_interval,tone_duration,ramp_duration,desired_rms,fs)
        all_seqs[cf_key] = this_cf_seqs

    return all_seqs


def get_repeat_idxs(pool,tarN):

    '''
    Return an array of repeat start index. 
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


def generate_trial_tonotopy_1back(params,seq_dict):

    '''
    This function is used for generating a tonotopy trial with task finding 1-back repeating pattern. 
    Each miniseq has 4 tones.  
    
    Each trial contains 1 stream. The number and locations of targets (and distractors) are randomly selected. 

    ====================
    Inputs:
    - params: a dictionary containing all parameters needed to customize a trial, except cue related variables
    - seq_dict: a dictionary containing all sequences with center frequency cf, key example: "up_seq_1" (non-spatialized)

    Outputs:
    - trial: a N*2 numpy array containing the trial 
    - trial_info: an dictionary include all information about one trial 
    '''

    # -------------- preparation ----------------

    # read parameters from params

    cf = params["cf"]
    tone_duration = params["tone_duration"]
    seq_interval = params["seq_interval"]
    seq_per_trial = params["seq_per_trial"]
    tarN_T = params["target_number_T"]
    fs = params["fs"]
    
    # get sequence pool for this center frequency 

    seq_pool = np.array(['seq'+str(n+1) for n in range(len(seq_dict))])  # 81

    # -------------- create trial without cue ----------------

    # target stream
    repeat_seq_keys_T = np.random.choice(seq_pool, size=3, replace=False)               # get tarN_T keys for target sequence                    
    repeat_loc_idxs_T = get_repeat_idxs(np.arange(seq_per_trial-1),tarN_T)              # get indexes for where the repeating pattern starts 

    nonrepeat_pool_T = [n for n in seq_pool if n not in repeat_seq_keys_T]              # get seq pool except those selected to be target 
    nonrepeat_seq_idxs_T = random.sample(nonrepeat_pool_T,seq_per_trial-2*tarN_T)       # get indexes for non-target sequence index 

    target_stream_order = (np.ones(seq_per_trial).astype(int)*99).astype('U21')
    for t in range(tarN_T):
        t_loc = repeat_loc_idxs_T[t]
        target_stream_order[t_loc:t_loc+2] = repeat_seq_keys_T[t] 
    
    target_stream_order[target_stream_order=='99'] = nonrepeat_seq_idxs_T

    # padding between intervals 
    seq_interval_padding = np.zeros((int(seq_interval*fs),2)) 

    # start generating the trial 
    target_stream = np.empty((0,2))
    target_seq_dict = seq_dict

    for i in range(seq_per_trial):

        this_target_key = target_stream_order[i]
        this_target = np.tile(target_seq_dict[this_target_key].reshape(-1,1),(1, 2)) # mono to stereo

        # concatenate this target to the stream
        target_stream = np.concatenate((target_stream,this_target),axis=0) 

        # add interval between mini-sequences 
        target_stream = np.concatenate((target_stream,seq_interval_padding),axis=0)

    trial = target_stream 

    # -------------- get target time ----------------

    # also initialize target time 
    target_index = np.sort(repeat_loc_idxs_T) 

    # seq_block_time is the time per mini-seq block (all time between 2 seq-intervals), here only 4 tone_duration 
    seq_block_time = tone_duration*4 
    target_time = (target_index+1)*(seq_block_time + seq_interval) + tone_duration*3 # target_index is the first mini-seq in repeat, add 1 for the second mini-seq

    # test for target time 
    target_time_testing = np.zeros(trial.shape[0])
    target_time_testing[(target_time*fs).astype(int)] = 1
    #plt.plot(trial);plt.plot(target_time_testing);plt.show()  # TODO: test target time 

    trial_info = {"cf": cf,\
                  "tone_dur":tone_duration,\
                  "seq_per_trial":seq_per_trial,\
                  "tarN_T": tarN_T,\
                  "target_index":target_index,\
                  "target_time":target_time} # onset of last tone in target seq, 0 as first tone onset 
    
    return trial, trial_info


def run_tonotopy_task(freq_cycle, dev_id, ref_rms, probe_ild, matched_dbs, cycle_per_run, round_idx, task_mode, save_path, logger):

    do_eyetracker = config['run-setting'][task_mode]['do_eyetracker']
    tone_duration = config['tonotopy']['tone_duration']
    tone_interval = config['tonotopy']['tone_interval']
    seq_interval = config['tonotopy']['seq_interval']
    seq_per_trial = config['tonotopy']['seq_per_trial']
    rt_good_delay = config['tonotopy']['rt_good_delay']
    fs = config['sound']['fs']
    subject = save_path.split('/')[-2]

    response_key_1 = config['keys']['response_key_1']
    response_key_2 = config['keys']['response_key_2']
    trigger_key = config['keys']['trigger_key']
    
    
    LEFT_EYE = config['eyetracker']['LEFT_EYE'] 
    RIGHT_EYE = config['eyetracker']['RIGHT_EYE'] 
    BINOCULAR = config['eyetracker']['BINOCULAR'] 
    el_trial = 1 # + (round_idx-1)*cycle_per_run*len(freq_cycle)

    if do_eyetracker:
        tonetype_str = 'pt' if len(freq_cycle) == 5 else 'ct'
        edf_file_name = subject + tonetype_str + str(round_idx) + '.EDF'

        if os.path.exists(edf_file_name):
            edf_file_name = edf_file_name.split('.')[0] + 'd.EDF'

        el_tracker = func_eyetracker.get_eyetracker() 
        el_tracker.openDataFile(edf_file_name) 

        # add a preamble text (data file header)
        preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
        el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

        logger.info("Eye tracker file opened!")


    # -------------------- initialization --------------------------

    # generate all mini sequences to use later

    all_seqs = generate_all_seqs(freq_cycle, matched_dbs, ref_rms)

    # initialize window and show instruction

    d = m.open_device(dev_id[0], dev_id[0], 2) 
    interface = theForm.Interface()

    interface.update_Prompt("Now starting tonotopy task run "+str(round_idx)+"\n\nHit a key when you hear a repeating pattern\n\nPress your button to start",show=True, redraw=True)
    interface.update_Title_Center("Tonotopy scan task")
    utils.wait_for_subject(interface)

    interface.update_Prompt("Waiting for trigger (t) to start...", show=True,
                            redraw=True)  # Hit a key to start this trial
    wait = True
    while wait:
        ret = interface.get_resp()
        if ret in [trigger_key]:
            trial_start_time = datetime.now()
            wait = False

    logger.info("Trigger received at %s"%trial_start_time.strftime("%H:%M:%S.%f"))

    # -------------------- start the experiment --------------------------

    for c in range(cycle_per_run): # e.g. 5 frequencies/cycle, 8 cycles/run, 4 runs

        logger.info("------------------------")
        logger.info("Now starting cycle %d..."%(c+1)) 
        interface.update_Prompt("Now starting cycle " + str(c + 1) + "...", show=True,redraw=True)
        time.sleep(2)

        seqs_keys = all_seqs.keys()

        for i_f, cf_key in enumerate(seqs_keys):

            logger.info("*** Starting frequency %s..."%cf_key)

            if do_eyetracker:

                el_tracker = func_eyetracker.get_eyetracker()

                # show some info about the current trial on the Host PC screen
                pars_to_show = ('tonotopy', i_f, len(seqs_keys), c, cycle_per_run, round_idx+1)
                status_message = 'Link event example, %s, Trial %d/%d, Cycle %d/%d, Run number %d' % pars_to_show
                el_tracker.sendCommand("record_status_message '%s'" % status_message)

                # log a TRIALID message to mark trial start, before starting to record.
                # EyeLink Data Viewer defines the start of a trial by the TRIALID message.
                el_tracker.sendMessage("TRIALID %d" % el_trial)
                el_trial += 1

                # clear tracker display to black
                el_tracker.sendCommand("clear_screen 0")

                # switch tracker to idle mode
                el_tracker.setOfflineMode()

                error = el_tracker.startRecording(1, 1, 1, 1)
                if error:
                    return error

            if 'c' in cf_key:
                cf = int(cf_key[:-1])
            else:
                cf = int(cf_key)

            # ------------ prepare stimuli -------------
            params = {
                "cf": cf,
                "tone_duration": tone_duration,
                "tone_interval": tone_interval,
                "seq_interval": seq_interval,
                "seq_per_trial": seq_per_trial,
                "target_number_T": np.random.choice(np.arange(3)+1),
                "fs": fs
            }
            trial, trial_info = generate_trial_tonotopy_1back(params,all_seqs[cf_key])

            # ------------ open a file -------------
            file_path = save_path + subject + '-tonotopy.csv'
            fid = open(file_path, 'a')
            word_line = f"{subject},{'tonotopy'},{trial_info['cf']},{trial_info['tone_dur']},{trial_info['seq_per_trial']},{trial_info['tarN_T']},\
                    {','.join(trial_info['target_time'].astype(str))}"
            fid.write(word_line + ',')

            # ------------ run this trial -------------

            responses = []
            valid_responses = []
            valid_response_count = 0

            if do_eyetracker:
                # log a message to mark the time at which the initial display came on
                el_tracker.sendMessage("SYNCTIME")

            interface.update_Prompt("Hit a key when you hear a repeating melody",show=True, redraw=True)
            time.sleep(0.8)

            interface.update_Prompt("   ██   \n   ██   \n████████\n   ██   \n   ██   ", show=True, redraw=True)

            target_times = trial_info['target_time']
            target_times_end = target_times.copy() + rt_good_delay

            audiodev = m.open_device(*dev_id)
            s = audiodev.open_array(trial, fs)

            mix_mat = utils.apply_probe_ild(np.zeros((2, 2)),probe_ild)
            s.mix_mat = mix_mat

            if do_eyetracker:
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

            dur_ms = len(trial) / fs * 1000
            this_wait_ms = 500
            s.play()

            start_ms = interface.timestamp_ms()
            while s.is_playing:
                ret = interface.get_resp(timeout=this_wait_ms / 1000)
                this_current_ms = interface.timestamp_ms()
                this_elapsed_ms = this_current_ms - start_ms
                if ret in [response_key_1,response_key_2]:  
                    resp = np.round(this_elapsed_ms / 1000, 3)
                    responses.append(str(resp))

                    # valid responses
                    bool_1 = (resp > target_times)
                    bool_2 = (resp <= target_times_end)
                    bool_valid = bool_1 * bool_2  # same as "AND"

                    if bool_valid.any():
                        valid_responses.append(str(resp)) # why this is not being saved to file? 
                        valid_response_count += 1
                        this_tar_idx = np.where(bool_valid)[0][0]  # index of first valid target
                        target_times = np.delete(target_times, this_tar_idx)
                        target_times_end = np.delete(target_times_end, this_tar_idx)

            fid = open(file_path,'a')
            word_line = f"{','.join(responses)}" + "," + trial_start_time.strftime("%H:%M:%S.%f")
            fid.write(word_line + "\n")

            interface.update_Prompt("Waiting...", show=True, redraw=True)
            time.sleep(0.8)

            logger.info("Target num: %d"%params['target_number_T'])
            logger.info("Responses received: %s"%(','.join(responses)))
            logger.info("Valid responses: %s"%(','.join(valid_responses)))
            logger.info("***")

            if do_eyetracker:
                el_active = pylink.getEYELINK()
                el_active.stopRecording()

                el_active.sendMessage("!V TRIAL_VAR el_trial %d" % el_trial)
                el_active.sendMessage("!V TRIAL_VAR task tonotopy") 
                el_active.sendMessage("!V TRIAL_VAR trial %d" % i_f)
                el_active.sendMessage("!V TRIAL_VAR cf %d" % cf)
                el_active.sendMessage("!V TRIAL_VAR trial_per_cycle %d" % len(freq_cycle))
                el_active.sendMessage("!V TRIAL_VAR cycle %d" % c)
                el_active.sendMessage("!V TRIAL_VAR cycle_per_run %d" % cycle_per_run)
                el_active.sendMessage("!V TRIAL_VAR run_number %d" % (round_idx+1))

                el_active.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)

                ret_value = el_active.getRecordingStatus()
                if (ret_value == pylink.TRIAL_OK):
                    el_active.sendMessage("TRIAL OK")

        logger.info("Finished cycle %d!"%(c+1)) 

    logger.info("==============")
    logger.info("Run finished!")
    logger.info("==============")

    if do_eyetracker:

        logger.info("Now closing and receiving eyetracker file...")
        logger.info("File name: %s"%edf_file_name)

        # send back file after each run
        el_active = pylink.getEYELINK()
        el_active.setOfflineMode()
        
        # Close the edf data file on the Host
        el_active.closeDataFile()

        local_file_name = os.path.join(save_path, edf_file_name)
        try:
            el_active.receiveDataFile(edf_file_name, local_file_name)
        except RuntimeError as error:
            print('ERROR:', error)

        logger.info("Done!")

    #interface.destroy()

    # no return for this task, data saved in file


