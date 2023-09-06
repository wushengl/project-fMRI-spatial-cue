'''
This version includes 1-back tonotopy task, and reversal pattern recognition main task 
'''

import numpy as np
import os
import soundfile as sf
import sounddevice as sd
import random
import time
from scipy.signal import windows

import matplotlib.pylab as plt
import pdb
import itertools

import curses
import medussa as m
import psylab
from gustav.forms import rt as theForm
from datetime import datetime

import pylink

###########################################################
# run_loudness_match and run_tonotopy
###########################################################


def get_loudness_match(ref, probe, audio_dev, fs=44100, tone_dur_s=.5, tone_level_start=.5, isi_s=.2, do_addnoise=False, step=0.5, round_idx=0, key_up=curses.KEY_UP, key_dn=curses.KEY_DOWN):
    """A loudness matching task

        Parameters
        ----------
        ref : numeric or 1-d numpy array
            If ref is a number, the reference signal will be a tone of that frequency. If ref is
            an array, it is taken as the reference signal.

        probe : numeric or 1-d array, or a list of same
            If probe is not a list, it is treated similarly to the ref parameter. If it is a list,
            each item will be looped through.

        audio_dev : int
            audio_dev should be an index to an audio device, as specified by
            medussa.print_available_devices()

        fs : int
            The sampling frequency to use

        tone_dur_s : float
            If tones are being used, the tone duration, in seconds

        tone_level_start : float
            If tones are being used, the tone level to start with

        isi_s : float
            The signals are presented 1 after the other in a looped fashion. isi_s is the amount
            of time, in seconds, to wait before playing each signal again.

        Returns
        -------
        responses : float or list of floats
            If probe is a number, response is the dB difference between the reference and the probe.
            If probe is a list, then response is a list of dB differences between the reference and
            each probe in the list.

    """

    d = m.open_device(audio_dev, audio_dev, 2) # TODO: change to 8?

    if not isinstance(probe, list):
        probes = [probe]
    else:
        probes = probe.copy()

    isi_sig = np.zeros(psylab.signal.ms2samp(isi_s * 1000, fs))
    interface = theForm.Interface()

    interface.update_Prompt("Now starting loudness matching round "+str(round_idx+1)+"\n\nHit a key to continue",show=True, redraw=True)
    ret = interface.get_resp()

    interface.update_Title_Center("Loudness Matching")
    interface.update_Prompt("Match the loudness of tone 2 to tone 1\n\nHit a key to continue\n(q or / to quit)",
                            show=True, redraw=True)
    ret = interface.get_resp()

    responses = []

    if ret not in ['q', '/']:
        for probe in probes:

            interface.update_Prompt(
                "Use up & down to match\nthe loudness of tone 2 to tone 1\n\nHit enter when finished\n(q or / to quit)",
                show=True, redraw=True)
            if isinstance(probe, (int, float, complex, list)):
                if isinstance(probe,list):
                    probe_sig_1 = psylab.signal.tone(probe[0], fs, tone_dur_s*1000, amp=0.5)
                    probe_sig_2 = psylab.signal.tone(probe[1], fs, tone_dur_s*1000, amp=0.5)

                    probe_sig_1 = probe_sig_1*tone_level_start/computeRMS(probe_sig_1)
                    probe_sig_2 = probe_sig_2*tone_level_start/computeRMS(probe_sig_2)

                    probe_sig = probe_sig_1 + probe_sig_2
                    probe_sig = probe_sig*tone_level_start/computeRMS(probe_sig)
                else:
                    # Assume tone
                    probe_sig = psylab.signal.tone(probe, fs, tone_dur_s * 1000, amp=0.5)
                    probe_sig = probe_sig * tone_level_start/computeRMS(probe_sig)
                probe_sig = psylab.signal.ramps(probe_sig, fs)
            else:
                # Assume signal
                probe_sig = probe

            if isinstance(ref, (int, float, complex)):
                # Rebuild ref_sig each time, otherwise it will leak
                ref_sig = psylab.signal.tone(ref, fs, tone_dur_s * 1000, amp=0.5)
                ref_sig = ref_sig * tone_level_start/computeRMS(ref_sig)
                ref_sig = psylab.signal.ramps(ref_sig, fs)
            else:
                ref_sig = ref

            pad_ref = np.zeros(ref_sig.size)
            pad_probe = np.zeros(probe_sig.size)

            ref_sig_build = np.concatenate((ref_sig, pad_probe, isi_sig))
            probe_sig_build = np.concatenate((pad_ref, probe_sig, isi_sig))
            while ref_sig_build.size < fs * 5:
                ref_sig_build = np.concatenate((ref_sig_build, ref_sig, pad_probe, isi_sig))
                probe_sig_build = np.concatenate((probe_sig_build, pad_ref, probe_sig, isi_sig))

            sig = np.vstack((ref_sig_build, probe_sig_build)).T  # (264600, 2)
            if do_addnoise:
                sig = get_trial_with_noise(sig)

            #print(sig.shape)

            stream = d.open_array(sig, fs)
            stream.loop(True)
            mix_mat = stream.mix_mat

            mix_mat[:] = 1

            stream.mix_mat = mix_mat

            stream.play()

            quit = False
            quit_request = False
            probe_level = 1
            while not quit:
                ret = interface.get_resp()
                interface.update_Status_Left(f'Enter: {ord(ret)}; {curses.KEY_ENTER}', redraw=True)
                if ret == 'q' or ret == '/':
                    quit = True
                    quit_request = True
                elif ord(ret) in (curses.KEY_ENTER, 10, 13):
                    interface.update_Status_Right('Enter', redraw=True)
                    quit = True
                elif ord(ret) == key_dn:  # Down
                    interface.update_Status_Right('Down', redraw=True)
                    probe_level = psylab.signal.atten(probe_level, step)
                elif ord(ret) == key_up:  # Up
                    interface.update_Status_Right('Up', redraw=True)
                    probe_level = psylab.signal.atten(probe_level, -step)
                mix_mat[:, 1] = probe_level
                stream.mix_mat = mix_mat
            if quit_request:
                break
            else:
                responses.append(20 * np.log10(probe_level / 1))
            stream.stop()
        interface.destroy()
        if len(responses) == 1:
            return responses[0]
        else:
            return responses



def run_tonotopy_task(cf_pool, audio_dev, exp, do_adjust_level, matched_dbs, do_addnoise=False, cycle_per_run=8, round_idx=1):

    do_add_eyetracker = exp.user.do_add_eyetracker
    LEFT_EYE = exp.user.LEFT_EYE
    RIGHT_EYE = exp.user.RIGHT_EYE
    BINOCULAR = exp.user.BINOCULAR

    tone_duration = 0.14
    ramp_duration = 0.04
    tone_interval = 0
    seq_interval = 0.24
    seq_per_trial = 14

    all_seqs = dict()

    for i, cf in enumerate(cf_pool):
        if do_adjust_level:
            this_level_adjust = matched_dbs[i]
            if isinstance(cf, list):
                cf_ratio = 3  # complex tone 
                cf = cf[0]
                cf_key = str(cf)+'c'
            else:
                cf_ratio = None  # pure tone
                cf_key = str(cf)
            desired_rms = attenuate_db(exp.stim.ref_rms,-this_level_adjust)
        else: # not adjusting level
            if isinstance(cf, list):
                cf_ratio = 3  # complex tone
                cf = cf[0]
                cf_key = str(cf)+'c'
            else:
                cf_ratio = None  # pure tone
                cf_key = str(cf)
            desired_rms = exp.stim.desired_rms
        this_cf_seqs = generate_miniseq_4tone(cf,exp.stim.semitone_step,cf_ratio,tone_interval,tone_duration,ramp_duration,desired_rms,exp.stim.fs)
        all_seqs[cf_key] = this_cf_seqs


    # -------------------- initialize GUI --------------------------

    d = m.open_device(audio_dev, audio_dev, 2) 

    interface = theForm.Interface()

    interface.update_Prompt("Now starting tonotopy scan task round "+str(round_idx+1)+"\n\nHit a key when you hear a repeating pattern\n\nHit Space to start",show=True, redraw=True)
    interface.update_Title_Center("Tonotopy scan task")

    #interface.update_Prompt("Hit a key when you hear a repeating pattern\n\nHit Space to continue",show=True, redraw=True)
    #ret = interface.get_resp()

    wait = True
    while wait:
        ret = interface.get_resp()
        if ret in [' ', exp.quitKey]:
            wait = False

    # -------------------- start the experiment--------------------------

    for c in range(cycle_per_run): # e.g. 5 frequencies/cycle, 8 cycles/run, 4 runs


        interface.update_Prompt("Now starting cycle " + str(c + 1) + "...", show=True,redraw=True)
        #ret = interface.get_resp()
        time.sleep(2)

        seqs_keys = all_seqs.keys()

        for i_f, cf_key in enumerate(seqs_keys):

            if do_add_eyetracker:

                # TODO: currently saving tonotopy to the same file as main task, not sure if this is easy for later analysis

                el_tracker = pylink.getEYELINK()

                if(not el_tracker.isConnected() or el_tracker.breakPressed()):
                    raise RuntimeError("Eye tracker is not connected!")

                # show some info about the current trial on the Host PC screen
                pars_to_show = ('tonotopy', i_f, len(seqs_keys), c, cycle_per_run, round_idx+1)
                status_message = 'Link event example, %s, Trial %d/%d, Cycle %d/%d, Round number %d' % pars_to_show
                el_tracker.sendCommand("record_status_message '%s'" % status_message)

                # log a TRIALID message to mark trial start, before starting to record.
                # EyeLink Data Viewer defines the start of a trial by the TRIALID message.
                print("exp.user.el_trial = %d" % exp.user.el_trial)
                el_tracker.sendMessage("TRIALID %d" % exp.user.el_trial)
                exp.user.el_trial += 1

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
                "fs": exp.stim.fs
            }
            trial, trial_info = generate_trial_tonotopy_1back(params,all_seqs[cf_key])

            fid = open(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv", 'a')
            word_line = f"{exp.subjID},{exp.name},{trial_info['cf']},{trial_info['tone_dur']},{trial_info['seq_per_trial']},{trial_info['tarN_T']},\
                    {','.join(trial_info['target_time'].astype(str))}"
            fid.write(word_line + ',')

            # ------------ run this trial -------------

            interface.update_Prompt("Waiting for trigger (t)\nto start new trial...", show=True, redraw=True)  # Hit a key to start this trial
            wait = True
            while wait:
                ret = interface.get_resp()
                if ret in ['t']:
                    trial_start_time = datetime.now()
                    wait = False


            responses = []
            valid_responses = []
            valid_response_count = 0

            if do_add_eyetracker:
                # log a message to mark the time at which the initial display came on
                el_tracker.sendMessage("SYNCTIME")

            interface.update_Prompt("Hit a key when you hear a repeating melody",show=True, redraw=True)
            time.sleep(0.8)

            interface.update_Prompt("   ██   \n   ██   \n████████\n   ██   \n   ██   ", show=True, redraw=True)

            target_times = trial_info['target_time']
            target_times_end = target_times.copy() + exp.stim.rt_good_delay

            s = exp.stim.audiodev.open_array(trial, exp.stim.fs)

            mix_mat = np.zeros((2, 2)) # TODO: check if this is correct 
            if exp.stim.probe_ild > 0:
                mix_mat[0, 0] = psylab.signal.atten(1, exp.stim.probe_ild)
                mix_mat[1, 1] = 1
            else:
                mix_mat[1, 1] = psylab.signal.atten(1, -exp.stim.probe_ild)
                mix_mat[0, 0] = 1
            s.mix_mat = mix_mat

            if do_add_eyetracker:
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

            dur_ms = len(trial) / exp.stim.fs * 1000
            this_wait_ms = 500
            s.play()
            #time.sleep(1)

            start_ms = interface.timestamp_ms()
            while s.is_playing:
                ret = interface.get_resp(timeout=this_wait_ms / 1000)
                this_current_ms = interface.timestamp_ms()
                this_elapsed_ms = this_current_ms - start_ms
                if ret in ['b','y']:
                    resp = np.round(this_elapsed_ms / 1000, 3)
                    responses.append(str(resp))

                    # valid responses
                    bool_1 = (resp > target_times)
                    bool_2 = (resp <= target_times_end)
                    bool_valid = bool_1 * bool_2  # same as "AND"

                    if bool_valid.any():
                        valid_responses.append(str(resp))
                        valid_response_count += 1
                        this_tar_idx = np.where(bool_valid)[0][0]  # index of first valid target
                        target_times = np.delete(target_times, this_tar_idx)
                        target_times_end = np.delete(target_times_end, this_tar_idx)

            fid = open(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv", 'a')
            word_line = f"{','.join(responses)}" + "," + trial_start_time.strftime("%H:%M:%S.%f")
            fid.write(word_line + "\n")

            interface.update_Prompt("Waiting...", show=True, redraw=True)
            time.sleep(0.8)

            if do_add_eyetracker:
                el_active = pylink.getEYELINK()
                el_active.stopRecording()

                el_active.sendMessage("!V TRIAL_VAR el_trial %d" % exp.user.el_trial)
                el_active.sendMessage("!V TRIAL_VAR task tonotopy") 
                el_active.sendMessage("!V TRIAL_VAR trial %d" % i_f)
                el_active.sendMessage("!V TRIAL_VAR cf %d" % cf)
                el_active.sendMessage("!V TRIAL_VAR trial_per_cycle %d" % len(cf_pool))
                el_active.sendMessage("!V TRIAL_VAR cycle %d" % c)
                el_active.sendMessage("!V TRIAL_VAR cycle_per_run %d" % cycle_per_run)
                el_active.sendMessage("!V TRIAL_VAR run_number %d" % (round_idx+1))

                el_active.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)

                ret_value = el_active.getRecordingStatus()
                if (ret_value == pylink.TRIAL_OK):
                    el_active.sendMessage("TRIAL OK")

    #interface.destroy()

    

    # no return for this task, data saved in file




###########################################################
# utils.py
###########################################################


def computeRMS(sig):
    return np.sqrt(np.mean(sig**2))


def attenuate_db(sig,db):
    '''attenuate sig by db'''
    out = sig * np.exp(np.float32(-db)/8.6860)
    return out


def generate_tone(f_l,f_h,duration,ramp,desired_rms,fs):

    sample_len = int(fs * duration)

    # create samples
    samples_low = (np.sin(2 * np.pi * np.arange(sample_len) * f_l / fs)).astype(np.float32)
    if f_h:
        samples_high =  (np.sin(2 * np.pi * np.arange(sample_len) * f_h / fs)).astype(np.float32)
        samples = samples_low + samples_high
    else:
        samples = samples_low

    # adjust rms
    samples = samples*desired_rms/computeRMS(samples) # np.max(samples)
    
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


def parse_trial_info_tonotopy(trial_info):

    '''
    trial_info = {"cf": cf,\
                  "tone_dur":tone_duration,\
                  "seq_per_trial":seq_per_trial,\
                  "tarN_T": tarN_T,\
                  "target_index":target_index,\
                  "target_time":target_time}
    '''

    cf_str = str(trial_info["cf"]) 
    tone_dur_str = '{}d{}'.format(*str(trial_info["tone_dur"]).split(".")) 
    seq_per_trial_str = str(trial_info["seq_per_trial"]) + "seq"
    tarN_T_str = str(trial_info["tarN_T"]) + "tarT"
    tar_loc_str = "tarlocs" + ('').join(list(np.array(trial_info["target_index"]).astype(str)))

    trial_info_str = '-'.join([cf_str,tone_dur_str,seq_per_trial_str,tarN_T_str,tar_loc_str]) 

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



###########################################################
# func_spatialization.py
###########################################################


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
        seq_ild_l = np.concatenate((attenuate_db(sig,-ild/2).reshape(-1,1),attenuate_db(sig,ild/2).reshape(-1,1)),axis=1)
        seq_ild_r = np.concatenate((attenuate_db(sig,ild/2).reshape(-1,1),attenuate_db(sig,-ild/2).reshape(-1,1)),axis=1)

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



###########################################################
# func_stimuli.py
###########################################################



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

    if cf_ratio:
        tone_1_high = tone_1_low * cf_ratio
        tone_2_high = tone_2_low * cf_ratio
        tone_3_high = tone_3_low * cf_ratio
    else:
        tone_1_high = None
        tone_2_high = None
        tone_3_high = None

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


def generate_miniseq_4tone(cf,step,cf_ratio,interval,duration,ramp,volume,fs):  
    '''
    creating sequence dict for 4-tone sequences, returning single channel sequences 
    '''

    tone_1_low = cf / step
    tone_2_low = cf
    tone_3_low = cf * step

    if cf_ratio:
        tone_1_high = tone_1_low * cf_ratio
        tone_2_high = tone_2_low * cf_ratio
        tone_3_high = tone_3_low * cf_ratio
    else:
        tone_1_high = None
        tone_2_high = None
        tone_3_high = None

    tone_1 = generate_tone(tone_1_low,tone_1_high,duration,ramp,volume,fs)
    tone_2 = generate_tone(tone_2_low,tone_2_high,duration,ramp,volume,fs)
    tone_3 = generate_tone(tone_3_low,tone_3_high,duration,ramp,volume,fs)
    interval_samps = np.zeros((int(fs * interval)))

    tone_pool = {
        'tone1':tone_1,
        'tone2':tone_2,
        'tone3':tone_3
    }

    tone_keys = list(tone_pool.keys())
    combinations = list(itertools.product(tone_keys, repeat=4)) # returns a list of tuples containing all possible combinations 
    seq_dict = {}

    for i, seq in enumerate(combinations):
        
        tone_loc1 = tone_pool[seq[0]]
        tone_loc2 = tone_pool[seq[1]]
        tone_loc3 = tone_pool[seq[2]]
        tone_loc4 = tone_pool[seq[3]]
        this_seq = np.concatenate((tone_loc1,interval_samps,tone_loc2,interval_samps,tone_loc3,interval_samps,tone_loc4))

        seq_key = 'seq' + str(i+1)
        seq_dict[seq_key] = this_seq

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


def generate_trial_tonotopy(params,seq_dict):
    '''
    This function is used for generating a tonotopy trial with task being find reversal pattern (zigzag pattern) from target direction.  
    
    Each trial contains 1 stream. The number and locations of targets (and distractors) are randomly selected. 
    The targets are randomly selected from zigzag pattern pools. The rest mini-sequences are selected from up/down pattern pools. 

    ====================
    Inputs:
    - params: a dictionary containing all parameters needed to customize a trial, except cue related variables
    - seq_dict: a dictionary containing all sequences with center frequency cf, key example: "up_seq_1_l"

    Outputs:
    - trial: a N*2 numpy array containing the trial 
    - trial_info: an dictionary include all information about one trial 
    '''

    # -------------- preparation ----------------

    # read parameters from params

    '''
    params = {
    "cf": cf,
    "tone_duration": tone_duration,
    "tone_interval": tone_interval,
    "seq_interval": seq_interval,
    "seq_per_trial": seq_per_trial,
    "target_number_T": target_number_T,
    "fs":fs
}
    '''

    cf = params["cf"]
    tone_duration = params["tone_duration"]
    tone_interval = params["tone_interval"]
    seq_interval = params["seq_interval"]
    seq_per_trial = params["seq_per_trial"]
    tarN_T = params["target_number_T"]
    fs = params["fs"]
    
    # prepare zigzag and non-zigzag sequence pools, where each "pool" is a list containing all seq names for seq in that pool 

    seq_pool_up = np.array(['up_seq_'+str(n+1) for n in range(7)])
    seq_pool_down = np.array(['down_seq_'+str(n+1) for n in range(7)])
    seq_pool_zigzag = np.array(['zigzag_seq_'+str(n+1) for n in range(10)])
    seq_pool_nonzigzag = np.concatenate((seq_pool_up,seq_pool_down))


    # -------------- create trial without cue ----------------

    target_num = tarN_T

    # location of zigzag patterns in each stream
    target_location_idxes = random.sample(range(0,seq_per_trial),target_num)

    # randomly select zigzag patterns for target and distractor streams
    target_pattern_idxes = random.sample(range(len(seq_pool_zigzag)),target_num)
    target_nonpattern_idxes = random.choices(range(len(seq_pool_nonzigzag)),k=int(seq_per_trial-target_num)) # nonzigzag pool size = 14, nonzigzag pattern num = 14, 15,16

    # create an array containing seq names 
    target_stream_seq_order = (99*np.ones(seq_per_trial).astype(int)).astype(str)
    target_stream_seq_order[np.array(target_location_idxes)] = seq_pool_zigzag[target_pattern_idxes] 
    target_stream_seq_order[target_stream_seq_order == "99"] = seq_pool_nonzigzag[target_nonpattern_idxes] 

    # padding between intervals 
    seq_interval_padding = np.zeros((int(seq_interval*fs),2)) 


    # start generating the trial 
    target_stream = np.empty((0,2))
    target_seq_dict = seq_dict

    for i in range(seq_per_trial):

        this_target_key = target_stream_seq_order[i]
        this_target = np.tile(target_seq_dict[this_target_key].reshape(-1,1),(1, 2)) # mono to stereo

        # here always set target stream leading is fine, since we'll randomly switch pairs later
        target_stream = np.concatenate((target_stream,this_target),axis=0) 

        # add interval between mini-sequences 
        target_stream = np.concatenate((target_stream,seq_interval_padding),axis=0)

    trial = target_stream 

    # -------------- randomly switch pair ----------------

    # also initialize target time 
    target_location_idxes.sort()
    target_index = np.array(target_location_idxes)

    # target time is computed if didn't switch last pair in the miniseq 
    seq_block_time = tone_duration*3
    target_time = target_index*(seq_block_time + seq_interval) + tone_duration*2

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
    tone_interval = params["tone_interval"]
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

    #pdb.set_trace()
    '''
    # location of zigzag patterns in each stream
    target_location_idxes = random.sample(range(0,seq_per_trial),target_num)

    # randomly select zigzag patterns for target and distractor streams
    target_pattern_idxes = random.sample(range(len(seq_pool_zigzag)),target_num)
    target_nonpattern_idxes = random.choices(range(len(seq_pool_nonzigzag)),k=int(seq_per_trial-target_num)) # nonzigzag pool size = 14, nonzigzag pattern num = 14, 15,16

    # create an array containing seq names 
    target_stream_seq_order = (99*np.ones(seq_per_trial).astype(int)).astype(str)
    target_stream_seq_order[np.array(target_location_idxes)] = seq_pool_zigzag[target_pattern_idxes] 
    target_stream_seq_order[target_stream_seq_order == "99"] = seq_pool_nonzigzag[target_nonpattern_idxes] '''

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



###########################################################
# run_generate_stimuli.py + run_generate_tonotopy_1back_4tone.py
###########################################################


if __name__ == '__main__':


    # ------------ main task setting -----------------

    f0 = 220 # Hz
    fs = 44100 # Hz

    low_pitch_cf_1 = f0 # cf for center frequency
    low_pitch_cf_2 = 3*f0
    high_pitch_cf_1 = 2*f0
    high_pitch_cf_2 = 6*f0 

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
    desired_rms = 0.3    # desired rms of pure tone, note that to avoid peak value (greater than rms) go above 1, this value should be set to smaller

    cue2stim_interval = 0.5

    # updated for matched level 
    ref_rms = 0.3        # this should be the same as tonotopy task ref_rms
    matched_dbs_complex = [-0.5,-0.3]  # this is random number for testing 
    do_adjust_level = True  # don't have both do_loudness_match and do_adjust_level in this script, it's in gustav script 

    ############### create minisequence ###############

    if do_adjust_level:
        desired_rms_low = matched_dbs_complex[0]
        desired_rms_high = matched_dbs_complex[1]
    else:
        desired_rms_low = desired_rms
        desired_rms_high = desired_rms

    low_pitch_seqs = generate_miniseq(low_pitch_cf_1, semitone_step, low_pitch_cf_ratio, tone_interval, tone_duration, ramp_duration, desired_rms_low, fs)
    high_pitch_seqs = generate_miniseq(high_pitch_cf_1, semitone_step, high_pitch_cf_ratio, tone_interval, tone_duration, ramp_duration, desired_rms_high, fs)

    # sd.play(low_pitch_seqs['up_seq_1'], fs) 

    #compare_samerms = np.concatenate((low_pitch_seqs['up_seq_1'],np.zeros(int(0.25*44100)),high_pitch_seqs['up_seq_1']))
    #sf.write("compare-high-low-same-rms.wav",compare_samerms,fs)
    #compare_samepeak = np.concatenate((low_pitch_seqs['up_seq_1'],np.zeros(int(0.25*44100)),high_pitch_seqs['up_seq_1']*np.max(low_pitch_seqs['up_seq_1'])/np.max(high_pitch_seqs['up_seq_1'])))
    #sf.write("compare-high-low-same-peak.wav",compare_samepeak,fs)


    ############### spatialize minisequence ############ 

    if do_adjust_level:
        low_pitch_seqs_ILD, low_pitch_seqs_ITD = spatialize_seq_matched(low_pitch_seqs,ild,itd,fs)
        high_pitch_seqs_ILD, high_pitch_seqs_ITD = spatialize_seq_matched(high_pitch_seqs,ild,itd,fs)
    else:
        low_pitch_seqs_ILD, low_pitch_seqs_ITD = spatialize_seq(low_pitch_seqs,ild,itd,fs)
        high_pitch_seqs_ILD, high_pitch_seqs_ITD = spatialize_seq(high_pitch_seqs,ild,itd,fs)

    # sd.play(low_pitch_seqs_ILD['up_seq_1_l'], fs) 
    # sd.play(low_pitch_seqs_ITD['up_seq_1_l'], fs) 

    #compare_spacond_1stream = np.concatenate((low_pitch_seqs_ILD['up_seq_1_l'],np.zeros((int(0.25*44100),2)),low_pitch_seqs_ITD['up_seq_1_l']),axis=0)
    #sf.write("compare-ild-itd-mean-rms.wav",compare_spacond_1stream,fs)

    pdb.set_trace()

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


    # ------------ tonotopy task setting -----------------


    cf_pool = [300,566,1068,2016,3805,[220,3*220],[440,3*440]]      # Hz
    fs = 44100                              # Hz
    cf_ratio = None                         # if None, generate tone with only cf; if not None, will generate complex tone with 2 frequencies 
    semitone_step = 2**(1/12)

    tone_duration = 0.14                    # s
    ramp_duration = 0.04                    # s, total ramp length in a tone (onset and offset)
    tone_interval = 0                       # s, set to 0 results in no interval between tones within a sequence 
    seq_interval = 0.24                     # s, time between sequences 

    seq_per_trial = 14                      # to match with Fred and Holt 11.2s per trial (4-tone sequence with 14 sequences per trial)

    target_number_T = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for target stream

    desired_rms = 0.3                            # peak of tone samples could be around 1.5, set volume to be around 0.6 times the peak value you want 

    # updated for perceptual matched loudness
    ref_rms = 0.3
    matched_dbs = [-5,-3,-1,0,-1,-10,-10]   # TODO: this is example for testing, should should be list/array with 5 elements, negative means need attenuate 
    do_adjust_level = True


    ############### create minisequence ###############


    all_seqs = dict()

    for i, cf in enumerate(cf_pool):
        if do_adjust_level:
            this_level_adjust = matched_dbs[i]
            if isinstance(cf, list):
                cf_ratio = 3  # complex tone 
                cf = cf[0]
                cf_key = str(cf)+'c'
            else:
                cf_ratio = None  # pure tone
                cf_key = str(cf)
            desired_rms = attenuate_db(ref_rms,-this_level_adjust) 
        this_cf_seqs = generate_miniseq_4tone(cf,semitone_step,cf_ratio,tone_interval,tone_duration,ramp_duration,desired_rms,fs) 
        all_seqs[cf_key] = this_cf_seqs

    '''keys: 300,566,1068,2016,3805,220c,440c'''
    # sd.play(all_seqs['300']['seq1'], fs)
    #pdb.set_trace()
    #print(20*np.log10(computeRMS(all_seqs['300']['seq1'])))
    #print(20*np.log10(computeRMS(all_seqs['566']['seq1'])))
    #print(20*np.log10(computeRMS(all_seqs['1068']['seq1'])))
    #print(20*np.log10(computeRMS(all_seqs['2016']['seq1'])))
    #print(20*np.log10(computeRMS(all_seqs['3805']['seq1'])))


    ############### create trials ######################
    ''''''

    cf = 566

    params = {
        "cf": cf,
        "tone_duration": tone_duration,
        "tone_interval": tone_interval,
        "seq_interval": seq_interval,
        "seq_per_trial": seq_per_trial,
        "target_number_T": target_number_T,
        "fs":fs
    }


    # ------------ find zigzag clean task -----------------

    # cf_pool = [300,566,1068,2016,3805] 

    test_trial, trial_info = generate_trial_tonotopy_1back(params,all_seqs[str(cf)])
    trial_info_str = parse_trial_info_tonotopy(trial_info)

    save_prefix = '../stimuli/tonotopy-1back-trial-'
    save_path = get_unrepeated_filename(trial_info_str,save_prefix)

    pdb.set_trace()

    sd.play(test_trial,fs)
    sf.write(save_path,test_trial,fs)

    pdb.set_trace()

    test_trial_with_noise = get_trial_with_noise(test_trial) # TODO: this function is always adding noise with SNR=0dB, not adding noise fixed as desired_rms
    save_prefix_noisy = '../stimuli/noisy-tonotopy-1back-trial-'
    save_path_noisy = get_unrepeated_filename(trial_info_str,save_prefix_noisy)

    sd.play(test_trial_with_noise,fs)
    sf.write(save_path_noisy,test_trial_with_noise,fs)

    pdb.set_trace()





