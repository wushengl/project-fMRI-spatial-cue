# -*- coding: utf-8 -*-

# Copyright (c) 2010-2012 Christopher Brown
#
# This file is part of Psylab.
#
# Psylab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Psylab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Psylab.  If not, see <http://www.gnu.org/licenses/>.
#
# Bug reports, bug fixes, suggestions, enhancements, or other 
# contributions are welcome. Go to http://code.google.com/p/psylab/ 
# for more information and to contribute. Or send an e-mail to: 
# cbrown1@pitt.edu.
# 
# Psylab is a collection of Python modules for handling various aspects 
# of psychophysical experimentation. Python is a powerful programming  
# language that is free, open-source, easy-to-learn, and cross-platform, 
# thus making it extremely well-suited to scientific applications. 
# There are countless modules written by other scientists that are  
# freely available, making Python a good choice regardless of your  
# particular field. Consider using Python as your scientific platform.
# 

# A Gustav experiment file!

import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import psylab
import gustav
from gustav.forms import rt as theForm
import medussa as m
import mri_tones
import random
from datetime import datetime
import subprocess
import glob
import logging

def setup(exp):

    #global dev_id
    
    # Machine-specific settings
    machine = psylab.config.local_settings(conf_file='config/psylab.conf')
    workdir = machine.get_path('workdir')
    dev_name = machine.get_str('audiodev_name')
    dev_ch = machine.get_int('audiodev_ch')
    devs = m.get_available_devices()
    dev_id = None
    for i,di in enumerate(devs):
        name = psylab.string.as_str(di.name)
        ch = di.maxOutputChannels
        if name.startswith(dev_name) and ch == 8:
            dev_id = i,i,ch
    if dev_id:
        exp.stim.audiodev = m.open_device(*dev_id)
    else:
        raise Exception(f"The audio device {dev_name} was not found")

    # General Experimental Variables
    exp.name = 'ild-fmri-2'
    exp.method = 'constant' # 'constant' for constant stimuli, or 'adaptive' for a staircase procedure (SRT, etc)
    # TODO: move logstring and datastring vars out of exp and into either method or experiment, so they can be properly enumerated at startup

    exp.logFile = os.path.join(workdir,'logs','$name_$date.log')  # Name and date vars only on logfile name
    exp.logConsoleDelay = True # Set to True if using a curses form
    exp.dataFile = os.path.join(workdir,'data','$name.csv')
    exp.recordData = True
    exp.dataString_header = "# A datafile created by Gustav!\n# \n# Experiment: $name\n# \n# $note\n# \n# $comments\n# \n\nS,Trial,Date,Block,Condition,@currentvars[],Times\n"
    exp.dataString_post_trial = "$subj,$trial,$date,$block,$condition,$currentvars[],$user[response]\n"
    exp.logString_pre_exp = "\nExperiment $name running subject $subj started at $time on $date\n"
    exp.logString_post_exp = "\nExperiment $name running subject $subj ended at $time on $date; Overall hit rate: $user[valid_response_count] / $user[target_number_count]\n"
    exp.logString_pre_block = "\n  Block $block of $blocks started at $time; Condition: $condition ; $currentvarsvals[' ; ']\n"
    exp.logString_post_trial = "" # "    Trial $trial, target stimulus: $user[trial_stimbase], valid response count: $user[valid_response_count] KWs correct: $response / possible: $user[trial_kwp] ($user[block_kwc] / $user[block_kwp]: $user[block_pc] %)\n"
    exp.logString_post_block = "  Block $block of $blocks ended at $time; Target number: $stim[tar_num]; Responses: $user[response]; Valid responses: $user[valid_response]\n" # Condition: $condition ; $currentvarsvals[' ; ']
    exp.frontend = 'tk'
    exp.debug = False

    exp.validKeys = '0,1,2,3,4,5,6,7,8,9'.split(',')
    exp.quitKey = '/'
    exp.note = 'no notes'
    exp.comments = '''
    no comments
    '''

    # get experiment info from keyboard
    if not exp.subjID:
        exp.subjID = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Enter a Subject ID:')

    if True:
        run_mode = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Enter a running mode:')


    # fixed experiment setting
    exp.stim.training_threshold = .5
    exp.stim.rt_good_delay = 1.5

    exp.stim.fs = 44100.
    f0 = 220 # Hz

    low_pitch_cf_1 = f0 # cf for center frequency
    low_pitch_cf_2 = 3*f0
    high_pitch_cf_1 = 2*f0
    high_pitch_cf_2 = 6*f0  # pilot data used 4*f0 for higher frequency component in high pitch tone

    low_pitch_cf_ratio = int(low_pitch_cf_2/low_pitch_cf_1)
    high_pitch_cf_ratio = int(high_pitch_cf_2/high_pitch_cf_1)

    exp.stim.tone_duration = 0.25 # s
    exp.stim.ramp_duration = 0.04 # s (this is total length for on and off ramps) # TODO weird tone with too short ramp

    exp.stim.tone_interval = exp.stim.tone_duration # this is offset to onset interval
    exp.stim.seq_interval = 0.8
    exp.stim.cue2stim_interval = 0.5

    exp.stim.seq_per_trial = 10
    #target_number_T = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for target stream 
    #target_number_D = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for distractor stream 

    # use matched ILD to ITD for all subjects 
    exp.stim.itd = 500e-6    # 685e-6 # s
    exp.stim.ild = 10        #15 # db

    exp.stim.semitone_step = 2**(1/12)
    exp.stim.volume = 0.3    # peak of tone samples could be around 1.5, set volume to be around 0.6 times the peak value you want 
    exp.stim.desired_rms = 0.3
    exp.stim.ref_rms = 0.3

    #global do_loudness_match, do_adjust_level, loudness_match_times, diff_thre, do_run_tonotopy, tonotopy_f_pool, tonotopy_matching_pool, matching_pool
    do_loudness_match = True
    do_adjust_level = True
    loudness_match_times = 2
    diff_thre = 25

    do_run_tonotopy = True
    tonotopy_f_pool = [300,566,1068,2016,3805]
    tonotopy_matching_pool = [300,566,1068,3805]
    complextone_matching_pool = [[low_pitch_cf_1,low_pitch_cf_2],[high_pitch_cf_1,high_pitch_cf_2]]
    matching_pool = tonotopy_matching_pool + complextone_matching_pool


    """EXPERIMENT VARIABLES
        There are 2 kinds of variables: factorial and covariable

        Levels added as 'factorial' variables will be factorialized with each
        other. So, if you have 2 fact variables A & B, each with 3 levels, you
        will end up with 9 conditions: A1B1, A1B2, A1B3, A2B1 etc..

        Levels added as 'covariable' variables will simply be listed (in parallel
        with the corresponding levels from the other variables) in the order
        specified. So, if you have 2 'covariable' variables A & B, each with 3
        levels, you will end up with 3 conditions: A1B1, A2B2, and A3B3. All
        'covariable' variables must have either the same number of levels, or
        exactly one level. When only one level is specified, that level will
        be used in all 'covariable' conditions. Eg., A1B1, A2B1, A3B1, etc.

        You can use both types of variables in the same experiment, but both
        factorial and covariable must contain exactly the same set of variable
        names. factorial levels are processed first, covariable levels are added
        at the end.

        Both factorial and covariable are Python ordered dicts, where the keys 
        are variable names, and the values are lists of levels. During the 
        experiment, you have access to the current level of each variable. For 
        example, if you have the following variable:
        
        exp.var.factorial['target'] = ['Male', 'Female']
        
        Then, you can find out what the level is at any point in the experiment 
        with exp.var.current['target'], which would return either 'Male' or 
        'Female' depending on what the condition happened to be. This is 
        probably most useful to generate your stimuli, eg., in the pre_trial 
        function. 
    """

    exp.var.factorial['cue'] = [
                                      'ITD500',
                                      'ILD10',
                                     ]

    exp.var.factorial['isTargetLeft'] = [
                                        'True',
                                        'False',
                                    ]

    exp.var.factorial['isLowLeft'] = [
                                        'True',
                                        'False',
                                    ]

    exp.var.factorial['noise'] = [
                                        'yes',
                                        #'no',
                                    ]


    """CONSTANT METHOD VARIABLES
        The method of constant limits requires three variables to be set.
            trialsperblock
            startblock [crash recovery]
            starttrial [crash recovery]
    """
    exp.var.constant = {
        'trialsperblock' : 1,
        'startblock' : 1,
        'starttrial' : 1,
        }
    
    """CONDITION PRESENTATION ORDER
        Use 'prompt' to prompt for condition on each block, 'random' to randomize
        condition order, 'menu' to be able to choose from a list of conditions at
        the start of a run, 'natural' to use natural order (1:end), or a
        print-range style string to specify the order ('1-10, 12, 15'). You can
        make the first item in the print range 'random' to randomize the specified
        range.
    """

    conds_pool = ['1','2','3','4','5','6','7','8']
    run_num = 10
    all_conds = []
    for i in range(run_num):
        random.shuffle(conds_pool)
        all_conds += conds_pool

    test_order = ','.join(all_conds)  # this version makes sure per block (8 trial) runs each condition once
    train_order = ','.join(conds_pool)  # this is already shuffled, containing all 8 conditions  # '6,5,2,1,7,4,3,8'  # use this as exp.var.order for training, also set noise to 'no'
    debug_order = '1,2'

    if run_mode == 'train':
        exp.var.order = train_order
    elif run_mode == 'debug':
        exp.var.order = debug_order
    else:
        exp.var.order = test_order


    """IGNORE CONDITIONS
        A list of condition numbers to ignore. These conditions will not be
        reflected in the total number of conditions to be run, etc. They will
        simply be skipped as they are encountered during a session.
    """
    exp.var.ignore = []

    """USER VARIABLES
        Add any additional variables you need here
    """

    total_trial_num = len(exp.var.order.split(','))
    print('Total trial number: %d'%total_trial_num)

    exp.var.constant['total_trial_num'] = total_trial_num
    exp.user.valid_response_count = 0
    exp.user.target_number_count = 0


    """SOME EXPERIMENTS BEFORE MAIN TASK
        Opening new command window, running loudness matching and tonotopy
    """

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M")
    pretask_log_path = './logs/log_pretask_'+time_str+'.log'
    logger = logging.getLogger('logger_pretask')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(pretask_log_path)
    logger.addHandler(file_handler)

    print('Open a new powershell and see the log with:\nGet-Content -Path '+pretask_log_path+' -Wait')

    input("Press Enter to continue...")


    ############### loudness matching ###############

    if do_loudness_match:
        logger.info("--------------------------------------------------------------")
        logger.info("Now start loudness matching...")
        logger.info("Matching pool: "+str(matching_pool))
        all_matched_levels = np.empty((0,len(matching_pool)))
        for i in range(loudness_match_times): # minimum match times
            logger.info("Now start matching round "+str(i+1))

            # matched_levels is a list of matched level difference, without 2016Hz
            matched_levels = mri_tones.get_loudness_match(2016,matching_pool,dev_id[0],tone_level_start=exp.stim.ref_rms,do_addnoise=True,round_idx=i)
            all_matched_levels = np.concatenate((all_matched_levels,np.array(matched_levels).reshape(1,-1)),axis=0)
            print(matched_levels)

            # Open a new command window and run this: Get-Content -Path "./logs/logfiletest.log" -Wait
            logger.info(matched_levels)

        # remove extremely large value anyway, or it might converge even variance is large and final averaged data would be not reliable
        all_matched_levels[abs(all_matched_levels) >= diff_thre] = np.nan

        # compute mean and std with minimum matched data
        last_mean = np.nanmean(all_matched_levels,axis=0)
        last_std = np.nanstd(all_matched_levels-last_mean) # ,axis=0, taking all samples' std to avoid harder convergence for low variance frequencies in baseline

        # create initial boundaries
        upper_bound = last_mean + last_std
        lower_bound = last_mean - last_std

        conv_check = np.zeros(len(last_mean)).astype(int)
        this_matching_pool = np.array(matching_pool.copy())

        logger.info("Finished baseline matching, now start adaptive matching...")

        while np.sum(conv_check)<len(conv_check): # stop when conv_check = len(conv_check), which means all 1

            print('-------------------------------------------')
            print('Starting new adaptive matching...')
            print('Current convergence status: ',conv_check.astype(bool))
            print('-------------------------------------------')

            logger.info("----------------------- new matching -------------------------")
            logger.info("Current convergence status: " + str(conv_check.astype(bool)))

            # get matching pool for this round
            this_matching_pool = np.array(matching_pool)[(1-conv_check).astype(bool)]

            # get new sample
            i+=1
            this_matched_levels = mri_tones.get_loudness_match(2016, list(this_matching_pool), dev_id[0],tone_level_start=exp.stim.ref_rms, do_addnoise=True,round_idx=i)

            # make full matched sample
            matched_levels = np.zeros(len(last_mean))
            matched_levels[conv_check.astype(bool)] = last_mean[conv_check.astype(bool)]  # set matched values to last mean if already converges
            matched_levels[(1-conv_check).astype(bool)] = this_matched_levels              # add this matched values to this sample

            all_matched_levels = np.concatenate((all_matched_levels, np.array(matched_levels).reshape(1, -1)), axis=0)
            print(matched_levels)

            # check if new sample within boundary
            conv_check = (matched_levels>=lower_bound).astype(int) * (matched_levels<=upper_bound).astype(int)

            logger.info("This matching pool: " + str(this_matching_pool))
            logger.info("New matched levels: " + str(this_matched_levels))
            logger.info("Upper bounds: " + str(upper_bound))
            logger.info("Lower bounds: " + str(lower_bound))
            logger.info("Updated convergence status: " + str(conv_check.astype(bool)))

            # update boundary
            last_mean = np.nanmean(all_matched_levels,axis=0)
            last_std = np.nanstd(all_matched_levels-last_mean) # ,axis=0
            upper_bound = last_mean + last_std
            lower_bound = last_mean - last_std


        matched_levels_ave = np.nanmean(all_matched_levels,axis=0)

        logger.info("--------------------------------------------------------------")
        logger.info("Matching finished!")
        logger.info("Averaged matched levels: " + str(matched_levels_ave))
        logger.info("--------------------------------------------------------------")

        ref_index = 3
        matched_levels_ave = np.insert(matched_levels_ave,ref_index,0) # inset level 0 for reference frequency
        print("Matched level differences:")
        print(matched_levels_ave)

        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M")
        matched_levels_save_path = os.path.join(workdir, 'data', exp.subjID + '-matched_levels-' + time_str + '.csv')
        np.savetxt(matched_levels_save_path,all_matched_levels,delimiter=',')
    else:
        if do_adjust_level:
            file_names = glob.glob(workdir+'\\'+'data'+'\\*' + exp.subjID + '-matched_levels*')
            print(file_names)  # return is a list
            matched_levels_all = np.loadtxt(file_names[-1], delimiter=',')  # if multiple, get the latest one

            matched_levels_all[abs(matched_levels_all)>=diff_thre] = np.nan
            matched_levels_ave = np.nanmean(matched_levels_all, axis=0)

            ref_index = 3
            matched_levels_ave = np.insert(matched_levels_ave, ref_index, 0)  # inset level 0 for reference frequency
            print("Matched level differences:")
            print(matched_levels_ave)
        else:
            matched_levels_ave = np.zeros(7)


    ############### tonotopy scan task ###############

    if not os.path.isfile(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv"):
        fid = open(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv", 'a')
        word_line = f"SubjectID,Trial,frequency,toneDur,seqPerTrial,TargetNum,TargetTime,ResponseTime"  # TODO: change header line here
        fid.write(word_line + "\n")
        fid.flush()

    cycle_per_run = 5
    run_num = 4

    if do_run_tonotopy:
        logger.info("Now start tonotopy task...")
        logger.info("cycles per run: "+str(cycle_per_run))
        logger.info("run number: "+str(run_num))
        for i in range(run_num):
            logger.info("Now running round "+str(i+1))
            #mri_tones.run_tonotopy_task(tonotopy_f_pool, dev_id[0], exp, do_adjust_level, matched_levels_ave[:len(tonotopy_f_pool)], cycle_per_run=cycle_per_run, round_idx=i)
            mri_tones.run_tonotopy_task(matching_pool, dev_id[0], exp, do_adjust_level,matched_levels_ave, cycle_per_run=cycle_per_run,round_idx=i)


    ############### create minisequence ###############
    if do_adjust_level:
        desired_rms_low = mri_tones.attenuate_db(exp.stim.ref_rms, -1*matched_levels_ave[-2])
        desired_rms_high = mri_tones.attenuate_db(exp.stim.ref_rms, -1*matched_levels_ave[-1])
    else:
        desired_rms_low = exp.stim.desired_rms
        desired_rms_high = exp.stim.desired_rms

    low_pitch_seqs = mri_tones.generate_miniseq(low_pitch_cf_1, exp.stim.semitone_step, low_pitch_cf_ratio,exp.stim.tone_interval, exp.stim.tone_duration,exp.stim.ramp_duration, desired_rms_low, exp.stim.fs)
    high_pitch_seqs = mri_tones.generate_miniseq(high_pitch_cf_1, exp.stim.semitone_step, high_pitch_cf_ratio,exp.stim.tone_interval, exp.stim.tone_duration,exp.stim.ramp_duration, desired_rms_high, exp.stim.fs)

    ############### spatialize minisequence ############
    if do_adjust_level:  # ITD condition use matched level, ILD condition use +/- 0.5*ILD for louder/weaker channel
        exp.stim.low_pitch_seqs_ILD, exp.stim.low_pitch_seqs_ITD = mri_tones.spatialize_seq_matched(low_pitch_seqs,exp.stim.ild, exp.stim.itd,exp.stim.fs)
        exp.stim.high_pitch_seqs_ILD, exp.stim.high_pitch_seqs_ITD = mri_tones.spatialize_seq_matched(high_pitch_seqs,exp.stim.ild,exp.stim.itd, exp.stim.fs)
    else:
        exp.stim.low_pitch_seqs_ILD, exp.stim.low_pitch_seqs_ITD = mri_tones.spatialize_seq(low_pitch_seqs,exp.stim.ild, exp.stim.itd,exp.stim.fs)
        exp.stim.high_pitch_seqs_ILD, exp.stim.high_pitch_seqs_ITD = mri_tones.spatialize_seq(high_pitch_seqs,exp.stim.ild,exp.stim.itd, exp.stim.fs)


    if not os.path.isfile(f"data/{exp.name}_times_{exp.subjID}.csv"):

        fid = open(f"data/{exp.name}_times_{exp.subjID}.csv", 'a')
        # word_line = f"{exp.run.trial},{trial_info['isTargetLeft']},{','.join(trial_info['target_time'])}"
        word_line = f"SubjectID,Trial,SpatialCond,isTargetLeft,isLowLeft,TargetNum,TargetTime,ResponseTime"  # TODO: change header line here
        fid.write(word_line + "\n")

def pre_exp(exp):
    try:
        ''' # TODO: add this to setup, debug why it's not staying, then add log info for loudness matching '''
        # start a new powershell showing log info
        log_file_path = exp.logFile  # r'logs\ild-fmri-2_2023-05-06.log'
        command = f'start powershell.exe -Command "Get-Content -Path "{log_file_path}" -Wait'
        subprocess.Popen(command, shell=True)

        exp.interface = theForm.Interface()
        exp.interface.update_Title_Center(exp.name)
        exp.interface.update_Title_Right(f"S {exp.subjID}", redraw=False)
        exp.interface.update_Prompt("Hit a key to begin", show=True, redraw=True)
        ret = exp.interface.get_resp()
        if ret == exp.quitKey:
            exp.gustav_is_go = False

    except Exception as e:
        exp.interface.destroy()
        raise e


def pre_block(exp):
    try:
        exp.user.block_kwp = 0
        exp.user.block_kwc = 0
        exp.user.block_pc = 0.
        exp.user.pract = 1
        exp.interface.update_Status_Left(f"Block {exp.run.block+1} of {exp.run.nblocks}")
    except Exception as e:
        exp.interface.destroy()
        raise e

"""PRE_TRIAL
    This function gets called on every trial to generate the stimulus, and do
    any other processing you need. All settings and variables are available. 
    For the current level of a variable, use exp.var.current['varname'].
"""
def pre_trial(exp):
    try:
        exp.interface.update_Status_Center(exp.var.current['cue'], redraw=True)

        ### Begin generate trial, code from mri_tones.py
        target_number_T = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for target stream 
        target_number_D = np.random.choice(np.arange(3)+1) # random embed 1~3 targets for distractor stream

        if exp.var.current['isTargetLeft'] == 'True':
            current_isTargetLeft = True
        else:
            current_isTargetLeft = False

        if exp.var.current['isLowLeft'] == 'True':
            current_isLowLeft = True
        else:
            current_isLowLeft = False

        params = {
            "spatial_condition": exp.var.current['cue'], # "ILD10" or "ITD500"
            "tone_duration": exp.stim.tone_duration,
            "tone_interval": exp.stim.tone_interval,
            "seq_interval": exp.stim.seq_interval,
            "seq_per_trial": exp.stim.seq_per_trial,
            "target_number_T": target_number_T,
            "target_number_D": target_number_D,
            "fs": exp.stim.fs,
            "isLowLeft":current_isTargetLeft,   # np.random.choice([True,False]),
            "isTargetLeft": current_isLowLeft,   # np.random.choice([True,False]),
            "isTargetPresent": True,
            "cue2stim_interval": exp.stim.cue2stim_interval
        }
        exp.stim.direction = params['isTargetLeft']
        exp.stim.tar_num = params['target_number_T']
        exp.user.target_number_count += target_number_T

        if params["spatial_condition"] == 'ILD10':
            low_pitch_seqs_dict = exp.stim.low_pitch_seqs_ILD
            high_pitch_seqs_dict = exp.stim.high_pitch_seqs_ILD
        else:
            low_pitch_seqs_dict = exp.stim.low_pitch_seqs_ITD
            high_pitch_seqs_dict = exp.stim.high_pitch_seqs_ITD

        # ------------ pattern task -----------------
        ''''''
        cue_pitch_seqs = mri_tones.generate_miniseq(330, 2**(1/12), 1.5, exp.stim.tone_interval, exp.stim.tone_duration, exp.stim.ramp_duration, exp.stim.volume, exp.stim.fs)
        cue_pitch_seqs_ILD, cue_pitch_seqs_ITD = mri_tones.spatialize_seq(cue_pitch_seqs,exp.stim.ild,exp.stim.itd,exp.stim.fs)
        if params["spatial_condition"] == 'ILD10':
            cue_pitch_seqs_dict = cue_pitch_seqs_ILD
        else:
            cue_pitch_seqs_dict = cue_pitch_seqs_ITD

        # Wusheng edited Apr 24: task changed to reversal pattern finding task
        test_trial, trial_info = mri_tones.generate_trial_findzigzag_clean(params,low_pitch_seqs_dict,high_pitch_seqs_dict,isCueIncluded=False) # isCueIncluded has to be True for this task
        if exp.var.current['noise'] =='yes':
            test_trial = mri_tones.get_trial_with_noise(test_trial)

        exp.stim.trial_info = trial_info

        trial_info_str = mri_tones.parse_trial_info_ptask(trial_info)
        #exp.user.response = ",".join(list(trial_info['target_time'].astype(str)))
        fid = open(f"data/{exp.name}_times_{exp.subjID}.csv", 'a')
        # word_line = f"{exp.run.trial},{trial_info['isTargetLeft']},{','.join(trial_info['target_time'])}"
        # word_line = f"SubjectID,Trial,SpatialCond,isTargetLeft,isLowLeft,TargetNum,TargetTime1,TargetTime2,TargetTime3,ResponseTime"
        word_line = f"{exp.subjID},{exp.name},{trial_info['spa_cond']},{trial_info['isTargetLeft']},{trial_info['isLowLeft']},{trial_info['tarN_T']},\
        {','.join(trial_info['target_time'].astype(str))}"

        fid.write(word_line+',') #+"\n"  # wait to change line until record response after trial

        #test_trial_with_noise = mri_tones.get_trial_with_noise(test_trial)
        ### End generate trial, code from mri_tones.py

        #exp.stim.out = test_trial_with_noise
        exp.stim.out = test_trial
        #plt.plot(test_trial)
        #plt.show()
    except Exception as e:
        exp.interface.destroy()
        raise e


def present_trial(exp):
    # This is a custom present_trial that records keypress times during playback

    exp.interface.update_Prompt("During the trial, hit a key when you hear a reversal melody. Hit a key to start trial", show=True, redraw=True)

#    wait = True
#    while wait:
    ret = exp.interface.get_resp()
#        if ret in ['l', 'r', 'b', exp.quitKey]:
#            wait = False

    if ret == exp.quitKey:
        exp.run.gustav_is_go = False
    else:
        try:
            '''
            TODO:
            1. add function receiving keypress from scanner to start a trial (should be after subject keypress)
            2. save all timesteps 
            3. do this for other tasks as well (do for tonotopy task, no need for loudness matching) 
            '''

            exp.user.side = ret
            #exp.interface.update_Prompt("Hit a key when you hear a reversal melody", show=True, redraw=True)
            #time.sleep(1)

            if exp.stim.direction:  # change font size of powershell to make this looks bigger
                exp.interface.update_Prompt('<- Listen Left', show=True, redraw=True)
            else:
                exp.interface.update_Prompt('Listen Right ->', show=True, redraw=True)
            time.sleep(.5)

            #exp.interface.update_Prompt("+", show=True, redraw=True)
            exp.interface.update_Prompt("   ██   \n   ██   \n████████\n   ██   \n   ██   ", show=True, redraw=True)
            #time.sleep(.5)

            #exp.interface.update_Notify_Left('Playing', show=False, redraw=True)
            #exp.interface.update_Status_Right("", redraw=True)
            responses = []
            valid_responses = []
#            exp.interface.showPlaying(True)
            if not exp.debug:

                target_times = exp.stim.trial_info['target_time']
                target_times_end = target_times.copy() + exp.stim.rt_good_delay

                s = exp.stim.audiodev.open_array(exp.stim.out,exp.stim.fs)
                #exp.interface.show_Notify_Left(show=True, redraw=True)
                dur_ms = len(exp.stim.out) / exp.stim.fs * 1000
                this_wait_ms = 500
                this_elapsed_ms = 0
                resp_percent = []
                s.play()
                #if triggers:
                #    exp.user.triggers.trigger(exp.stim.trigger)
                start_ms = exp.interface.timestamp_ms()
                while s.is_playing:
                    ret = exp.interface.get_resp(timeout=this_wait_ms/1000)
                    this_current_ms = exp.interface.timestamp_ms()
                    this_elapsed_ms = this_current_ms - start_ms
                    this_elapsed_percent = this_elapsed_ms / dur_ms * 100
                    if ret:
                        resp = np.round(this_elapsed_ms/1000, 3)
                        responses.append(str(resp))
                        resp_percent.append(this_elapsed_ms / dur_ms * 100)

                        # valid responses
                        bool_1 = (resp > target_times)
                        bool_2 = (resp <= target_times_end)
                        bool_valid = bool_1 * bool_2   # same as "AND"

                        if bool_valid.any():
                            valid_responses.append(str(resp))
                            exp.user.valid_response_count += 1
                            this_tar_idx = np.where(bool_valid)[0][0]   # index of first valid target
                            target_times = np.delete(target_times,this_tar_idx)
                            target_times_end = np.delete(target_times_end,this_tar_idx)


                    progress = psylab.string.prog(this_elapsed_percent, width=50, char_done="=", spec_locs=resp_percent, spec_char="X")
                    #exp.interface.update_Prompt(progress, show=True, redraw=True)
                exp.user.response = ",".join(responses)
                exp.user.valid_response = ",".join(valid_responses)

                #fid = open(exp.dataFile, 'a')  # TODO: this seems like it's not saving
                fid = open(f"data/{exp.name}_times_{exp.subjID}.csv", 'a')
                word_line = f"{','.join(responses)}"  # TODO: could add valid response as well, for sanity check, to compare with R code
                fid.write(word_line+"\n")
#            exp.interface.showPlaying(False)
        except Exception as e:
            exp.interface.destroy()
            raise e


"""CUSTOM PROMPT
    If you want a custom response prompt, define a function for it
    here. exp.run.response should receive the response as a string, and
    if you want to cancel the experiment, set both exp.run.block_on and
    exp.run.pylab_is_go to False
"""

def prompt_response(exp):
    pass
    # while True:
    #     ret = exp.interface.get_resp()
    #     if ret in exp.validKeys:
    #         exp.run.response = ret
    #         break
    #     elif ret in ['/','q']:
    #         exp.run.gustav_is_go = False
    #         break

def post_trial(exp):

    # exp.run.trials_exp is 0 based, and this is in "post trial"
    if exp.run.trials_exp == exp.var.constant['total_trial_num']-1:
        exp.interface.update_Prompt("Congratulations! You've finished the study!", show=True, redraw=True)
        time.sleep(3)
        exp.interface.update_Prompt("", show=True, redraw=True)
    elif (exp.run.trials_exp+1)%8==0:
        exp.interface.update_Prompt("Congratulations! You've just finished a block, please take a break.", show=True, redraw=True)
        time.sleep(5)  # TODO: update this time
        exp.interface.update_Prompt("", show=True, redraw=True)
    else:
        #if not exp.gustav_is_go:
        exp.interface.update_Prompt("Waiting 1 sec...", show=True, redraw=True)
        time.sleep(1)
        exp.interface.update_Prompt("", show=False, redraw=True)

def post_block(exp):
    pass
#    exp.interface.updateInfo_BlockScore(f"Prev Condition # {exp.run.condition+1}\nScore: {exp.user.block_kwc:.1f} / {exp.user.block_kwp:.1f} ({exp.user.block_pc} %%)")

def post_exp(exp):
#    pass
#    exp.interface.dialog.isPlaying.setText("Finished")
#    exp.interface.showPlaying(True)
    exp.interface.destroy()

if __name__ == '__main__':
    argv = sys.argv[1:]
    argv.append(f"--experimentFile={os.path.realpath(__file__)}")
    gustav.gustav.main(argv)

