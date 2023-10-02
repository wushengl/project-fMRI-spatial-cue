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
import psytasks
import random
from datetime import datetime
import subprocess
import glob
import logging
import pdb
import pandas as pd

from screeninfo import get_monitors
from tkinter import messagebox

def setup(exp):

    try:
        import pylink
        exp.user.pylink = pylink
        exp.user.do_eyetracker = True
    except ImportError:
        exp.user.do_eyetracker = False

    # ----------------------- Eye tracker settings --------------------------
    #global do_add_eyetracker, el_tracker, results_folder, edf_file_name, LEFT_EYE, RIGHT_EYE, BINOCULAR
    global el_tracker, results_folder, edf_file_name, LEFT_EYE, RIGHT_EYE, BINOCULAR

    #do_add_eyetracker = True
    do_use_extend_monitor = True
    el_trial = 1  # initiate trial number for eye tracker data

    #exp.user.do_add_eyetracker = do_add_eyetracker
    exp.user.el_trial = el_trial
    

    if exp.user.do_eyetracker:

        # set up eye tracker display on extend screen 

        if do_use_extend_monitor:
            os.environ['SDL_VIDEO_FULLSCREEN_HEAD'] = '1'

            if len(get_monitors()) >1:
                extended_monitor = get_monitors()[0] # BRIDGE center: display 1 (left) is mirrored into the scanner
            else:
                print("No extended monitor founded!")

        # some global constants

        LEFT_EYE = 0
        RIGHT_EYE = 1
        BINOCULAR = 2

        exp.user.LEFT_EYE = LEFT_EYE
        exp.user.RIGHT_EYE = RIGHT_EYE
        exp.user.BINOCULAR = BINOCULAR


        # set the screen size (0 for current defualt, which is primary screen size)
        SCN_WIDTH = extended_monitor.width #0
        SCN_HEIGHT = extended_monitor.height #0

        # set up a folder to store the data files
        results_folder = 'results'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)


    # ----------------------- Machine-specific settings --------------------------

    machine = psylab.config.local_settings(conf_file='config/psylab_booth3.conf')
    workdir = machine.get_path('workdir')
    dev_name = machine.get_str('audiodev_name')
    dev_ch = machine.get_int('audiodev_ch')
    devs = m.get_available_devices()
    dev_id = None
    for i,di in enumerate(devs):
        name = psylab.string.as_str(di.name)
        ch = di.maxOutputChannels
        if name.startswith(dev_name) and ch == 2: #8
            dev_id = i,i,ch
            out_id = i
    if dev_id:
        exp.stim.audiodev = m.open_device(*dev_id)
    else:
        raise Exception(f"The audio device {dev_name} was not found")

    # -------------------------- General Experimental Variables --------------------------

    exp.name = 'ild-fmri-2'
    exp.method = 'constant' # 'constant' for constant stimuli, or 'adaptive' for a staircase procedure (SRT, etc)

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

    

    # -------------------------- get experiment info from keyboard --------------------------

    if not exp.subjID:
        exp.subjID = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Enter a Subject ID: ')

    ret = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Session (1 | 2; 0 to choose tasks): ')
    exp.sesNum = ret
    if ret == '1':
        run_mode = 'none'
        match_mode = 'y'
        tonotopy_mode = 'y'
    elif ret == '2':
        run_mode = 'task'
        match_mode = 'n'
        tonotopy_mode = 'n'
    else:

        if True:
            run_mode = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Enter a running mode (task | train | debug): ')

        if True:
            match_mode = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Do loudness match (y | n): ')

        if True:
            tonotopy_mode = exp.term.get_input(parent=None, title = "Gustav!", prompt = 'Run tonotopy task (y | n): ')

    ret = exp.term.get_input(parent=None, title="Gustav!", prompt='Run volume adjust and get centered image? (y | n): ')
    if ret == 'y':
        do_get_comfortable_level = True
        do_get_centered_image = True
    elif ret == 'n':
        do_get_comfortable_level = False
        do_get_centered_image = False

        # load existing files
        try:
            soundtest_file_path = "./logs/soundtest_sub-" + exp.subjID + "_ses-0" + exp.sesNum + ".csv"
            df = pd.read_csv(soundtest_file_path)
        except:
            ret = exp.term.get_input(parent=None, title="Gustav!", prompt='Soundtest data not exist for this session, use data from ses-01? (y | n): ')
            if ret == 'y':
                soundtest_file_path = "./logs/soundtest_sub-" + exp.subjID + "_ses-01.csv"
                df = pd.read_csv(soundtest_file_path)
            else:
                raise ValueError("No soundtest data saved for this session!")

        exp.stim.ref_rms = df['ref_rms'][0]
        exp.stim.probe_ild = df['probe_ild'][0]

    messagebox.showinfo("Volume adjust reminder", "Make sure the system volume is set to a safe level! (30~45)")


    # -------------------------- eye tracker stuff --------------------------

    if exp.user.do_eyetracker:

        # Step 1: initialize a tracker object with a Host IP address
        try:
            el_tracker = exp.user.pylink.EyeLink("100.1.1.1")
        except RuntimeError as error:
            print('ERROR:', error)
            sys.exit()

        # Step 2: Initializes the graphics (for calibration)
        exp.user.pylink.openGraphics((SCN_WIDTH, SCN_HEIGHT), 32)

        # Press ENTER to show the camera image, C to calibrate, V to validate the tracker. Press Esc to quit eyetracker GUI.

        # Step 3: open EDF file on Host PC
        edf_file_name = exp.subjID + time.strftime("%H%M") + ".EDF" # "TEST.EDF"
        el_tracker.openDataFile(edf_file_name)

        # add a preamble text (data file header)
        preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
        el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

        # Step 4: setting up tracking, recording and calibration options
        exp.user.pylink.flushGetkeyQueue()
        el_tracker.setOfflineMode() 

        # send resolution of the screen to tracker
        pix_msg = "screen_pixel_coords 0 0 %d %d" % (SCN_WIDTH - 1, SCN_HEIGHT - 1)
        el_tracker.sendCommand(pix_msg)

        # send resolution of the screen to data viewer
        dv_msg = "DISPLAY_COORDS  0 0 %d %d" % (SCN_WIDTH - 1, SCN_HEIGHT - 1)
        el_tracker.sendMessage(dv_msg)

        # Get the software version
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        # print out some version info in the shell
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

        # Select what data to save in the EDF file, for a detailed discussion
        # of the data flags, see the EyeLink User Manual, "Setting File Contents"
        file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        if eyelink_ver < 4:
            file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
        el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)

        # Select what data is available over the link (for online data accessing)
        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        if eyelink_ver < 4:
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
        el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

        # Set the calibration target and background color
        exp.user.pylink.setCalibrationColors((0, 0, 0), (128, 128, 128))
        el_tracker.sendCommand("calibration_area_proportion = 0.5 0.5")

        # select best size for calibration target
        exp.user.pylink.setTargetSize(int(SCN_WIDTH/70.0), int(SCN_WIDTH/300.))

        # Set the calibraiton and drift correction sound
        exp.user.pylink.setCalibrationSounds("", "", "")
        exp.user.pylink.setDriftCorrectSounds("", "", "")

        # Step 5: Do the tracker setup at the beginning of the experiment.
        el_tracker.doTrackerSetup()

        exp.user.pylink.closeGraphics()


    # -------------------------- fixed experiment setting --------------------------

    exp.stim.training_threshold = .5
    exp.stim.rt_good_delay = 1.5

    exp.stim.fs = 44100.
    f0 = 220 # Hz

    low_pitch_cf_1 = f0         # cf for center frequency
    low_pitch_cf_2 = 3*f0
    high_pitch_cf_1 = 2*f0
    high_pitch_cf_2 = 6*f0      # pilot data used 4*f0 for higher frequency component in high pitch tone

    low_pitch_cf_ratio = int(low_pitch_cf_2/low_pitch_cf_1)
    high_pitch_cf_ratio = int(high_pitch_cf_2/high_pitch_cf_1)

    exp.stim.tone_duration = 0.25 # s
    exp.stim.ramp_duration = 0.04 # s (this is total length for on and off ramps)

    exp.stim.tone_interval = exp.stim.tone_duration # this is offset to onset interval
    exp.stim.seq_interval = 0.8
    exp.stim.cue2stim_interval = 0.5

    exp.stim.seq_per_trial = 10

    # use matched ILD to ITD for all subjects 
    exp.stim.itd = 500e-6    # 685e-6 # s
    exp.stim.ild = 10        #15 # db

    exp.stim.semitone_step = 2**(1/12)
    exp.stim.volume = 0.3       # Not used probably
    exp.stim.desired_rms = 0.3  # In case we don't do loudness matching
    #exp.stim.ref_rms = 0.3      # Set this to the desired rms and all tones will be matched in loudness


    psytasks.test_keyboard_input(keys=[ord('b'), ord('y'), ord('g')], labels=["Blue (Right)", "Yellow (Left)", "Enter (TODO)"])

    soundtest_dict = {}
    if do_get_comfortable_level:
        ret = psytasks.get_comfortable_level(2016, out_id, fs=44100, tone_dur_s=1, tone_level_start=1, atten_start=50, ear='both', key_up=ord('y'), key_dn=ord('b'))
        exp.stim.ref_rms = psylab.signal.atten(.707, float(ret))
        soundtest_dict['ref_rms'] = [exp.stim.ref_rms]
    if do_get_centered_image:
        #sig = ""  # TODO: should we use pure tone or complex tone or maybe syllables? since tones are not that good for spatialiation?
        exp.stim.probe_ild = psytasks.get_centered_image(2016, out_id, tone_level_start=exp.stim.ref_rms, adj_step=0.5, key_l=ord('y'), key_r=ord('b'))
        soundtest_dict['probe_ild'] = [exp.stim.probe_ild]

    if (do_get_comfortable_level and do_get_centered_image):
        print(soundtest_dict)
        soundtest_df = pd.DataFrame(data=soundtest_dict)

        soundtest_file_path = "./logs/soundtest_sub-" + exp.subjID + "_ses-0" + exp.sesNum + ".csv"
        soundtest_df.to_csv(soundtest_file_path)

        ''' usage of prob_ild: 
        if probe_ild > 0:
            mix_mat[0,0] = psylab.signal.atten(1, probe_ild)
            mix_mat[1,1] = 1
        else:
            mix_mat[1,1] = psylab.signal.atten(1, -probe_ild)
            mix_mat[0,0] = 1
        '''

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
                                        #'yes',
                                        'no',
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
    # CAB EDIT 2023-05-22
    #pretask_log_path = './logs/log_pretask_'+time_str+'.log'
    pretask_log_path = os.path.join('logs',f'log_pretask_{time_str}.log')
    logger = logging.getLogger('logger_pretask')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(pretask_log_path)
    logger.addHandler(file_handler)


    # -------------------------- Experiment setting --------------------------

    if match_mode == 'y':
        do_loudness_match = True
    else:
        do_loudness_match = False

    if tonotopy_mode == 'y':
        do_run_tonotopy = True
    else:
        do_run_tonotopy = False

    do_adjust_level = True
    loudness_match_times = 2  # baseline matching times, this means start adaptive matching after getting 2 matches
    diff_thre = 25            # this is the threshold to throw away a match, common values are +- 10dB for largest diff
    tonotopy_f_pool = [300, 566, 1068, 2016, 3805]
    tonotopy_matching_pool = [300, 566, 1068, 3805]
    complextone_matching_pool = [[low_pitch_cf_1, low_pitch_cf_2], [high_pitch_cf_1, high_pitch_cf_2]]
    matching_pool = tonotopy_matching_pool + complextone_matching_pool
    tonotopy_pool = tonotopy_f_pool + complextone_matching_pool

    if do_loudness_match or do_run_tonotopy:
        print('Open a new powershell and see the log with:\nGet-Content -Path ' + pretask_log_path + ' -Wait')
        input("Press Enter to continue...")


    ##############################################################
    ### loudness matching
    ##############################################################

    if do_loudness_match:
        logger.info("--------------------------------------------------------------")
        logger.info("Now start loudness matching...")
        logger.info("Matching pool: "+str(matching_pool))
        all_matched_levels = np.empty((0,len(matching_pool)))
        for i in range(loudness_match_times): # minimum match times
            logger.info("Now start matching round "+str(i+1))

            # matched_levels is a list of matched level difference, without 2016Hz
            matched_levels = mri_tones.get_loudness_match(2016,matching_pool,dev_id[0],tone_level_start=exp.stim.ref_rms,round_idx=i, key_up=ord('y'), key_dn=ord('b'))
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

            logger.info("----------------------- new adaptive matching -------------------------")
            logger.info("Current convergence status: " + str(conv_check.astype(bool)))

            # get matching pool for this round
            this_matching_pool = np.array(matching_pool)[(1-conv_check).astype(bool)]

            # get new sample
            i+=1
            this_matched_levels = mri_tones.get_loudness_match(2016, list(this_matching_pool), dev_id[0],tone_level_start=exp.stim.ref_rms, round_idx=i,key_up=ord('y'), key_dn=ord('b'))

            # make full matched sample
            matched_levels = np.zeros(len(last_mean))
            matched_levels[conv_check.astype(bool)] = last_mean[conv_check.astype(bool)]  # set matched values to last mean if already converges
            matched_levels[(1-conv_check).astype(bool)] = this_matched_levels              # add this matched values to this sample

            all_matched_levels = np.concatenate((all_matched_levels, np.array(matched_levels).reshape(1, -1)), axis=0)
            print(matched_levels)

            # check if new sample within boundary
            conv_check = (matched_levels>=lower_bound).astype(int) * (matched_levels<=upper_bound).astype(int)

            logger.info("This matching pool: " + str(this_matching_pool))
            logger.info("New matched levels: " + str(np.round(this_matched_levels,3)))
            logger.info("Upper bounds: " + str(upper_bound))
            logger.info("Lower bounds: " + str(lower_bound))
            logger.info("Updated convergence status: " + str(conv_check.astype(bool)))

            # update boundary
            last_mean = np.nanmean(all_matched_levels,axis=0)
            last_std = np.nanstd(all_matched_levels-last_mean) # ,axis=0
            upper_bound = last_mean + last_std
            lower_bound = last_mean - last_std


        matched_levels_ave = np.nanmean(all_matched_levels,axis=0)

        ref_index = 3
        matched_levels_ave = np.insert(matched_levels_ave, ref_index, 0)  # inset level 0 for reference frequency

        logger.info("--------------------------------------------------------------")
        logger.info("Matching finished!")
        logger.info("Averaged matched levels: " + str(matched_levels_ave))
        logger.info("--------------------------------------------------------------")
        print("Matched level differences:")
        print(matched_levels_ave)

        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M")
        # CAB EDIT 2023-05-22
        #matched_levels_save_path = os.path.join(workdir, 'data', exp.subjID + '-matched_levels-' + time_str + '.csv')
        matched_levels_save_path = os.path.join(workdir, 'data', f'{exp.subjID}-matched_levels-{time_str}.csv')
        np.savetxt(matched_levels_save_path,all_matched_levels,delimiter=',')
    else:
        if do_adjust_level:

            logger.info("--------------------------------------------------------------")
            logger.info("Loading previous loudness matching...")

            # CAB EDIT 2023-05-22
            #file_names = glob.glob(workdir+'\\'+'data'+'\\*' + exp.subjID + '-matched_levels*')
            file_names = glob.glob(os.path.join(workdir, 'data', f'*{exp.subjID}-matched_levels*'))
            #print(file_names)  # return is a list
            if len(file_names) == 0:
                matched_levels_ave = np.zeros(7)
            else:
                matched_levels_all = np.loadtxt(file_names[-1], delimiter=',')  # if multiple, get the latest one

                matched_levels_all[abs(matched_levels_all)>=diff_thre] = np.nan
                matched_levels_ave = np.nanmean(matched_levels_all, axis=0)

                ref_index = 3
                matched_levels_ave = np.insert(matched_levels_ave, ref_index, 0)  # inset level 0 for reference frequency

                print("Matched level differences:")
                print(matched_levels_ave)

                logger.info("Loaded file name: "+file_names[-1])
                logger.info("Averaged matched levels: " + str(matched_levels_ave))
                logger.info("--------------------------------------------------------------")
        else:
            matched_levels_ave = np.zeros(7)


    ##############################################################
    ### tonotopy scan task
    ##############################################################

    # CAB EDIT 2023-05-22
    tono_1back_filename = os.path.join('data', f'{exp.name}_times_{exp.subjID}_tonotopy-1back.csv')
    #if not os.path.isfile(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv"):
    if not os.path.isfile(tono_1back_filename):
        #fid = open(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv", 'a')
        fid = open(tono_1back_filename, 'a')
        word_line = f"SubjectID,Trial,frequency,toneDur,seqPerTrial,TargetNum,TargetTime,ResponseTime,TrialStartTime"  # TODO: change header line here
        fid.write(word_line + "\n")
        fid.flush()
    elif do_run_tonotopy:
        #fid = open(f"data/{exp.name}_times_{exp.subjID}_tonotopy-1back.csv", 'a')
        fid = open(tono_1back_filename, 'a')
        fid.write("\n\n")
        fid.flush()


    if  run_mode == 'debug':
        cycle_per_run_puretone = 1
        run_num_puretone = 1
        cycle_per_run_ctone = 1
        run_num_ctone = 1
    else:

        cycle_per_run_puretone = 8
        run_num_puretone = 4

        cycle_per_run_ctone = 16
        run_num_ctone = 2

    switch_step_puretone = [0,0,1,1] # control whether shift steps for each run
    switch_step_ctone = [0,1]

    if do_run_tonotopy:

        logger.info("Now start tonotopy task...")

        logger.info("--------------------------------------------------------------")
        logger.info("Now start pure tone...")
        logger.info("cycles per run: " + str(cycle_per_run_puretone))
        logger.info("run number: " + str(run_num_puretone))

        if exp.user.do_eyetracker:
            el_active = exp.user.pylink.getEYELINK()
            if not el_active.isConnected():
                raise RuntimeError("Eye tracker is not connected!")

        for i in range(run_num_puretone):

            logger.info("Now running pure tone round "+str(i+1))

            this_pool = tonotopy_f_pool.copy()
            do_switch_step = switch_step_puretone[i]
            if do_switch_step:
                this_pool.reverse()
            mri_tones.run_tonotopy_task(this_pool, dev_id[0], exp, do_adjust_level,matched_levels_ave[:-2], cycle_per_run=cycle_per_run_puretone,round_idx=i)

        logger.info("--------------------------------------------------------------")
        logger.info("Now start complex tone...")
        logger.info("cycles per run: " + str(cycle_per_run_ctone))
        logger.info("run number: " + str(run_num_ctone))

        if exp.user.do_eyetracker:
            el_active = exp.user.pylink.getEYELINK()
            if not el_active.isConnected():
                raise RuntimeError("Eye tracker is not connected!")

        for j in range(run_num_ctone):

            logger.info("Now running complex tone round "+str(j+1))

            this_pool = complextone_matching_pool.copy()
            do_switch_step = switch_step_ctone[j]
            if do_switch_step:
                this_pool.reverse()
            mri_tones.run_tonotopy_task(this_pool, dev_id[0], exp, do_adjust_level,matched_levels_ave[5:], cycle_per_run=cycle_per_run_ctone,round_idx=(i+j+1))



    # -------------------- create minisequence ------------------------

    if do_adjust_level:
        desired_rms_low = mri_tones.attenuate_db(exp.stim.ref_rms, -1*matched_levels_ave[-2])
        desired_rms_high = mri_tones.attenuate_db(exp.stim.ref_rms, -1*matched_levels_ave[-1])
    else:
        desired_rms_low = exp.stim.desired_rms
        desired_rms_high = exp.stim.desired_rms

    low_pitch_seqs = mri_tones.generate_miniseq(low_pitch_cf_1, exp.stim.semitone_step, low_pitch_cf_ratio,exp.stim.tone_interval, exp.stim.tone_duration,exp.stim.ramp_duration, desired_rms_low, exp.stim.fs)
    high_pitch_seqs = mri_tones.generate_miniseq(high_pitch_cf_1, exp.stim.semitone_step, high_pitch_cf_ratio,exp.stim.tone_interval, exp.stim.tone_duration,exp.stim.ramp_duration, desired_rms_high, exp.stim.fs)

    # ------------------------ spatialize minisequence ------------------------

    if do_adjust_level:  # ITD condition use matched level, ILD condition use +/- 0.5*ILD for louder/weaker channel
        exp.stim.low_pitch_seqs_ILD, exp.stim.low_pitch_seqs_ITD = mri_tones.spatialize_seq_matched(low_pitch_seqs,exp.stim.ild, exp.stim.itd,exp.stim.fs)
        exp.stim.high_pitch_seqs_ILD, exp.stim.high_pitch_seqs_ITD = mri_tones.spatialize_seq_matched(high_pitch_seqs,exp.stim.ild,exp.stim.itd, exp.stim.fs)
    else:
        exp.stim.low_pitch_seqs_ILD, exp.stim.low_pitch_seqs_ITD = mri_tones.spatialize_seq(low_pitch_seqs,exp.stim.ild, exp.stim.itd,exp.stim.fs)
        exp.stim.high_pitch_seqs_ILD, exp.stim.high_pitch_seqs_ITD = mri_tones.spatialize_seq(high_pitch_seqs,exp.stim.ild,exp.stim.itd, exp.stim.fs)


    # CAB EDIT 2023-05-22
    exp.stim.times_fname = os.path.join('data', f'{exp.name}_times_{exp.subjID}.csv')
    #if not os.path.isfile(f"data/{exp.name}_times_{exp.subjID}.csv"):
    if not os.path.isfile(exp.stim.times_fname):
        #fid = open(f"data/{exp.name}_times_{exp.subjID}.csv", 'a')
        fid = open(exp.stim.times_fname, 'a')
        # word_line = f"{exp.run.trial},{trial_info['isTargetLeft']},{','.join(trial_info['target_time'])}"
        word_line = f"SubjectID,Trial,SpatialCond,isTargetLeft,isLowLeft,TargetNum,TargetTime,ResponseTime,TrialStartTime"  # TODO: change header line here
        fid.write(word_line + "\n")
    else:
        #fid = open(f"data/{exp.name}_times_{exp.subjID}.csv", 'a')
        fid = open(exp.stim.times_fname, 'a')
        fid.write("\n\n")


    if run_mode == 'none':
        exp.run.gustav_is_go = False

def pre_exp(exp):
    try:

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
        #exp.interface.destroy()
        post_exp(exp)
        raise e


def pre_block(exp):
    try:
        exp.user.block_kwp = 0
        exp.user.block_kwc = 0
        exp.user.block_pc = 0.
        exp.user.pract = 1
        exp.interface.update_Status_Left(f"Block {exp.run.block+1} of {exp.run.nblocks}")
    except Exception as e:
        #exp.interface.destroy()
        post_exp(exp)
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

        test_trial, trial_info = mri_tones.generate_trial_findzigzag_clean(params,low_pitch_seqs_dict,high_pitch_seqs_dict,isCueIncluded=False) # isCueIncluded has to be True for this task
        if exp.var.current['noise'] =='yes':
            test_trial = mri_tones.get_trial_with_noise(test_trial)

        exp.stim.trial_info = trial_info

        trial_info_str = mri_tones.parse_trial_info_ptask(trial_info)
        # CAB EDIT 2023-05-22
        #fid = open(f"data/{exp.name}_times_{exp.subjID}.csv", 'a')
        fid = open(exp.stim.times_fname, 'a')
        word_line = f"{exp.subjID},{exp.name},{trial_info['spa_cond']},{trial_info['isTargetLeft']},{trial_info['isLowLeft']},{trial_info['tarN_T']},\
        {','.join(trial_info['target_time'].astype(str))}"

        fid.write(word_line+',')

        exp.stim.out = test_trial

        if exp.user.do_eyetracker:
            el_tracker = exp.user.pylink.getEYELINK()

            if(not el_tracker.isConnected() or el_tracker.breakPressed()):
                raise RuntimeError("Eye tracker is not connected!")

            # show some info about the current trial on the Host PC screen
            pars_to_show = ('main', exp.run.trials_exp, exp.var.constant['total_trial_num']) # TODO: check current trial number
            status_message = 'Link event example, %s, Trial %d/%d' % pars_to_show
            el_tracker.sendCommand("record_status_message '%s'" % status_message)

            # log a TRIALID message to mark trial start, before starting to record.
            # EyeLink Data Viewer defines the start of a trial by the TRIALID message.
            print("exp.user.el_trial = %d" % exp.user.el_trial)
            el_tracker.sendMessage("TRIALID %d" % exp.user.el_trial) # exp.run.trials_exp
            exp.user.el_trial += 1

            # clear tracker display to black
            el_tracker.sendCommand("clear_screen 0")

            # switch tracker to idle mode
            el_tracker.setOfflineMode()

            error = el_tracker.startRecording(1, 1, 1, 1)
            if error:
                return error

    except Exception as e:
        #exp.interface.destroy()
        post_exp(exp)
        raise e


def present_trial(exp):
    # This is a custom present_trial that records keypress times during playback
    try:
        if exp.run.trials_exp%8 == 0:

            exp.interface.update_Prompt("Waiting for trigger\n\nHit a key when you hear a zig-zag melody", show=True, redraw=True)
    
            wait = True
            while wait:
                ret = exp.interface.get_resp()
                if ret in ['t', exp.quitKey]:
                    trial_start_time = datetime.now()
                    wait = False
            if exp.user.do_eyetracker:
                # log a message to mark the time at which the initial display came on
                el_tracker.sendMessage("SYNCTIME")
        else:
            trial_start_time = None

        #exp.user.side = yb
        #exp.interface.update_Prompt("Hit a key when you hear a reversal melody", show=True, redraw=True)
        #time.sleep(1)

        if exp.stim.direction:
            exp.interface.update_Prompt('<- Listen Left', show=True, redraw=True)
            #exp.interface.update_Notify_Left('Listen Left', show=True, redraw=True)
        else:
            exp.interface.update_Prompt('Listen Right ->', show=True, redraw=True)
            #exp.interface.update_Notify_Right('Listen Right', show=True, redraw=True)
        time.sleep(2)

        exp.interface.update_Prompt("   ██   \n   ██   \n████████\n   ██   \n   ██   ", show=True, redraw=True)
        #time.sleep(.5)

        responses = []
        valid_responses = []

        if not exp.debug:

            target_times = exp.stim.trial_info['target_time']
            target_times_end = target_times.copy() + exp.stim.rt_good_delay

            s = exp.stim.audiodev.open_array(exp.stim.out,exp.stim.fs)

            # TODO: check if this is working correctly
            mix_mat = np.zeros((2, 2))
            if exp.stim.probe_ild > 0:
                mix_mat[0, 0] = psylab.signal.atten(1, exp.stim.probe_ild)
                mix_mat[1, 1] = 1
            else:
                mix_mat[1, 1] = psylab.signal.atten(1, -exp.stim.probe_ild)
                mix_mat[0, 0] = 1
            s.mix_mat = mix_mat

            if exp.user.do_eyetracker:
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
                    return exp.user.pylink.TRIAL_ERROR

            dur_ms = len(exp.stim.out) / exp.stim.fs * 1000
            this_wait_ms = 500
            this_elapsed_ms = 0
            resp_percent = []
            s.play()
            #time.sleep(1)

            start_ms = exp.interface.timestamp_ms()
            while s.is_playing:

                if exp.user.do_eyetracker:
                    error = el_tracker.isRecording()
                    if error != exp.user.pylink.TRIAL_OK:
                        el_active = exp.user.pylink.getEYELINK()
                        el_active.stopRecording()
                        raise RuntimeError("Recording stopped!")

                ret = exp.interface.get_resp(timeout=this_wait_ms/1000)
                this_current_ms = exp.interface.timestamp_ms()
                this_elapsed_ms = this_current_ms - start_ms
                this_elapsed_percent = this_elapsed_ms / dur_ms * 100
                if ret in ['b','y']:
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

            if trial_start_time:
                fid = open(exp.stim.times_fname, 'a')
                word_line = f"{','.join(responses)}" + "," + trial_start_time.strftime("%H:%M:%S.%f")
                fid.write(word_line+"\n")

    except Exception as e:
        post_exp(exp)
        #exp.interface.destroy()
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
        exp.interface.update_Prompt("Block finished.\nHit space to continue", show=True, redraw=True)
        #time.sleep(5)
        #exp.interface.update_Prompt("", show=True, redraw=True)
        wait = True
        while wait:
            ret = exp.interface.get_resp()
            if ret in [' ', exp.quitKey]:
                wait = False
    else:
        #if not exp.gustav_is_go:
        exp.interface.update_Prompt("Waiting 2 sec...", show=True, redraw=True)
        time.sleep(2)
        exp.interface.update_Prompt("", show=False, redraw=True)

    if exp.user.do_eyetracker:

        el_active = exp.user.pylink.getEYELINK()
        el_active.stopRecording()

        # record the trial variable in a message recognized by Data Viewer
        el_active.sendMessage("!V TRIAL_VAR el_trial %d" % exp.user.el_trial)
        el_active.sendMessage("!V TRIAL_VAR trial %d" % exp.run.trials_exp)
        el_active.sendMessage("!V TRIAL_VAR task main") 

        el_active.sendMessage('TRIAL_RESULT %d' % exp.user.pylink.TRIAL_OK)

        ret_value = el_active.getRecordingStatus()
        if (ret_value == exp.user.pylink.TRIAL_OK):
                el_tracker.sendMessage("TRIAL OK")

        


def post_block(exp):
    pass
#    exp.interface.updateInfo_BlockScore(f"Prev Condition # {exp.run.condition+1}\nScore: {exp.user.block_kwc:.1f} / {exp.user.block_kwp:.1f} ({exp.user.block_pc} %%)")

def post_exp(exp):
#    pass
#    exp.interface.dialog.isPlaying.setText("Finished")
#    exp.interface.showPlaying(True)

    if exp.user.do_eyetracker:

        el_active = exp.user.pylink.getEYELINK()
        el_active.setOfflineMode()
        
        # Close the edf data file on the Host
        el_active.closeDataFile()

        local_file_name = os.path.join(results_folder, edf_file_name)
        try:
            el_active.receiveDataFile(edf_file_name, local_file_name)
        except RuntimeError as error:
            print('ERROR:', error)

        # close EyeLink connection and quit display-side graphics
        el_active.close()

    exp.interface.destroy()

if __name__ == '__main__':
    argv = sys.argv[1:]
    argv.append(f"--experimentFile={os.path.realpath(__file__)}")
    gustav.gustav.main(argv)

