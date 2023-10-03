import utils






def run_tonotopy_task(cf_pool, audio_dev, exp, do_adjust_level, matched_dbs, do_addnoise=False, cycle_per_run=8, round_idx=1):

    do_eyetracker = exp.user.do_eyetracker
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
            desired_rms = utils.attenuate_db(exp.stim.ref_rms,-this_level_adjust)
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

    interface.update_Prompt("Now starting tonotopy scan task run "+str(round_idx+1)+"\n\nHit a key when you hear a repeating pattern\n\nHit Space to start",show=True, redraw=True)
    interface.update_Title_Center("Tonotopy scan task")

    #interface.update_Prompt("Hit a key when you hear a repeating pattern\n\nHit Space to continue",show=True, redraw=True)
    #ret = interface.get_resp()

    wait = True
    while wait:
        ret = interface.get_resp()
        if ret in [' ', exp.quitKey]:
            wait = False


    interface.update_Prompt("Waiting for trigger (t)\nto start new trial...", show=True,
                            redraw=True)  # Hit a key to start this trial
    wait = True
    while wait:
        ret = interface.get_resp()
        if ret in ['t']:
            trial_start_time = datetime.now()
            wait = False

    # -------------------- start the experiment--------------------------

    for c in range(cycle_per_run): # e.g. 5 frequencies/cycle, 8 cycles/run, 4 runs


        interface.update_Prompt("Now starting cycle " + str(c + 1) + "...", show=True,redraw=True)
        #ret = interface.get_resp()
        time.sleep(2)

        seqs_keys = all_seqs.keys()

        for i_f, cf_key in enumerate(seqs_keys):

            if do_eyetracker:

                # TODO: currently saving tonotopy to the same file as main task, not sure if this is easy for later analysis

                el_tracker = exp.user.pylink.getEYELINK()

                if(not el_tracker.isConnected() or el_tracker.breakPressed()):
                    raise RuntimeError("Eye tracker is not connected!")

                # show some info about the current trial on the Host PC screen
                pars_to_show = ('tonotopy', i_f, len(seqs_keys), c, cycle_per_run, round_idx+1)
                status_message = 'Link event example, %s, Trial %d/%d, Cycle %d/%d, Run number %d' % pars_to_show
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
                    return exp.user.pylink.TRIAL_ERROR

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

            if do_eyetracker:
                el_active = exp.user.pylink.getEYELINK()
                el_active.stopRecording()

                el_active.sendMessage("!V TRIAL_VAR el_trial %d" % exp.user.el_trial)
                el_active.sendMessage("!V TRIAL_VAR task tonotopy") 
                el_active.sendMessage("!V TRIAL_VAR trial %d" % i_f)
                el_active.sendMessage("!V TRIAL_VAR cf %d" % cf)
                el_active.sendMessage("!V TRIAL_VAR trial_per_cycle %d" % len(cf_pool))
                el_active.sendMessage("!V TRIAL_VAR cycle %d" % c)
                el_active.sendMessage("!V TRIAL_VAR cycle_per_run %d" % cycle_per_run)
                el_active.sendMessage("!V TRIAL_VAR run_number %d" % (round_idx+1))

                el_active.sendMessage('TRIAL_RESULT %d' % exp.user.pylink.TRIAL_OK)

                ret_value = el_active.getRecordingStatus()
                if (ret_value == exp.user.pylink.TRIAL_OK):
                    el_active.sendMessage("TRIAL OK")

    #interface.destroy()

    

    # no return for this task, data saved in file


