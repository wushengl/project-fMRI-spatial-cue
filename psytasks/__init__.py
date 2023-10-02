import numpy as np
import curses
import medussa as m
import psylab
import gustav
from gustav.forms import rt as theForm

# TODO: Add a practice function, allowing experimenter to play soundfiles one at a time, repeat, etc., like listplayer


def do_breathing(blocks=11, breaths_per_block=5, inhale_dur=5, exhale_dur=5, hold_dur=15, callback=None):

    interface = theForm.Interface()
    interface.update_Title_Center("Breath Holding")
    interface.update_Prompt("Breath Holding\n\nHit a key to continue\n(q or / to quit)", show=True, redraw=True)
    ret = interface.get_resp()
    if not ret in ['q', '/']:
        quit = False
        for i in range(blocks):
            if quit:
                interface.update_Prompt(f"{inhale_dur} sec inhale, {exhale_dur} sec exhale ({breaths_per_block} times), then {hold_dur} sec breath hold\n({i+1} of {blocks}; hit a key to start)", show=True, redraw=True)
                ret = interface.get_resp()
                if ret in ['q', '/']:
                    quit = True
                else:
                    for j in range(breaths_per_block):
                        hale_cur = 0
                        prompt = f"Inhale ({j+1}/{breaths_per_block})..."
                        time_init = interface.timestamp_ms()
                        if callback:
                            #exp.interface.update_Status_Right(f"trigger {exp.stim.trigger_dict['Inhale']}")
                            #exp.user.triggers.trigger(exp.stim.trigger_dict['Inhale'])
                            callback('Inhale')
                        while hale_cur < inhale_dur:
                            hale_cur = np.minimum((interface.timestamp_ms() - time_init) / 1000, inhale_dur)
                            hale_cur = np.maximum(hale_cur, 0)
                            progress = psylab.string.prog(hale_cur, width=50, char_done="=", maximum=inhale_dur)
                            this_prompt = f"{prompt}\n{progress} {np.int(hale_cur)} / {inhale_dur}"
                            interface.update_Prompt(this_prompt, show=True, redraw=True)
                            time.sleep(.2)
                        hale_cur = exhale_dur
                        prompt = f"Exhale ({j+1}/{breaths_per_block})..."
                        time_init = interface.timestamp_ms()
                        if callback:
                            #exp.interface.update_Status_Right(f"trigger {exp.stim.trigger_dict['Exhale']}")
                            #exp.user.triggers.trigger(exp.stim.trigger_dict['Exhale'])
                            callback('Exhale')
                        while hale_cur > 0:
                            hale_cur = np.minimum(exhale_dur - ((interface.timestamp_ms() - time_init) / 1000), exhale_dur)
                            hale_cur = np.maximum(hale_cur, 0)
                            progress = psylab.string.prog(hale_cur, width=50, char_done="=", maximum=exhale_dur)
                            this_prompt = f"{prompt}\n{progress} {exhale_dur - np.int(hale_cur)} / {exhale_dur}"
                            interface.update_Prompt(this_prompt, show=True, redraw=True)
                            time.sleep(.2)
                    hale_cur = 0
                    prompt = f"Inhale (then hold)..."
                    time_init = interface.timestamp_ms()
                    if callback:
                        #exp.interface.update_Status_Right(f"trigger {exp.stim.trigger_dict['Inhale']}")
                        #exp.user.triggers.trigger(exp.stim.trigger_dict['Inhale'])
                        callback('Inhale')
                    while hale_cur < inhale_dur:
                        hale_cur = np.minimum((interface.timestamp_ms() - time_init) / 1000, inhale_dur)
                        hale_cur = np.maximum(hale_cur, 0)
                        progress = psylab.string.prog(hale_cur, width=50, char_done="=", maximum=inhale_dur)
                        this_prompt = f"{prompt}\n{progress} {np.int(hale_cur)} / {inhale_dur}"
                        interface.update_Prompt(this_prompt, show=True, redraw=True)
                        time.sleep(.2)
                    hold_cur = hold_dur
                    prompt = f"Hold ({hold_dur} sec)..."
                    time_init = interface.timestamp_ms()
                    if callback:
                        #exp.interface.update_Status_Right(f"trigger {exp.stim.trigger_dict['Hold_Breath']}")
                        #exp.user.triggers.trigger(exp.stim.trigger_dict['Hold_Breath'])
                        callback('Hold')
                    while hold_cur > 0:
                        hold_cur = np.minimum(hold_dur - ((interface.timestamp_ms() - time_init) / 1000), hold_dur)
                        hold_cur = np.maximum(hold_cur, 0)
                        progress = psylab.string.prog(hold_cur, width=50, char_done="=", maximum=hold_dur)
                        this_prompt = f"{prompt}\n{progress} {hold_dur - np.int(hold_cur)} / {hold_dur}"
                        interface.update_Prompt(this_prompt, show=True, redraw=True)
                        time.sleep(.2)
                    hale_cur = exhale_dur
                    prompt = f"Exhale ({j+1}/{breaths_per_block})..."
                    time_init = interface.timestamp_ms()
                    if callback:
                        #exp.interface.update_Status_Right(f"trigger {exp.stim.trigger_dict['Exhale']}")
                        #exp.user.triggers.trigger(exp.stim.trigger_dict['Exhale'])
                        callback('Exhale')
                    while hale_cur > 0:
                        hale_cur = np.minimum(exhale_dur - ((interface.timestamp_ms() - time_init) / 1000), exhale_dur)
                        hale_cur = np.maximum(hale_cur, 0)
                        progress = psylab.string.prog(hale_cur, width=50, char_done="=", maximum=exhale_dur)
                        this_prompt = f"{prompt}\n{progress} {exhale_dur - np.int(hale_cur)} / {exhale_dur}"
                        interface.update_Prompt(this_prompt, show=True, redraw=True)
                        time.sleep(.2)


def test_keyboard_input(keys=[curses.KEY_UP, curses.KEY_DOWN, curses.KEY_ENTER], labels=["Up", "Down","Enter"]):

    interface = theForm.Interface()
    interface.update_Title_Center("Input Test")
    interface.update_Prompt("Keyboard Input Test\n\nHit a key to continue\n(q or / to quit)", show=True, redraw=True)
    ret = interface.get_resp()
    if ret in ['q', '/']:
        quit = True
    else:
        quit = False
    probe_ild = 0
    for key,label in zip(keys, labels):
        if not quit:
            interface.update_Prompt(f"Press the {label} key\n\n(q or / to quit)", show=True, redraw=True)
            got_key = False
            while not got_key:
                ret = interface.get_resp()
                interface.update_Status_Left(f"Key: {ord(ret)}: '{ret}'", redraw=True)
                if ret == 'q' or ret == '/':
                    got_key = True
                    quit = True
                elif ord(ret) == key:
                    got_key = True
    interface.destroy()


def get_comfortable_level(sig, audio_dev, fs=44100, tone_dur_s=1, tone_level_start=1, atten_start=50, ear='both', key_up=curses.KEY_UP, key_dn=curses.KEY_DOWN):
    """A task to allow a participant to adjust the signal to a comfortable level

        Parameters
        ----------
        sig : numeric or 1-d numpy array
            If sig is a number, the signal will be a tone of that frequency. If sig is
            an array, it is taken as the signal

        audio_dev : int
            audio_dev should be an index to an audio device, as specified by 
            medussa.print_available_devices()

        fs : int
            The sampling frequency to use

        tone_dur_s : float
            If a tone is being used, the tone duration, in seconds

        atten_start : float
            The amount of attenuation, in dB, to start with. Be careful with
            this parameter, make it a large number since you don't want to 
            start too loud.

        ear : str
            The ear to present to. Should be one of: 'left', 'right', 'both'

        key_up : int
            The key to accept to increase signal level. Default is the up key

        key_dn : int
            The key to accept to decrease signal level. Default is the down key

        Returns
        -------
        atten : float
            The amount of attenuation, in dB, that should be applied for the signal
            level to be comfortable to the subject. 
    """
    d = m.open_device(audio_dev, audio_dev, 2)

    interface = theForm.Interface()
    interface.update_Title_Center("MCL1")
    interface.update_Prompt("Now you will adjust the volume so that it's at a comfortable level\n\nPress your button to continue\n(q or / to quit)", show=True, redraw=True)
    ret = interface.get_resp()

    response = False

    if ret not in ['q', '/']:
        #if ret not in ['t']: # the problem is ret only listen for 1 input?
        listen_moveon = True
        while listen_moveon:
            ret = interface.get_resp()
            if ret in ['b','y','g']:
                listen_moveon = False
        interface.update_Prompt("Adjust the volume with your buttons to \nincrease (button which?) or decrease (button which?) the volume,\nuntil it's at a comfortable level.\n\nPress (button which?) when finished\n(q or / to quit)", show=True, redraw=True)
        if isinstance(sig, (int, float, complex)):
            # Assume tone
            probe_sig = psylab.signal.tone(sig, fs, tone_dur_s*1000, amp=tone_level_start)
            probe_sig = psylab.signal.ramps(probe_sig, fs)
        else:
            # Assume signal
            probe_sig = sig

        stream = d.open_array(probe_sig, fs)
        mix_mat = stream.mix_mat
        if ear == 'left':
            mix_mat[:,1] = 0
        elif ear == 'right':
            mix_mat[:,0] = 0            # Left channel off
            if mix_mat.shape[1] == 2:   # If 2-channel sig
                mix_mat[1,1] = 1        # then route sig channel 2 to right (stereo)
            else:                       # otherwise
                mix_mat[1] = 1          # route sig channel 1 to right (diotic)
        else:
            if mix_mat.shape[1] == 1:   # route to both ears, so if 1 channel
                mix_mat[1] = 1          # then diotic. 2-channel should be done already

        probe_atten = atten_start
        stream.mix_mat = psylab.signal.atten(mix_mat, probe_atten)
        stream.loop(True)
        stream.play()

        quit = False
        while not quit:
            ret = interface.get_resp()
            interface.update_Status_Left(f'Enter: {ord(ret)}; {curses.KEY_ENTER}', redraw=True)
            if ret == 'q' or ret == '/':
                quit = True
            elif ord(ret) in (curses.KEY_ENTER, 10, 13, ord('g')):
                interface.update_Status_Right('Enter', redraw=True)
                quit = True
            elif ord(ret) == key_up:                               # Volume up
                response = True
                interface.update_Status_Right('Up', redraw=True)
                probe_atten -= 1
                if probe_atten <= 0:
                    probe_atten = 0
                    interface.update_Status_Center(f'0 Attenuation!', redraw=True)
            elif ord(ret) == key_dn:                               # Volume down
                response = True
                interface.update_Status_Right('Down', redraw=True)
                probe_atten += 1
            stream.mix_mat = psylab.signal.atten(mix_mat, probe_atten)

        stream.stop()
    interface.destroy()
    return probe_atten


def get_centered_image(sig, audio_dev, adj_step, fs=44100, tone_dur_s=1, tone_level_start=1, key_l=curses.KEY_LEFT, key_r=curses.KEY_RIGHT):
    """A task to allow a participant to center a stereo image in their head (using ILDs)

        Parameters
        ----------
        sig : numeric or 1-d numpy array
            If sig is a number, the signal will be a tone of that frequency. If sig is
            an array, it is taken as the signal

        audio_dev : int
            audio_dev should be an index to an audio device, as specified by 
            medussa.print_available_devices()

        fs : int
            The sampling frequency to use

        tone_dur_s : float
            If a tone is being used, the tone duration, in seconds

        tone_level_start : float
            If a tone is being used, the tone level to start with. Values 0 > 1 are
            treated as peak voltages. Values <=0 are treated as dB values relative
            to peak (+/-1).

        key_l : int
            The key to accept to move the image to the left. Default is the left key

        key_r : int
            The key to accept to move the image to the right. Default is the right key

        Returns
        -------
        ild : float
            The interaural difference in dB that resulted in a centered image. Negative 
            values indicate that attenuation should be applied to the left ear (ie., the 
            original image was to the left, thus the left ear 
            should be attenuated by that amount). 
    """
    d = m.open_device(audio_dev, audio_dev, 2)

    interface = theForm.Interface()
    interface.update_Title_Center("Stereo Centering")
    interface.update_Prompt("Now you will move the sound image to the center\n\nPress your button to continue\n(q or / to quit)", show=True, redraw=True)
    ret = interface.get_resp()

    response = False

    if ret not in ['q', '/']:
        #if ret not in ['t']:
        listen_moveon = True
        while listen_moveon:
            ret = interface.get_resp()
            if ret in ['b','y','g']:
                listen_moveon = False
        interface.update_Prompt("Use your buttons to shift the location of the sound\nto the left (button which?) or to the right (button which?),\nadjust to make the sound centered in your head\n\nPress (button which?) when finished\n(q or / to quit)", show=True, redraw=True)
        if isinstance(sig, (int, float, complex)):
            # Assume tone
            probe_sig = psylab.signal.tone(sig, fs, tone_dur_s*1000, amp=tone_level_start)
            probe_sig = psylab.signal.ramps(probe_sig, fs)
        else:
            # Assume signal
            probe_sig = sig

        sig_out = np.vstack((probe_sig, probe_sig)).T

        stream = d.open_array(sig_out, fs)
        stream.loop(True)
        stream.play()

        quit = False
        probe_ild = 0
        while not quit:
            ret = interface.get_resp()
            interface.update_Status_Left(f'Enter: {ord(ret)}; {curses.KEY_ENTER}', redraw=True)
            if ret == 'q' or ret == '/':
                quit = True
            elif ord(ret) in (curses.KEY_ENTER, 10, 13, ord('g')):
                interface.update_Status_Right('Enter', redraw=True)
                quit = True
            elif ord(ret) == key_l:                               # Left
                response = True
                interface.update_Status_Right('Left', redraw=True)
                probe_ild -= adj_step
            elif ord(ret) == key_r:                               # Right
                response = True
                interface.update_Status_Right('Right', redraw=True)
                probe_ild += adj_step

            mix_mat = np.zeros((2,2))
            if probe_ild > 0:
                mix_mat[0,0] = psylab.signal.atten(1, probe_ild)
                mix_mat[1,1] = 1
            else:
                mix_mat[1,1] = psylab.signal.atten(1, -probe_ild)
                mix_mat[0,0] = 1
            stream.mix_mat = mix_mat
        stream.stop()
    interface.destroy()
    if response:
        return probe_ild
    else:
        return None


def get_loudness_match(ref, probe, audio_dev, fs=44100, tone_dur_s=.5, tone_level_start=1, isi_s=.2, step=1, key_up=curses.KEY_UP, key_dn=curses.KEY_DOWN):
    """A loudness matching task

        Parameters
        ----------
        ref : numeric or 1-d numpy array
            If ref is a number, the reference signal will be a tone of that frequency. If ref is
            an array, it is taken as the reference signal

        probe : numeric or 1-d array, or a list of same
            If probe is not a list, it is treated similarly to the ref parameter. If it is a list, 
            each item will be looped through

        audio_dev : int
            audio_dev should be an index to an audio device, as specified by 
            medussa.print_available_devices()

        fs : int
            The sampling frequency to use

        tone_dur_s : float
            If tones are being used, the tone duration, in seconds

        tone_level_start : float
            If tones are being used, the tone level to start with. Values 0 > 1 are
            treated as peak voltages. Values <=0 are treated as dB values relative
            to peak (+/-1).

        isi_s : float
            The signals are presented 1 after the other in a looped fashion. isi_s is
            the amount of time, in seconds, to wait before playing each signal again

        step : float
            The stepsize to use when increasing or decreasing the level

        key_up : int
            The key to accept to increase volume. Default is the up key

        key_dn : dn
            The key to accept to decrease volume. Default is the down key

        Returns
        -------
        responses : float or list of floats
            If probe is a number, response is the dB difference between the reference and the probe.
            If probe is a list, then response is a list of dB differences between the reference and
            each probe in the list

    """

    d = m.open_device(audio_dev, audio_dev, 2)

    if not isinstance(probe, list):
        probes = [probe]
    else:
        probes = probe.copy()

    isi_sig = np.zeros(psylab.signal.ms2samp(isi_s*1000, fs))
    interface = theForm.Interface()
    interface.update_Title_Center("Loudness Matching")
    interface.update_Prompt("Match the loudness of tone 2 to tone 1\n\nHit a key to continue\n(q or / to quit)", show=True, redraw=True)
    interface.update_Prompt(f"{probes}", show=True, redraw=True)
    ret = interface.get_resp()

    responses = []

    if ret not in ['q', '/']:

        for this_probe in probes:

            #interface.update_Prompt("Use up & down to match\nthe loudness of tone 2 to tone 1\n\nHit enter when finished\n(q or / to quit)", show=True, redraw=True)
            if isinstance(this_probe, (int, float, complex)):
                # Assume tone
                probe_sig = psylab.signal.tone(this_probe, fs, tone_dur_s*1000, amp=tone_level_start)
                probe_sig = psylab.signal.ramps(probe_sig, fs)
            else:
                # Assume signal
                probe_sig = this_probe

            if isinstance(ref, (int, float, complex)):
                # Rebuild ref_sig each time, otherwise it will leak
                ref_sig = psylab.signal.tone(ref, fs, tone_dur_s*1000, amp=tone_level_start)
                ref_sig = psylab.signal.ramps(ref_sig, fs)
            else:
                ref_sig = ref

            pad_ref = np.zeros(ref_sig.size)
            pad_probe = np.zeros(probe_sig.size)

            ref_sig_build = np.concatenate((ref_sig, pad_probe, isi_sig))
            probe_sig_build = np.concatenate((pad_ref, probe_sig, isi_sig))
            while ref_sig_build.size < fs*5:
                ref_sig_build = np.concatenate((ref_sig_build, ref_sig, pad_probe, isi_sig))
                probe_sig_build = np.concatenate((probe_sig_build, pad_ref, probe_sig, isi_sig))
            
            sig = np.vstack((ref_sig_build, probe_sig_build)).T

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
                elif ord(ret) == key_dn:                               # Down
                    interface.update_Status_Right('Down', redraw=True)
                    probe_level = psylab.signal.atten(probe_level, step)
                elif ord(ret) == key_up:                               # Up
                    interface.update_Status_Right('Up', redraw=True)
                    probe_level = psylab.signal.atten(probe_level, -step)
                interface.update_Status_Left(f'{probe_level}', redraw=True)
                mix_mat[:,1] = probe_level
                stream.mix_mat = mix_mat
            if quit_request:
                break
            else:
                responses.append(20*np.log10(probe_level/1))
            stream.stop()
        interface.destroy()
        if len(responses) == 1:
            return responses[0]
        else:
            return responses
