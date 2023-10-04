import curses
import medussa as m
import psylab
import gustav
from gustav.forms import rt as theForm
import numpy as np

from functions import utils


def get_comfortable_level(sig, dev_id, fs=44100, tone_dur_s=1, tone_level_start=1, atten_start=50, key_up='', key_dn='', key_enter=''):
    """A task to allow a participant to adjust the signal to a comfortable level

        Parameters
        ----------
        sig : numeric or 1-d numpy array
            If sig is a number, the signal will be a tone of that frequency. If sig is
            an array, it is taken as the signal

        dev_id : tuple
            dev_id is tuple for (id, id, ch) 
            medussa.print_available_devices()

        atten_start : float
            The amount of attenuation, in dB, to start with. Be careful with
            this parameter, make it a large number since you don't want to 
            start too loud.

        Returns
        -------
        atten : float
            The amount of attenuation, in dB, that should be applied for the signal
            level to be comfortable to the subject. 
    """

    d = m.open_device(*dev_id)

    # Initialize window and show instruction
    interface = theForm.Interface()
    interface.update_Title_Center("MCL1")
    interface.update_Prompt("Now you will adjust the volume so that it's at a comfortable level\n\nPress your button to continue", show=True, redraw=True)
    utils.wait_for_subject(interface)

    # start the task
    interface.update_Prompt("Adjust the volume with your buttons to \nincrease (button which?) or decrease (button which?) the volume,\nuntil it's at a comfortable level.\n\nPress (button which?) when finished", show=True, redraw=True)
    
    if isinstance(sig, (int, float, complex)):
        # Assume tone
        probe_sig = psylab.signal.tone(sig, fs, tone_dur_s*1000, amp=tone_level_start)
        probe_sig = psylab.signal.ramps(probe_sig, fs)
    else:
        # Assume signal
        probe_sig = sig

    stream = d.open_array(probe_sig, fs)
    mix_mat = stream.mix_mat

    if mix_mat.shape[1] == 1:   # route to both ears, so if 1 channel
        mix_mat[1] = 1          # then diotic. 2-channel should be done already

    probe_atten = atten_start
    stream.mix_mat = psylab.signal.atten(mix_mat, probe_atten)
    stream.loop(True)
    stream.play()

    quit = False
    while not quit:
        ret = interface.get_resp()
        interface.update_Status_Left(f'Entered: {ret}', redraw=True)

        if ret == key_enter:
            interface.update_Status_Right('Enter', redraw=True)
            quit = True
        elif ret == key_up:                               # Volume up
            interface.update_Status_Right('Up', redraw=True)
            probe_atten -= 1
            if probe_atten <= 0:
                probe_atten = 0
                interface.update_Status_Center(f'0 Attenuation!', redraw=True)
        elif ret == key_dn:                               # Volume down
            interface.update_Status_Right('Down', redraw=True)
            probe_atten += 1
        stream.mix_mat = psylab.signal.atten(mix_mat, probe_atten)

    stream.stop()
    interface.destroy()

    return probe_atten



def get_centered_image(sig, dev_id, adj_step, fs=44100, tone_dur_s=1, tone_level_start=1, key_l='', key_r='', key_enter=''):
    """A task to allow a participant to center a stereo image in their head (using ILDs)

        Parameters
        ----------
        sig : numeric or 1-d numpy array
            If sig is a number, the signal will be a tone of that frequency. If sig is
            an array, it is taken as the signal

        audio_dev : int
            audio_dev should be an index to an audio device, as specified by 
            medussa.print_available_devices()

        tone_dur_s : float
            If a tone is being used, the tone duration, in seconds

        tone_level_start : float
            If a tone is being used, the tone level to start with. Values 0 > 1 are
            treated as peak voltages. Values <=0 are treated as dB values relative
            to peak (+/-1).

        Returns
        -------
        ild : float
            The interaural difference in dB that resulted in a centered image. Negative 
            values indicate that attenuation should be applied to the left ear (ie., the 
            original image was to the left, thus the left ear 
            should be attenuated by that amount). 
    """
    d = m.open_device(*dev_id)

    # initialize window and instructions 

    interface = theForm.Interface()
    interface.update_Title_Center("Stereo Centering")
    interface.update_Prompt("Now you will move the sound image to the center\n\nPress your button to continue\n(q or / to quit)", show=True, redraw=True)
    utils.wait_for_subject(interface)

    # start of the task

    interface.update_Prompt("Use your buttons to shift the location of the sound\nto the left (button which?) or to the right (button which?),\nadjust to make the sound centered in your head\n\nPress (button which?) when finished\n(q or / to quit)", show=True, redraw=True)
    if isinstance(sig, (int, float, complex)):
        # Assume tone
        probe_sig = psylab.signal.tone(sig, fs, tone_dur_s*1000, amp=tone_level_start)
        probe_sig = psylab.signal.ramps(probe_sig, fs)
        # probe_sig is (n_sample,) signal
    else:
        # Assume signal
        probe_sig = sig

    sig_out = np.vstack((probe_sig, probe_sig)).T
    # sig_out is (n_sample, 2) signal

    stream = d.open_array(sig_out, fs)
    stream.loop(True)
    stream.play()

    logger = utils.init_logger('test','soundtest','./debug/')

    quit = False
    probe_ild = 0
    while not quit:
        ret = interface.get_resp()
        interface.update_Status_Left(f'Enter: {ret}', redraw=True)

        if ret == key_enter:
            interface.update_Status_Right('Enter', redraw=True)
            quit = True
        elif ret == key_l:                               # Left
            interface.update_Status_Right('Left', redraw=True)
            probe_ild -= adj_step
            # probe_ild goes negative when moving left
        elif ret == key_r:                               # Right
            interface.update_Status_Right('Right', redraw=True)
            probe_ild += adj_step
            # probe_ild goes positive when moving right

        logger.info("Key pressed: %s, probe_ild: %f"%(ret, probe_ild))

        # mix_mat is 2*2 array when the sound stream opened is stereo
        mix_mat = utils.apply_probe_ild(np.zeros((2,2)), probe_ild)
        stream.mix_mat = mix_mat

        logger.info("Updated mix_mat: %s"%str(mix_mat))

    stream.stop()
    interface.destroy()

    return probe_ild
