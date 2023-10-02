import curses
import medussa as m
import psylab
import gustav
from gustav.forms import rt as theForm
import numpy as np

from functions import utils

# TODO: make sure sound is playing correctly


def get_comfortable_level(sig, audio_dev, fs=44100, tone_dur_s=1, tone_level_start=1, atten_start=50, ear='both', key_up=curses.KEY_UP, key_dn=curses.KEY_DOWN, key_enter=''):
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

    # Initialize window and show instruction
    interface = theForm.Interface()
    interface.update_Title_Center("MCL1")
    interface.update_Prompt("Now you will adjust the volume so that it's at a comfortable level\n\nPress your button to continue", show=True, redraw=True)
    utils.wait_for_subject(interface=interface)

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

    response = False
    quit = False
    while not quit:
        ret = interface.get_resp()
        interface.update_Status_Left(f'Enter: {ord(ret)}; {curses.KEY_ENTER}', redraw=True)
        if ret == 'q' or ret == '/':
            quit = True
        elif ord(ret) in (curses.KEY_ENTER, 10, 13, ord(key_enter)):
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



def get_centered_image(sig, audio_dev, adj_step, fs=44100, tone_dur_s=1, tone_level_start=1, key_l=curses.KEY_LEFT, key_r=curses.KEY_RIGHT, key_enter=''):
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
    else:
        # Assume signal
        probe_sig = sig

    sig_out = np.vstack((probe_sig, probe_sig)).T

    stream = d.open_array(sig_out, fs)
    stream.loop(True)
    stream.play()

    response = False
    quit = False
    probe_ild = 0
    while not quit:
        ret = interface.get_resp()
        interface.update_Status_Left(f'Enter: {ord(ret)}; {curses.KEY_ENTER}', redraw=True)
        if ret == 'q' or ret == '/':
            quit = True
        elif ord(ret) in (curses.KEY_ENTER, 10, 13, ord(key_enter)):
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