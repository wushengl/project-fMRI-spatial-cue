import numpy as np
import curses
import medussa as m
import psylab
import gustav
from gustav.forms import rt as theForm

def get_loudness_match(ref, probe, audio_dev, fs=44100, tone_dur_s=.5, tone_level_start=.5, isi_s=.2):
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

    d = m.open_device(audio_dev, audio_dev, 2)

    if not isinstance(probe, list):
        probes = [probe]
    else:
        probes = probe.copy()

    isi_sig = np.zeros(psylab.signal.ms2samp(isi_s*1000, fs))
    interface = theForm.Interface()
    interface.update_Title_Center("Loudness Matching")
    interface.update_Prompt("Match the loudness of tone 2 to tone 1\n\nHit a key to continue\n(q or / to quit)", show=True, redraw=True)
    ret = interface.get_resp()

    responses = []

    if ret not in ['q', '/']:
        for probe in probes:

            interface.update_Prompt("Use up & down to match\nthe loudness of tone 2 to tone 1\n\nHit enter when finished\n(q or / to quit)", show=True, redraw=True)
            if isinstance(probe, (int, float, complex)):
                # Assume tone
                probe_sig = psylab.signal.tone(probe, fs, tone_dur_s*1000, amp=tone_level_start)
                probe_sig = psylab.signal.ramps(probe_sig, fs)
            else:
                # Assume signal
                probe_sig = probe

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
                elif ord(ret) == curses.KEY_DOWN:                            # Down
                    interface.update_Status_Right('Down', redraw=True)
                    probe_level = psylab.signal.atten(probe_level, 1)
                elif ord(ret) == curses.KEY_UP:                              # Up
                    interface.update_Status_Right('Up', redraw=True)
                    probe_level = psylab.signal.atten(probe_level, -1)
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


if __name__ == '__main__':
    get_loudness_match(2016,566,)