import numpy as np
import medussa as m
import psylab
from gustav.forms import rt as theForm
from . import utils

def find_convergence_bounds(all_matched_levels):
    '''
    This function is used for finding the boundaries for determining whether the next match converges or not. 
    E.g. return lower bound and upper bound, if next matched value falls in this range then decide it's converged. 
    '''
    # compute mean and std with minimum matched data
    last_mean = np.nanmean(all_matched_levels,axis=0)
    last_std = np.nanstd(all_matched_levels-last_mean) # ,axis=0, taking all samples' std to avoid harder convergence for low variance frequencies in baseline

    # create initial boundaries
    upper_bound = last_mean + last_std
    lower_bound = last_mean - last_std

    return lower_bound, upper_bound, last_mean


def fill_matched_levels(this_matched_levels, conv_check, last_mean):
    '''
    Given new matched levels for part of the matching pool, this function create an array with same size as 
    matching pool, fill in new matched levels for those frequencies having new matches, and fill in past average 
    matched values for those frequencies that the matching has already converged.
    '''
    matched_levels = np.zeros(len(conv_check))
    matched_levels[conv_check.astype(bool)] = last_mean[conv_check.astype(bool)] # set matched values to last mean if already converges
    matched_levels[(1-conv_check).astype(bool)] = this_matched_levels              # add this matched values to this sample

    return matched_levels

def get_loudness_match(ref, probe, dev_id, fs=44100, tone_dur_s=.5, tone_level_start=.5, isi_s=.2, do_addnoise=False, step=0.5, round_idx=0, key_up='', key_dn='', key_enter=''):
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

        tone_level_start : float
            If tones are being used, the tone level to start with

        Returns
        -------
        responses : float or list of floats
            If probe is a number, response is the dB difference between the reference and the probe.
            If probe is a list, then response is a list of dB differences between the reference and
            each probe in the list.

    """

    d = m.open_device(*dev_id) 

    if not isinstance(probe, list):
        probes = [probe]
    else:
        probes = probe.copy()

    isi_sig = np.zeros(psylab.signal.ms2samp(isi_s * 1000, fs))
    
    interface = theForm.Interface()
    interface.update_Prompt("Now starting loudness matching round "+str(round_idx+1)+"\n\nPress your button to continue",show=True, redraw=True)
    utils.wait_for_subject(interface)

    responses = []

    for probe in probes:
        interface.update_Prompt(
            "Use up (button which?) & down (button which?) to match\nthe loudness of tone 2 to tone 1,\nuntil they sound samely loud to you\n\nPress (button which?) when finished",
            show=True, redraw=True)
        if isinstance(probe, (int, float, complex, list)):
            if isinstance(probe,list): # complex tone
                probe_sig_1 = psylab.signal.tone(probe[0], fs, tone_dur_s*1000, amp=0.5)
                probe_sig_2 = psylab.signal.tone(probe[1], fs, tone_dur_s*1000, amp=0.5)

                probe_sig_1 = probe_sig_1*tone_level_start/utils.computeRMS(probe_sig_1)
                probe_sig_2 = probe_sig_2*tone_level_start/utils.computeRMS(probe_sig_2)

                probe_sig = probe_sig_1 + probe_sig_2
                probe_sig = probe_sig*tone_level_start/utils.computeRMS(probe_sig)
            else: # pure tone
                probe_sig = psylab.signal.tone(probe, fs, tone_dur_s * 1000, amp=0.5)
                probe_sig = probe_sig * tone_level_start/utils.computeRMS(probe_sig)
            probe_sig = psylab.signal.ramps(probe_sig, fs)
        else:
            # Assume signal
            probe_sig = probe

        if isinstance(ref, (int, float, complex)):
            # Rebuild ref_sig each time, otherwise it will leak
            ref_sig = psylab.signal.tone(ref, fs, tone_dur_s * 1000, amp=0.5)
            ref_sig = ref_sig * tone_level_start/utils.computeRMS(ref_sig)
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

        stream = d.open_array(sig, fs)
        stream.loop(True)

        mix_mat = stream.mix_mat
        mix_mat[:] = 1
        stream.mix_mat = mix_mat

        stream.play()

        quit = False
        probe_level = 1
        while not quit:
            ret = interface.get_resp()
            interface.update_Status_Left(f'Entered: {ret}', redraw=True)

            if ret == key_enter:
                interface.update_Status_Right('Enter', redraw=True)
                quit = True
            elif ret == key_dn:  # Down
                interface.update_Status_Right('Down', redraw=True)
                probe_level = psylab.signal.atten(probe_level, step)
            elif ret == key_up:  # Up
                interface.update_Status_Right('Up', redraw=True)
                probe_level = psylab.signal.atten(probe_level, -step)
            mix_mat[:, 1] = probe_level
            stream.mix_mat = mix_mat

        responses.append(20 * np.log10(probe_level / 1))
        stream.stop()
    interface.destroy()

    return responses