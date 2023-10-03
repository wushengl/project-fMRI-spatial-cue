

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


