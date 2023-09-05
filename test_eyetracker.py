# this script is modified from link_sample.py, and is used for testing, 

from __future__ import division
from __future__ import print_function

import time
import sys
import os
import pylink

from screeninfo import get_monitors


#############################################################
## SETUP 
#############################################################

os.environ['SDL_VIDEO_FULLSCREEN_HEAD'] = '1'
do_use_extend_monitor = True

if do_use_extend_monitor:
    if len(get_monitors()) >1:
        extended_monitor = get_monitors()[1]
    else:
        print("No extended monitor founded!")

# so resource stimuli like images and sounds can be located with relative paths
script_path = os.path.dirname(sys.argv[0]) # argv[0] is the 0th input in command line, which is the script name 
if len(script_path) != 0:
    os.chdir(script_path)

# some global constants
RIGHT_EYE = 1
LEFT_EYE = 0
BINOCULAR = 2

N_TRIALS = 3
TRIAL_DUR = 5000


if sys.version_info > (3,0):
    ok = input("\nPress 'Y' to continue, 'N' to quit: ")
else: 
    ok = raw_input("\nPress 'Y' to continue, 'N' to quit: ")
if ok not in ['Y', 'y']:
    sys.exit()

SCN_WIDTH = extended_monitor.width #0
SCN_HEIGHT = extended_monitor.height #0


trial_condition = ['condition1', 'condition2', 'condition3']

# set up a folder to store the data files
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


#############################################################
## FUNCTIONS
#############################################################

def run_trials():
    """ 
    Handle trials
    """

    # get the currently active tracker object (connection)
    el_tracker = pylink.getEYELINK()

    for trial in range(N_TRIALS):
        # first check if the connection to the tracker is still alive
        if(not el_tracker.isConnected() or el_tracker.breakPressed()):
            break

        while True:

        	# run trial
            ret_value = do_trial(trial)

            if (ret_value == pylink.TRIAL_OK):
                el_tracker.sendMessage("TRIAL OK")
                break
            elif (ret_value == pylink.SKIP_TRIAL):
                el_tracker.sendMessage("TRIAL ABORTED")
                break
            elif (ret_value == pylink.ABORT_EXPT):
                el_tracker.sendMessage("EXPERIMENT ABORTED")
                return pylink.ABORT_EXPT
            elif (ret_value == pylink.REPEAT_TRIAL):
                el_tracker.sendMessage("TRIAL REPEATED")
            else:
                el_tracker.sendMessage("TRIAL ERROR")
                break
    return 0


def do_trial(trial):
    """ Run a single trial """

    # get the currently active tracker object (connection)
    el_active = pylink.getEYELINK()

    # show some info about the current trial on the Host PC screen
    pars_to_show = (trial_condition[trial], trial + 1, N_TRIALS)
    status_message = 'Link event example, %s, Trial %d/%d' % pars_to_show
    el_tracker.sendCommand("record_status_message '%s'" % status_message)

    el_tracker.sendMessage("TRIALID %d" % trial)
    el_tracker.sendCommand("clear_screen 0")
    el_tracker.setOfflineMode()

    # start recording samples and events; save them to the EDF file and
    # make them available over the link
    error = el_tracker.startRecording(1, 1, 1, 1)
    if error:
        return error

    # INSERT CODE TO DRAW INITIAL DISPLAY HERE

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

    # reset keys and buttons on tracker
    el_tracker.flushKeybuttons(0)

    # get trial start time
    start_time = pylink.currentTime()
    # poll link events and samples
    while True:
        # first check if recording is aborted
        # (returns 0 if no error, otherwise return codes, e.g.,
        # REPEAT_TRIAL, SKIP_TRIAL, ABORT_EXPT, TRIAL_ERROR )
        error = el_tracker.isRecording()
        if error != pylink.TRIAL_OK:
            end_trial()
            return error

        # check if trial duration exceeded
        if pylink.currentTime() > (start_time + TRIAL_DUR):
            el_tracker.sendMessage("TIMEOUT")
            end_trial()
            break

        # program termination or ALT-F4 or CTRL-C keys
        if el_tracker.breakPressed():
            end_trial()
            return pylink.ABORT_EXPT

        # check for local ESC key to abort trial (useful in debugging)
        elif el_tracker.escapePressed():
            end_trial()
            return pylink.SKIP_TRIAL
        

    # record the trial variable in a message recognized by Data Viewer
    el_active.sendMessage("!V TRIAL_VAR trial %d" % trial)

    msg_test = trial+100
    el_active.sendMessage("!V TRIAL_VAR msg_test %d" % msg_test)

    # return exit record status
    ret_value = el_active.getRecordingStatus()

    return ret_value


def end_trial():
    """Ends recording

    We add 100 msec of data to catch final events"""

    # get the currently active tracker object (connection)
    el_active = pylink.getEYELINK()
    el_active.stopRecording()

    while el_active.getkey():
        pass



###########################################################
## main
###########################################################

try:
    el_tracker = pylink.EyeLink("100.1.1.1")
except RuntimeError as error:
    print('ERROR:', error)
    sys.exit()


pylink.openGraphics((SCN_WIDTH, SCN_HEIGHT), 32) # 32 bits

edf_file_name = "TEST.EDF"
el_tracker.openDataFile(edf_file_name)

# add a preamble text (data file header)
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)


pylink.flushGetkeyQueue()
el_tracker.setOfflineMode()


pix_msg = "screen_pixel_coords 0 0 %d %d" % (SCN_WIDTH - 1, SCN_HEIGHT - 1)
el_tracker.sendCommand(pix_msg)
# The Data Viewer software also needs to know the screen
# resolution for correct visualization
dv_msg = "DISPLAY_COORDS  0 0 %d %d" % (SCN_WIDTH - 1, SCN_HEIGHT - 1)
el_tracker.sendMessage(dv_msg)


vstr = el_tracker.getTrackerVersionString()
eyelink_ver = int(vstr.split()[-1].split('.')[0])
# print out some version info in the shell
print('Running experiment on %s, version %d' % (vstr, eyelink_ver))


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
pylink.setCalibrationColors((0, 0, 0), (128, 128, 128))

# select best size for calibration target
pylink.setTargetSize(int(SCN_WIDTH/70.0), int(SCN_WIDTH/300.))

# Set the calibraiton and drift correction sound
pylink.setCalibrationSounds("", "", "")
pylink.setDriftCorrectSounds("", "", "")

# Step 5: Do the tracker setup at the beginning of the experiment.
el_tracker.doTrackerSetup()

# Step 6: Run trials. make sure display-tracker connection is established
# and no program termination or ALT-F4 or CTRL-C pressed
if el_tracker.isConnected() and not el_tracker.breakPressed():
    run_trials()

print("Finished running trials.")


if el_tracker is not None:
    el_tracker.setOfflineMode()
    pylink.msecDelay(500)

    # Close the edf data file on the Host
    el_tracker.closeDataFile()

    # transfer the edf file to the Display PC and rename it
    local_file_name = os.path.join(results_folder, edf_file_name)

    try:
        el_tracker.receiveDataFile(edf_file_name, local_file_name)
    except RuntimeError as error:
        print('ERROR:', error)

# Step 8: close EyeLink connection and quit display-side graphics
el_tracker.close()
# Close the experiment graphics
pylink.closeGraphics()



