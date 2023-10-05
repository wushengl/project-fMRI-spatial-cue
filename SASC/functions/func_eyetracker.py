import pylink
import os
from screeninfo import get_monitors
import sys
from . import utils

config_file = 'config/config.json'
config = utils.get_config()

def init_eyetracker():
    
    # Step 1: initialize a tracker object with a Host IP address
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        sys.exit()

    return el_tracker


def init_eyetracker_graphics():

    os.environ['SDL_VIDEO_FULLSCREEN_HEAD'] = '1'

    if len(get_monitors()) >1:
        extended_monitor = get_monitors()[0] # BRIDGE center: display 1 (left) is mirrored into the scanner
    else:
        print("No extended monitor founded!")

    SCN_WIDTH = extended_monitor.width #0
    SCN_HEIGHT = extended_monitor.height #0

    pylink.openGraphics((SCN_WIDTH, SCN_HEIGHT), 32)

    return SCN_WIDTH, SCN_HEIGHT


def send_initial_info(el_tracker, SCN_WIDTH, SCN_HEIGHT):

    # Step 4: setting up tracking, recording and calibration options
    pylink.flushGetkeyQueue()
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
    pylink.setCalibrationColors((0, 0, 0), (128, 128, 128))

    area_proportion_x = config['eyetracker']['area_proportion_x']
    area_proportion_y = config['eyetracker']['area_proportion_x']
    el_tracker.sendCommand("calibration_area_proportion = %s %s"%(str(area_proportion_x), str(area_proportion_y)))

    # select best size for calibration target
    pylink.setTargetSize(int(SCN_WIDTH/70.0), int(SCN_WIDTH/300.))

    # Set the calibraiton and drift correction sound
    pylink.setCalibrationSounds("", "", "")
    pylink.setDriftCorrectSounds("", "", "")


def get_eyetracker():
    el_tracker = pylink.getEYELINK()

    if(not el_tracker.isConnected() or el_tracker.breakPressed()):
        raise RuntimeError("Eye tracker is not connected!")
    
    return el_tracker

if __name__ == '__main__':
    
    init_eyetracker()
