# this script is used for testing eyetracker commands 

# TODO: where is eyelink installed on bridge center computer? do we launch through conda or powershell?

from cmath import atan
import os
from pickle import FALSE
import time
import numpy as np
import pylink 

from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from gustav.forms import rt as theForm

from psychopy import monitors, visual

# EyeLinkCoreGraphicsPsychoPy.py is a script included in EDK psychopy example folders
# it is used for calibration along with 3 other video files 

# setup edf file, filename cannot exceed 8 characters 
edf_filename = 'TEST' + time.strftime("%H%M") + '.edf'

##########################################
# connect to tracker 
##########################################

try:
    el_tracker = pylink.EyeLink("100.1.1.1")
except RuntimeError as error:
    print('ERROR: ', error)

##########################################
# open edf file 
##########################################

try: 
    el_tracker.openDataFile(edf_filename)
except RuntimeError as err:
    print('ERROR: ', err)
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# add header information in EDF file, this will strip out current script name
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

##########################################
# configure eyetracker 
##########################################

# put to offline mode before change parameters
el_tracker.setOfflineMode()

vDistance = 107.5           # scanner viewing distance
dWidth = 41.5               # scanner display width
win_w = 800                 # TODO: rect(3) in Screen rect output psychtoolbox
win_h = 600                 # TODO: rect(4) in Screen rect output psychtoolbox
ppd = np.pi * win_w / atan(dWidth/vDistance/2) /360     # TODO: why pixels per degree compute this way?

# print software version info
vstr = el_tracker.getTrackerVersionString()
EyeLink_ver = int(vstr.split()[-1].split('.')[0])
print("Running experiment on %s, version %d" % (vstr, EyeLink_ver))

# what eye events to save in the edf file 
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'

# what eye events to make available over the link
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

# what sample data to save in the EDF data file and make available over the link
file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'

el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# optional set sample rate el_tracker.sendCommand("sample_rate 1000")

# choose calibration type
el_tracker.sendCommand("calibration_type = HV9")

# TODO: if default Enter is not working, then add button for accepting fixation
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")


##########################################
# open a window 
##########################################

# we need to open a window to present stimuli and calibrate the tracker
# we need to send correct screen resolution to the Host PC with the screen_pixel_coords 
# and log this information in EDF file with DISPLAY_COORDS message 

# TODO: pass the display pixel coordinates (top left, bottom right)
el_coords = "screen_pixel_coords = 0 0 %d %d" % (win_w - 1, win_h - 1)
el_tracker.sendCommand(el_coords)

# write DISPLAY_COORDS message to EDF file
dv_coords = "DISPLAY_COORDS 0 0 %d %d" % (win_w - 1, win_h - 1)
el_tracker.sendMessage(dv_coords) 


##########################################
# calibration
##########################################

# configure calibration graphics before calibration
# Pylink will use EyeLinkCoreGraphicsPsychoPy.py to draw camera image/target/play beeps

# gustav won't work for this calibration process as well 
# win = theForm.Interface()

# TODO: psychopy monitors.getAllMonitors() can only find one monitor, always show on default screen
# could it be getAllMonitors can only find monitors for which calibration files exist? 
# what is this calibration file? 
mon = monitors.Monitor('testMonitor', width = dWidth, distance = vDistance) 
win = visual.Window(size=(1920,1200), screen=1) # fullscr=FALSE, monitor=mon, winType='pyglet', units='pix'
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)

foreground_color = (-1,-1,-1) # black
background_color = win.color 
genv.setCalibrationColors(foreground_color,background_color)
# not sure if setTargetType and setTargetSize and setCalibrationSounds are necessary

# request pylink to use CoreGraphics for calibration
pylink.openGraphicsEx(genv)


try:
    el_tracker.doTrackerSetup() 
except RuntimeError as err:
    print("ERROR: ", err)
    el_tracker.exitCalibration() 



##########################################
# running trials
##########################################

# TODO: do we need to close psychopy window as we're using gustav? will that affect host screen? 

# TODO: eye tracker events during trials