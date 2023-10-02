
# README

SASC is short for Spatial Attention Spatial Cue. 

This folder contains re-structured code for Python tasks, including 
- keyboard test 
- get comfortable level 
- get centered image 
- get loudness match 
- tonotopy (1-back working memory)
- main task (zig-zag pattern recognition)

The tasks will be arranged as
- 1 config file for all running settings 
- 1 function folder for all function scripts and test scripts
- 1 data folder for all data (1 subfolder for each subject)
- separate run scripts for each tasks

### Function folder
- func_soundtest.py
- func_matchtone.py
- func_tonotopy.py
- func_zigzagtask.py
- utils.py (helper functions)
- stimuli.py (generate zigzag stream and spatialization)

### Data folder
- s001
  - s001_soundtest.csv (ref_rms, probe_ild)
  - s001_matchtone.csv (matched loudness w.r.t. ref_rms)
  - s001_tonotopy.csv (behavior and timestamp)
  - s001_zigzagtask.csv (behavior and timestamp)
  - s001TIME.bdf (eye tracker)
  - log_s001_prpetask.log 
  - log_s001_zigzagtask.log

### To run a subject
- For each session, MUST run soundtest first, which will save a ref_rms and a probe_ild for adjusting the sound level and get centered image. 