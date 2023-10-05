
import numpy as np
import matplotlib.pyplot as plt
import psylab
import gustav
from gustav.forms import rt as theForm
import pdb
import time


interface = theForm.Interface()
interface.update_Title_Center("fixation")
interface.update_Prompt("   ██   \n   ██   \n████████\n   ██   \n   ██   ", show=True, redraw=True)
wait = True
while wait:
    ret = interface.get_resp()
    if ret in [' ', '/']:
        wait = False
        interface.destroy()
