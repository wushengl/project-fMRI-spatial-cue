import curses
import gustav
from gustav.forms import rt as theForm

def test_keyboard_input(keys=[curses.KEY_UP, curses.KEY_DOWN, curses.KEY_ENTER], labels=["Up", "Down","Enter"]):

    interface = theForm.Interface()
    interface.update_Title_Center("Input Test")
    interface.update_Prompt("Keyboard Input Test\n\nHit a key to continue\n(q or / to quit)", show=True, redraw=True)
    ret = interface.get_resp()
    if ret in ['q', '/']:
        quit = True
    else:
        quit = False
    probe_ild = 0
    for key,label in zip(keys, labels):
        if not quit:
            interface.update_Prompt(f"Press the {label} key\n\n(q or / to quit)", show=True, redraw=True)
            got_key = False
            while not got_key:
                ret = interface.get_resp()
                interface.update_Status_Left(f"Key: {ord(ret)}: '{ret}'", redraw=True)
                if ret == 'q' or ret == '/':
                    got_key = True
                    quit = True
                elif ord(ret) == key:
                    got_key = True
    interface.destroy()


