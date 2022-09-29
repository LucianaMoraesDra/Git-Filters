import PySimpleGUI as sg
from Lu_defs_test_Lu import *

sg.theme('GrayGrayGray')   # Add a touch of color

# All the stuff inside your window.
font = ("Helvetica", 16)
# sg.set_options(font=font)
layout = [ 
    [sg.Text("Selectionez votre filter", font=font)], # line
    [sg.Text()],  # line

    [sg.Button("mirror"), sg.Button("glow"),  sg.Button("sepia"), sg.Button("lips")],  
    [sg.Button("blackwhite"), sg.Button("x_ray"),  sg.Button("cartoon")], 
    [sg.Button("draw"), sg.Button("thermal_camera"), sg.Button("cat")],  
    [sg.Button("ghost_color"), sg.Button("kaleidoscope"),  sg.Button("red_nose")],  
    [sg.Button("flashing_nose"), sg.Button("groucho_marx"),  sg.Button("carnaval")], 
    [sg.Button("Annuler")], 

]

# Create the Window
window = sg.Window("As-tu souri aujourd'hui ?", layout,  margins=(100, 100))

filter_dic = {"mirror": mirror, "glow": glow, "sepia": sepia, "blackwhite": blackwhite, "x_ray": x_ray,
        "cartoon": cartoon, "draw": draw, "thermal_camera": thermal_camera, "lips": lips, "cat": cat, "ghost_color": ghost_color,
        "kaleidoscope": kaleidoscope, "red_nose": red_nose, "flashing_nose": flashing_nose, "groucho_marx": groucho_marx,
        "carnaval": carnaval}

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Annuler': # if user closes window or clicks cancel
        break

    else:
        filter_choise(filter_dic[event])

window.close()
