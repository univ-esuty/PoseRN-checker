import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

from compScale import GetScale_OpenPose, GetScale_MoCap
from getPoints import Mocap, Openpose3d
from getPly import GetPly

mocap_conf = {
    'DATASET_DIR_ROOT': 'input_data/opt-mocap',
    'DATASET_FILE' : 'optmocap.trc',
    'JOINT_IDX' : 29,
}

openpose_conf = {
    'DATASET_DIR_ROOT': 'input_data/mv-openpose/3dpose',
    'JOINT_IDX' : 8,
    'FRAME_NUM' : 750,
    'ADJ_SCALE' : 0.9,
    'ADJ_CENTER_Z': 0.012, 
}

export_conf = {
    'EXPORT_DIR' : 'keyframes',
    'FRAME_NUM' : [[0,0]]  # Do not change the parameters.
}

## set scale.
openpose_scale, openpose_center = GetScale_OpenPose(openpose_conf)
mocap_scale, mocap_center = GetScale_MoCap(mocap_conf)

## adjust
tmp = [openpose_scale, openpose_center]
openpose_scale = tmp[0] * openpose_conf['ADJ_SCALE']
openpose_center[2] = tmp[1][2] + openpose_conf['ADJ_CENTER_Z']

def scalize(scale, x, y, z):
    return x / scale, y / scale, z / scale

def centerlize(scale, c, x, y, z):
    c_x, c_y, c_z = scalize(scale, c[0], c[1], c[2])
    return x - c_x, y - c_y, z - c_z


## main()
mocap = Mocap(mocap_conf)
mocap.importData()
openpose = Openpose3d(openpose_conf)

## draw 3d pose point 
fig = plt.figure()
ax = fig.gca(projection='3d')

def update_frame(m_fc, o_fc):
    ax.clear()
    #ax.view_init(elev=30, azim=-90)
    ax.set_xlim(-2, 2); ax.set_ylim(-2.5, 2.5); ax.set_zlim(-1, 1)
    ax.set_xlabel("x", size = 14, weight = "light"); ax.set_ylabel("y", size = 14, weight = "light"); ax.set_zlabel("z", size = 14, weight = "light")

    ## visualize mocap points
    X, Y, Z = mocap.loadPoints(int(m_fc*36))
    X, Y, Z = scalize(mocap_scale, X, Y, Z)
    X, Y, Z = centerlize(mocap_scale, mocap_center, X, Y, Z)
    ax.plot(X, Y, Z, 'k.')

    ## visualize openpose3d points
    X, Y, Z = openpose.loadPoints(o_fc)
    X, Y, Z = scalize(openpose_scale, X, Y, Z)
    X, Y, Z = centerlize(openpose_scale, openpose_center, X, Y, Z)
    ax.plot(X, Y, Z, 'k.')

    X, Y, Z = openpose.setLines(X, Y, Z)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        line = art3d.Line3D(x, y, z, color=plt.cm.jet(255//len(Openpose3d._BonesV2)*i))
        ax.add_line(line)

    #plt.show()
    plt.pause(.01)

## Slider Controller.
from tkinter import *
from tkinter import ttk

root = Tk()
root.title('Configs')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Frame
frame = ttk.Frame(root, padding=10)
frame.grid(sticky=(N, W, S, E))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

val1 = DoubleVar()
val2 = DoubleVar()

sc1 = ttk.Scale(
    frame,
    variable=val1,
    orient=HORIZONTAL,
    length=999,
    from_=0,
    to=749,
    command=lambda e: sync_e1(val1.get()) )

sc2 = ttk.Scale(
    frame,
    variable=val2,
    orient=HORIZONTAL,
    length=999,
    from_=0,
    to=33315//36,
    command=lambda e:  sync_e2(val2.get()) )

lb1 = ttk.Label(frame, text="MocapFrame   ")
lb2 = ttk.Label(frame, text="openposeFrame")

lb1.grid(row=0, column=0)
lb2.grid(row=1, column=0)

sc1.grid(row=0, column=1, sticky=(N, E, S, W))
sc2.grid(row=1, column=1, sticky=(N, E, S, W))

e1 = ttk.Entry(frame)
e2 = ttk.Entry(frame)

e1.grid(row=0, column=2)
e2.grid(row=1, column=2)

def setParam():
    _val1 = int(float(e1.get()))
    _val2 = int(float(e2.get()))
    val1.set(_val1)
    val2.set(_val2)
    update_frame(int(val2.get()), int(val1.get()))

def exportply():
    _val1 = int(float(e1.get()))
    _val2 = int(float(e2.get()))
    val1.set(_val1)
    val2.set(_val2)
    update_frame(int(val2.get()), int(val1.get()))

    export_conf['FRAME_NUM'][0][0] = _val2
    export_conf['FRAME_NUM'][0][1] = _val1

    print(_val1, _val2)

    _ = GetPly(mocap_conf, openpose_conf, export_conf)
    print(f"exported successfully!")

def sync_e1(val):
    e1.delete(0, END)
    e1.insert(END, int(val))
    update_frame(int(val2.get()), int(val1.get()))

def sync_e2(val):
    e2.delete(0, END)
    e2.insert(END, int(val))
    update_frame(int(val2.get()), int(val1.get()))

button_execute = ttk.Button(frame, text="visualize", command=setParam)
button_execute.grid(row=4, column=1)

button_export = ttk.Button(frame, text="export to ply", command=exportply)
button_export.grid(row=5, column=1)

root.mainloop()