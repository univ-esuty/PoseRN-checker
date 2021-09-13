import datetime
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d

from compScale import GetScale_OpenPose, GetScale_MoCap
from getPoints import Mocap, Openpose3d

SKIP_OPT_CAP_FRAME = 39 
SKIP_OPT_CAP_FRAME_START = 2650 

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
    'EXPORT_DIR' : 'results',
    'FRAME_RATE' : 30,
    'EXPORT_FORMAT' : 'gif'
}


## set scale.
openpose_scale, openpose_center = GetScale_OpenPose(openpose_conf)
mocap_scale, mocap_center = GetScale_MoCap(mocap_conf)

## adjust
openpose_scale *= mocap_conf['ADJ_SCALE']
openpose_center[2] += mocap_conf['ADJ_CENTER_Z']

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

def update_frame(fc):
    ax.clear()
    ax.view_init(elev=30, azim=-90)
    ax.set_xlim(-2, 2); ax.set_ylim(-2.5, 2.5); ax.set_zlim(-1, 1)
    ax.set_xlabel("x", size = 14, weight = "light"); ax.set_ylabel("y", size = 14, weight = "light"); ax.set_zlabel("z", size = 14, weight = "light")
    ax.set_title("Frame: {}".format(fc))

    ## visualize mocap points
    X, Y, Z = mocap.loadPoints(int(fc*SKIP_OPT_CAP_FRAME+SKIP_OPT_CAP_FRAME_START))
    X, Y, Z = scalize(mocap_scale, X, Y, Z)
    X, Y, Z = centerlize(mocap_scale, mocap_center, X, Y, Z)
    ax.plot(X, Y, Z, 'k.', color='b', markersize=2)

    X, Y, Z = mocap.setLines(X, Y, Z)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        line = art3d.Line3D(x, y, z, color="b", lw=1)
        ax.add_line(line)

    ## visualize openpose3d points
    X, Y, Z = openpose.loadPoints(fc)
    X, Y, Z = scalize(openpose_scale, X, Y, Z)
    X, Y, Z = centerlize(openpose_scale, openpose_center, X, Y, Z)
    XYZ = np.array([X, Y, Z]).T

    # ## use icp parameters
    transform_array = np.load('results/transform.npy')
    XYZ = np.dot(transform_array, XYZ.T).T
    X = XYZ[:, 0]; Y = XYZ[:, 1]; Z = XYZ[:, 2]

    ax.plot(X, Y, Z, 'k.', color='r', markersize=3)

    X, Y, Z = openpose.setLines(X, Y, Z)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        # line = art3d.Line3D(x, y, z, color=plt.cm.jet(255//len(Openpose3d._BonesV2)*i))
        line = art3d.Line3D(x, y, z, color='r', lw=1)
        ax.add_line(line)

ani = animation.FuncAnimation(fig, update_frame, frames=int(openpose_conf['FRAME_NUM']) , interval=100)
plt.show()

## output gif animation file (optional)
# videopath = '{}/movie_{}.gif'.format(export_conf["EXPORT_DIR"], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
# ani.save(videopath, writer='PillowWriter', fps=10)