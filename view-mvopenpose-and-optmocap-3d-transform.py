import datetime
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d

from compScale import GetScale_OpenPose, GetScale_MoCap

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

## MoCap class
class Mocap:
    _BonesMocap = [
        #HEAD
        [0,1],[0,2],[1,3],[2,3],
        #CROTCH
        [2,24],[3,24],[22,23],[22,24],[24,25],[25,26],[26,27],[27,28],[23,29],[29,32],[29,33],
        #LEFT-ARM
        [25,15],[13,14],[14,15],[22,13],[15,16],[16,17],[13,18],[17,20],[18,19],[19,21],[20,21],
        #RIGHT-ARM
        [25,6],[4,5],[5,6],[22,4],[6,7],[7,8],[4,9],[9,10],[8,11],[10,12],[11,12],
        #WAIST
        [30,31],[30,32],[31,33],[32,33],
        #LEFT-LEG
        [31,48],[33,48],[31,49],[33,49],[48,50],[49,51],[50,52],[51,53],[53,54],[54,55],[52,55],
        #LEFT-FOOT
        [55,56],[56,57],[55,58],[58,59],[59,60],[58,61],
        #RIGHT-LEG
        [30,34],[32,34],[30,35],[34,36],[35,37],[36,38],[37,39],[39,40],[38,41],[40,41],
        #RIGHT-FOOT
        [41,42],[41,43],[41,44],[44,45],[45,46],[44,47]
    ]

    #define instance variable
    def __init__(self, configs):
        self.data = []
        self.configs = configs

    #data import
    def importData(self):
        path = self.configs['DATASET_DIR_ROOT'] + '/' + self.configs['DATASET_FILE']
        with open(path) as f:
            self.data = [s.strip() for s in f.readlines()]
            ## delete header
            for i in range(6):
                self.data.pop(0)

    ## load joint points
    def loadPoints(self, fc):
        isYinverse = -1
        isXinverse = -1
        points = self.data[fc].split()
        numJoints = (len(points) - 2) // 3

        X = np.zeros(numJoints, dtype=float)
        Y = np.zeros(numJoints, dtype=float)
        Z = np.zeros(numJoints, dtype=float)
        idx = 0; ct = 2
        while ct < len(points):
            X[idx] = float(points[ct]); ct += 1
            Y[idx] = float(points[ct]); ct += 1
            Z[idx] = float(points[ct]); ct += 1
            idx += 1

        return isXinverse*X, isYinverse*Y, Z
    
    def setLines(self, X, Y, Z):
        num = len(Mocap._BonesMocap)
        lineX = np.zeros(num*2).reshape(num, 2)
        lineY = np.zeros(num*2).reshape(num, 2)
        lineZ = np.zeros(num*2).reshape(num, 2)
        
        for i, bone in enumerate(Mocap._BonesMocap): 
            lineX[i][0] = X[bone[0]]; lineX[i][1] = X[bone[1]]
            lineY[i][0] = Y[bone[0]]; lineY[i][1] = Y[bone[1]] 
            lineZ[i][0] = Z[bone[0]]; lineZ[i][1] = Z[bone[1]] 

        return lineX, lineY, lineZ

## OpenPose3d class
class Openpose3d:
    _BonesV2 = [
        #NECK
        [1,0],[1,8],[1,2],[1,5],
        #HEAD
        [0,15],[15,17],[0,16],[16,18],
        #CROTCH
        [8,9],[8,12],
        #LEFT-ARM
        [2,3],[3,4],
        #RIGHT-ARM
        [5,6],[6,7],
        #LEFT-LEG
        [9,10],[10,11],[11,22],[22,23],[11,24],
        #RIGHT-LEG
        [12,13],[13,14],[14,19],[19,20],[14,21]
    ]

    def __init__(self, configs):
        self.configs = configs

    def loadPoints(self, idx):
        isY_reverse = -1   # 1 is not reverse.
        path = '{}/pose{:04d}.txt'.format(self.configs['DATASET_DIR_ROOT'], idx)
        point_array = np.loadtxt(path)
        return point_array[:, 0], isY_reverse*point_array[:, 2], point_array[:, 1]

    def setLines(self, X, Y, Z):
        num = len(Openpose3d._BonesV2)
        lineX = np.zeros(num*2).reshape(num, 2)
        lineY = np.zeros(num*2).reshape(num, 2)
        lineZ = np.zeros(num*2).reshape(num, 2)
        
        for i, bone in enumerate(Openpose3d._BonesV2): 
            lineX[i][0] = X[bone[0]]; lineX[i][1] = X[bone[1]]
            lineY[i][0] = Y[bone[0]]; lineY[i][1] = Y[bone[1]] 
            lineZ[i][0] = Z[bone[0]]; lineZ[i][1] = Z[bone[1]] 

        return lineX, lineY, lineZ


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