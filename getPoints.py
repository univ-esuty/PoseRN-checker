import sys
import numpy as np

from compScale import GetScale_OpenPose, GetScale_MoCap

def scalize(scale, x, y, z):
    return x / scale, y / scale, z / scale

def centerlize(scale, c, x, y, z):
    c_x, c_y, c_z = scalize(scale, c[0], c[1], c[2])
    return x - c_x, y - c_y, z - c_z

## MoCap class
class Mocap:
    #define instance variable
    def __init__(self, configs):
        self.data = []
        self.configs = configs
        self.mocap_scale, self.mocap_center = GetScale_MoCap(configs)

    #data import
    def importData(self):
        path = self.configs['DATASET_DIR_ROOT'] + '/' + self.configs['DATASET_FILE']
        with open(path) as f:
            self.data = [s.strip() for s in f.readlines()]
            ## delete header
            for i in range(6):
                self.data.pop(0)

    ## load joint points
    def loadPoints(self, fc, delta_scale, isReverse=True, isScale=True, isCenter=True):
        isXreverse = -1
        isYinverse = -1
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

        if isReverse:
            X = isXreverse*X,
            Y = isYinverse*Y

        if isScale:
            X, Y, Z = scalize(self.mocap_scale + delta_scale, X, Y, Z)

        if isCenter:
            X, Y, Z = centerlize(self.mocap_scale + delta_scale, self.mocap_center, X, Y, Z)

        return X, Y, Z

## OpenPose3d class
class Openpose3d:
    _BonesV2 = [
        #NECK
        [1,0],[1,8],[1,2],[1,5],
        #HEAD
        [0,15],[15,16],[0,17],[17,18],
        #CROTCH
        [8,9],[8,12],
        #LEFT-ARM
        [2,3],[3,4],
        #RIGHT-ARM
        [5,6],[6,7],
        #LEFT-LEG
        [9,10],[10,11],[11,22],[11,23],[11,24],
        #RIGHT-LEG
        [12,13],[13,14],[14,19],[14,20],[14,21]
    ]

    def __init__(self, configs):
        self.configs = configs

        ## set scale.
        self.openpose_scale, self.openpose_center = GetScale_OpenPose(configs)
        ## adjust
        tmp = [self.openpose_scale, self.openpose_center]
        self.openpose_scale = tmp[0] * 0.9
        self.openpose_center[2] = tmp[1][2] + 0.012


    def loadPoints(self, fc, isScale=True, isCenter=True):
        isY_reverse = -1   # 1 is not reverse.
        path = '{}/pose{:04d}.txt'.format(self.configs['DATASET_DIR_ROOT'], fc)
        point_array = np.loadtxt(path)
        
        X = point_array[:, 0]
        Y = isY_reverse*point_array[:, 2], 
        Z = point_array[:, 1]

        if isScale:
            X, Y, Z = scalize(self.openpose_scale, X, Y, Z) 
        
        if isCenter:
            X, Y, Z = centerlize(self.openpose_scale, self.openpose_center, X, Y, Z)

        return X, Y, Z

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