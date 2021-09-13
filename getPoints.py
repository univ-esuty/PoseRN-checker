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