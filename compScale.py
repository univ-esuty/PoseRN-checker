import math
import numpy as np

## -----------------##
## tools
## -----------------##

## get distance from two coordinates
def getDist(p1, p2):
    dx = (p2[0] - p1[0]) * (p2[0] - p1[0])
    dy = (p2[1] - p1[1]) * (p2[1] - p1[1])
    dz = (p2[2] - p1[2]) * (p2[2] - p1[2])
    return math.sqrt(dx + dy + dz)


##-------------------##
## MoCap-Data-Scale
##-------------------##

def GetScale_MoCap(configs):
    DATASET_DIR_ROOT = configs['DATASET_DIR_ROOT']
    DATASET_FILE = configs['DATASET_FILE']
    JOINT_IDX = configs['JOINT_IDX']

    ## gloabl
    data = []

    ## load-mocap-file
    def loadCsv(fpath):
        with open(fpath) as f:
            loaddata = [s.strip() for s in f.readlines()]
            
            ## delete header
            for i in range(6):
                loaddata.pop(0)
        
            return loaddata

    ## load mocap-joint points
    def loadPoints(fc):
        points = data[fc].split()
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

        return X[JOINT_IDX], Y[JOINT_IDX], Z[JOINT_IDX]

    ## load dataset file
    path = DATASET_DIR_ROOT + '/' + DATASET_FILE
    data = loadCsv(path)

    numFrame = len(data)
    x_array = np.zeros(numFrame, dtype=float)
    y_array = np.zeros(numFrame, dtype=float)
    z_array = np.zeros(numFrame, dtype=float)
    dist_array = np.zeros(numFrame, dtype=float)

    for i in range(numFrame):
        x_array[i], y_array[i], z_array[i] = loadPoints(i)

    center = np.array([np.mean(x_array), np.mean(y_array), np.mean(z_array)])

    for i in range(numFrame):
        point = np.array([x_array[i], y_array[i], z_array[i]])
        dist_array[i] = getDist(center, point)

    radius = np.mean(dist_array)

    return radius, center


##-------------------##
## OpenPose-Data-Scale
##-------------------##

def GetScale_OpenPose(configs):
    DATASET_DIR_ROOT = configs['DATASET_DIR_ROOT']
    JOINT_IDX = configs['JOINT_IDX']
    FRAME_NUM = configs['FRAME_NUM']

    def loadPoints(idx):
        isY_reverse = -1   # 1 is not reverse.
        path = '{}/pose{:04d}.txt'.format(DATASET_DIR_ROOT, idx)
        point_array = np.loadtxt(path)
        return point_array[JOINT_IDX, 0], isY_reverse*point_array[JOINT_IDX, 2], point_array[JOINT_IDX, 1]


    numFrame = FRAME_NUM
    x_array = np.zeros(numFrame, dtype=float)
    y_array = np.zeros(numFrame, dtype=float)
    z_array = np.zeros(numFrame, dtype=float)
    dist_array = np.zeros(numFrame, dtype=float)

    for i in range(numFrame):
        x_array[i], y_array[i], z_array[i] = loadPoints(i)

    center = np.array([np.mean(x_array), np.mean(y_array), np.mean(z_array)])

    for i in range(numFrame):
        point = np.array([x_array[i], y_array[i], z_array[i]])
        dist_array[i] = getDist(center, point)

    radius = np.mean(dist_array)
    return radius, center

##-------------------##
## Debugging
##-------------------##
if __name__ == '__main__':
    configs = {
        'DATASET_DIR_ROOT': 'input_data/opt-mocap',
        'DATASET_FILE' : 'optmocap.trc',
        'JOINT_IDX' : 29,
    }
    print(GetScale_MoCap(configs))

    configs = {
        'DATASET_DIR_ROOT': 'input_data/mv-openpose/3dpose',
        'JOINT_IDX' : 8,
        'FRAME_NUM' : 750
    }
    print(GetScale_OpenPose(configs))
