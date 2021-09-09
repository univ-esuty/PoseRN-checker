import numpy as np
import datetime

from compScale import GetScale_OpenPose, GetScale_MoCap

## generally, optical motion capture samples data very frequently
## adjust the parameter and fit to sampleing rate of MV-OpenPose 
SKIP_OPT_CAP_FRAME = 36

## MoCap class
class Mocap:
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

        return isXreverse*X, isYinverse*Y, Z

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

class GetPly():
    def __init__(self, mocap_conf, openpose_conf, export_conf, batch=False):
        self.mocap_conf = mocap_conf
        self.openpose_conf = openpose_conf
        self.export_conf = export_conf

        ## set scale.
        self.openpose_scale, self.openpose_center = GetScale_OpenPose(self.openpose_conf)
        self.mocap_scale, self.mocap_center = GetScale_MoCap(self.mocap_conf)

        ## adjust
        tmp = [self.openpose_scale, self.openpose_center]
        self.openpose_scale = tmp[0] * 0.9
        self.openpose_center[2] = tmp[1][2] + 0.012

        self.mocap = Mocap(self.mocap_conf)
        self.mocap.importData()
        self.openpose = Openpose3d(self.openpose_conf)
        

        for i, f_num in enumerate(self.export_conf['FRAME_NUM']):
            a, b = self.get_ply(f_num[0], f_num[1])

            if batch:
                self.export_txt("{}/opt-mocap/m_{:2d}.ply".format(self.export_conf['EXPORT_DIR'], i), a)
                self.export_txt("{}/mv-openpose/o_{:2d}.ply".format(self.export_conf['EXPORT_DIR'], i), b)
            else:
                self.export_txt("{}/opt-mocap/m_{}.ply".format(self.export_conf['EXPORT_DIR'], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), a)
                self.export_txt("{}/mv-openpose/o_{}.ply".format(self.export_conf['EXPORT_DIR'], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), b)    

    def scalize(self, scale, x, y, z):
        return x / scale, y / scale, z / scale

    def centerlize(self, scale, c, x, y, z):
        c_x, c_y, c_z = self.scalize(scale, c[0], c[1], c[2])
        return x - c_x, y - c_y, z - c_z


    def mk_ply(self, X, Y, Z, vertex_num, col):
        ply = []
        ply.append("ply\n")
        ply.append("format ascii 1.0\n")
        ply.append("element vertex {:d}\n".format(vertex_num))
        ply.append("property float x\n")
        ply.append("property float y\n")
        ply.append("property float z\n")
        ply.append("property uchar red\n")
        ply.append("property uchar green\n")
        ply.append("property uchar blue\n")
        ply.append("end_header\n")

        for x, y, z in zip(X, Y, Z):
            ply.append("{} {} {} {} {} {}\n".format(x, y, z, col[0], col[1], col[2]))

        return ply

    def get_ply(self, m_fc, o_fc):
        ## visualize mocap points
        X, Y, Z = self.mocap.loadPoints(int(m_fc*SKIP_OPT_CAP_FRAME))
        X, Y, Z = self.scalize(self.mocap_scale, X, Y, Z)
        X, Y, Z = self.centerlize(self.mocap_scale, self.mocap_center, X, Y, Z)
        ply_mocap = self.mk_ply(X, Y, Z, 62, (255,0,0))
        
        ## visualize openpose3d points
        X, Y, Z = self.openpose.loadPoints(o_fc)
        X, Y, Z = self.scalize(self.openpose_scale, X, Y, Z)
        X, Y, Z = self.centerlize(self.openpose_scale, self.openpose_center, X, Y, Z)
        ply_openpose = self.mk_ply(X, Y, Z, 25, (0,0,255))

        return ply_mocap, ply_openpose

    def export_txt(self, filename, txt_arr):
        with open(filename, mode='w') as f:
            f.writelines(txt_arr)


## ------------ ##
## batch procss ##
## ------------ ##
if __name__ == '__main__':
    mocap_conf = {
        'DATASET_DIR_ROOT': 'input_data/opt-mocap',
        'DATASET_FILE' : 'optmocap.trc',
        'JOINT_IDX' : 29,
    }

    openpose_conf = {
        'DATASET_DIR_ROOT': 'input_data/mv-openpose/3dpose',
        'JOINT_IDX' : 8,
        'FRAME_NUM' : 750
    }

    export_conf = {
        'EXPORT_DIR' : 'keyframes',
        # pairs of frame [MV-openpose, Optical-Motion-Capture]
        'FRAME_NUM' : [[163,78],[178,92],[193,107],[208,121],[224,139],[239,152],[259,172],[277,183]] 
    }

    _ = GetPly(mocap_conf, openpose_conf, export_conf, batch=True)
    print(f"exported successfully!")
