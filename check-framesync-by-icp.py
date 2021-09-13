import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from getPoints import Mocap, Openpose3d
from icp import ICP

SKIP_OPT_CAP_FRAME = 36

MOCAP_CONF = {
    'DATASET_DIR_ROOT': 'input_data/opt-mocap',
    'DATASET_FILE' : 'optmocap.trc',
    'JOINT_IDX' : 29,
}

OPENPOSE_CONF = {
    'DATASET_DIR_ROOT': 'input_data/mv-openpose/3dpose',
    'JOINT_IDX' : 8,
    'FRAME_NUM' : 750
}

## set each of your keyframe numbers.
FRAME_CONF = {
    'FRAME_NUM' : 8,
    'MOCAP_IDX' : [163, 178, 193, 208, 224, 239, 259, 277], #for optical motion capture
    'OP_IDX'    : [ 78,  92, 107, 121, 139, 152, 172, 183]  #for MV-OpenPose
}

def scheduler(scope, steps):
    pos = np.arange(steps, scope+steps, steps)
    neg = -1 * pos.copy()[::-1]
    return np.concatenate([neg, np.array([0]), pos])
    
FRAME_RANGE = 100
FRAME_PER_STEP = 2
frame_idx = [x for x in scheduler(FRAME_RANGE, FRAME_PER_STEP)]

SCALE_RANGE = 120*4
SCALE_PER_STEP = 8*4
scale_idx = [x for x in scheduler(SCALE_RANGE, SCALE_PER_STEP)]

mocap = Mocap(MOCAP_CONF)
mocap.importData()
openpose = Openpose3d(OPENPOSE_CONF)

cost = np.zeros(len(frame_idx))

steps_report = tqdm(total = len(frame_idx))
step_counter = 0
min_val = 10e+5
min_idx = [-1, -1]


### Search
for i, f_idx in enumerate(frame_idx):
    ## MoCap
    tX=[None]*FRAME_CONF['FRAME_NUM']
    tY=[None]*FRAME_CONF['FRAME_NUM'] 
    tZ=[None]*FRAME_CONF['FRAME_NUM']
    for idx, m_fc in enumerate(FRAME_CONF['MOCAP_IDX']):
        tX[idx], tY[idx], tZ[idx] = mocap.loadPoints(int(m_fc*SKIP_OPT_CAP_FRAME)+f_idx, 0)

    X = np.concatenate(tX).flatten()
    Y = np.concatenate(tY).flatten()
    Z = np.concatenate(tZ).flatten()
    mc_points = np.stack([X, Y, Z]).T.copy()

    ## OpenPose
    tX=[None]*FRAME_CONF['FRAME_NUM']
    tY=[None]*FRAME_CONF['FRAME_NUM']
    tZ=[None]*FRAME_CONF['FRAME_NUM']
    for idx, o_fc in enumerate(FRAME_CONF['OP_IDX']):
        tX[idx], tY[idx], tZ[idx] = openpose.loadPoints(o_fc)
    X = np.concatenate(tX).flatten()
    Y = np.concatenate(tY).flatten()
    Z = np.concatenate(tZ).flatten()
    op_points = np.stack([X, Y, Z]).T.copy()

    ## icp = ICP(dst, src)
    icp = ICP(mc_points, op_points)
    icp.icp_calculate_s(100)

    icp_cost = icp.calc_icpcost()
    cost[i] = icp_cost

    if icp_cost < min_val:
        min_val = icp_cost
        min_idx = f_idx
        min_step = step_counter

    steps_report.update(1)
    step_counter += 1

print('min_cost:', min_val, 'step, min_idx.frame:', min_step, min_idx)

plt.plot(frame_idx, cost)
plt.xlabel("Frame")
plt.ylabel("Cost")
plt.grid(True)
plt.show()


f_idx = min_idx

## opt-mocap
tX=[None]*FRAME_CONF['FRAME_NUM']
tY=[None]*FRAME_CONF['FRAME_NUM'] 
tZ=[None]*FRAME_CONF['FRAME_NUM']
for idx, m_fc in enumerate(FRAME_CONF['MOCAP_IDX']):
    tX[idx], tY[idx], tZ[idx] = mocap.loadPoints(int(m_fc*SKIP_OPT_CAP_FRAME)+f_idx, 0)

X = np.concatenate(tX).flatten()
Y = np.concatenate(tY).flatten()
Z = np.concatenate(tZ).flatten()
mc_points = np.stack([X, Y, Z]).T.copy()

## mv-OpenPose
tX=[None]*FRAME_CONF['FRAME_NUM']
tY=[None]*FRAME_CONF['FRAME_NUM']
tZ=[None]*FRAME_CONF['FRAME_NUM']
for idx, o_fc in enumerate(FRAME_CONF['OP_IDX']):
    tX[idx], tY[idx], tZ[idx] = openpose.loadPoints(o_fc)
X = np.concatenate(tX).flatten()
Y = np.concatenate(tY).flatten()
Z = np.concatenate(tZ).flatten()
op_points = np.stack([X, Y, Z]).T.copy()

## icp = ICP(dst, src)
icp = ICP(mc_points, op_points, configs=FRAME_CONF)
icp.icp_calculate_s(100)
icp_cost = icp.calc_icpcost()

print("icp_cost:", icp_cost)
icp.graph_plot(isSave=False)  # <- draw a graph



            


            