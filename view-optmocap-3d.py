import argparse
import datetime
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


## config (set path directly)
DATASET_DIR_ROOT = 'input_data/opt-mocap'
DATASET_FILE = 'optmocap.trc'
MOVIE_OUT_DIR = 'results'

## args setting.
parser = argparse.ArgumentParser(
    prog='',
    usage='',
    description='', 
    epilog='end', 
    add_help=True, 
)
 
parser.add_argument('-d', '--dir', help='dataset root dir', default=DATASET_DIR_ROOT)
parser.add_argument('-f', '--file', help='dataset file name', default=DATASET_FILE)
parser.add_argument('-o', '--out', help='export video path', default=MOVIE_OUT_DIR)
args = parser.parse_args()

DATASET_DIR_ROOT = args.dir
DATASET_FILE = args.file
MOVIE_OUT_DIR = args.out

## load dataset file
path = DATASET_DIR_ROOT + '/' + DATASET_FILE
data = []

with open(path) as f:
    data = [s.strip() for s in f.readlines()]
    
    ## delete header
    for i in range(6):
        data.pop(0)

## load joint points
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

    return X, Y, Z

## draw 3d pose point 
fig = plt.figure()
ax = fig.gca(projection='3d')

def update_frame(fc):
    ax.clear()
    ax.view_init(elev=30, azim=-90)
    ax.set_xlim(-2000, 2000); ax.set_ylim(-2500, 2500); ax.set_zlim(0, 2000)
    ax.set_xlabel("x", size = 14, weight = "light"); ax.set_ylabel("y", size = 14, weight = "light"); ax.set_zlabel("z", size = 14, weight = "light")

    X, Y, Z = loadPoints(fc*30)
    ax.plot(X, Y, Z, 'k.')

ani = animation.FuncAnimation(fig, update_frame, frames=len(data)//30, interval=33)
plt.show()

## output gif animation file (optional)
# videopath = '{}/movie_{}.gif'.format(MOVIE_OUT_DIR, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
# ani.save(videopath, writer='PillowWriter', fps=60)