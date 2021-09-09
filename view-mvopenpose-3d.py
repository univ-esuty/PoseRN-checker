import sys
import datetime
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d

pose3d_dir = ''
movie_dir = ''

test = np.zeros(750*25*3).reshape(750*25, 3)

def loadPoints(idx):
    isY_reverse = -1   # 1 is not reverse.
    path = '{}/pose{:04d}.txt'.format(pose3d_dir, idx)
    point_array = np.loadtxt(path)
    test[idx*25:idx*25+25, 0] = point_array[:, 0]
    test[idx*25:idx*25+25, 1] = point_array[:, 1]
    test[idx*25:idx*25+25, 2] = point_array[:, 2]

    return point_array[:, 0], isY_reverse*point_array[:, 2], point_array[:, 1]

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

def setLines(X, Y, Z):
    num = len(_BonesV2)
    lineX = np.zeros(num*2).reshape(num, 2)
    lineY = np.zeros(num*2).reshape(num, 2)
    lineZ = np.zeros(num*2).reshape(num, 2)
    
    for i, bone in enumerate(_BonesV2): 
        lineX[i][0] = X[bone[0]]; lineX[i][1] = X[bone[1]]
        lineY[i][0] = Y[bone[0]]; lineY[i][1] = Y[bone[1]] 
        lineZ[i][0] = Z[bone[0]]; lineZ[i][1] = Z[bone[1]] 

    return lineX, lineY, lineZ

## draw 3d pose point 
fig = plt.figure()
ax = fig.gca(projection='3d')

def update_frame(fc):
    ax.clear()
    ax.view_init(elev=30, azim=-90)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, 2); ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel("x", size = 14, weight = "light"); ax.set_ylabel("y", size = 14, weight = "light"); ax.set_zlabel("z", size = 14, weight = "light")


    X, Y, Z = loadPoints(fc)
    ax.plot(X, Y, Z, 'k.')
    # for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        # ax.text(x, y, z, i, size=4)

    X, Y, Z = setLines(X, Y, Z)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        line = art3d.Line3D(x, y, z, color=plt.cm.jet(255//len(_BonesV2)*i))
        ax.add_line(line)

## main function
if(len(sys.argv) != 3):
    print('python **.py [3dpose-data-dir] [FRAMENUM]')
    pose3d_dir = 'input_data/mv-openpose/3dpose'
    movie_dir = 'results'
    row_num = 750

else:
    pose3d_dir = sys.argv[1] + '/mv-openpose/3dpose'
    movie_dir = sys.argv[1] + '/results'
    row_num = int(sys.argv[2])
    
ani = animation.FuncAnimation(fig, update_frame, frames=row_num, interval=100)
plt.show()

## output gif animation file (optional)
# videopath = '{}/movie_{}.gif'.format(movie_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
# ni.save(videopath, writer='PillowWriter', fps=10)

## output mp4 animation file (optional)
# HTML(ani.to_html5_video())
# ani.save('output_{}.mp4'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), writer="mpeg4")