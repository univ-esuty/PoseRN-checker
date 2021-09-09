from scipy.spatial import KDTree
import numpy as np

def calcRigidTranformation(MatA, MatB):
    A, B = np.copy(MatA).astype('float64'), np.copy(MatB).astype('float64')

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A -= centroid_A
    B -= centroid_B

    H = np.dot(A.T, B)
    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T, U.T)
    T = np.dot(-R, centroid_A) + centroid_B

    return R, T

def calcAffineTransformation(MatA, MatB):
    A, B = np.copy(MatA).astype('float64').T, np.copy(MatB).astype('float64').T
    TRS = np.matmul(B, np.linalg.pinv(A))
    return TRS

def get_transform(MatA, MatB):
    return np.matmul(MatB, np.linalg.pinv(MatA))

class ICP(object):
    def __init__(self, pointsA, pointsB):
        self.pointsA = pointsA
        self.pointsB = pointsB
        self.kdtree = KDTree(self.pointsA)

    def calculate(self, iter): ## 剛体変換
        old_points = np.copy(self.pointsB)
        new_points = np.copy(self.pointsB)

        for i in range(iter):
            neighbor_idx = self.kdtree.query(old_points)[1]
            targets = self.pointsA[neighbor_idx]
            R, T = calcRigidTranformation(old_points, targets)
            new_points = np.dot(R, old_points.T).T + T

            if  np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = np.copy(new_points)

        return new_points

    def calculate_s(self, iter): ## Affine変換
        old_points = np.copy(self.pointsB)
        new_points = np.copy(self.pointsB)

        for i in range(iter):
            # 3dim
            neighbor_idx = self.kdtree.query(old_points)[1]
            targets = self.pointsA[neighbor_idx]

            # 4dim 
            source = np.hstack((old_points, np.ones((old_points.shape[0], 1))))
            targets = np.hstack((targets, np.ones((targets.shape[0], 1))))

            TRS = calcAffineTransformation(source, targets)
            new_points = np.dot(TRS, source.T).T
            new_points = np.delete(new_points, -1, axis=1)

            # 3dim
            if  np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = np.copy(new_points)
        
        return new_points

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

def setLines_at_openpose(X, Y, Z):
    num = len(_BonesV2)
    lineX = np.zeros(25*8*2).reshape(25*8, 2)
    lineY = np.zeros(25*8*2).reshape(25*8, 2)
    lineZ = np.zeros(25*8*2).reshape(25*8, 2)

    for j in range(8):
        for i, bone in enumerate(_BonesV2): 
            lineX[i+25*j][0] = X[bone[0]+25*j]; lineX[i+25*j][1] = X[bone[1]+25*j]
            lineY[i+25*j][0] = Y[bone[0]+25*j]; lineY[i+25*j][1] = Y[bone[1]+25*j] 
            lineZ[i+25*j][0] = Z[bone[0]+25*j]; lineZ[i+25*j][1] = Z[bone[1]+25*j] 

    return lineX, lineY, lineZ

# op = OpenPose, mc = MotionCapture
def icp_run(ply_mc, ply_op, iter=100, scale_ignore=True, isSave=True):
    def load_ply(f_name):
        with open(f_name) as f:
            points = [s.strip() for s in f.readlines()]; del points[:10]
            output = np.zeros((len(points), 3))
            for idx, point in enumerate(points):
                output[idx] = np.array(list(map(float, point.split())))[:3]
            return output

    mc_points = load_ply(ply_mc)
    op_points = load_ply(ply_op)

    ## run icp ICP(dst, src)
    icp = ICP(mc_points, op_points)
    if scale_ignore:
        icp_points = icp.calculate(iter)
    else:
        icp_points = icp.calculate_s(iter)

    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    import mpl_toolkits.mplot3d.art3d as art3d

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_label("x - axis")
    ax.set_label("y - axis")
    ax.set_label("z - axis")
    
    ax.plot(op_points[:,0], op_points[:,1], op_points[:,2], "o", color="#0000ff", ms=4, mew=0.5)
    ax.plot(mc_points[:,0], mc_points[:,1], mc_points[:,2], "o", color="#ff0000", ms=4, mew=0.5)
    ax.plot(icp_points[:,0], icp_points[:,1], icp_points[:,2], "o", color="#000000", ms=4, mew=0.5)

    X, Y, Z = setLines_at_openpose(op_points[:,0], op_points[:,1], op_points[:,2])
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        line = art3d.Line3D(x, y, z, color=pyplot.cm.jet(255//len(_BonesV2)*(i%24)))
        ax.add_line(line)

    pyplot.show()

    final_result = calcAffineTransformation(op_points, icp_points)

    if not isSave:
        print('transform parameter:', final_result)
    else:
        np.save('results/transform', final_result)

## main
if __name__ == '__main__':
    import sys
    args = sys.argv

    ## icp_run(dst, src)
    if len(args) == 2 and args[1] == '--scale_False':
        icp_run('keyframes/opt-mocap.ply', 'keyframes/mv-openpose.ply', scale_ignore=True)
    else:
        icp_run('keyframes/opt-mocap.ply', 'keyframes/mv-openpose.ply')
    