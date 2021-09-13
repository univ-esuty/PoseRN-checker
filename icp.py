from scipy.spatial import KDTree
from matplotlib import pyplot
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from math import sin, cos

import numpy as np
from draw import setLines_at_openpose, setLines_at_optmocap

class ICP(object):
    def __init__(self, points_dst, points_src, configs=None):
        self.configs = configs

        self.points_dst = points_dst
        self.points_src = points_src
        self.icp_points = np.array([None])
        self.kdtree = KDTree(self.points_dst)

    def calcRigidTranformation(self, MatA, MatB):
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

    def calcAffineTransformation(self, MatA, MatB):
        A, B = np.copy(MatA).astype('float64').T, np.copy(MatB).astype('float64').T
        TRS = np.matmul(B, np.linalg.pinv(A))
        return TRS

    def icp_calculate(self, iter):
        old_points = np.copy(self.points_src)
        new_points = np.copy(self.points_src)

        for i in range(iter):
            dist, neighbor_idx = self.kdtree.query(old_points)
            targets = self.points_dst[neighbor_idx]
            R, T = self.calcRigidTranformation(old_points, targets)
            new_points = np.dot(R, old_points.T).T + T
            
            if  np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = np.copy(new_points)

        self.icp_points = new_points

    def icp_calculate_s(self, iter): ## Affine変換
        old_points = np.copy(self.points_src)
        new_points = np.copy(self.points_src)

        for i in range(iter):
            # 3dim
            neighbor_idx = self.kdtree.query(old_points)[1]
            targets = self.points_dst[neighbor_idx]

            # 4dim 
            source = np.hstack((old_points, np.ones((old_points.shape[0], 1))))
            targets = np.hstack((targets, np.ones((targets.shape[0], 1))))

            TRS = self.calcAffineTransformation(source, targets)
            new_points = np.dot(TRS, source.T).T
            new_points = np.delete(new_points, -1, axis=1)

            # 3dim
            if  np.sum(np.abs(old_points - new_points)) < 0.000000001:
                break

            old_points = np.copy(new_points)
        
        self.icp_points = new_points

    def calc_icpcost(self):
        if self.icp_points.all() != None:
            dist, neighbor_idx = self.kdtree.query(self.icp_points)
            return np.sum(np.square(dist))
        else:
            return -1  

    def graph_plot(self, isSave=False, index=0):
        if self.icp_points.all() != None:
            fig = pyplot.figure()
            ax = Axes3D(fig)
            ax.set_label("x")
            ax.set_label("y")
            ax.set_label("z")
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.view_init(elev=90, azim=-90)
            
            ax.plot(self.points_dst[:,0], self.points_dst[:,1], self.points_dst[:,2], "o", color="#ff0000", ms=1, mew=0.5)
            # ax.plot(self.points_src[:,0], self.points_src[:,1], self.points_src[:,2], "o", color="#0000ff", ms=1, mew=0.5)
            ax.plot(self.icp_points[:,0], self.icp_points[:,1], self.icp_points[:,2], "o", color="#000000", ms=1, mew=0.5)

            if self.configs:
                X, Y, Z = setLines_at_openpose(self.icp_points[:,0], self.icp_points[:,1], self.icp_points[:,2], self.configs['FRAME_NUM'])
                for i, (x, y, z) in enumerate(zip(X, Y, Z)):
                    line = art3d.Line3D(x, y, z, color='red')
                    ax.add_line(line)

                X, Y, Z = setLines_at_optmocap(self.points_dst[:,0], self.points_dst[:,1], self.points_dst[:,2], self.configs['FRAME_NUM']):
                for i, (x, y, z) in enumerate(zip(X, Y, Z)):
                    line = art3d.Line3D(x, y, z, color='blue')
                    ax.add_line(line)
            
            if not isSave: 
                pyplot.show()
            else:
                fig.savefig("results/{:04d}.png".format(index))
                pyplot.close()

        else:
            print("Graph plot Error.")
