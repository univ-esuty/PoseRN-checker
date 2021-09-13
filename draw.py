import numpy as np

def setLines_at_openpose(X, Y, Z, numFrames):
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

    num = len(_BonesV2)
    lineX = np.zeros(25*numFrames*2).reshape(25*numFrames, 2)
    lineY = np.zeros(25*numFrames*2).reshape(25*numFrames, 2)
    lineZ = np.zeros(25*numFrames*2).reshape(25*numFrames, 2)

    for j in range(numFrames):
        for i, bone in enumerate(_BonesV2): 
            lineX[i+25*j][0] = X[bone[0]+25*j]; lineX[i+25*j][1] = X[bone[1]+25*j]
            lineY[i+25*j][0] = Y[bone[0]+25*j]; lineY[i+25*j][1] = Y[bone[1]+25*j] 
            lineZ[i+25*j][0] = Z[bone[0]+25*j]; lineZ[i+25*j][1] = Z[bone[1]+25*j] 

    return lineX, lineY, lineZ

def setLines_at_optmocap(X, Y, Z, numFrames):
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

    num = len(_BonesV2)
    lineX = np.zeros(62*numFrames*2).reshape(62*numFrames, 2)
    lineY = np.zeros(62*numFrames*2).reshape(62*numFrames, 2)
    lineZ = np.zeros(62*numFrames*2).reshape(62*numFrames, 2)

    for j in range(numFrames):
        for i, bone in enumerate(_BonesMocap): 
            lineX[i+62*j][0] = X[bone[0]+62*j]; lineX[i+62*j][1] = X[bone[1]+62*j]
            lineY[i+62*j][0] = Y[bone[0]+62*j]; lineY[i+62*j][1] = Y[bone[1]+62*j] 
            lineZ[i+62*j][0] = Z[bone[0]+62*j]; lineZ[i+62*j][1] = Z[bone[1]+62*j] 

    return lineX, lineY, lineZ