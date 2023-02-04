"""
Original Matlab project template by: M. Subbarao, ECE, SBU

"""

import sys
import numpy as np
import cv2
import math

from skimage.color.rgb_colors import red

'''
function for rotation and translation
'''


def Map2Da(K, R, T, Vi):
    T_transpose = np.transpose(np.atleast_2d(T))  # numpy needs to treat 1D as 2D to transpose
    V_transpose = np.transpose(np.atleast_2d(np.append(Vi, [1])))
    RandTappended = np.append(R, T_transpose, axis=1)
    P = K @ RandTappended @ V_transpose  # @ is the matrix mult operator for numpy arrays
    P = np.asarray(P).flatten()  # just to make it into a flat array

    w1 = P[2]
    v = [None] * 2  # makes an empty array of size 2
    # map Vi = (X, Y, Z) to v = (x, y)
    v[0] = P[0] / w1  # v[0] is the x-value for the 2D point v
    v[1] = P[1] / w1


    return v


'''
function for mapping image coordinates in mm to
row and column index of the image, with pixel size p mm and
image center at [r0,c0]

u : the 2D point in mm space
[r0, c0] : the image center
p : pixel size in mm

@return : the 2D point in pixel space
'''


def MapIndex(u, c0, r0, p):
    v = [None] * 2
    v[0] = round(r0 - u[1] / p)
    v[1] = round(c0 + u[0] / p)

    return v


'''
Wrapper for drawing line cv2 draw line function
Necessary to flip the coordinates b/c of how Python indexes pixels on the screen >:(

A : matrix to draw a line in
vertex1 : terminal point for the line
vertex2 : other terminal point for the line
thickness : thickness of the line(default = 3)
color : RGB tuple for the line to be drawn in (default = (255, 255, 255) ie white)

@return : the matrix with the line drawn in it

NOTE: order of vertex1 and vertex2 does not change the line drawn
'''


def drawLine(A, vertex1, vertex2, color=(255, 255, 255), thickness=3):
    v1 = list(reversed(vertex1))
    v2 = list(reversed(vertex2))
    return cv2.line(A, v1, v2,color, thickness)  # replace this


def main():
    length = 10  # length of an edge in mm
    # the 8 3D points of the cube in mm:
    V1 = np.array([0, 0, 0])
    V2 = np.array([0, length, 0])
    V3 = np.array([length, length, 0])
    V4 = np.array([length, 0, 0])
    V5 = np.array([length, 0, length])
    V6 = np.array([0, length, length])
    V7 = np.array([0, 0, length])
    V8 = np.array([length, length, length])

    blueTranslation = np.array([10,0,0])

    V1R = np.array([0, 0, 0])
    V2R = np.array([0, length, 0])
    V3R = np.array([length, length, 0])
    V4R = np.array([length, 0, 0])
    V5R = np.array([length, 0, length])
    V6R = np.array([0, length, length])
    V7R = np.array([0, 0, length])
    V8R = np.array([length, length, length])

    '''
    Find the unit vector u81 (N0) corresponding to the axis of rotation which is along (V8-V1).
    From u81, compute the 3x3 matrix N in Eq. 2.32 used for computing the rotation matrix R in eq. 2.34
    '''

    V81sumR = pow((V8R - V1R)[0], 2) + pow((V8R - V1R)[1], 2) + pow((V8R - V1R)[2], 2)

    u81R = (V8R - V1R) / pow((V81sumR), 1 / 2)

    NRight = np.stack(([0, -u81R[2], u81R[1]], [u81R[2], 0, -u81R[0]], [-u81R[1], u81R[0], 0]))

    V81sum = pow((V8 - V1)[0], 2) + pow((V8 - V1)[1], 2) + pow((V8 - V1)[2], 2)

    u81 = (V8 - V1) / pow((V81sum), 1 / 2)

    N = np.stack(([0, -u81[2], u81[1]], [u81[2], 0, -u81[0]], [-u81[1], u81[0], 0]))



    T0 = np.array([-20, -25, 500])  # origin of object coordinate system in mm

    T0R = np.array([-10, -25, 500])
    f = 40  # focal length in mm
    velocity = np.array([2, 9, 7])  # translational velocity
    acc = np.array([0.0, -0.80, 0])  # acceleration
    theta0 = 0  # initial angle of rotation is 0 (in degrees)
    w0 = 20  # angular velocity in deg/sec
    p = 0.01  # pixel size(mm)
    Rows = 600  # image size
    Cols = 600  # image size

    r0 = np.round(Rows / 2)  # x-value of center of image
    # r0 = int(Rows/2)
    c0 = np.round(Cols / 2)  # y-value of center of image
    #c0 = int(Cols/2)
    time_range = np.arange(0.0, 24.2, 0.2)

    K = np.stack(([f, 0, 0], [0, f, 0], [0, 0, 1]))


    # This section handles mapping the texture to one face:

    # You are given a face of a cube in 3D space specified by its
    # corners at 3D position vectors V1, V2, V3, V4.
    # You are also given a square graylevel image tmap of size r x c
    # This image is to be "painted" on the face of the cube:
    # for each pixel at position (i,j) of tmap,
    # the corresponding 3D coordinates
    # X(i,j), Y(i,j), and Z(i,j), should be computed,
    # and that 3D point is
    # associated with the brightness given by tmap(i,j).
    #

    # Find h, w: the height and width of the face
    # Find the unit vectors u21 and u41 which correspond to (V2-V1) and (V4-V1)
    # u21 = (V2-V1) / h ; u41 = (V4 - V1) / w


    h = pow(pow((V2 - V1)[0], 2) + pow((V2 - V1)[1], 2) + pow((V2 - V1)[2], 2), 1 / 2)
    w = pow(pow((V4 - V1)[0], 2) + pow((V4 - V1)[1], 2) + pow((V4 - V1)[2], 2), 1 / 2)

    hRight = pow(pow((V2R - V1R)[0], 2) + pow((V2R - V1R)[1], 2) + pow((V2R - V1R)[2], 2), 1 / 2)
    wRight = pow(pow((V4R - V1R)[0], 2) + pow((V4R - V1R)[1], 2) + pow((V4R - V1R)[2], 2), 1 / 2)

    u21 = (V2-V1) / h
    u41 = (V4-V1) / w

    u21R = (V2R - V1R) / hRight
    u41R = (V4R - V1R) / wRight




    # We use u21 and u41 to iteratively discover each point of the face below:
    # Finding the 3D points of the face bounded by V1, V2, V3, V4
    # and associating each point with a color from texture:
    tmap = cv2.imread('einstein132.jpg')  # texture map image
    if tmap is None:
        print("image file can not be found on path given. Exiting now")
        sys.exit(1)
    img = cv2.imread('background.jpg')

    if img is None:
        print("image file can not be found on path given. Exiting now")
        sys.exit(1)
    #background = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    r, c, colors = tmap.shape
    # We keep three arrays of size (r, c) to store the (X, Y, Z) points cooresponding
    # to each pixel on the texture
    X = np.zeros((r, c), dtype=np.float64)
    Y = np.zeros((r, c), dtype=np.float64)
    Z = np.zeros((r, c), dtype=np.float64)
    XR = np.zeros((r, c), dtype=np.float64)
    YR = np.zeros((r, c), dtype=np.float64)
    ZR = np.zeros((r, c), dtype=np.float64)

    for i in range(0, r):
        for j in range(0, c):
            p1 = V1 + (i) * u21 * (h / r) + (j) * u41 * (w / c)

            p1R = V1R + (i) * u21R * (hRight / r) + (j) * u41R * (wRight / c)

            X[i, j] = p1[0]
            Y[i, j] = p1[1]
            Z[i, j] = p1[2]

            XR[i, j] = p1R[0]
            YR[i, j] = p1R[1]
            ZR[i, j] = p1R[2]


    for t in time_range:  # Generate a sequence of images as a function of time


        theta = theta0 + w0 * t

        T = T0 + velocity * t + 0.5 * acc * t * t
        TRight = T0R + velocity * t + 0.5 * acc * t * t


        I = np.stack(([1, 0, 0], [0, 1, 0], [0, 0, 1]))
        sinTheta = round(math.sin(math.radians(theta)),3)
        cosTheta = round(math.cos(math.radians(theta)),3)
        Nsquared = N@N
        R = I + sinTheta * N + (1 - cosTheta) * Nsquared

        NsquaredRight = NRight @ NRight
        RRight = I + sinTheta * NRight + (1 - cosTheta) * NsquaredRight


        # find the image position of vertices


        v = Map2Da(K, R, T, V1)

        v1 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V2)
        v2 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V3)
        v3 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V4)
        v4 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V5)
        v5 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V6)
        v6 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V7)
        v7 = MapIndex(v, c0, r0, p)


        v = Map2Da(K, R, T, V8)
        v8 = MapIndex(v, c0, r0, p)




        vr = Map2Da(K, RRight, TRight, V1R)
        v1r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V2R)
        v2r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V3R)
        v3r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V4R)
        v4r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V5R)
        v5r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V6R)
        v6r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V7R)
        v7r = MapIndex(vr, c0, r0, p)

        vr = Map2Da(K, RRight, TRight, V8R)
        v8r = MapIndex(vr, c0, r0, p)




        # Draw edges of the cube

        # color = (0, 0, 255) #note, CV uses BGR by default, not RGB. This is Red.
        color = (255, 0, 0)  # note, CV uses BGR by default, not gray=(R+G+B)/3. This is Red.
        colorb = (0, 0, 255)
        thickness = 2
        A = np.zeros((Rows, Cols, 3),
                     dtype=np.uint8)
        # array which stores the image at this time step; (Rows x Cols) pixels, 3 channels per pixel

        newimg = np.copy(img)
        newimg = drawLine(newimg, v1, v2, color, thickness)
        newimg = drawLine(newimg, v2, v3, color, thickness)
        newimg = drawLine(newimg, v3, v4, color, thickness)
        newimg = drawLine(newimg, v1, v4, color, thickness)
        newimg = drawLine(newimg, v4, v5, color, thickness)
        newimg = drawLine(newimg, v5, v7, color, thickness)
        newimg = drawLine(newimg, v1, v7, color, thickness)
        newimg = drawLine(newimg, v7, v6, color, thickness)
        newimg = drawLine(newimg, v6, v8, color, thickness)
        newimg = drawLine(newimg, v8, v5, color, thickness)
        newimg = drawLine(newimg, v8, v3, color, thickness)
        newimg = drawLine(newimg, v6, v2, color, thickness)

        newimg = drawLine(newimg, v1r, v2r, colorb, thickness)
        newimg = drawLine(newimg, v2r, v3r, colorb, thickness)
        newimg = drawLine(newimg, v3r, v4r, colorb, thickness)
        newimg = drawLine(newimg, v1r, v4r, colorb, thickness)
        newimg = drawLine(newimg, v4r, v5r, colorb, thickness)
        newimg = drawLine(newimg, v5r, v7r, colorb, thickness)
        newimg = drawLine(newimg, v1r, v7r, colorb, thickness)
        newimg = drawLine(newimg, v7r, v6r, colorb, thickness)
        newimg = drawLine(newimg, v6r, v8r, colorb, thickness)
        newimg = drawLine(newimg, v8r, v5r, colorb, thickness)
        newimg = drawLine(newimg, v8r, v3r, colorb, thickness)
        newimg = drawLine(newimg, v6r, v2r, colorb, thickness)





        # Now we must add the texture to the face bounded by v1-4:
        for i in range(r):
            for j in range(c):
                p1 = [X[i, j], Y[i, j], Z[i, j]]
                p1R = [XR[i, j], YR[i, j], ZR[i, j]]
                # p1 now stores the world point on the cubic face which
                # corresponds to (i, j) on the texture

                v = Map2Da(K, R, T, p1)
                pimg = MapIndex(v, c0, r0, p)

                vr = Map2Da(K, RRight, TRight, p1R)
                pimgR = MapIndex(vr, c0, r0, p)

                (irR, jrR) = pimgR
                (ir, jr) = pimg
                #print(pimg)
                if ((ir >= 0) and (jr >= 0) and (ir < Rows) and (jr < Cols)):
                    tmapval = tmap[i, j, 2]
                    newimg[ir, jr] = [tmapval, 0, 0]  # gray here, but [0, 0, tmpval] for red color output
                if ((irR >= 0) and (jrR >= 0) and (irR < Rows) and (jrR < Cols)):
                    tmapval = tmap[i, j, 2]
                    newimg[irR, jrR] = [0, 0, tmapval]  # gray here, but [0, 0, tmpval] for red color output
        #cv2.imwrite(A,background)
        #imagefinal = cv2.addWeighted(A,0.9,img,0.7,1)
        #imagefinal = cv2.add(A,img)
        cv2.imshow("Display Window", newimg)
        #cv2.waitKey(0)
        #cv2.imshow("Display Window", A)


        #cv2.waitKey(0)
        # ^^^ uncomment if you want to display frame by frame
        # and press return(or any other key) to display the next frame
        # by default just waits 1 ms and goes to next frame
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
