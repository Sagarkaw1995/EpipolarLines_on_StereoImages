from tkinter import Tk, Canvas, NW, Button, CENTER, messagebox, NE, E, W
import numpy as np
from PIL import Image, ImageTk
import cv2 as cv
from matplotlib import pyplot as plt

LeftPoints = []
RightPoints = []
A = []
LC = []
RealTime_LeftPoint = []
RealTime_RightPoint = []


def imgLeft():
    img1 = Image.open("Capture1.png")
    img1 = img1.resize((700, 600))
    img1.save("Cap1.png")


def imgRight():
    img2 = Image.open("Capture2.png")
    img2 = img2.resize((700, 600))
    img2.save("Cap2.png")


def on_mouseclick1(mc):
    cord_x, cord_y = int(canvas.canvasx(mc.x)), int(canvas.canvasy(mc.y))
    if cord_x <= 700:
        LeftPoints.append(cord_x)
        LeftPoints.append(cord_y)
    else:
        RightPoints.append(cord_x - 700)
        RightPoints.append(cord_y)
    canvas.create_rectangle(cord_x - 5, cord_y - 5, cord_x + 5, cord_y + 5, fill='yellow')


def on_mouseclick2(mc):
    cord_x, cord_y = int(canvas.canvasx(mc.x)), int(canvas.canvasy(mc.y))
    if cord_x <= 700:
        RealTime_LeftPoint.append(cord_x)
        RealTime_LeftPoint.append(cord_y)
    else:
        RealTime_RightPoint.append(cord_x - 700)
        RealTime_RightPoint.append(cord_y)
    canvas.create_rectangle(cord_x - 5, cord_y - 5, cord_x + 5, cord_y + 5, fill='green')


def calc_fm():
    print("LeftPoints: ", end="")
    for x in LeftPoints:
        print(x, end=" ")
    print("\n")
    print("RightPoints: ", end="")
    for y in RightPoints:
        print(y, end=" ")
    print("\n")
    if len(LeftPoints) != len(RightPoints):
        messagebox.showerror("Error", "Select equal number of points in both the images.")
    elif len(LeftPoints) < 20 or len(RightPoints) < 20:
        messagebox.showerror("Error", "Both images must have atleast 10 points")
    else:
        for i in range(0, len(LeftPoints), 2):
            A.append([RightPoints[i] * LeftPoints[i], RightPoints[i] * LeftPoints[i + 1], RightPoints[i],
                      RightPoints[i + 1] * LeftPoints[i], RightPoints[i + 1] * LeftPoints[i + 1], RightPoints[i + 1],
                      LeftPoints[i], LeftPoints[i + 1], 1])

        Matrix_A = np.array(A)

        U, Sigma, V = np.linalg.svd(Matrix_A)

        Min_Sigma_Value_Index = np.argmin(Sigma)

        Fundamental_Matrix = V[Min_Sigma_Value_Index].reshape(3, 3)

        return Fundamental_Matrix


def epipolarlines(image, lines, points):
    r, c, _ = image.shape
    for r, pt1 in zip(lines, points):
        color = (0, 255, 0)
        a1, b1 = map(int, [0, -r[2] / r[1]])
        a2, b2 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        image = cv.line(image, (a1, b1), (a2, b2), color, 2)
        image = cv.circle(image, tuple(pt1), 2, color, 2)
    return image


def plot_epilines():
    F = calc_fm()
    LP = np.array(LeftPoints).reshape(int(len(LeftPoints) / 2), 2)
    RP = np.array(RightPoints).reshape(int(len(RightPoints) / 2), 2)

    imgLeft = plt.imread("Cap1.png")
    imgRight = plt.imread("Cap2.png")

    onesL = np.ones((int(len(LeftPoints) / 2), 1))
    onesR = np.ones((int(len(RightPoints) / 2), 1))

    LP_final = np.append(LP, onesL, axis=1)
    LP_Epilines = np.dot(F, LP_final.transpose())

    RP_final = np.append(RP, onesR, axis=1)
    RP_Epilines = np.dot(RP_final, F)

    fig1 = plt.figure()
    Right_Img_with_Epilines = epipolarlines(imgRight, LP_Epilines.transpose(), RP)
    plt.imshow(Right_Img_with_Epilines)
    fig1.savefig("Res2.png")

    fig2 = plt.figure()
    Left_Img_with_Epilines = epipolarlines(imgLeft, RP_Epilines, LP)
    plt.imshow(Left_Img_with_Epilines)
    fig2.savefig("Res1.png")

    Res_LeftImage = cv.imread('Res1.png')
    Res_RightImage = cv.imread('Res2.png')

    Res_Combined = np.hstack((Res_LeftImage, Res_RightImage))
    cv.namedWindow("Epilines")
    cv.imshow("Epilines", Res_Combined)
    k = cv.waitKey(0)
    if k == ord('c'):
        cv.destroyAllWindows()


def plot_epilines_RealTime():
    F = calc_fm()
    LP = np.array(RealTime_LeftPoint).reshape(int(len(RealTime_LeftPoint) / 2), 2)
    RP = np.array(RealTime_RightPoint).reshape(int(len(RealTime_RightPoint) / 2), 2)
    imgLeft = plt.imread("Cap1.png")
    imgRight = plt.imread("Cap2.png")

    onesL = np.ones((int(len(RealTime_LeftPoint) / 2), 1))
    onesR = np.ones((int(len(RealTime_RightPoint) / 2), 1))

    LP_final = np.append(LP, onesL, axis=1)
    LP_Epilines = np.dot(F, LP_final.transpose()) # LP_final @ F

    RP_final = np.append(RP, onesR, axis=1)
    RP_Epilines = np.dot(RP_final, F) #RP_final @ F

    fig1 = plt.figure()
    Right_Img_with_Epilines = epipolarlines(imgRight, LP_Epilines.transpose(), LP)
    plt.imshow(Right_Img_with_Epilines)
    fig1.savefig("Res4.png")

    fig2 = plt.figure()
    Left_Img_with_Epilines = epipolarlines(imgLeft, RP_Epilines, RP)
    plt.imshow(Left_Img_with_Epilines)
    fig2.savefig("Res3.png")

    Res_LeftImage = cv.imread('Res3.png')
    Res_RightImage = cv.imread('Res4.png')

    Res_Combined = np.hstack((Res_LeftImage, Res_RightImage))
    cv.namedWindow("Epipolar Lines")
    cv.imshow("Epipolar Lines", Res_Combined)
    k = cv.waitKey(0) & 0xFF
    if k == ord('c'):
        cv.destroyAllWindows()


GUI = Tk()

FM = Button(GUI, text="Fundamental Matrix", command=calc_fm)
FM.pack(anchor=CENTER)

EL = Button(GUI, text="Calculate Epilines", command=plot_epilines)
EL.pack(anchor=CENTER)

Realtime_EL = Button(GUI, text="Calculate Realtime Epilines", command=plot_epilines_RealTime)
Realtime_EL.pack(anchor=CENTER)

imgLeft()
imgRight()
img1 = Image.open("Cap1.png")
img2 = Image.open("Cap2.png")

picL = ImageTk.PhotoImage(img1)
picR = ImageTk.PhotoImage(img2)

canvas = Canvas(GUI, width=1400, height=600)
canvas.pack()

image_Container1 = canvas.create_image(0, 0, anchor=NW, image=picL)
image_Container2 = canvas.create_image(700, 0, anchor=NW, image=picR)

canvas.bind("<Button-1>", on_mouseclick1)
canvas.bind("<Button-2>", on_mouseclick2)

GUI.mainloop()
