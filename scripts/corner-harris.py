import numpy as np
import cv2 as cv

def onCook(scriptOp):
    img = scriptOp.inputs[0].numpyArray(delayed=True)
    if img is None:
        return
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    img[dst>0.01*dst.max()]=[0,0,255, 255]
    scriptOp.copyNumpyArray(img)