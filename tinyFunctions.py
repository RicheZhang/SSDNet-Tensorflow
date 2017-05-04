# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: tinyFunctions.py
   create time: 2017年05月03日 星期三 16时30分57秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#some tiny functions, such as NMS, IOU...
import cv2
from config import inputSize
def drawBBox(I, r, color, thickness=1):
    """drawBBox: image, point, color, thickness"""
    if abs(sum(r)) < 100: # conditional to prevent min/max error
        cv2.rectangle(I, (int(r[0] * inputSize), int(r[1] * inputSize)),
                      (int((r[0] + max(r[2], 0)) * inputSize), int((r[1] + max(r[3], 0)) * inputSize)),
                      color, thickness)

def drawResult(I, r, text, color=(255, 0, 255), confidence=-1):
    """draw BBox and annotation"""
    drawBBox(I, r, color)
    #draw annotation box
    cv2.rectangle(I, (int(r[0] * inputSize), int(r[1] * inputSize - 15)),
                  (int(r[0] * inputSize + 100), int(r[1] * inputSize)), color, -1)
    ann = text
    if confidence >= 0:
        ann += ": %0.2f" % confidence

    cv2.putText(I, ann, (int(r[0] * inputSize), int((r[1]) * inputSize)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def formatOutput(labels, bBoxes, ):

