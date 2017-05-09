# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: tinyFunctions.py
   create time: 2017年05月03日 星期三 16时30分57秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#some tiny functions, such as NMS, IOU...
import numpy as np
import cv2
from config import inputSize, layerBoxesNum, outShapes
def calOffset(default, truth):
    """cal offset between box1 and box2"""
    return [truth[0] - default[0], truth[1] - default[1], truth[2] - default[2], truth[3] - default[3]]

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

def resizeBoxes(resized, original, boxes):
    """resize boxes on resized image to boxes on origin image"""
    scale_x = original.shape[1] / float(resized.shape[1])
    scale_y = original.shape[0] / float(resized.shape[0])

    for o in range(len(layerBoxesNum)):
        for y in range(outShapes[o][2]):
            for x in range(outShapes[o][1]):
                for i in range(layerBoxesNum[o]):
                    boxes[o][x][y][i][0] *= scale_x
                    boxes[o][x][y][i][1] *= scale_y
                    boxes[o][x][y][i][2] *= scale_x
                    boxes[o][x][y][i][3] *= scale_y

def clipBox(box):
    """prevent negative width and height"""
    return [box[0], box[1], max(box[2], 0.01), max(box[3], 0.01)]

def calIOU(box1, box2):
    """IOU"""
    a = clipBox(box1)
    b = clipBox(box2)
    left = max(a[0], b[0])
    right = min(a[0] + a[2], b[0] + b[2])
    top = max(a[1], b[1])
    bottom = max(a[1] + a[3], b[1] + b[3])
    ab = 0
    if left < right and top < bottom:
        ab = (right - left) * (bottom - top)
    return ab / (a[2] * a[3] + b[2] * b[3] -ab)

def NMS(boxes, threshold, nums):
    """NMS"""
    filtered = []
    for box, conf, label in boxes:
        if len(filtered) >= nums:
            break
        can = True
        for box2, conf2, label2 in filtered:
            if label == label2 and calIOU(box, box2) > threshold:
                can = False
                break
        if can:
            filtered.append((box, conf, label))
    return filtered

def centerToCorner(rect):
    """box center to box corner"""
    return [rect[0] - rect[2]/2.0, rect[1] - rect[3]/2.0, rect[2], rect[3]]

def cornerToCenter(rect):
    """box corner to box center"""
    return [rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0, rect[2], rect[3]]