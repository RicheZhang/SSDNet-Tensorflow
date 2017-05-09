# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: matcher.py
   create time: 2017年05月08日 星期一 19时22分35秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#matcher: default box and groundtruth box
import config
from config import layerBoxesNum, classNum, negPosRatio #cant import out_shapes and defaults here since its still not initialized
from tinyFunctions import centerToCorner, calIOU
import numpy as np
class matcher:
    def __init__(self):
        self.indexToIndices = []
        #form the list
        for o_i in range(len(layerBoxesNum)):
            for y in range(config.outShapes[o_i][2]):
                for x in range(config.outShapes[o_i][1]):
                    for i in range(layerBoxesNum[o_i]):
                        self.indexToIndices.append([o_i, y, x, i])

    def matchBoxes(self, predLabels, annotations):
        """match default box and groundtruth box, generate negative sample"""
        #form the list
        matches = [[[[None for i in range(config.layerBoxesNum[o])] for x in range(config.outShapes[o][1])]
                    for y in range(config.outShapes[o][2])]for o in range(len(layerBoxesNum))]
        posCnt = 0

        for (groundTruthBox, id) in annotations:
            topMatch = (None, 0)
            for o in range(len(layerBoxesNum)):
                #cal x,y range
                x1 = max(int(groundTruthBox[0] / (1.0 / config.outShapes[o][2])), 0)
                y1 = max(int(groundTruthBox[1] / (1.0 / config.outShapes[o][1])), 0)
                x2 = min(int((groundTruthBox[0] + groundTruthBox[2]) / (1.0 / config.outShapes[o][2])) + 2,
                         config.outShapes[o][2])
                y2 = min(int((groundTruthBox[1] + groundTruthBox[3]) / (1.0 / config.outShapes[o][1])) + 2,
                         config.outShapes[o][1])

                for y in range(y1, y2):
                    for x in range(x1, x2):
                        for i in range(layerBoxesNum[o]):
                            box = config.defaults[o][x][y][i]
                            IOU = calIOU(groundTruthBox, centerToCorner(box))  # groundTruth is corner
                            if IOU >= config.thresholdIOU: #match positive sample and default box
                                matches[o][x][y][i] = (groundTruthBox, id)
                                posCnt += 1
                            if IOU > topMatch[1]: #current IOU is better
                                topMatch = ([o, x, y, i], IOU)

            topBox = topMatch[0]
            # if box's IOU is <0.5 but is the best
            if topBox is not None and matches[topBox[0]][topBox[1]][topBox[2]][topBox[3]] is None:
                posCnt += 1
                matches[topBox[0]][topBox[1]][topBox[2]][topBox[3]] = (groundTruthBox, id)

        negativeMax = posCnt * negPosRatio
        negCnt = 0

        confidences = getTopConfidences(predLabels, negativeMax)

        for i in confidences:
            indices = self.indexToIndices[i]
            #predict area don't have any object in ground truth, predict label isn't background, it's negative sample
            if matches[indices[0]][indices[1]][indices[2]][indices[3]] is None and np.argmax(predLabels[i]) != classNum:
                matches[indices[0]][indices[1]][indices[2]][indices[3]] = -1
                negCnt += 1
                if negCnt >= negativeMax:
                    break

        return matches


def getTopConfidences(predLabels, topK):
    """get top confidence"""
    confidences = []
    for logits in predLabels:
        probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)
        topLabel = np.amax(probs)
        confidences.append(topLabel)
    k = min(topK, len(confidences))
    topConfidences = np.argpartition(np.asarray(confidences), -k)[-k:]
    return topConfidences