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

    
