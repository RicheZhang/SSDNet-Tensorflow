# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: addSSD.py
   create time: 2017年04月26日 星期三 16时24分20秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#box definition and additional layer for SSD
import tensorflow as tf
import numpy as np
from config import *
#additional part for SSD, including BN and convolution
def batchNorm(x, train_phase, decay=0.9, eps=1e-5):
    """batch normalization"""
    shape = x.get_shape().as_list()

    assert len(shape) in [2, 4]

    n_out = shape[-1]
    beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0))
    gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.constant_initializer(1))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)


def convLayerSSD(x, train, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", reluFlag = True):
    """convolution"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum],
                            initializer = tf.contrib.layers.xavier_initializer_conv2d())
        #b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        #out = tf.nn.bias_add(featureMap, b)
        bn = batchNorm(featureMap, train)
        if reluFlag:
            return tf.nn.relu(bn, name = scope.name)
        else:
            return bn

def boxScale(k):
    """calculate box scale"""
    m = 6.0
    sk = sMin + (sMax - sMin) * (k - 1.0) / (m - 1)
    return sk

def defaultBox(shapes):
    """generate default box"""
    boxes = []
    for index, shape in enumerate(shapes):
        layerBoxes = []
        sk = boxScale(index + 1)
        sk1 = boxScale(index + 2)
        for x in range(shape[1]):
            xBoxes = []
            for t in range(shape[2]):
                yBoxes = []
                #ratios
                if index == 0: #conv4_3
                    rs = conv4_3Ratios
                else:
                    rs = boxRatios
                for i in range(len(rs)):
                    if index == 0:
                        scale = conv4_3Scale
                    else:
                        scale = sk
                        #additional scale when ratios == 1
                        if i == 0:
                            scale = np.sqrt(sk * sk1)
                    width = scale * np.sqrt(rs[i])
                    height = scale / np.sqrt(rs[i])
                    #box center
                    centerX = (x + 0.5) / float(shape[1])
                    centerY = (x + 0.5) / float(shape[2])
                    yBoxes.append([centerX, centerY, width, height])
                xBoxes.append(yBoxes)
            layerBoxes.append(xBoxes)
        boxes.append(layerBoxes)
    return boxes