# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: loss.py
   create time: 2017年04月26日 星期三 19时20分45秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
from config import *
def smoothL1(x):
    """L reg"""
    a = tf.multiply(0.5, tf.pow(x, 2.0))
    b = tf.subtract(tf.abs(x), 0.5)
    condition = tf.less(tf.abs(x), 1.0)
    return tf.cond(condition, a, b)

def loss(labels, bBoxes, totalBBoxes):
    """define loss"""
    pos = tf.placeholder(tf.float32, [None, totalBBoxes])
    neg = tf.placeholder(tf.float32, [None, totalBBoxes])
    groundTruthLabels = tf.placeholder(tf.float32, [None, totalBBoxes])
    groundTruthBBoxes = tf.placeholder(tf.float32, [None, totalBBoxes, 4])
    sample = pos + neg

    bBoxesLoss = tf.reduce_sum(smoothL1(bBoxes - groundTruthBBoxes), reduction_indices = 2) * pos
    bBoxesLoss = tf.reduce_sum(bBoxesLoss, reduction_indices = 1) / (1e-5 + tf.reduce_sum(pos, reduction_indices = 1))
    classLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, groundTruthLabels) * sample
    classLoss = tf.reduce_sum(classLoss, reduction_indices = 1) / (1e-5 + tf.reduce_sum(sample, reduction_indices = 1))
    totalLoss = tf.reduce_sum(bBoxLossAlpha * bBoxesLoss + classLoss)

    return pos, neg, groundTruthLabels, groundTruthBBoxes,\
           totalLoss, tf.reduce_mean(classLoss), tf.reduce_mean(bBoxesLoss)
