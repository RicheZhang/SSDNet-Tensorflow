# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: model.py
   create time: 2017年04月25日 星期二 17时27分20秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#SSD model
import tensorflow as tf
import vgg.vggSSD as vggSSD
import model.addSSD as addSSD
from config import *

def model(sess):
    images = tf.placeholder("float", [None, inputSize, inputSize, 3])
    bnTrain = tf.placeholder(tf.bool)

    vgg = vggSSD.VGG19()

    with tf.variable_scope("SSD_extension"):
        conv6 = addSSD.convLayerSSD(vgg.conv5_4, bnTrain, 3, 3, 1, 1, 1024, name = "conv6")
        conv7 = addSSD.convLayerSSD(conv6, bnTrain, 1, 1, 1, 1, 1024, name = "conb7")

        conv8_1 = addSSD.convLayerSSD(conv7, bnTrain, 1, 1, 1, 1, 256, name = "conv8_1")
        conv8_2 = addSSD.convLayerSSD(conv8_1, bnTrain, 3, 3, 2, 2, 512, name = "conv8_2")

        conv9_1 = addSSD.convLayerSSD(conv8_2, bnTrain, 1, 1, 1, 1, 128, name = "conv9_1")
        conv9_2 = addSSD.convLayerSSD(conv9_1, bnTrain, 3, 3, 2, 2, 256, name = "conv9_2")

        conv10_1 = addSSD.convLayerSSD(conv9_2, bnTrain, 1, 1, 1, 1, 128, name = "conv10_1")
        conv10_2 = addSSD.convLayerSSD(conv10_1, bnTrain, 3, 3, 2, 2, 256, name = "conv10_2")

        pool11 = tf.nn.avg_pool(conv10_2, [1, 3, 3, 1], [1, 1, 1, 1], "valid")

        classes = classNum + 1

        out1 = addSSD.convLayerSSD(vgg.conv4_4, bnTrain, 3, 3, 1, 1, 3 * (classes + 4),
                                   name = "out1", reluFlag = False)
        out2 = addSSD.convLayerSSD(conv7, bnTrain, 3, 3, 1, 1, 6 * (classes + 4),
                                   name = "out2", reluFlag = False)
        out3 = addSSD.convLayerSSD(conv8_2, bnTrain, 3, 3, 1, 1, 6 * (classes + 4),
                                   name="out3", reluFlag=False)
        out4 = addSSD.convLayerSSD(conv9_2, bnTrain, 3, 3, 1, 1, 6 * (classes + 4),
                                   name="out4", reluFlag=False)
        out5 = addSSD.convLayerSSD(conv10_2, bnTrain, 3, 3, 1, 1, 6 * (classes + 4),
                                   name="out5", reluFlag=False)
        out6 = addSSD.convLayerSSD(pool11, bnTrain, 1, 1, 1, 1, 6 * (classes + 4),
                                   name="out6", reluFlag=False)

    newVars = tf.get_collection(tf.GraphKeys.VARIABLES, scope = "SSD_extension")
    sess.run(tf.initialize_variables(newVars))

    outs = [out1, out2, out3, out4, out5, out6]
    outfs = []
    for i, out in zip(range(len(outs)), outs):
        height = out.get_shape().as_list()[1]
        width = out.get_shape().as_list()[2]
        if i == 0:
            outfs.append(tf.reshape(out, [-1, width * height * 3, classes + 4]))
        else:
            outfs.append(tf.reshape(out, [-1, width * height * 6, classes + 4]))

    outCube = tf.concat(1, outfs)
    labels = outCube[:,:,:classes]
    bBoxes = outCube[:,:,classes:]

    return images, bnTrain, outs, labels, bBoxes
