# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: train/trainer.py
   create time: 2017年05月02日 星期二 16时18分58秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
#SSD trainer
import tensorflow as tf
import cv2
import pickle
import config
import model.model as model
import model.addSSD as addSSD
import train.loss as loss
import tinyFunctions
from tinyFunctions import drawResult

flags = tf.app.flags
FLAGS = flags.FLAGS
i2name = pickle.load(open("i2name.p", "rb"))

class SSD:
    def __init__(self, modelDir = FLAGS.modelDir):
        # set GPU fraction
        gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction = config.GpuMemory)
        # allow tf to allocate device automatically
        conf = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpuOptions)
        self.sess = tf.Session(config = conf)
        #get result
        self.images, self.bn, self.outs, self.labels, self.bBoxes = model.model(self.sess)
        boxNum = self.labels.get_shape().as_list()[1]

        #calculate loss
        self.pos, self.neg, self.groundTruthLabels, self.groundTruthBBoxes, self.totalLoss, self.classLoss,\
        self.bBoxesLoss = loss.loss(self.labels, self.bBoxes, boxNum)

        outShapes = [t.get_shape().as_list() for t in self.outs]
        config.outShapes = outShapes
        #generate default box
        config.defaults = addSSD.defaultBox(outShapes)

        #initialize train parameters
        with tf.variable_scope("optimizer"):
            self.global_step = tf.Variable(0)
            self.learningRate = tf.placeholder(tf.float32, shape=[])

            self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.totalLoss, global_step=self.global_step)
        newVars = tf.get_collection(tf.GraphKeys.VARIABLES, scope="optimizer")
        self.sess.run(tf.initialize_variables(newVars))

        checkPoint = tf.train.get_checkpoint_state(modelDir)
        self.saver = tf.train.Saver()
        #if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)

    def singleImage(self, sample):
        """single image test"""
        dst = cv2.resize(sample, (config.inputSize, config.inputSize))
        labels, bBoxes, step = self.sess.run([self.labels, self.bBoxes, self.global_step],
                                             feed_dict = {self.images: dst, self.bn: False})
        tinyFunctions.formatOutput(labels, bBoxes)

def train():
    """train model"""

def test(path):
    """test model"""
    testSample = cv2.imread(path)
    ssd = SSD()
    bBoxes, score = ssd.singleImage(testSample)



if __name__ == "__main__":
    flags.DEFINE_string("modelDir", "summaries/test0", "model directory")
    flags.DEFINE_integer("batchSize", 32, "batch size")
    flags.DEFINE_boolean("display", True, "display relevant windows")
    flags.DEFINE_string("mode", "", "train or test")
    flags.DEFINE_string("imagePath", "", "path to image")

    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "test":
        test(FLAGS.imagePath)