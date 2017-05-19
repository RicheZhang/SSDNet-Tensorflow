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
import numpy as np
import cv2
import pickle
import config as cfg
import time
import signal
import sys
import cocoTools as coco
import model.model as model
import model.addSSD as addSSD
import train.loss as loss
import train.matcher as matcher
import tinyFunctions


flags = tf.app.flags
FLAGS = flags.FLAGS
i2name = pickle.load(open("i2name.p", "rb"))

class SSD:
    def __init__(self, modelDir = None):
        # set GPU fraction
        gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction = cfg.GpuMemory)
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
        cfg.outShapes = outShapes
        #generate default box
        cfg.defaults = addSSD.defaultBox(outShapes)

        #initialize train parameters
        with tf.variable_scope("optimizer"):
            self.global_step = tf.Variable(0)
            self.learningRate = tf.placeholder(tf.float32, shape=[])

            self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.totalLoss, global_step=self.global_step)
        newVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="optimizer")
        self.sess.run(tf.variables_initializer(newVars))
        if modelDir is None:
            modelDir = FLAGS.modelDir

        checkPoint = tf.train.get_checkpoint_state(modelDir)
        self.saver = tf.train.Saver()
        #if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)

    def singleImage(self, sample):
        """single image test"""
        dst = cv2.resize(sample, (cfg.inputSize, cfg.inputSize))
        labelsGroup, bBoxesIncrement, step = self.sess.run([self.labels, self.bBoxes, self.global_step],
                                             feed_dict = {self.images: [dst], self.bn: False})
        bBoxes, confidences = addSSD.formatOutput(labelsGroup, bBoxesIncrement)
        tinyFunctions.resizeBoxes(dst, sample, bBoxes)
        return filterBox(bBoxes, confidences)

def filterBox(bBoxes, confidences):
    """NMS and filter boxes with low confidence"""
    filtered = []
    for box, c, label in confidences:
        #not background
        if c >= cfg.confidence and label != cfg.classNum:
            coords = bBoxes[box[0]][box[1]][box[2]][box[3]]
            coords = tinyFunctions.centerToCorner(coords)
            filtered.append((coords, c, label))
    return tinyFunctions.NMS(filtered, cfg.nmsIOU, cfg.nmsNum)

def train():
    """train model"""
    ssd =SSD()
    t = time.time()
    step = 0

    # noinspection PyUnusedLocal
    def signalHandle(sigNum, frame):
        """define SIGINT action"""
        print("Ctrl + c, training stopped!")
        ssd.saver.save(ssd.sess, "%s/checkPoint" % FLAGS.modleDir, step)
        sys.exit(0)

    signal.signal(signal.SIGINT, signalHandle)

    summaryWriter = tf.summary.FileWriter(FLAGS.modelDir)

    trainLoader = coco.Loader(True)
    #i2name = trainLoader.i2name
    trainBatches = trainLoader.create_batches(FLAGS.batchSize, shuffle = True)
    boxMatcher = matcher.matcher()

    while True:
        batch = trainBatches.__next__()
        images, annotations = trainLoader.preprocess_batch(batch)
        labelsGroup, bBoxesIncrement, step = ssd.sess.run([ssd.labels, ssd.bBoxes, ssd.global_step],
                                                           feed_dict={ssd.images: images, ssd.bn: False})
        # noinspection PyUnusedLocal
        batchValues = [None for i in range(FLAGS.batchSize)]

        # noinspection PyShadowingNames
        def matchBoxes(index):
            """get matches matrix and change the matrix to the format of feeding data"""
            matches = boxMatcher.matchBoxes(labelsGroup[index], annotations[index])
            posFeed, negFeed, groundTruthLabelFeed, groundTruthBoxFeed = prepareFeed(matches)
            batchValues[index] = (posFeed, negFeed, groundTruthLabelFeed, groundTruthBoxFeed)

        for index in range(FLAGS.batchSize):
            matchBoxes(index)

        posFeed, negFeed, groundTruthLabelFeed, groundTruthBoxFeed = [np.stack(m) for m in zip(*batchValues)]

        if step < 100:
            learningRate = 1e-2
        elif step < 800:
            learningRate = 1e-3
        elif step < 3000:
            learningRate = 1e-4
        else:
            learningRate = 1e-5

        temp, classLoss, bBoxesLoss, totalLoss, step = ssd.sess.run(
            [ssd.optimizer, ssd.classLoss, ssd.bBoxesLoss, ssd.totalLoss, ssd.global_step],
            feed_dict = {ssd.images: images, ssd.bn: True, ssd.pos: posFeed, ssd.neg: negFeed,
                         ssd.groundTruthLabels: groundTruthLabelFeed, ssd.groundTruthBBoxes: groundTruthBoxFeed,
                         ssd.learningRate: learningRate})

        t = time.time() - t
        print("%i: %f (%f secs)" % (step, totalLoss, t))
        t = time.time()

        summaryFloat(step, "loss", totalLoss, summaryWriter)
        summaryFloat(step, "class loss", classLoss, summaryWriter)
        summaryFloat(step, "bBox loss", bBoxesLoss, summaryWriter)

        if step % 1000 == 0:
            ssd.saver.save(ssd.sess, "%s/checkPoint" % FLAGS.modelDir, step)

def prepareFeed(matches):
    """matches matrix to sample array"""
    pos = []
    neg = []
    groundTruthLabel = []
    groundTruthBox = []
    for o in range(len(cfg.layerBoxesNum)):
        for y in range(cfg.outShapes[o][2]):
            for x in range(cfg.outShapes[o][1]):
                for i in range(cfg.layerBoxesNum[o]):
                    match = matches[o][x][y][i]
                    # there is a ground truth assigned to this default box
                    if isinstance(match, tuple):
                        pos.append(1)
                        neg.append(0)
                        groundTruthLabel.append(match[1])
                        default = cfg.defaults[o][x][y][i]
                        groundTruthBox.append(tinyFunctions.calOffset(default, tinyFunctions.cornerToCenter(match[0])))
                    # this default box was chosen to be a negative
                    elif match == -1:
                        pos.append(0)
                        neg.append(1)
                        groundTruthLabel.append(cfg.classNum)  # background ID
                        groundTruthBox.append([0] * 4)
                    # no influence for this training step
                    else:
                        pos.append(0)
                        neg.append(0)
                        groundTruthLabel.append(cfg.classNum)  # background ID
                        groundTruthBox.append([0] * 4)
    return np.asarray(pos), np.asarray(neg), np.asarray(groundTruthLabel), np.asarray(groundTruthBox)

def summaryFloat(step, name, value, summaryWriter):
    """add summary"""
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summaryWriter.add_summary(summary, global_step=step)

def test(path):
    """test model"""
    testSample = cv2.imread(path)
    ssd = SSD()
    filtered = ssd.singleImage(testSample)
    for box, conf, label in filtered:
        col = tuple((255.0 / cfg.classNum * label, 255, 255))
        print(box, i2name[label], col, conf)
        tinyFunctions.drawResult(testSample, box, i2name[label], conf)
    cv2.imshow("demo", testSample)
    cv2.waitKey(0)


if __name__ == "__main__":
    flags.DEFINE_string("modelDir", "summaries/test0", "model directory")
    flags.DEFINE_integer("batchSize", 8, "batch size")
    flags.DEFINE_string("mode", "train", "train or test")
    flags.DEFINE_string("imagePath", "/home/hjp/deepLearning/dataset/coco/val2014/COCO_val2014_000000000073.jpg", "path to image")
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "test":
        test(FLAGS.imagePath)