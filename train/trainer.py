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
import time
import signal
import sys
import cocoTools as coco
import model.model as model
import model.addSSD as addSSD
import train.loss as loss
import tinyFunctions

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
        labelsGroup, bBoxesIncrement, step = self.sess.run([self.labels, self.bBoxes, self.global_step],
                                             feed_dict = {self.images: dst, self.bn: False})
        bBoxes, confidences = addSSD.formatOutput(labelsGroup, bBoxesIncrement)
        tinyFunctions.resizeBoxes(dst, sample, bBoxes)
        return filterBox(bBoxes, confidences)

def filterBox(bBoxes, confidences):
    """NMS and filter boxes with low confidence"""
    filtered = []
    for box, c, label in confidences:
        #not background
        if c >= config.confidence and label != config.classNum:
            coords = bBoxes[box[0]][box[1]][box[2]][box[3]]
            coords = tinyFunctions.centerToCorner(coords)
            filtered.append((coords, c, label))
    return tinyFunctions.NMS(filtered, config.nmsIOU, config.nmsNum)

def train():
    """train model"""
    ssd =SSD()
    t = time.time()
    step = 0

    def signalHandle(sigNum, frame):
        """define SIGINT action"""
        print("Ctrl + c, training stopped!")
        ssd.saver.save(ssd.sess, "%s/ckpt" % FLAGS.modleDir, step)
        sys.exit(0)

    signal.signal(signal.SIGINT, signalHandle)

    summaryWriter = tf.summary.FileWriter(FLAGS.modelDir)
    boxMatcher = matcher()

    trainLoader = coco.Loader(True)
    i2name = trainLoader.i2name
    trainBatches = trainLoader.create_batches(FLAGS.batchSize, shuffle = True)

    while True:
        batch = trainBatches.next()
        images, annotations = trainLoader.preprocess_batch(batch)
        labelsGroup, bBoxesIncrement, step = ssd.sess.run([ssd.labels, ssd.bBoxes, ssd.global_step],
                                                           feed_dict={ssd.images: images, ssd.bn: False})
        batchValue = [None for i in range(FLAGS.batchSize)]


        #################################################


        #################################################




        if step < 2000:
            learningRate = 1e-3
        elif step < 40000:
            learningRate = 1e-4
        else:
            learningRate = 1e-5

        temp, classLoss, bBoxesLoss, totalLoss, step = ssd.sess.run(
            [ssd.optimizer, ssd.classLoss, ssd.bBoxesLoss, ssd.totalLoss, ssd.global_step],
            feed_dict = {ssd.images: images, ssd.bn: True,
                         ssd.pos: positives_f, ssd.neg: negatives_f,
                         ssd.groundTruthLabels: true_labels_f, ssd.groundTruthBBoxes: true_locs_f,
                         ssd.learningRate: learningRate})

        t = time.time() - t
        print("%i: %f (%f secs)" % (step, totalLoss, t))
        t = time.time()

        summary_float(step, "loss", totalLoss, summaryWriter)
        summary_float(step, "class loss", classLoss, summaryWriter)
        summary_float(step, "bBox loss", bBoxesLoss, summaryWriter)

        if step % 1000 == 0:
            ssd.saver.save(ssd.sess, "%s/ckpt" % FLAGS.model_dir, step)

def summary_float(step, name, value, summaryWriter):
    """add summary"""
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=float(value))])
    summaryWriter.add_summary(summary, global_step=step)

def test(path):
    """test model"""
    testSample = cv2.imread(path)
    ssd = SSD()
    filtered = ssd.singleImage(testSample)
    for box, conf, label in filtered:
        col = tuple((255.0 / config.classNum * label, 255, 255))
        tinyFunctions.drawResult(testSample, box, i2name[label], col, conf)
    cv2.imshow("demo", testSample)
    cv2.waitKey(0)


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