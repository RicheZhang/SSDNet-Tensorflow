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
import numpy as np
import model.model
import train.loss
from config import *

class SSD:
#def __init__(self):