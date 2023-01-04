import os
import random
import time
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.python.ops import control_flow_ops
from absl import flags
from absl import logging



 #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` characters only.")
flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
flags.DEFINE_integer('max_steps', 16002, 'the max training steps ')
flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
flags.DEFINE_integer('save_steps', 500, "the steps to save")

flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
flags.DEFINE_string('train_data_dir', './data/train/', 'the train dataset dir')
flags.DEFINE_string('test_data_dir', './data/test/', 'the test dataset dir')

flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
flags.DEFINE_string('mode', 'validation', 'Running mode. One of {"train", "valid", "test"}')

FLAGS = flags.FLAGS


def loadTR( TrFileName):
    rawDataset = tf.data.TFRecordDataset(TrFileName)    # 读取 TFRecord 文件
    feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    @staticmethod
    def dataAugmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def parseExample(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        imageExample= tf.image.resize_images(
            tf.image.convert_image_dtype(
                tf.io.decode_png(feature_dict['image'],channels=1),tf.float32
            ) ,
            tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        )# 解码PNG图片
        return dataAugmentation(imageExample), feature_dict['label']
    return rawDataset.map(parseExample)

class CWCR(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,             # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],     # 感受野大小
            padding='same',         # padding策略（vaild 或 same）
            activation=tf.nn.relu6   # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu6
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu6
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv4 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            padding='same',
        )
        self.conv5 = tf.keras.layers.Conv2D(
            filters=512,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu6
        )
#        print(self.conv5.input_shape)
        self.flatten = tf.keras.layers.Reshape(target_shape=(8*8*512,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu6)
        self.dense2 = tf.keras.layers.Dense(units=3755)

    def call(self, inputs):
        x = self.conv1(inputs)                  # [batch_size, 28, 28, 32]
        x = self.pool1(x)                       # [batch_size, 14, 14, 32]
        x = self.conv2(x)                       # [batch_size, 14, 14, 64]
        x = self.pool2(x)                       # [batch_size, 7, 7, 64]
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)                     # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)                      # [batch_size, 1024]
        x = self.dense2(x)                      # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

model = CWCR()
#model.compute_output_shape(input_shape=(1,64,64,1))
#for layer in model.layers:
#    print(layer.output_shape)

