import os
import sys
import random
import time
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import matplotlib.pyplot as plt

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from tensorflow.python.ops import control_flow_ops
from absl import flags
from absl import logging

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` characters only.")
flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
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
FLAGS(sys.argv)


def loadTR(TrFileName):
    rawDataset = tf.data.TFRecordDataset(TrFileName)  # 读取 TFRecord 文件
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
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
        imageExample = tf.image.resize(
            tf.image.convert_image_dtype(
                tf.io.decode_png(feature_dict['image'], channels=1), tf.float32
            ),
            [FLAGS.image_size, FLAGS.image_size]
        )  # 解码PNG图片
        imageExample = dataAugmentation(imageExample)
        return tf.expand_dims(imageExample, axis=0), tf.expand_dims(feature_dict['label'], axis=0)

    return rawDataset.map(parseExample)


def CWCR(inputShape=None):
    inputShape = imagenet_utils.obtain_input_shape(
        inputShape,
        default_size=64,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=True
    )
    imgInput = tf.keras.layers.Input(inputShape)
    x = tf.keras.layers.Conv2D(
        64,  # 卷积层神经元（卷积核）数目
        (5, 5),  # 感受野大小
        padding='same',  # padding策略（vaild 或 same）
        activation=tf.nn.relu6,  # 激活函数
        name='conv0'
    )(imgInput)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding='same',
        activation=tf.nn.relu6,
        name='conv1'
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=(2, 2), padding='same', name='pool2')(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=[3, 3],
        padding='same',
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu6
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=(2, 2), padding='same', name='pool3')(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[3, 3],
        padding='same',
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu6
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Dense(units=3755, activation=tf.nn.softmax)(x)
    return training.Model(imgInput, x, name="CWCR")


trainDataSet = loadTR('train0.tfr')
testDataSet = loadTR('test2.tfr')
model = CWCR((64, 64, 1))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['sparse_categorical_accuracy'],
)
checkpoint_filepath = './checkpoint/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    save_best_only=True)
model.fit(trainDataSet.shuffle(buffer_size=3755*4), batch_size=32, steps_per_epoch=3755,epochs=200,callbacks=[model_checkpoint_callback], validation_data=testDataSet.shuffle(100),validation_steps=100)
#model.load_weights(checkpoint_filepath)

