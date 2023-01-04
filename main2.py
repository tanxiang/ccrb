import os
import random
import time
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from keras.engine import training
from tensorflow.python.ops import control_flow_ops
from absl import flags
from absl import logging

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


flags.DEFINE_boolean('--random_flip_up_down', False, "Whether to random flip up down")
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
            size=(64, 64)
        )  # 解码PNG图片
        return dataAugmentation(imageExample), feature_dict['label']

    return rawDataset.map(parseExample)


def CWCR(input=None):
    x = tf.keras.layers.Conv2D(
        name='conv1',
        filters=64,  # 卷积层神经元（卷积核）数目
        kernel_size=[5, 5],  # 感受野大小
        padding='same',  # padding策略（vaild 或 same）
        activation=tf.nn.relu6  # 激活函数
    )(input)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu6
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu6
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
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
    #        print(self.conv5.input_shape)
    x = tf.keras.layers.Reshape(target_shape=(8 * 8 * 256,))(x)
    x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu6)(x)
    x = tf.keras.layers.Dense(units=3755)(x)
    return training.Model(input, x, name="CWCR")


model = CWCR(tf.keras.Input((64,64,1)))
model.compile()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)
data_loader = loadTR('train.tfr')
for batch_index in range(1, 5000):
    X, y = data_loader.get_batch(128)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    if batch_index % 100 == 0:
        # 使用CheckpointManager保存模型参数到文件并自定义编号
        path = manager.save(checkpoint_number=batch_index)
        print("model saved to %s" % path)
#tf.keras.applications.VGG16()
# for layer in model.layers:
#    print(layer.output_shape)
