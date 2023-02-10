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

flags.DEFINE_string('mode', 'validation', 'Running mode. One of {"train", "valid", "test"}')

FLAGS = flags.FLAGS
FLAGS(sys.argv)


def loadTR(TrFileName):
    rawDataset = tf.data.TFRecordDataset(TrFileName)  # 读取 TFRecord 文件
    feature_description = {
        'image/encoded':
            tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(tf.int64)}
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
        imageExample = tf.image.convert_image_dtype(
                tf.io.decode_png(feature_dict['image/encoded']), tf.float32
            )
        imageExample = dataAugmentation(imageExample)
        return imageExample, feature_dict['image/object/class/label']

    return rawDataset.map(parseExample).batch(128)


trainDataSet = loadTR('ssd0.tfr')
testDataSet = loadTR('ssdt1.tfr')
from keras.applications.efficientnet_v2 import EfficientNetV2

modelBB = EfficientNetV2(
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=224,
    model_name="efficientnetv2-b0",
    activation=tf.nn.relu6,
    include_top=False,
    weights=None,
    input_shape=(None, None, 1),
    pooling=None,
    classes=3755,
    classifier_activation=None,
    include_preprocessing=False, )

checkpoint_filepath = './checkpointE60/'

checkpointEBB = tf.train.latest_checkpoint(checkpoint_filepath)
if checkpointEBB:
    modelBB.load_weights(tf.train.latest_checkpoint(checkpoint_filepath))
print(modelBB.summary())
converter = tf.lite.TFLiteConverter.from_keras_model(modelBB)
tflite_model_quant = converter.convert()

open("modelbbqr0.tflite", "wb").write(tflite_model_quant)

modelBB.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath, "m{loss:.2f}.fs"),
    verbose=1,
    save_weights_only=True,
    save_best_only=True)
# 897758 sample
modelBB.fit(trainDataSet.repeat(), steps_per_epoch=7014, epochs=1, callbacks=[model_checkpoint_callback],
          validation_data=testDataSet, validation_steps=500)


# model.fit(trainDataSet.repeat(),steps_per_epoch=7014,epochs=2,callbacks=[model_checkpoint_callback])
# modelEval = model.evaluate(testDataSet)
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()

open("modelq0.tflite", "wb").write(tflite_model_quant)
