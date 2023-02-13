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
import neural_structured_learning as nsl


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
        return imageExample, feature_dict['label']
    return rawDataset.map(parseExample).batch(128)


def convert_to_adversarial_training_dataset(dataset):
  def to_dict(x, y):
    return {'input_1': x, 'label': y}

  return dataset.map(to_dict)

trainDataSet = loadTR('train0.tfr')
testDataSet = loadTR('test1.tfr')

from keras.applications.efficientnet_v2 import EfficientNetV2

model = EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        model_name="efficientnetv2-b0",
        activation = tf.nn.relu6,
        include_top=True,
        weights=None,
        input_shape=(64, 64, 1),
        classes=3755,
        include_preprocessing=False, )

print(model.summary())
adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
adv_model = nsl.keras.AdversarialRegularization(model,adv_config=adv_config)

adv_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
checkpoint_filepath = './checkpointE60nsl/'

checkpointEBB = tf.train.latest_checkpoint(checkpoint_filepath)
if checkpointEBB:
    adv_model.load_weights(tf.train.latest_checkpoint(checkpoint_filepath))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_filepath,"m{loss:.2f}.fs"),
    verbose=1,
    save_weights_only=True,
    save_best_only=True)
#897758 sample
advData = convert_to_adversarial_training_dataset(trainDataSet)
adv_model.fit(advData,batch_size=128,callbacks=[model_checkpoint_callback],steps_per_epoch=7014)
#model.fit(trainDataSet.repeat(),steps_per_epoch=7014,epochs=2,callbacks=[model_checkpoint_callback])
#modelEval = model.evaluate(testDataSet)
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

#converter = tf.lite.TFLiteConverter.from_keras_model(adv_model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = representative_data_gen

#tflite_model_quant = converter.convert()

#open("modelq0R6NSL.tflite", "wb").write(tflite_model_quant)

