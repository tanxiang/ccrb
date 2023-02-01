import io
import matplotlib.pyplot as plt
import tensorflow as tf
from absl import flags
import sys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def loadTR(TrFileName):
    rawDataset = tf.data.TFRecordDataset(TrFileName)  # 读取 TFRecord 文件
    def parseExample(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
            'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        imageExample = tf.io.decode_image(feature_dict['image/encoded'])
        xmin = feature_dict['image/object/bbox/xmin']
        ymin = feature_dict['image/object/bbox/ymin']
        xmax = feature_dict['image/object/bbox/xmax']
        ymax = feature_dict['image/object/bbox/ymax']
        print(feature_dict['image/object/class/label'])
        return imageExample, xmin,feature_dict['image/object/class/label']
    return rawDataset.map(parseExample).batch(1)




trainDataSet = loadTR('ssd0.tfr')
testDataSet = loadTR('ssdt1.tfr')


for image,xmin,label in testDataSet:
    print(xmin)
    print(label)
    plt.imshow(image[0,:, :, :])
    plt.show()