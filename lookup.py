import matplotlib.pyplot as plt
import tensorflow as tf
from absl import flags
import sys

flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

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


trainDataSet = loadTR('train0.tfr')
testDataSet = loadTR('test2.tfr')

for image, label in trainDataSet:
    plt.title(label.numpy())
    plt.imshow(image[0,:, :, 0])
    plt.show()
