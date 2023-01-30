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
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def parseExample(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        imageExample =  tf.image.convert_image_dtype(
                tf.io.decode_image(feature_dict['image'], channels=1), tf.float32
            )
        return imageExample, feature_dict['label']
    return rawDataset.map(parseExample).batch(1)


def draw_line(draw, size,  num):
    for i in range(num):
        begin = (random.randint(0, size[0] / 4), random.randint(0, size[1] / 2))
        end = (random.randint(size[0] / 2, size[0]), random.randint(0, size[1]))
        draw.line([begin, end], fill=0.0, width=3)

def draw_image(size,dataSet):
    """
    生成图片
    :param size: 大小,(width, height)
    """
    width, height = size
    image = Image.new('F', size, 1.0)
    # 设置字体，将字体库复制到当前目录
    #font = ImageFont.truetype('ariali.ttf', int(size[1] * 0.8))
    # 创建画笔
    # 生成文字
    # 获取文字长度与高度
    #font_width, font_height = font.getsize(text)
    # 填充文字
    draw = ImageDraw.Draw(image)
    #draw.text(((width - font_width) / number, (height - font_height) / number),
    #          text, font=font, fill=fontcolor)
    draw_line(draw, size,  3)
    #image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    data = dataSet.take(1)

    for imgs, labels in data:
        im = imgs[0,:, :, :]
        x = 0
        y = 0
        draw.point((x,y),0.0)
    # 保存
    return image

trainDataSet = loadTR('train0.tfr')
testDataSet = loadTR('test2.tfr')
plt.imshow(np.asarray(draw_image([300,300],trainDataSet)))
plt.show()
