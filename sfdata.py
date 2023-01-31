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
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def parseExample(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        imageExample =  tf.io.decode_image(feature_dict['image'], channels=1)
        return imageExample, feature_dict['label']
    return rawDataset.map(parseExample).batch(1)


def draw_line(draw, size,  num):
    for i in range(num):
        begin = (random.randint(0, size[0] / 2), random.randint(0, size[1] / 2))
        end = (random.randint(size[0] / 2, size[0]), random.randint(size[1] / 2, size[1]))
        draw.line([begin, end], fill=0, width=2)

def xxbox(boxAs,boxB):
    for boxA in boxAs:
        [AXmin,AYmin,AXmax,AYmax] = boxA
        [BXmin,BYmin,BXmax,BYmax] = boxB
        if AXmin>BXmax or BXmin>AXmax or AYmin>BYmax or BYmin>AYmax:
            continue
        if AXmin > BXmin and BXmax - AXmin < 8:
            continue
        if BXmin > AXmin and AXmax - BXmin < 8:
            continue
        return True
    return False

def draw_image(size,dataSet,writer):
    """
    生成图片
    :param size: 大小,(width, height)
    """
    image = Image.new('L', size, 255)
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

    boxes =[]
    label =[]
    addSample = 0
    for imgs, labels in dataSet:
        im = imgs[0,:, :, :]
        imShape = tf.shape(im)
        minx = random.randint(0, size[0]-imShape[1])
        miny = random.randint(0, size[1]-imShape[0])
        maxx = minx + imShape[1]
        maxy = miny + imShape[0]
        boxAdded =False
        for testTime in range(6):
            if not xxbox(boxes, [minx, miny, maxx, maxy]):
                boxAdded=True
                break
            minx = random.randint(0, size[0] - imShape[1])
            miny = random.randint(0, size[1] - imShape[0])
            maxx = minx + imShape[1]
            maxy = miny + imShape[0]
        if not boxAdded:
            imgMemory = io.BytesIO()
            image.save(imgMemory, 'PNG')
            addSample += len(label)
            print(label,addSample)
            feature_dict = {
                'image/encoded':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgMemory.getvalue()])),
                'image/object/bbox/xmin':
                    tf.train.Feature(float_list=tf.train.FloatList(value=[b[0] for b in boxes])),
                'image/object/bbox/ymin':
                    tf.train.Feature(float_list=tf.train.FloatList(value=[b[1] for b in boxes])),
                'image/object/bbox/xmax':
                    tf.train.Feature(float_list=tf.train.FloatList(value=[b[2] for b in boxes])),
                'image/object/bbox/ymax':
                    tf.train.Feature(float_list=tf.train.FloatList(value=[b[3] for b in boxes])),
                'image/object/class/label':
                    tf.train.Feature(int64_list=tf.train.Int64List(value=label))}
            #plt.imshow(np.asarray(image))
            #plt.show()
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))  # 通过字典建立 Example
            writer.write(example.SerializeToString())
            image = Image.new('L', size, 255)
            draw = ImageDraw.Draw(image)
            draw_line(draw, size, 3)
            boxes = []
            label = []

        boxes.append([minx, miny, maxx, maxy])
        label.append(labels.numpy()[0])
        imp = im.numpy()
        for y in range(imShape[0]):
            for x in range(imShape[1]):
                pixel = image.getpixel((minx+x,miny+y))
                draw.point((minx+x,miny+y),int(pixel-(255- imp[y, x][0])))


trainDataSet = loadTR('train0.tfr')
testDataSet = loadTR('test2.tfr')
with tf.io.TFRecordWriter('ssd.tfr') as writer:
    draw_image([300,300],trainDataSet,writer)
