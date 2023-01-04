import tensorflow as tf
import argparse
import os

parser = argparse.ArgumentParser(description='Process some args.')
parser.add_argument('--data', default='./data/train/')
parser.add_argument('--output', default='train.tfr')
args = parser.parse_args()

def buildTR(dataDir, TrFileName):
    imageNames = []
    for rootDir, dirList, fileList in os.walk(dataDir):
        imageNames += [os.path.join(rootDir, filePath) for filePath in fileList]
    imageLables = [int(fileName[len(dataDir):].split(os.sep)[0]) for fileName in imageNames]
    with tf.io.TFRecordWriter(TrFileName) as writer:
        for filename, label in zip(imageNames, imageLables):
            image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
            feature = {                             # 建立 tf.train.Feature 字典
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
            writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件

buildTR(args.data,args.output)
