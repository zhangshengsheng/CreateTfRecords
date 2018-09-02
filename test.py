# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:03:43 2018

@author: 流羽秋
"""
import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET #操作xml文件

#我的标签定义只有两类，根据自己的图片而定
VOC_LABELS = {
    'none': (0, 'Background'),
    '丑时': (2, '文字'),
    '长安酒店': (3, '文字'),
    '任务': (4, '文字'),
    '师门任务': (5, '文字'),
    '镇元大仙': (6, '文字'),
    '前': (7, '文字'),
    '往': (8, '文字'),
    '师父': (9, '文字'),
    '处': (10, '文字'),
    '领': (11, '文字'),
    '取': (12, '文字')
}

# 图片和标签存放的文件夹.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# 随机种子.
RANDOM_SEED = 4242  
SAMPLES_PER_FILES = 200  #每个文件的样本数

#生成整数型，浮点型和字符串型的属性
def int64_feature(value):
    
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

#图片处理
def _process_image(directory, name):
   
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('utf-8')) #变为ascii格式

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        a=float(bbox.find('ymin').text) / shape[0]
        b=float(bbox.find('xmin').text) / shape[1]
        a1=float(bbox.find('ymax').text) / shape[0]
        b1=float(bbox.find('xmax').text) / shape[1]
        a_e=a1-a
        b_e=b1-b
        if abs(a_e)<1 and abs(b_e)<1:
            bboxes.append((a,b,a1,b1))
        
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated

#转化样例
def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example
#增加到tfrecord
def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())
#name为转化文件的前缀
def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)

def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)  

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path)) #排序
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)
   
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write(' Converting image %d/%d \n' % (i+1, len(filenames)))#终端打印，类似print
                sys.stdout.flush() #缓冲

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
            
    print('\nFinished converting the Pascal VOC dataset!')

#原数据集路径，输出路径以及输出文件名
dataset_dir="./VOC2017/"
output_dir="./TFRecords"
name="voc_train"
def main(_):
     run(dataset_dir, output_dir,name)

if __name__ == '__main__':
    tf.app.run()
    
