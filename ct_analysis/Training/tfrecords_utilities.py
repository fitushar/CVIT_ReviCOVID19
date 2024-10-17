from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from config import*
from volumentations import *


########################-------Fucntions for tf records
# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#GaussianNoise(var_limit=(0, 5), p=0.2),
#RandomGamma(gamma_limit=(0.5, 1.5), p=0.2)
def get_augmentation():
    return Compose([Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
                    Flip(2, p=0.5),
                    Flip(1, p=0.5),
                    RandomRotate90((1, 2), p=0.5)], p=0.5)
def agunentor(img):
    aug = get_augmentation()
    data = {'image': img}
    aug_data = aug(**data)
    img     = aug_data['image']
    return np.ndarray.astype(img , np.float32)

@tf.function
def decode_ct_withAugmentation(Serialized_example):

    features = {
                'covid_lbl': tf.io.FixedLenFeature([],tf.int64),
                'image' : tf.io.FixedLenFeature([],tf.string),
                'PatchY': tf.io.FixedLenFeature([],tf.int64),
                'PatchX': tf.io.FixedLenFeature([],tf.int64),
                'PatchZ': tf.io.FixedLenFeature([],tf.int64),
                'label_shape': tf.io.FixedLenFeature([],tf.int64),
                'Sub_id': tf.io.FixedLenFeature([],tf.string)
                    }
    examples=tf.io.parse_single_example(Serialized_example,features)
    ##Decode_image_float
    image_1 = tf.io.decode_raw(examples['image'], float)
    img_shape=[examples['PatchZ'],examples['PatchY'],examples['PatchX']]
    print(img_shape)
    #Resgapping_the_data
    img1=tf.reshape(image_1,img_shape)
    result_tensor=tf.numpy_function(agunentor, [img1], tf.float32)
    result_tensor.set_shape([96,160,160])
    img=tf.reshape(result_tensor,[96,160,160])
    #Because CNN expect(batch,H,W,D,CHANNEL)
    img=tf.expand_dims(img1, axis=-1)
    img=tf.cast(img, tf.float32)
    lbl=examples['covid_lbl']
    return img,lbl

@tf.function
def decode_ct_withoutAug(Serialized_example):

    features = {
                'covid_lbl': tf.io.FixedLenFeature([],tf.int64),
                'image' : tf.io.FixedLenFeature([],tf.string),
                'PatchY': tf.io.FixedLenFeature([],tf.int64),
                'PatchX': tf.io.FixedLenFeature([],tf.int64),
                'PatchZ': tf.io.FixedLenFeature([],tf.int64),
                'label_shape': tf.io.FixedLenFeature([],tf.int64),
                'Sub_id': tf.io.FixedLenFeature([],tf.string)
                    }
    examples=tf.io.parse_single_example(Serialized_example,features)
    ##Decode_image_float
    image_1 = tf.io.decode_raw(examples['image'], float)
    img_shape=[examples['PatchZ'],examples['PatchY'],examples['PatchX']]
    print(img_shape)
    #Resgapping_the_data
    img=tf.reshape(image_1,img_shape)
    #Because CNN expect(batch,H,W,D,CHANNEL)
    img=tf.expand_dims(img, axis=-1)
    img=tf.cast(img, tf.float32)
    lbl=examples['covid_lbl']
    return img,lbl
