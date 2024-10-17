from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import tensorflow as tf


#function-(1)
#-|This function resample images to desired sapcing spacing using simpleitk
# [ref:SimpleITK;https://github.com/SimpleITK/SimpleITK]
def resample_img2mm(itk_image, out_spacing=[2.0, 2.0, 5.0], is_label=False):
    """
    inputs: a.itk_image   (simpleITK image)     : image to be resample.
            b.out_spacing (float;list or tuple) : Spacing to which to rsample;type:float,[height,width,depth]
            c.is_label    (True/False)          : Define resapling a table image or not. Default=False

    output (simpleITK image)                    : Resampled image

    """
    original_spacing = itk_image.GetSpacing() #-|Getting original image spacing
    original_size    = itk_image.GetSize()    #-|Getting original image size

    #-| Calculating resampled size
    out_size         = [
                      int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                      int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                      int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    #-|Making Resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    #-|If Lable is True the interpolation performed is Nearest Neighbour
    #  Otherwise it's BSpline interpolation.
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    #-|Returned resampled itk-image
    return resample.Execute(itk_image)

#function-(2)
#-|This function normalize images[type:numpy.arrays] to fit [0, 1] range
# [ref:DLTK;https://dltk.github.io/]
def normalise_zero_one(image):
    """
    input (np.ndarray) : image to be normalized
    output(np.ndarray) : Normalized image fitted to [0,1]
    """
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret

#function-(3)
#-|This function normalize images[type:numpy.arrays] to fit [-1, 1] range
# [ref:DLTK;https://dltk.github.io/]
def normalise_one_one(image):
    """
    input (np.ndarray) : image to be normalized
    output(np.ndarray) : Normalized image fitted to [-1,1]
    """
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret

#function-(4)
#-|This function resize image by cropping or padding dimension to fit specified size
# [ref:DLTK;https://dltk.github.io/]
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """.
    inputs:
            a. image    (np.ndarray): image to be resized
            b. img_size (list or tuple): new image size
            c. kwargs (): additional arguments to be passed to np.pad
    output:
    np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)

#function-(5)
#-|This function Load an nifti image and performe the listed pre-processing
def Load_and_preprocess_nifti_resample_clip_normalize(img_path,resampling_sapcing,lbl_flag,flip_img,hu_cliping_range):

    """
    inputs: a. image_path         (strin)               : image path to read.
            b. resampling_sapcing (float;list or tuple) : Spacing to which to rsample;type:float,[height,width,depth]
            c. lbl_flag           (True/False)          : Define resapling a table image or not. Default=False.
            d. flip_img           (True/False)          : flip the image to be in RAI orientation.
            e. hu_cliping_range   (float;list of two)   : Upper and Lower boung to clip te HU units;e.g.,[-1000.,500.]

    output (np.array)  : Pre-processed image

    """

    img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_sitk = resample_img2mm(img_sitk,out_spacing=resampling_sapcing,is_label=lbl_flag)
    image    = sitk.GetArrayFromImage(img_sitk)
    if flip_img == True:
        image    = image[:, ::-1, :]
    image    = np.clip(image, hu_cliping_range[0], hu_cliping_range[1]).astype(np.float32)
    image    = normalise_one_one(image)
    return image

#function-(6)
#-|This function Load an numpy image and give the patch to fit in DL model.
def COVID_PATCH_EXTRACTION(IMAGE_CT,PATCH_SIZE,PADDING_CONSTENT_VALUE):
    """
    inputs: a. IMAGE_CT               (np.ndarray)     : image to be resized
            b. PATCH_SIZE             (list or tuple)  : new image size
            c. PADDING_CONSTENT_VALUE (int)            : padding pixel value

    output (np.ndarray): Cropped or padded image.
    """
    ## Patch Parameters
    PatchZ=PATCH_SIZE[0] # z-Depth
    PatchY=PATCH_SIZE[1] # y-Height
    PatchX=PATCH_SIZE[2] # x-Wight
    ex_image=resize_image_with_crop_or_pad(IMAGE_CT, [PatchZ,PatchY,PatchX], mode='constant',constant_values=PADDING_CONSTENT_VALUE)
    return ex_image

##function-group(7)--decode tfrecords
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


@tf.function
def decode_COVID_PATCH(Serialized_example):

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
    img=tf.cast(img, tf.float32)
    lbl=examples['covid_lbl']
    sub_id=examples['Sub_id']
    return img,lbl,sub_id


def give_numpy_img(tf_path):
    raw_image_dataset = tf.data.TFRecordDataset(tf_path)
    dataset=raw_image_dataset.map(decode_COVID_PATCH)
    for data in dataset:
        ct_img=data[0].numpy()
        ct_lbl=data[1].numpy()
        #ct_id=data[2].numpy()
        ct_id= tf_path.split('/')[-1].split('.tfrecords')[0].split('DukeSim_')[-1]
    return ct_img,ct_lbl,ct_id
