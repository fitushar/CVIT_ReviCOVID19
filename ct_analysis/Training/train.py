from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_addons as tfa
import random
random.seed(200)

tf.config.optimizer.set_jit(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0:3], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from config import*
from loss_funnction_And_matrics import*
from Attention_utility import*
import os
import datetime
from Resnet_3D import Resnet3D
from tfrecords_utilities import*
import numpy as np
import random
import pathlib


from sklearn.metrics import roc_auc_score
def auc_score_ml(y_true, y_pred):
    if len(np.unique(y_true[:,0])) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred)
def auc_ml(y_true, y_pred):
    return tf.compat.v1.py_func(auc_score_ml, (y_true, y_pred), tf.double)




def getting_list(path):
    a=[file for file in os.listdir(path) if file.endswith('.tfrecords')]
    all_tfrecoeds=random.sample(a, len(a))
    #all_tfrecoeds.sort(key=lambda f: int(filter(str.isdigit, f)))
    list_of_tfrecords=[]
    for i in range(len(all_tfrecoeds)):
        tf_path=path+all_tfrecoeds[i]
        list_of_tfrecords.append(tf_path)
    return list_of_tfrecords

def getting_list_combined_data(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    all_a_b_c_d=a+b+c+d
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords

def getting_list_combined_data_ALL(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")
    h=glob.glob(list_of_path[7]+"*.tfrecords")

    all_a_b_c_d=a+b+c+d+e+f+g+h
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords

def getting_list_mid_ctmd_mosmed_NIHa_eff_covidct_wsct(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")


    all_a_b_c_d=a+b+c+d+e+f+g
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords

def mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_wsct_nysub_lhcancer(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")
    h=glob.glob(list_of_path[7]+"*.tfrecords")
    i=glob.glob(list_of_path[8]+"*.tfrecords")
    j=glob.glob(list_of_path[9]+"*.tfrecords")
    all_a_b_c_d=a+b+c+d+e+f+g+h+i+j
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords

def mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")
    h=glob.glob(list_of_path[7]+"*.tfrecords")
    i=glob.glob(list_of_path[8]+"*.tfrecords")
    j=glob.glob(list_of_path[9]+"*.tfrecords")
    all_a_b_c_d=a+b+c+d+e+f+g+h+i+j
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords

def mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")
    h=glob.glob(list_of_path[7]+"*.tfrecords")
    i=glob.glob(list_of_path[8]+"*.tfrecords")
    all_a_b_c_d=a+b+c+d+e+f+g+h+i
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords

def mid_ctmd_mosmed_NIHa_eff_covidct_nysub_lhcancer(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")
    h=glob.glob(list_of_path[7]+"*.tfrecords")

    all_a_b_c_d=a+b+c+d+e+f+g+h
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords



def mid_ctmd_mosmed_NIHa_eff_covidct_wsct_nysub_lhcancer(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    d=glob.glob(list_of_path[3]+"*.tfrecords")
    e=glob.glob(list_of_path[4]+"*.tfrecords")
    f=glob.glob(list_of_path[5]+"*.tfrecords")
    g=glob.glob(list_of_path[6]+"*.tfrecords")
    h=glob.glob(list_of_path[7]+"*.tfrecords")
    i=glob.glob(list_of_path[8]+"*.tfrecords")
    all_a_b_c_d=a+b+c+d+e+f+g+h+i
    list_of_tfrecords=random.sample(all_a_b_c_d, len(all_a_b_c_d))
    print(len(list_of_tfrecords))
    return list_of_tfrecords


def getting_list_combined_data123(list_of_path):
    a=glob.glob(list_of_path[0]+"*.tfrecords")
    b=glob.glob(list_of_path[1]+"*.tfrecords")
    c=glob.glob(list_of_path[2]+"*.tfrecords")
    all_a_b_c=a+b+c
    list_of_tfrecords=random.sample(all_a_b_c, len(all_a_b_c))
    print(len(list_of_tfrecords))
    return list_of_tfrecords



#--Traing Decoder
def load_training_tfrecords(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(lambda x: tf.data.TFRecordDataset(x),cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    dataset=dataset.map(decode_ct_withAugmentation,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).take(len(record_mask_file)).cache(DATA_CACHE_TR).repeat(-1).batch(batch_size)
    #dataset=dataset.map(decode_ct_withAugmentation,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(-1).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset

#--Validation Decoder
def load_validation_tfrecords(record_mask_file,batch_size):
    dataset=tf.data.Dataset.list_files(record_mask_file).interleave(tf.data.TFRecordDataset,cycle_length=NUMBER_OF_PARALLEL_CALL,num_parallel_calls=NUMBER_OF_PARALLEL_CALL)
    dataset=dataset.map(decode_ct_withoutAug,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).take(len(record_mask_file)).cache(DATA_CACHE_VAL).repeat(-1).batch(batch_size)
    #dataset=dataset.map(decode_ct_withoutAug,num_parallel_calls=NUMBER_OF_PARALLEL_CALL).repeat(-1).batch(batch_size)
    batched_dataset=dataset.prefetch(PARSHING)
    return batched_dataset






def Training():

    #TensorBoard
    logdir = os.path.join(LOG_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
    ##csv_logger
    csv_logger = tf.keras.callbacks.CSVLogger(TRAINING_CSV)
    ##Model-checkpoings
    path=TRAINING_SAVE_MODEL_PATH
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    model_path=os.path.join(path,MODEL_SAVING_NAME)
    Model_callback= tf.keras.callbacks.ModelCheckpoint(filepath=model_path,save_best_only=False,save_weights_only=True,monitor=ModelCheckpoint_MOTITOR,verbose=1)
    LIST_OF_CALLBACKS=[tensorboard_callback,csv_logger,Model_callback]
    if LEARNING_RATE_ST=='decay':
        def lr_schedule(epoch):
            learning_rate = DECAY_STR_LR
            if epoch > 20:
                learning_rate = 1e-3
            if epoch > 100:
                learning_rate = 1e-5
            tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            return learning_rate
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        LIST_OF_CALLBACKS=[tensorboard_callback,csv_logger,Model_callback,lr_callback]




    ###-----Resampling BLOCK-------#############
    if DATA_SET_TO_USE=='Combined_Data':
        tf_train=getting_list_combined_data(TRAINING_TF_RECORDS)
        tf_val=getting_list_combined_data(VALIDATION_TF_RECORDS)
    elif  DATA_SET_TO_USE== 'mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_wsct_nysub_lhcancer':
        tf_train = mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_wsct_nysub_lhcancer(TRAINING_TF_RECORDS)
        tf_val   = mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_wsct_nysub_lhcancer(VALIDATION_TF_RECORDS)
    elif  DATA_SET_TO_USE== 'mid_ctmd_mosmed_NIHa_eff_covidct_nysub_lhcancer':
        tf_train = mid_ctmd_mosmed_NIHa_eff_covidct_nysub_lhcancer(TRAINING_TF_RECORDS)
        tf_val   = mid_ctmd_mosmed_NIHa_eff_covidct_nysub_lhcancer(VALIDATION_TF_RECORDS)
    elif  DATA_SET_TO_USE== 'mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer':
        tf_train = mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer(TRAINING_TF_RECORDS)
        tf_val   = mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer(VALIDATION_TF_RECORDS)
    elif  DATA_SET_TO_USE== 'mid_ctmd_mosmed_NIHa_eff_covidct_wsct_nysub_lhcancer':
        tf_train = mid_ctmd_mosmed_NIHa_eff_covidct_wsct_nysub_lhcancer(TRAINING_TF_RECORDS)
        tf_val   = mid_ctmd_mosmed_NIHa_eff_covidct_wsct_nysub_lhcancer(VALIDATION_TF_RECORDS)
    elif  DATA_SET_TO_USE== 'mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc':
        tf_train = mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc(TRAINING_TF_RECORDS)
        tf_val   = mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc(VALIDATION_TF_RECORDS)
    elif DATA_SET_TO_USE=='Combined_Data_NIHa_eff_covidct_wsct':
        tf_train = getting_list_combined_data_ALL(TRAINING_TF_RECORDS)
        tf_val   = getting_list_combined_data_ALL(VALIDATION_TF_RECORDS)
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_NIHa_eff_covidct_wsct':
        tf_train = getting_list_mid_ctmd_mosmed_NIHa_eff_covidct_wsct(TRAINING_TF_RECORDS)
        tf_val   = getting_list_mid_ctmd_mosmed_NIHa_eff_covidct_wsct(VALIDATION_TF_RECORDS)
    elif DATA_SET_TO_USE=='Combined_Data123':
        tf_train = getting_list_combined_data123(TRAINING_TF_RECORDS)
        tf_val   = getting_list_combined_data123(VALIDATION_TF_RECORDS)
    else:
        tf_train=getting_list(TRAINING_TF_RECORDS)
        tf_val=getting_list(VALIDATION_TF_RECORDS)

    traing_data=load_training_tfrecords(tf_train,BATCH_SIZE)
    Val_batched_dataset=load_validation_tfrecords(tf_val,BATCH_SIZE)

    if (NUM_OF_GPU==1):
        if RESUME_TRAINING==1:
            inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
            if MODEL_TO_USE=='Resnet3D':
                Model_3D=Resnet3D(inputs,num_classes=NUMBER_OF_CLASSES)


            Model_3D.load_weights(RESUME_TRAIING_MODEL)
            initial_epoch_of_training=TRAINING_INITIAL_EPOCH
            Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),auc_ml,tf.keras.metrics.BinaryAccuracy()])
            Model_3D.summary()
        else:
            initial_epoch_of_training=0
            inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
            if MODEL_TO_USE=='Resnet3D':
                Model_3D=Resnet3D(inputs,num_classes=NUMBER_OF_CLASSES)

            Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),auc_ml,tf.keras.metrics.BinaryAccuracy()])
            Model_3D.summary()
        Model_3D.fit(traing_data,
                   steps_per_epoch=TRAINING_STEP_PER_EPOCH,
                   epochs=TRAING_EPOCH,
                   initial_epoch=initial_epoch_of_training,
                   validation_data=Val_batched_dataset,
                   validation_steps=VALIDATION_STEP,
                   validation_freq=1,
                   callbacks=LIST_OF_CALLBACKS)

    ###Multigpu----
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy(["gpu:0","gpu:1","gpu:2"])
        with mirrored_strategy.scope():
                if RESUME_TRAINING==1:
                    inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
                    if MODEL_TO_USE=='Resnet3D':
                        Model_3D=Resnet3D(inputs,num_classes=NUMBER_OF_CLASSES)

                    Model_3D.load_weights(RESUME_TRAIING_MODEL)
                    initial_epoch_of_training=TRAINING_INITIAL_EPOCH
                    print('Resume-Training From-Epoch{}-Loading-Model-from_{}'.format(initial_epoch_of_training,RESUME_TRAIING_MODEL))
                    Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),auc_ml,tf.keras.metrics.BinaryAccuracy()])
                    Model_3D.summary()
                else:
                    initial_epoch_of_training=0
                    inputs = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
                    if MODEL_TO_USE=='Resnet3D':
                        Model_3D=Resnet3D(inputs,num_classes=NUMBER_OF_CLASSES)
                    Model_3D.compile(optimizer=OPTIMIZER, loss=[TRAIN_CLASSIFY_LOSS], metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),auc_ml,tf.keras.metrics.BinaryAccuracy()])
                    Model_3D.summary()
                Model_3D.fit(traing_data,
                           steps_per_epoch=TRAINING_STEP_PER_EPOCH,
                           epochs=TRAING_EPOCH,
                           initial_epoch=initial_epoch_of_training,
                           validation_data=Val_batched_dataset,
                           validation_steps=VALIDATION_STEP,validation_freq=1,
                           callbacks=LIST_OF_CALLBACKS)

if __name__ == '__main__':
   Training()
