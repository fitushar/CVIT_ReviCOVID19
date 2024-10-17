import tensorflow as tf
import tensorflow_addons as tfa
from loss_funnction_And_matrics import*
import math
import numpy as np
import pandas as pd
import glob
import os
###---Number-of-GPU
NUM_OF_GPU=4
DISTRIIBUTED_STRATEGY_GPUS=["gpu:0","gpu:1","gpu:2"]
############################-------Model To use-----#####################################################
DATA_SET_TO_USE='bimcv'#['midric_ricord','mosmed','bimcv','covidctmd','Combined_Data123','Combined_Data','mid_ctmd_mosmed_NIHa_eff_covidct_wsct_nysub_lhcancer']
MODEL_TO_USE='Multi_REsolution_FeatureAggre'                                 #['Resnet3D','DenseNet3D','Inception3D','VGG3D','TinyResnet_3D',DeConVoNet3D,DySegClf_withDenseVnet]
OPTIMIZER_TO_USE='SGD'                                                      #'Adam','SGD'
LEARNING_RATE_ST='ExponentialDecay'                                         #['decay','cyclic','CosineDecayRestarts',constant]
RESUME_TRAINING=0
TRAINING_INITIAL_EPOCH=300
RAW_MODEL_DIRECTORY='/data2/usr/ft42/CVIT/tushar_Nov26th_Covid_Run/ExpoLargerModel/WBCE_SGD/'
DATA_CACHE_TR="/Local/nobackup/COVID_CHASE/"+DATA_SET_TO_USE+"_train"
DATA_CACHE_VAL="/Local/nobackup/COVID_CHASE/"+DATA_SET_TO_USE+"_val"
INIT_LR=1e-6
MAX_LR=1e-4
DECAY_STR_LR=1e-2
BATCH_SIZE=24
TRAING_EPOCH=300
NUMBER_OF_PARALLEL_CALL=3*BATCH_SIZE
PARSHING=3*BATCH_SIZE
CYCLIC_MULTIPLIER=2

############################-------Data paths-----#####################################################
if DATA_SET_TO_USE=='covidctmd':
    TRAINING_TF_RECORDS   ='/Local/nobackup/COVID19_CT_DATA/covidctmd_tfrecords_96x160x160/Train_tfrecords/'
    VALIDATION_TF_RECORDS ='/Local/nobackup/COVID19_CT_DATA/covidctmd_tfrecords_96x160x160/Validation_tfrecords/'
    training_data_loader_csv=pd.read_csv('training_losswight_calc_csv/COVID-CT-MD_Train_Sep28-2021.csv')

if DATA_SET_TO_USE=='midric_ricord':
    TRAINING_TF_RECORDS   ='/Local/nobackup/COVID19_CT_DATA/midric_ricord_tfrecords_96x160x160/Train_tfrecords/'
    VALIDATION_TF_RECORDS ='/Local/nobackup/COVID19_CT_DATA/midric_ricord_tfrecords_96x160x160/Validation_tfrecords/'
    training_data_loader_csv=pd.read_csv('training_losswight_calc_csv/MIDRC_RICORD_1A1B_Train_October06-2021.csv')

if DATA_SET_TO_USE=='mosmed':
    TRAINING_TF_RECORDS   ='/Local/nobackup/COVID19_CT_DATA/mosmed_tfrecords_96x160x160/Train_tfrecords/'
    VALIDATION_TF_RECORDS ='/Local/nobackup/COVID19_CT_DATA/mosmed_tfrecords_96x160x160/Validation_tfrecords/'
    training_data_loader_csv=pd.read_csv('training_losswight_calc_csv/MosMedData_Train_Sep28-2021.csv')

if DATA_SET_TO_USE=='bimcv':
    TRAINING_TF_RECORDS   ='/Local/nobackup/COVID19_CT_DATA/bimcv_tfrecords_96x160x160/Train_tfrecords/'
    VALIDATION_TF_RECORDS ='/Local/nobackup/COVID19_CT_DATA/bimcv_tfrecords_96x160x160/Validation_tfrecords/'
    training_data_loader_csv=pd.read_csv('training_losswight_calc_csv/BIMCV-PosiNeg_Train_October07-2021.csv')


if DATA_SET_TO_USE=='mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc':
    #---Validation_tfpaths:
    TRAINING_TF_RECORDS_1    ='/Local/nobackup/COVID19_CT_DATA/covidctmd_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_2    ='/Local/nobackup/COVID19_CT_DATA/midric_ricord_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_3    ='/Local/nobackup/COVID19_CT_DATA/mosmed_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_4   ='/Local/nobackup/COVID19_CT_DATA/bimcv_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_5    ='/Local/nobackup/COVID19_CT_DATA/ct_NIHa_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_6    ='/Local/nobackup/COVID19_CT_DATA/effusion_nclcl_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_7    ='/Local/nobackup/COVID19_CT_DATA/covidctdata_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_9    ='/Local/nobackup/COVID19_CT_DATA/ny_sub_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_10   ='/Local/nobackup/COVID19_CT_DATA/lgcancer_tfrecords_96x160x160/Train_tfrecords/'
    TRAINING_TF_RECORDS_11   ='/Local/nobackup/COVID19_CT_DATA/lidi_idri_tfrecords_96x160x160/Train_tfrecords/'

    #---Validation_tfpaths:
    VALIDATION_TF_RECORDS_1  ='/Local/nobackup/COVID19_CT_DATA/covidctmd_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_2  ='/Local/nobackup/COVID19_CT_DATA/midric_ricord_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_3  ='/Local/nobackup/COVID19_CT_DATA/mosmed_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_4 ='/Local/nobackup/COVID19_CT_DATA/bimcv_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_5  ='/Local/nobackup/COVID19_CT_DATA/ct_NIHa_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_6  ='/Local/nobackup/COVID19_CT_DATA/effusion_nclcl_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_7  ='/Local/nobackup/COVID19_CT_DATA/covidctdata_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_9  ='/Local/nobackup/COVID19_CT_DATA/ny_sub_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_10 ='/Local/nobackup/COVID19_CT_DATA/lgcancer_tfrecords_96x160x160/Validation_tfrecords/'
    VALIDATION_TF_RECORDS_11 ='/Local/nobackup/COVID19_CT_DATA/lidi_idri_tfrecords_96x160x160/Validation_tfrecords/'


    TRAINING_TF_RECORDS  =[TRAINING_TF_RECORDS_1,   TRAINING_TF_RECORDS_2,   TRAINING_TF_RECORDS_3,   TRAINING_TF_RECORDS_4,    TRAINING_TF_RECORDS_5,   TRAINING_TF_RECORDS_6,   TRAINING_TF_RECORDS_7,   TRAINING_TF_RECORDS_9 ,  TRAINING_TF_RECORDS_10,   TRAINING_TF_RECORDS_11]
    VALIDATION_TF_RECORDS=[VALIDATION_TF_RECORDS_1, VALIDATION_TF_RECORDS_2, VALIDATION_TF_RECORDS_3, VALIDATION_TF_RECORDS_4,  VALIDATION_TF_RECORDS_5, VALIDATION_TF_RECORDS_6, VALIDATION_TF_RECORDS_7, VALIDATION_TF_RECORDS_9 ,VALIDATION_TF_RECORDS_10, VALIDATION_TF_RECORDS_11]

    ##-----covid-cr_mdread
    training_data_loader_csv_covidctmd=pd.read_csv('training_losswight_calc_csv/COVID-CT-MD_Train_Sep28-2021.csv')
    total_number_of_cases_covidctmd=len(training_data_loader_csv_covidctmd)
    total_postive_cases_covidctmd=training_data_loader_csv_covidctmd['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_covidctmd=training_data_loader_csv_covidctmd['ct_label'].value_counts(normalize=False)[0]
    ##-----midric_ricord
    training_data_loader_csv_midric_ricord=pd.read_csv('training_losswight_calc_csv/MIDRC_RICORD_1A1B_Train_October06-2021.csv')
    total_number_of_cases_midric_ricord=len(training_data_loader_csv_midric_ricord)
    total_postive_cases_midric_ricord=training_data_loader_csv_midric_ricord['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_midric_ricord=training_data_loader_csv_midric_ricord['ct_label'].value_counts(normalize=False)[0]
    ##--mosmed-read
    training_data_loader_csv_mosmed=pd.read_csv('training_losswight_calc_csv/MosMedData_Train_Sep28-2021.csv')
    total_number_of_cases_mosmed=len(training_data_loader_csv_mosmed)
    total_postive_cases_mosmed=training_data_loader_csv_mosmed['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_mosmed=training_data_loader_csv_mosmed['ct_label'].value_counts(normalize=False)[0]

    ##-----bimcv
    training_data_loader_csv_bimcv=pd.read_csv('training_losswight_calc_csv/BIMCV-PosiNeg_Train_October07-2021.csv')
    total_number_of_cases_bimcv=len(training_data_loader_csv_bimcv)
    total_postive_cases_bimcv=training_data_loader_csv_bimcv['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_bimcv=training_data_loader_csv_bimcv['ct_label'].value_counts(normalize=False)[0]


##--------ClasswEIGHTSCALCULATION-------###


if DATA_SET_TO_USE=='mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc':
    total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + total_number_of_cases_bimcv + 391  + 241 + 604 + 739 + 479 + 611
    total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + total_postive_cases_bimcv   + 391  +  0  + 604 + 739 + 0   +  0
    total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + total_negative_cases_bimcv  + 0    + 241 +  0  + 0   + 479 + 611
    WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
    WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
    CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
else:
    total_number_of_cases=len(training_data_loader_csv)
    total_postive_cases=training_data_loader_csv['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases=training_data_loader_csv['ct_label'].value_counts(normalize=False)[0]
    WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
    WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
    CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}



#-----Loss function---
@tf.function(experimental_relax_shapes=True)
def WBCE( y_true, y_pred, weight1=WEIGHT_FOR_1, weight0=WEIGHT_FOR_0 ):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred , tf.float32)
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=-1)

def custom_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred , tf.float32)
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def MeanSqrError_WBCE(y_true, y_pred,weight1=WEIGHT_FOR_1, weight0=WEIGHT_FOR_0):
    LOSS1=WBCE(y_true, y_pred, weight1=WEIGHT_FOR_1, weight0=WEIGHT_FOR_0)
    LOSS2=custom_mean_squared_error(y_true, y_pred)
    final_loss=LOSS1+LOSS2
    return final_loss

#-------Computing the number of training and validation tfrecords-
if DATA_SET_TO_USE== 'mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc' :
    NUM_OF_TR_TFRED   = len([file for file in os.listdir(TRAINING_TF_RECORDS_1) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_2) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_3) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_4) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_5) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_6) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_7) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_9) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_10) if file.endswith('.tfrecords')])+\
                        len([file for file in os.listdir(TRAINING_TF_RECORDS_11) if file.endswith('.tfrecords')])

    NUM_OF_VAL_TFRED =   len([file for file in os.listdir(VALIDATION_TF_RECORDS_1) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_2) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_3) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_4) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_5) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_6) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_7) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_9) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_10) if file.endswith('.tfrecords')])+\
                         len([file for file in os.listdir(VALIDATION_TF_RECORDS_11) if file.endswith('.tfrecords')])
    TRAINING_STEP_PER_EPOCH=math.ceil((NUM_OF_TR_TFRED)/BATCH_SIZE)
    VALIDATION_STEP=math.ceil((NUM_OF_VAL_TFRED)/BATCH_SIZE)
else:
    NUM_OF_TR_TFRED=len([file for file in os.listdir(TRAINING_TF_RECORDS) if file.endswith('.tfrecords')])
    NUM_OF_VAL_TFRED=len([file for file in os.listdir(VALIDATION_TF_RECORDS) if file.endswith('.tfrecords')])
    TRAINING_STEP_PER_EPOCH=math.ceil((NUM_OF_TR_TFRED)/BATCH_SIZE)
    VALIDATION_STEP=math.ceil((NUM_OF_VAL_TFRED)/BATCH_SIZE)


print('Training-Model-{},For epoch-{}'.format(MODEL_TO_USE,TRAING_EPOCH))
print('Number of training tfrecords={},Batch size={}, Training step/epoch={}'.format(NUM_OF_TR_TFRED,BATCH_SIZE,TRAINING_STEP_PER_EPOCH))
print('Number of val tfrecords={},Batch size={},      Val step/epoch={}'.format(NUM_OF_VAL_TFRED,BATCH_SIZE,VALIDATION_STEP))


##-----Network Configuration----#####
NUMBER_OF_CLASSES=1
INPUT_PATCH_SIZE=(96,160,160,1) #[z,y,x]
TRAIN_NUM_RES_UNIT=1
TRAIN_NUM_FILTERS=(8,16,32)
TRAIN_STRIDES=((1, 1, 1),(2, 2, 2),(2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')

#--DySegClf_withDenseVnet
SEGMENTATION_MODEL_PATH='eff_nclcl_segct_DenseVnet3D_0.02_874.h5'
SEG_NUMBER_OF_CLASSES=1
SEG_INPUT_PATCH_SIZE=(96,160,160, 1)
NUM_DENSEBLOCK_EACH_RESOLUTION=(4, 8, 16)
NUM_OF_FILTER_EACH_RESOLUTION=(12,24,24)
DILATION_RATE=(5, 10, 10)
DROPOUT_RATE=0.25

#DeConVoNet3D
DeConVoNet3D_NUMBER_OF_CLASSES=1
DeConVoNet3D_INPUT_PATCH_SIZE=(96,160,160, 1) #[z,y,x]
DeConVoNet3D_TRAIN_NUM_RES_UNIT=1
DeConVoNet3D_TRAIN_NUM_FILTERS=(16,64,128)
DeConVoNet3D_TRAIN_STRIDES=((1, 1, 1),(2, 2, 2),(2, 2, 2),(2,2,2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
# DenseNet
DENSE_NET_BLOCKS = 3
DENSE_NET_BLOCK_LAYERS = 5
DENSE_NET_INITIAL_CONV_DIM = 16
DENSE_NET_GROWTH_RATE = DENSE_NET_INITIAL_CONV_DIM // 2
DENSE_NET_ENABLE_BOTTLENETCK = False # called DenseNet-BC if ENABLE_BOTTLENETCK and COMPRESSION < 1 in paper
DENSE_NET_TRANSITION_COMPRESSION = 1.0
DENSE_NET_ENABLE_DROPOUT = True
DENSE_NET_DROPOUT = 0.5
#-------Inception3D----#####
INCEPTION_BLOCKS = 3
INCEPTION_REDUCTION_STEPS = 2
INCEPTION_KEEP_FILTERS = 32
INCEPTION_ENABLE_DEPTHWISE_SEPARABLE_CONV_SHRINKAGE = 0.333
INCEPTION_ENABLE_SPATIAL_SEPARABLE_CONV = True
INCEPTION_DROPOUT = 0.5
#---------VGG3D----####
TRAIN_CLASSIFY_USE_BN = False


############################-------Callbacks-----#####################################################
if MODEL_TO_USE=='Resnet3D':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='DySegClf_withDenseVnet':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='DyFSegClf_withDenseVnet3D':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='DySegF2CHClf_withDenseVnet3D':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='DySegDyFAClf_withDenseVnet3D':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='DySegSyFAClf_withDenseVnet3D':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='Multi_REsolution_FeatureAggre':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='Multi_REsolution_SegClassifier':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='Multi_REsolution_OnlyFeature':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}_{}_D{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}_{}_D{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT,DECAY_STR_LR,CYCLIC_MULTIPLIER)
elif MODEL_TO_USE=='TinyResnet3D':
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}_d{}_u{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}_d{}_u{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}_d{}_u{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST,TRAIN_NUM_FILTERS[-1],TRAIN_NUM_RES_UNIT)
else:
    MODEL_SAVE_PATH_NAME='Model_{}_{}_{}_{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST)
    LOG_NAME=RAW_MODEL_DIRECTORY+'Log_{}_{}_{}_{}'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST)
    MODEL_SAVING_NAME=DATA_SET_TO_USE+"_"+MODEL_TO_USE+"_{val_loss:.2f}_{epoch}.h5"
    TRAINING_CSV=RAW_MODEL_DIRECTORY+'Training_{}_{}_{}_{}.csv'.format(MODEL_TO_USE,DATA_SET_TO_USE,OPTIMIZER_TO_USE,LEARNING_RATE_ST)



ModelCheckpoint_MOTITOR='loss'
TRAINING_SAVE_MODEL_PATH=RAW_MODEL_DIRECTORY+MODEL_SAVE_PATH_NAME+'/'
if RESUME_TRAINING==1:
    WEIGHT_TO_RESUME_NAME=glob.glob(TRAINING_SAVE_MODEL_PATH+DATA_SET_TO_USE+"_"+MODEL_TO_USE+"**_{}.h5".format(TRAINING_INITIAL_EPOCH))[0]
    RESUME_TRAIING_MODEL=WEIGHT_TO_RESUME_NAME


#------Learning Rate Slection-
if LEARNING_RATE_ST=='cyclic':
    LR_USE = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,maximal_learning_rate=MAX_LR,scale_fn=lambda x: 1/(2.**(x-1)),step_size=CYCLIC_MULTIPLIER*TRAINING_STEP_PER_EPOCH)
if LEARNING_RATE_ST=='decay':
    LR_USE = DECAY_STR_LR
if LEARNING_RATE_ST=='CosineDecayRestarts':
    LR_USE=tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=DECAY_STR_LR, t_mul=2.00, m_mul=1.00, alpha=1e-3,first_decay_steps=TRAINING_STEP_PER_EPOCH*CYCLIC_MULTIPLIER)
if LEARNING_RATE_ST=='constant':
    LR_USE = DECAY_STR_LR
if LEARNING_RATE_ST=='ExponentialDecay':
    LR_USE = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=DECAY_STR_LR,decay_steps=TRAINING_STEP_PER_EPOCH*CYCLIC_MULTIPLIER,decay_rate=0.90,staircase=True)



#-----# OPTIMIZER selection-
if OPTIMIZER_TO_USE=='Adam':
    OPTIMIZER=tf.keras.optimizers.Adam(learning_rate=LR_USE,beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
if OPTIMIZER_TO_USE=='AdamW':
    OPTIMIZER=tfa.optimizers.AdamW(learning_rate=LR_USE,epsilon=1e-5,amsgrad=True,weight_decay=1e-2)
if OPTIMIZER_TO_USE=='SGD':
    OPTIMIZER=tf.keras.optimizers.SGD(learning_rate=LR_USE,momentum=0.9,nesterov=True)
#-----Loss function and matric
TRAIN_CLASSIFY_LOSS=WBCE
#tf.keras.losses.MeanSquaredError()
#[WBCE
#tf.keras.losses.MeanSquaredError()
#tf.keras.losses.BinaryCrossentropy()
#tf.keras.losses.MeanSquaredError()
#tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)]
TRAIN_CLASSIFY_METRICS=tf.keras.metrics.AUC()


print('NUM_OF_TR_TFRED-{}'.format(NUM_OF_TR_TFRED))
print('NUM_OF_VAL_TFRED-{}'.format(NUM_OF_VAL_TFRED))
print('TRAINING_STEP_PER_EPOCH-{}'.format(TRAINING_STEP_PER_EPOCH))
print('VALIDATION_STEP-{}'.format(VALIDATION_STEP))
print('WEIGHT_FOR_0-{}'.format(WEIGHT_FOR_0))
print('WEIGHT_FOR_1-{}'.format(WEIGHT_FOR_0))
print('total_number_of_cases-{}'.format(total_number_of_cases))
print('total_postive_cases-{}'.format(total_postive_cases))
print('total_negative_cases-{}'.format(total_negative_cases))
