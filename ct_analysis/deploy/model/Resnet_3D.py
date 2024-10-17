from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import math




############################-------Data paths-----#####################################################
def calculate_class_weight_based_of_dataset(DATA_SET_TO_USE):

    training_data_loader_csv_covidctmd=pd.read_csv('model/training_losswight_calc_csv/COVID-CT-MD_Train_Sep28-2021.csv')
    total_number_of_cases_covidctmd=len(training_data_loader_csv_covidctmd)
    total_postive_cases_covidctmd=training_data_loader_csv_covidctmd['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_covidctmd=training_data_loader_csv_covidctmd['ct_label'].value_counts(normalize=False)[0]
    ##-----midric_ricord
    training_data_loader_csv_midric_ricord=pd.read_csv('model/training_losswight_calc_csv/MIDRC_RICORD_1A1B_Train_October06-2021.csv')
    total_number_of_cases_midric_ricord=len(training_data_loader_csv_midric_ricord)
    total_postive_cases_midric_ricord=training_data_loader_csv_midric_ricord['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_midric_ricord=training_data_loader_csv_midric_ricord['ct_label'].value_counts(normalize=False)[0]
    ##--mosmed-read
    training_data_loader_csv_mosmed=pd.read_csv('model/training_losswight_calc_csv/MosMedData_Train_Sep28-2021.csv')
    total_number_of_cases_mosmed=len(training_data_loader_csv_mosmed)
    total_postive_cases_mosmed=training_data_loader_csv_mosmed['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_mosmed=training_data_loader_csv_mosmed['ct_label'].value_counts(normalize=False)[0]
    ##-----bimcv
    training_data_loader_csv_bimcv=pd.read_csv('model/training_losswight_calc_csv/BIMCV-PosiNeg_Train_October07-2021.csv')
    total_number_of_cases_bimcv=len(training_data_loader_csv_bimcv)
    total_postive_cases_bimcv=training_data_loader_csv_bimcv['ct_label'].value_counts(normalize=False)[1]
    total_negative_cases_bimcv=training_data_loader_csv_bimcv['ct_label'].value_counts(normalize=False)[0]

    if DATA_SET_TO_USE=='covidctmd':
        training_data_loader_csv=pd.read_csv('model/training_losswight_calc_csv/COVID-CT-MD_Train_Sep28-2021.csv')

    if DATA_SET_TO_USE=='midric_ricord':
        training_data_loader_csv=pd.read_csv('model/training_losswight_calc_csv/MIDRC_RICORD_1A1B_Train_October06-2021.csv')

    if DATA_SET_TO_USE=='mosmed':
        training_data_loader_csv=pd.read_csv('model/training_losswight_calc_csv/MosMedData_Train_Sep28-2021.csv')

    if DATA_SET_TO_USE=='bimcv':
        training_data_loader_csv=pd.read_csv('model/training_losswight_calc_csv/BIMCV-PosiNeg_Train_October07-2021.csv')

    ##--------ClasswEIGHTSCALCULATION-------###

    if DATA_SET_TO_USE=='Combined_Data123':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='Combined_Data':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + total_number_of_cases_bimcv
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + total_postive_cases_bimcv
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + total_negative_cases_bimcv
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='Combined_Data_NIHa_eff_covidct_wsct':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + total_number_of_cases_bimcv + 391  + 241 + 604 + 3508
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + total_postive_cases_bimcv   + 391  +  0  + 604 +  0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + total_negative_cases_bimcv  + 0    + 241 +  0  + 3508
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_NIHa_eff_covidct_wsct':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + 391  + 241 + 604 + 3508
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + 391  +  0  + 604 +  0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + 0    + 241 +  0  + 3508
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_wsct_nysub_lhcancer':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + total_number_of_cases_bimcv + 391  + 241 + 604 + 3508 + 739 + 479
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + total_postive_cases_bimcv   + 391  +  0  + 604 +  0   + 739 + 0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + total_negative_cases_bimcv  + 0    + 241 +  0  + 3508 + 0   + 479
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_NIHa_eff_covidct_wsct_nysub_lhcancer':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + 391  + 241 + 604 + 3508 + 739 + 479
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + 391  +  0  + 604 +  0   + 739 + 0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + 0    + 241 +  0  + 3508 + 0   + 479
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + total_number_of_cases_bimcv + 391  + 241 + 604 + 739 + 479
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + total_postive_cases_bimcv   + 391  +  0  + 604 + 739 + 0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + total_negative_cases_bimcv  + 0    + 241 +  0  + 0   + 479
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_bimcv_NIHa_eff_covidct_nysub_lhcancer_lidc':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + total_number_of_cases_bimcv + 391  + 241 + 604 + 739 + 479 + 611
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + total_postive_cases_bimcv   + 391  +  0  + 604 + 739 + 0   +  0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + total_negative_cases_bimcv  + 0    + 241 +  0  + 0   + 479 + 611
        WEIGHT_FOR_0 =math.ceil((1 /total_negative_cases)*(total_number_of_cases)/2.0)
        WEIGHT_FOR_1 =math.ceil((1 /total_postive_cases) *(total_number_of_cases)/2.0)
        CLASS_WEIGHT = {0:WEIGHT_FOR_0, 1:WEIGHT_FOR_1}
    elif DATA_SET_TO_USE=='mid_ctmd_mosmed_NIHa_eff_covidct_nysub_lhcancer':
        total_number_of_cases =  total_number_of_cases_covidctmd + total_number_of_cases_midric_ricord + total_number_of_cases_mosmed + 391  + 241 + 604 + 739 + 479
        total_postive_cases   =  total_postive_cases_covidctmd   + total_postive_cases_midric_ricord   + total_postive_cases_mosmed   + 391  +  0  + 604 + 739 + 0
        total_negative_cases  =  total_negative_cases_covidctmd  + total_negative_cases_midric_ricord  + total_negative_cases_mosmed  + 0    + 241 +  0  + 0   + 479
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


    return WEIGHT_FOR_0,WEIGHT_FOR_1








###Residual Block
def Residual_Block(inputs,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 use_bias=False,
                 activation=tf.nn.relu6,
                 kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
                 bias_regularizer=None,
                 **kwargs):


    conv_params={'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}

    in_filters = inputs.get_shape().as_list()[-1]
    x=inputs
    orig_x=x

    ##building
    # Adjust the strided conv kernel size to prevent losing information
    k = [s * 2 if s > 1 else k for k, s in zip(kernel_size, strides)]

    if np.prod(strides) != 1:
            orig_x = tf.keras.layers.MaxPool3D(pool_size=strides,strides=strides,padding='valid')(orig_x)

    ##sub-unit-0
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=k,strides=strides,**conv_params)(x)

    ##sub-unit-1
    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(x)

        # Handle differences in input and output filter sizes
    if in_filters < out_filters:
        orig_x = tf.pad(tensor=orig_x,paddings=[[0, 0]] * (len(x.get_shape().as_list()) - 1) + [[
                    int(np.floor((out_filters - in_filters) / 2.)),
                    int(np.ceil((out_filters - in_filters) / 2.))]])

    elif in_filters > out_filters:
        orig_x = tf.keras.layers.Conv3D(filters=out_filters,kernel_size=kernel_size,strides=(1,1,1),**conv_params)(orig_x)

    x += orig_x
    return x



def Resnet3D(inputs,
              num_classes,
              num_res_units=1,
              filters=(8,16,32),
              strides=((1, 1, 1),(2, 2, 2),(2, 2, 2)),
              use_bias=False,
              activation=tf.nn.relu6,
              kernel_initializer=tf.keras.initializers.VarianceScaling(distribution='uniform'),
              bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=tf.keras.regularizers.l2(l=0.001),
              bias_regularizer=None,
              **kwargs):
    conv_params = {'padding': 'same',
                   'use_bias': use_bias,
                   'kernel_initializer': kernel_initializer,
                   'bias_initializer': bias_initializer,
                   'kernel_regularizer': kernel_regularizer,
                   'bias_regularizer': bias_regularizer}


    ##building
    k = [s * 2 if s > 1 else 3 for s in strides[0]]


    #Input
    x = inputs
    #1st-convo
    x=tf.keras.layers.Conv3D(filters[0], k, strides[0], **conv_params)(x)

    for res_scale in range(1, len(filters)):
        x = Residual_Block(
                inputs=x,
                out_filters=filters[res_scale],
                strides=strides[res_scale],
                activation=activation,
                name='unit_{}_0'.format(res_scale))
       
        for i in range(1, num_res_units):
            x = Residual_Block(
                    inputs=x,
                    out_filters=filters[res_scale],
                    strides=(1, 1, 1),
                    activation=activation,
                    name='unit_{}_{}'.format(res_scale, i)) 



    x=tf.keras.layers.BatchNormalization()(x)
    x=activation(x)
    x=tf.keras.layers.GlobalAveragePooling3D()(x)
    x =tf.keras.layers.Dropout(0.5)(x) #training=True)
    classifier=tf.keras.layers.Dense(units=num_classes,activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=classifier)
    return model



def get_Resnet3D_to_pridict(batched_img_ct, INPUT_PATCH_SIZE,NUMBER_OF_CLASSES,TRAIN_NUM_RES_UNIT,TRAIN_NUM_FILTERS,TRAIN_STRIDES,MODEL_PATH,DATASET_TO_RUN_ON):
    WEIGHT_FOR_0,WEIGHT_FOR_1 = calculate_class_weight_based_of_dataset(DATASET_TO_RUN_ON)
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

    inputs         = tf.keras.Input(shape=INPUT_PATCH_SIZE, name='CT')
    Model_3D       = Resnet3D(inputs,
                              num_classes   = NUMBER_OF_CLASSES,
                              num_res_units = TRAIN_NUM_RES_UNIT,
                              filters       = TRAIN_NUM_FILTERS,
                              strides       = TRAIN_STRIDES)

    Model_3D.load_weights(MODEL_PATH)
    Model_3D.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4,momentum=0.9,nesterov=True),
                     loss=[WBCE],
                     metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    #Model_3D.summary()
    predict_model=Model_3D.predict(batched_img_ct)
    tf.keras.backend.clear_session()
    return predict_model
