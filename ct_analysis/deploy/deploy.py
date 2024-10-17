from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import random
random.seed(200)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ="3"

'''Source=[https://github.com/tensorflow/tensorflow/issues/27023]
In detail:-
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
import tensorflow as tf
tf.config.optimizer.set_jit(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
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


import glob
import time
import json
import argparse
import dicom2nifti
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from utlities.preprocesing import Load_and_preprocess_nifti_resample_clip_normalize
from utlities.preprocesing import COVID_PATCH_EXTRACTION
from utlities.preprocesing import decode_COVID_PATCH
from utlities.preprocesing import give_numpy_img
from model.Resnet_3D import get_Resnet3D_to_pridict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)







'''
COVID-19 Classification Using Puplic and DukeSim.
copyright:            CVIT,DUKE.
Script:               Inference Script
Contributor:          Fakrul Islam Tushar (fakrulislam.tushar@duke.edu)
Target Organ:         Lungs
Classification Type:  COVID+/COVID-
Target Classes:       COVID-(0), COVID+(1)

Input:                Hyperperameters, model weight
Output:               CSV file with File Name and Prediction.

Start Date:           10/15/2021
Version:              v0.1
'''



def get_datalist_from_csv_or_directory(args, config):
    #---if data is reading from the csv
    if config["INPUT_FROM"]=='csv':

        reading_df=pd.read_csv(config["ROI_CSV"])
        if config["INFERENCE_MODE"]=="TEST_WOLBL":
            path_list        = reading_df[config["IMAGE_PATH_INDEX_NAME"]].tolist()
            return path_list
        else:
            path_list        = reading_df[config["IMAGE_PATH_INDEX_NAME"]].tolist()
            lbl_list         = reading_df[config["LBL_INDEX_NAME"]].tolist()
            return path_list,lbl_list

    if config["INPUT_FROM"]=="Directory":
        if config["INPUT_FORMAT"]=="nifti":
            path_list        = [ f for f in os.listdir(config["ROI_DIRECTORY"]) if f.endswith(config["NIFTI_EXTN"])]
            return path_list
        if config["INPUT_FORMAT"]=="dicom":
            path_list        = [ f.path for f in os.scandir(config["ROI_DIRECTORY"]) if f.is_dir()]
            path_list        = [ s + "/" for s in path_list]
            return path_list
        if config["INPUT_FORMAT"]=="tfrecords":

            DukeSim_20mAs_tfrecors_list   = glob.glob("/data2/usr/ft42/nobackup/DukeSim_20mAs_tfrecords/*.tfrecords")

            DukeSim_28p5mAsnoTCM_tfrecors_list  = glob.glob("/data2/usr/ft42/nobackup/28p5mAsnoTCM_tfrecords/*.tfrecords")
            DukeSim_57mAsnoTCM_tfrecors_list    = glob.glob("/data2/usr/ft42/nobackup/5p7mAsnoTCM_tfrecords/*.tfrecords")
            DukeSim_100mAsnoTCM_tfrecors_list   = glob.glob("/data2/usr/ft42/nobackup/100mAsnoTCM_tfrecords/*.tfrecords")
            DukeSim_200mAsnoTCM_tfrecors_list   = glob.glob("/data2/usr/ft42/nobackup/200mAsnoTCM_tfrecords/*.tfrecords")

            MosMed_tfrecords_list         = glob.glob("/data2/usr/ft42/nobackup/mosmed_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            MIDRC_RICORD_tfrecords_list   = glob.glob("/data2/usr/ft42/nobackup/midric_ricord_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            covidctmd_CN_tfrecords_list   = glob.glob("/data2/usr/ft42/nobackup/covidctmd_tfrecords_96x160x160/Test_CvsN_tfrecords/*.tfrecords")
            covidctmd_tfrecords_list      = glob.glob("/data2/usr/ft42/nobackup/covidctmd_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            bimcv_tfrecords_list          = glob.glob("/data2/usr/ft42/nobackup/bimcv_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            covidctdata_tfrecords_list    = glob.glob("/data2/usr/ft42/nobackup/covidctdata_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            NIHa_tfrecords_list           = glob.glob("/data2/usr/ft42/nobackup/ct_NIHa_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            effusion_tfrecords_list       = glob.glob("/data2/usr/ft42/nobackup/effusion_nclcl_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            lgcancer_tfrecords_list       = glob.glob("/data2/usr/ft42/nobackup/lgcancer_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            nysub_tfrecords_list          = glob.glob("/data2/usr/ft42/nobackup/ny_sub_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            wsct_duke_tfrecords_list      = glob.glob("/data2/usr/ft42/nobackup/wsct_duke_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            lidc_duke_tfrecords_list      = glob.glob("/data2/usr/ft42/nobackup/lidi_idri_tfrecords_96x160x160/Test_tfrecords/*.tfrecords")
            covidctmd_trvalts_tfrecords_list   = glob.glob("/data2/usr/ft42/nobackup/covidctmd_tfrecords_96x160x160/CovidCTMd_All_tf/*.tfrecords")

            if config["ROI_DIRECTORY"]=='covidctmd':
                path_list = covidctmd_tfrecords_list
            elif config["ROI_DIRECTORY"]=='DukeSim_28p5mAsnoTCM':
                path_list = DukeSim_28p5mAsnoTCM_tfrecors_list
            elif config["ROI_DIRECTORY"]=='DukeSim_57mAsnoTCM':
                path_list = DukeSim_57mAsnoTCM_tfrecors_list
            elif config["ROI_DIRECTORY"]=='DukeSim_100mAsnoTCM':
                path_list = DukeSim_100mAsnoTCM_tfrecors_list
            elif config["ROI_DIRECTORY"]=='DukeSim_200mAsnoTCM':
                path_list = DukeSim_200mAsnoTCM_tfrecors_list
            elif config["ROI_DIRECTORY"]=='DukeSim_noTCM':
                path_list = DukeSim_200mAsnoTCM_tfrecors_list + DukeSim_100mAsnoTCM_tfrecors_list + DukeSim_28p5mAsnoTCM_tfrecors_list + DukeSim_57mAsnoTCM_tfrecors_list
            elif config["ROI_DIRECTORY"]=='covidctmd_CvsN':
                path_list = covidctmd_CN_tfrecords_list
            elif config["ROI_DIRECTORY"]=='midric_ricord':
                path_list = MIDRC_RICORD_tfrecords_list
            elif config["ROI_DIRECTORY"]=='mosmed':
                path_list = MosMed_tfrecords_list
            elif config["ROI_DIRECTORY"]=='bimcv':
                path_list = bimcv_tfrecords_list
            elif config["ROI_DIRECTORY"]=='covidctdata':
                path_list = covidctdata_tfrecords_list
            elif config["ROI_DIRECTORY"]=='NIHa':
                path_list = NIHa_tfrecords_list
            elif config["ROI_DIRECTORY"]=='eff':
                path_list = effusion_tfrecords_list
            elif config["ROI_DIRECTORY"]=='lgcancer':
                path_list = lgcancer_tfrecords_list
            elif config["ROI_DIRECTORY"]=='nysub':
                path_list = nysub_tfrecords_list
            elif config["ROI_DIRECTORY"]=='wsct':
                path_list = wsct_duke_tfrecords_list
            elif config["ROI_DIRECTORY"]=='covidctmd_all':
                path_list = covidctmd_trvalts_tfrecords_list
            elif config["ROI_DIRECTORY"]=='U_data_1': #U_data_1
                path_list = covidctmd_tfrecords_list + MosMed_tfrecords_list + MIDRC_RICORD_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_2': #U_data_2
                path_list = covidctmd_tfrecords_list + MosMed_tfrecords_list + MIDRC_RICORD_tfrecords_list + bimcv_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_3': #U.data-3
                path_list = covidctmd_tfrecords_list + MosMed_tfrecords_list + MIDRC_RICORD_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list + covidctmd_tfrecords_list +wsct_duke_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_4': #U.data-4
                path_list = covidctmd_tfrecords_list + MosMed_tfrecords_list + MIDRC_RICORD_tfrecords_list + bimcv_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list + covidctmd_tfrecords_list +wsct_duke_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_5':#U.data-5
                path_list = MIDRC_RICORD_tfrecords_list + covidctmd_tfrecords_list + MosMed_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list + covidctmd_tfrecords_list + nysub_tfrecords_list + lgcancer_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_6':#U.data-6
                path_list = MIDRC_RICORD_tfrecords_list + covidctmd_tfrecords_list + MosMed_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list + covidctmd_tfrecords_list +wsct_duke_tfrecords_list + nysub_tfrecords_list  + lgcancer_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_7':#U.data-7
                path_list = MIDRC_RICORD_tfrecords_list + covidctmd_tfrecords_list + MosMed_tfrecords_list + bimcv_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list+ covidctmd_tfrecords_list     + nysub_tfrecords_list  + lgcancer_tfrecords_list

            elif config["ROI_DIRECTORY"]=='U_data_8':#U.data-8
                path_list = MIDRC_RICORD_tfrecords_list + covidctmd_tfrecords_list + MosMed_tfrecords_list + bimcv_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list + covidctmd_tfrecords_list +wsct_duke_tfrecords_list + nysub_tfrecords_list + lgcancer_tfrecords_list
            elif config["ROI_DIRECTORY"]=='U_data_10':#U.data-8
                path_list = MIDRC_RICORD_tfrecords_list + covidctmd_tfrecords_list + MosMed_tfrecords_list + bimcv_tfrecords_list + NIHa_tfrecords_list + effusion_tfrecords_list + covidctmd_tfrecords_list + nysub_tfrecords_list + lgcancer_tfrecords_list + lidc_duke_tfrecords_list
            else:
                path_list        = [ f for f in os.listdir(config["ROI_DIRECTORY"]) if f.endswith(config["TF_EXTN"])]
            return path_list




def deploy(args, config):
    #-|Starting the running log
    now             = datetime.now()                    # Getting daya and time
    data_and_time   = now.strftime("%m/%d/%Y %H:%M:%S") # dd/mm/YY H:M:S
    saved_log_path  = config["PATH_TO_SAVE_LOGS"]
    f_log           = open(saved_log_path + config["NAME_TO_SAVE"]+'.log','a')

    #-| Writing Script info
    f_log.write("COVID-19 Classification Using Puplic and DukeSim.\n")
    f_log.write("copyright:            CVIT,DUKE.\n")
    f_log.write("Script:               Inference Script\n")
    f_log.write("Contributor:          Fakrul Islam Tushar (fakrulislam.tushar@duke.edu)\n")
    f_log.write("Target Organ:         Lungs\n")
    f_log.write("Classification Type:  COVID+/COVID-\n")
    f_log.write("Input:                Hyperperameters, model weight\n")
    f_log.write("Output:               CSV file with File Name and Prediction.\n")
    f_log.write("Version:              v0.1\n")
    f_log.write("Date and Time:        {}\n".format(data_and_time))
    f_log.write("\n")
    #----Model-Hyperperameters
    f_log.write("|----Prediction---|\n")
    f_log.write("\n")

    #---csv
    pred_subject_id=[]
    pred_lbl=[]
    pred_prediction=[]

    problem_pred_subject_id=[]
    problem_pred_lbl=[]
    problem_pred_subject_path=[]




    if config["INFERENCE_MODE"]=="TEST_WOLBL" or config["INPUT_FROM"]=="Directory":
        path_list=get_datalist_from_csv_or_directory(args, config)
        print('Total Number of Test Data={}'.format(len(path_list)))
        print(path_list)
    else:
        path_list,lbl_list=get_datalist_from_csv_or_directory(args, config)
        print('Total Number of Test Data={}'.format(len(path_list)))
        print('Total Number of Test Label={}'.format(len(lbl_list)))
        #print(path_list)


    for ct_data_count in range(0,len(path_list)):
        try:

            if config["INPUT_FORMAT"]=="dicom":
                subject_id    = path_list[ct_data_count].split('/')[-2]
                scans_path    = path_list[ct_data_count]
                output_file   ='tep_nifti/temp.nii.gz'
                #output_file   ='tep_nifti/{}.nii.gz'.format(subject_id)
                dicom2nifti.dicom_series_to_nifti(scans_path,output_file, reorient_nifti=False)
                img_path      = output_file
                #print('({})----subject-id->{}----Saved---.nii.gz={}'.format(ct_data_count+1,subject_id,scans_path))

            elif config["INPUT_FORMAT"]=="nifti":
                subject_id   = path_list[ct_data_count].split('/')[-1].split('.')[0]
                img_path     = path_list[ct_data_count]
                #print('({})---subject_id->{}-------.nii.gz={}'.format(ct_data_count+1,subject_id,img_path))


            if config["INPUT_FORMAT"]=="tfrecords":

                ct_covid_patch,ct_lbl,subject_id=give_numpy_img(path_list[ct_data_count])

            else:
                ct_covid_img     = Load_and_preprocess_nifti_resample_clip_normalize(img_path           = img_path,
                                                                                 resampling_sapcing = config["RESAMPLING"],
                                                                                 lbl_flag           = config["MASKF_FLAG"],
                                                                                 flip_img           = config["FLIPING_IMG"],
                                                                                 hu_cliping_range   = config["HU_SLIPING"])

                ct_covid_patch   = COVID_PATCH_EXTRACTION(IMAGE_CT               = ct_covid_img,
                                                          PATCH_SIZE             = config["PATCH_SIZE"],
                                                          PADDING_CONSTENT_VALUE = config["PADDING_CONSTENT_VALUE"])

            ct_covid_patch_channeled=img=tf.expand_dims(ct_covid_patch, axis=-1)
            ct_covid_patch_batched=img=tf.expand_dims(ct_covid_patch_channeled, axis=0)

            prediction=get_Resnet3D_to_pridict(batched_img_ct     = ct_covid_patch_batched,
                                               INPUT_PATCH_SIZE   = config["INPUT_PATCH_SIZE"],
                                               NUMBER_OF_CLASSES  = config["NUMBER_OF_CLASSES"],
                                               TRAIN_NUM_RES_UNIT = config["TRAIN_NUM_RES_UNIT"],
                                               TRAIN_NUM_FILTERS  = config["TRAIN_NUM_FILTERS"],
                                               TRAIN_STRIDES      = config["TRAIN_STRIDES"],
                                               MODEL_PATH         = config["MODEL_PATH" ],
                                               DATASET_TO_RUN_ON  = config["DATASET_TO_RUN_ON"]
                                               )
            if config["INFERENCE_MODE"]=="TEST_WOLBL":
                print('({})----|subject-id->{}---|predict->{:.3f}\n'.format(ct_data_count+1,subject_id,prediction[0][0]))
                f_log.write('({})----|subject-id->{}---|predict->{:.3f}\n'.format(ct_data_count+1,subject_id,prediction[0][0]))
                pred_subject_id.append(subject_id)
                pred_prediction.append(prediction[0][0])
            else:
                if config["INPUT_FORMAT"]=="tfrecords":
                    roi_ct_lbl = ct_lbl
                else:
                    roi_ct_lbl = lbl_list[ct_data_count]
                print('({})----|subject-id->{}---|predict->{:.3f}-|Label->{}\n'.format(ct_data_count+1,subject_id,prediction[0][0],roi_ct_lbl))
                f_log.write('({})----|subject-id->{}---|predict->{:.3f}-|Label->{}\n'.format(ct_data_count+1,subject_id,prediction[0][0],roi_ct_lbl))
                pred_subject_id.append(subject_id)
                pred_lbl.append(roi_ct_lbl)
                pred_prediction.append(prediction[0][0])
        except:
            if config["INFERENCE_MODE"]=="TEST_WOLBL":
                print('({})----|subject-id->{}---|-Problem!!!!!\n'.format(ct_data_count+1,subject_id))
                f_log.write('({})----|subject-id->{}|-Problem!!!!\n'.format(ct_data_count+1,subject_id))
                problem_pred_subject_id.append(subject_id)
                problem_pred_subject_path.append(path_list[ct_data_count])
            else:
                if config["INPUT_FORMAT"]=="tfrecords":
                    roi_ct_lbl = ct_lbl
                else:
                    roi_ct_lbl = lbl_list[ct_data_count]
                print('({})----|subject-id->{}---|-Problem!!!-|Label->{}\n'.format(ct_data_count+1,subject_id,roi_ct_lbl))
                f_log.write('({})----|subject-id->{}---|Problem!!!-|Label->{}\n'.format(ct_data_count+1,subject_id,roi_ct_lbl))
                problem_pred_subject_id.append(subject_id)
                problem_pred_lbl.append(roi_ct_lbl)
                problem_pred_subject_path.append(path_list[ct_data_count])

    if config["INFERENCE_MODE"]=="TEST_WOLBL":
        predicted_info_DataFrame=pd.DataFrame(list(zip(pred_subject_id,pred_prediction)),columns=['id','pred'])
        predicted_info_DataFrame.to_csv(saved_log_path+config["NAME_TO_SAVE"]+'.csv',encoding='utf-8',index=False)
        problamatic_info_DataFrame=pd.DataFrame(list(zip(problem_pred_subject_id,problem_pred_subject_path)),columns=['id','path'])
        problamatic_info_DataFrame.to_csv(saved_log_path+config["NAME_TO_SAVE"]+'-Problematic.csv',encoding='utf-8',index=False)
    else:
        predicted_info_DataFrame=pd.DataFrame(list(zip(pred_subject_id,pred_lbl,pred_prediction)),columns=['id','ct_label','pred'])
        predicted_info_DataFrame.to_csv(saved_log_path+config["NAME_TO_SAVE"]+'.csv',encoding='utf-8',index=False)
        problamatic_info_DataFrame=pd.DataFrame(list(zip(problem_pred_subject_id,problem_pred_lbl,problem_pred_subject_path)),columns=['id','ct_label','path'])
        problamatic_info_DataFrame.to_csv(saved_log_path+config["NAME_TO_SAVE"]+'-Problematic.csv',encoding='utf-8',index=False)

    if config["COMPUTE_AUC"]=="True":
        y_true      = predicted_info_DataFrame['ct_label']
        y_pred      = predicted_info_DataFrame['pred']
        fpr,tpr,_   = roc_curve(y_true, y_pred,pos_label=1)
        evaluate_auc= auc(fpr, tpr)
        postive_cases  = predicted_info_DataFrame['ct_label'].value_counts(normalize=False)[1]
        negative_cases = predicted_info_DataFrame['ct_label'].value_counts(normalize=False)[0]

        print('Volume-Based-Evaluatio|Total # of cases={}|# of positives={}|# of negatives={}|AUC={:.3f}'.format(len(predicted_info_DataFrame),postive_cases,negative_cases,evaluate_auc))
        f_log.write('Volume-Based-Evaluatio|Total # of cases={}|# of positives={}|# of negatives={}|AUC={:.3f}'.format(len(predicted_info_DataFrame),postive_cases,negative_cases,evaluate_auc))

    f_log.close()
    return

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='COVID-19 Classification deploy script')
    parser.add_argument('--gpu',   type=int, default='0', help='define the gpu to use')
    parser.add_argument('--config', default='config_all.json')
    args = parser.parse_args()


    # Parse the run config
    with open(args.config) as f:
        config = json.load(f)

    # Call Deploy
    deploy(args, config)
