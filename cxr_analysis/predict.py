import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from model import efficientnetv2_s as create_model


def main(args, test_data_list):

    train_data_name = args.train_data_name                          
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ROOT_PATH_WEIGHTS =args.weights_path
    ROOT_PATH_DATA = args.data_path
    SAVE_PATH_CSV =  args.save_path

    for test_data_name in test_data_list:
        print ('Starting inference for Dataset - ', test_data_name)
        print (50 * '*')
        status_list = [0, 1]

        for is_positive in status_list:

            print ('Currently predicting for Label', is_positive) 

            if is_positive:
                covid_status = 'positive'    
            else:
                covid_status = 'negative'
                
            #device = torch.device("cpu")

            img_size = {"s": [300, 384],  # train_size, val_size
                        "m": [384, 480],
                        "l": [384, 480]}
            num_model = "s"

            data_transform = transforms.Compose(
                [transforms.Resize(img_size[num_model][1]),
                transforms.CenterCrop(img_size[num_model][1]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            
            # create model
            model = create_model(num_classes=2).to(device)
            # load model weights
            model_weight_path = os.path.join(ROOT_PATH_WEIGHTS, [filename for filename in os.listdir(ROOT_PATH_WEIGHTS) if filename.startswith(train_data_name)][0])
            print ('Currently predicting from', model_weight_path)
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()

            test_path = os.path.join(ROOT_PATH_DATA, test_data_name , 'test',covid_status + '/' )
            
            if os.path.exists(SAVE_PATH_CSV) is False:
                os.makedirs(SAVE_PATH_CSV)

            test_img_list = os.listdir(test_path)
            predict_all =[]
            predict_prob =[]
            fname_list = []
            gt_list =[]
            print ('Total files for inference ', len(test_img_list))
            for fname in test_img_list:
                #assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                img = Image.open(test_path + fname)
                #print ('Currently predicting', fname)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                #plt.imshow(img)
                # [N, C, H, W]
                img = data_transform(img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)
                
                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()
                    predict_all.append(predict_cla)
                    predict_prob.append(predict[1].numpy())
                    fname_list.append(fname)
                    if is_positive:
                        gt_list.append(1)
                    else:
                        gt_list.append(0)

            prob_result = list(zip(fname_list, predict_all,predict_prob,gt_list))
            prob_df = pd.DataFrame(prob_result, columns=['filename','prediction','covid_prob', 'gt'])
            prob_df.to_csv(SAVE_PATH_CSV + 'training_' + train_data_name + '_'+ 'efficientnetv2' +  '_'+ test_data_name + '_' + covid_status + '.csv')
            print ('Prediction complete for Covid-19 Images', covid_status) 
            print ('Total positive predicted', sum(predict_all))
            print ('Total images', len(test_img_list))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights_path', type=str, default="/codebase/virtual_vs_real_covid_classification/weights/covid-classification-models/")    
    parser.add_argument('--data_path', type=str, default="/dataset/chest_xrays/old/custom_dataset/")   
    parser.add_argument('--save_path', type=str, default="/output/virtual_vs_reality_covid_classification/")  
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--train_data_name', type=str, default="u-3-filter")    # Choice : covidx-cxr2 or rafael or u-3-filter or bimcv2
    
    opt = parser.parse_args()
    
    test_data_list = ['xcat-carestream-uint8-crop', 'xcat-siemens-uint8-crop', 'u-3-filter', 'covidx-cxr2', 'rafael',  'bimcv2']

    main(opt,test_data_list)

