import argparse
import time

import os
import torch
import shutil
import pandas as pd
import numpy as np

from pathlib import Path
from ultralytics import YOLO
from data.data import CustomDataset, frame_extraction

from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction

def train_yolov8(args):
    val_df = pd.DataFrame(columns = ['All'] + [str(i) for i in range(1, 10)])
    if args.val_type=='kfold':
        dataset_yamls = list(Path('/user/datasets/kfold/cfg').glob("*.yaml"))
        for k in range(args.ksplit):
            dataset_yaml = dataset_yamls[k]

            ###########
            #<<keep training>>
            # pretrained_model_path = f'practices/split_14/weights/best.pt'
            
            # model = YOLO(pretrained_model_path, task='train')
            # model.train(data=dataset_yaml,
            #             epochs=20,  # adding 20 epoch 
            #             batch=16,
            #             device=[0, 1],
            #             save=True,
            #             save_period=1,
            #             seed=0,
            #             project="practices_continued",
            #             name=f"split_{k+1}_continued"
            #             )
            
            # model = YOLO(f'practices_continued/split_{k+1}_continued/weights/best.pt')
            # metrics = model.val() 

            ############
            model = YOLO('yolov8x.pt', task='detect')
            model.train(data=dataset_yaml, 
                    epochs=15,
                    batch=16,
                    #imgsz=(1280, 720), 
                    device=[0, 1],
                    save=True,
                    save_period=1,
                    seed=0,
                    project="practices",
                    name=f"sahi_final_aug_{k+1}"
                    )
            
            model = YOLO(f'practices/sahi_final_aug_{k+1}/weights/best.pt')
            metrics = model.val()
            all_map = metrics.box.map
            class_maps = metrics.box.maps
            maps = np.concatenate([[all_map], class_maps])
            maps_df = pd.DataFrame([maps], columns=val_df.columns, index=[f'split_{k+1}'])
            val_df = pd.concat([val_df, maps_df], axis=0, ignore_index=False)
        
        val_df.to_csv('validation_result.csv',encoding='utf-8', index=True)

        # ####<<keep training>>
        # model = YOLO(pretrained_model_path, task='detect')

        # model.train(data='/user/cfg/yolov8.yaml', 
        #         epochs=20, 
        #         batch=16,
        #         device=[0, 1],  
        #         save=True,
        #         save_period=1,
        #         seed=0,
        #         val=True,
        #         project="practices_continued", 
        #         name="accessible_continued")

        model = YOLO('yolov8x.pt', task='detect')
        model.train(data='/user/cfg/yolov8.yaml', 
                    epochs=15,
                    batch=16,
                    #imgsz=(1280, 720), 
                    device=[0, 1],
                    save=True,
                    save_period=1,
                    seed=0,
                    val=False,
                    project="practices",
                    name=f"sahi_final_accessible")
        print(f'Final validation mAP50-95: {val_df.All.mean()}')
            
    elif args.val_type=='holdout':
        model = YOLO('yolov8x.pt', task='detect')
        model.train(data="/user/cfg/yolov8.yaml", 
                    epochs=30,
                    batch=16, 
                    #imgsz=(1280, 720), 
                    device=[0, 1],
                    save=True,
                    save_period=1,
                    seed=0,
                    project="practices",
                    name="practice"
                    )
    else:
        print("The training was not conducted. Please select either 'kfold' or 'holdout' for the 'val_type' argument.")
        return
                    
def submission(dataset_path, weight_path, save_path, filename):
    # submission format: 〈video_id〉, 〈frame〉, 〈bb_left〉, 〈bb_top〉, 〈bb_width〉, 〈bb_height〉, 〈class_id〉, 〈confidence〉
    model = YOLO(weight_path)
    
    save_path = Path(save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    submit_file = os.path.join(save_path, filename)
    
    test_dir = Path(dataset_path) / 'images' / 'val'
    test_imgnames = os.listdir(test_dir)
    test_imgs = [f'{test_dir}/{imgname}' for imgname in test_imgnames]
    for i, (test_img, test_imgname) in enumerate((zip(test_imgs, test_imgnames))):
        if (i+1)%10==0:
            print(f"Complete {i+1}/{len(test_imgnames)} img")
        video_id, frame = os.path.splitext(test_imgname)[0].replace('v','').replace('f','').split('_')

        ###sahi prediction
        detection_model= Yolov8DetectionModel(
            model_path=weight_path,
            confidence_threshold=0.3,
            image_size = (1920,1080),
        )

        sahi_result = get_sliced_prediction(
            test_img,
            detection_model,
            slice_height = 800,
            slice_width = 800,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2
        )
        sahi_bbox= []
        sahi_cls =[]
        sahi_conf = []
        sahi_result_list = sahi_result.object_prediction_list
        for i in range(len(sahi_result_list)):
            sahi_width=sahi_result_list[i].bbox.maxx - sahi_result_list[i].bbox.minx
            sahi_height = sahi_result_list[i].bbox.maxy - sahi_result_list[i].bbox.miny   
            sahi_onebox=torch.tensor([[sahi_result_list[i].bbox.minx,sahi_result_list[i].bbox.miny, sahi_width, sahi_height]])
            sahi_bbox.append(sahi_onebox)

            sahi_onecls = torch.tensor([sahi_result_list[i].category.id +1], dtype=torch.float64)
            sahi_cls.append(sahi_onecls)

            sahi_score =  torch.tensor([sahi_result_list[i].score.value], dtype=torch.float64)
            sahi_conf.append(sahi_score)


        if not sahi_bbox:
            continue

        final_bbox = torch.cat(sahi_bbox, dim=0)
        final_cls = torch.stack(sahi_cls, dim=0)
        final_conf = torch.stack(sahi_conf, dim=0)

        submits = torch.cat((final_bbox, final_cls, final_conf), dim=1)
        submits_array = submits.numpy().astype(str)
        
        with open(submit_file, 'a') as file:
            for row in submits_array:
                row = [video_id, frame] + list(row)
                row[-2] = str(int(float(row[-2])))
                file.write(','.join(row) + '\n')
                
    # sorting
    submit_file = os.path.join(save_path, filename)
    with open(submit_file, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        ann = line.split(',')
        video_id = int(ann[0].strip())
        frame = int(ann[1].strip())
        data.append((video_id, frame, line))
       
    data.sort(key=lambda x:(x[0], x[1]))
    new_submit_file = f'{os.path.splitext(submit_file)[0]}_sorted.txt'
    with open(new_submit_file, 'w') as file:
        for item in data:
            file.write(item[2])
            
    print(f"Complete to generate submission file.")
    
def main(fe=False, cl=False, sahi=False, sd=False, tm=False, gs=True):
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/user/datasets/", help="")
    parser.add_argument("--ksplit", type=int, default=5, help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--val_type", type=str, default="kfold", help="select either 'kfold' or 'holdout'")
    args = parser.parse_args()
    
    # STEP 1. Frame extraction : fe
    # select either 'train' or 'test' for the 'data' argument.
    if fe:
        frame_extraction(dataset_path=args.dataset_path, fps='fps=10', data='train')
    
    custom_dataset = CustomDataset(args)
    # STEP 2. convert the labels: cl
    if cl:
        custom_dataset.convert_data_to_yolo()

    # STEP 3. applying SAHI
    if sahi:
        custom_dataset.slicing_with_sahi()
        
    # STEP 4. split the dataset: sd
    if sd:
        if args.val_type=='kfold':
            custom_dataset.kfold_val_split(yaml_file='/user/cfg/yolov8.yaml')
        elif args.val_type=='holdout':
            custom_dataset.holdout_val_split()
        else:
            print("Splitting the data was not conducted. Please select either 'kfold' or 'holdout' for the 'val_type' argument.")
    
    # STEP 5. Train the model: tm
    if tm:
        train_yolov8(args)
    
    # STEP 6. Generate the submission: gs
    if gs:
        submission(dataset_path=args.dataset_path,
                   weight_path="/user/best.pt",
                   save_path='/user/submissions_sahi_more_epoch',
                   filename='submission.txt')
    
    end = time.time()
    print(f"Running Time: {int((end-start)//60)}m {int((end-start)%60)}s")
    
if __name__=="__main__":
    main(gs=True)
    
