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

# 추가로 할 일: train 하이퍼파라미터 args로 통합
def train_yolov8(args):
    # default imgsz(nHD): 640*260, original imgsz(FHD) = 1920 * 1080, input imgsz(HD) = 1280 * 720
    val_df = pd.DataFrame(columns = ['All'] + [str(i) for i in range(1, 10)])
    if args.val_type=='kfold':
        dataset_yamls = list(Path('/home/kongminseok/aicity/datasets/kfold/cfg').glob("*.yaml"))
        for k in range(args.ksplit):
            dataset_yaml = dataset_yamls[k]
            model = YOLO('yolov8x.pt', task='detect')
            model.train(data=dataset_yaml, 
                    epochs=30,
                    batch=16,
                    #imgsz=(1280, 720), 
                    device=[0, 1],
                    save=True,
                    save_period=1,
                    seed=0,
                    project="practices",
                    name=f"split_{k+1}"
                    )
            
            model = YOLO(f'practices/split_{k+1}/weights/best.pt')
            metrics = model.val() # 추가로 할 일: validation run 저장 위치 변경 어떻게 하는지 몰루?
            all_map = metrics.box.map
            class_maps = metrics.box.maps
            maps = np.concatenate([[all_map], class_maps])
            maps_df = pd.DataFrame([maps], columns=val_df.columns, index=[f'split_{k+1}'])
            val_df = pd.concat([val_df, maps_df], axis=0, ignore_index=False)
        
        val_df.to_csv('validation_result.csv',encoding='utf-8', index=True)
        model = YOLO('yolov8x.pt', task='detect')
        model.train(data='/home/kongminseok/aicity/cfg/yolov8.yaml', 
                    epochs=30,
                    batch=16,
                    #imgsz=(1280, 720), 
                    device=[0, 1],
                    save=True,
                    save_period=1,
                    seed=0,
                    val=False,
                    project="practices",
                    name=f"accessible")
        print(f'Final validation mAP50-95: {val_df.All.mean()}')
            
    elif args.val_type=='holdout':
        model.train(data="/home/kongminseok/aicity/cfg/yolov8.yaml", 
                    epochs=1,
                    batch=16, 
                    #imgsz=(1280, 720), 
                    device=[0, 1],
                    save=True,
                    save_period=1,
                    seed=0,
                    project="practices",
                    name="practice_1"
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
    
    test_dir = Path(dataset_path) / 'images' / 'test'
    test_imgnames = os.listdir(test_dir)
    test_imgs = [f'{test_dir}/{imgname}' for imgname in test_imgnames]
    for i, (test_img, test_imgname) in enumerate((zip(test_imgs, test_imgnames))):
        if (i+1)%10==0:
            print(f"Complete {i+1}/{len(test_imgnames)} img")
        video_id, frame = os.path.splitext(test_imgname)[0].replace('v','').replace('f','').split('_')
        result = model(test_img, verbose=False)
        boxes = result[0].boxes
        
        # restore original bboxes coordinates
        boxes_xywh = boxes.xywh.to('cpu')
        bboxes = torch.empty_like(boxes_xywh)
        bboxes[:, 0] = boxes_xywh[:, 0] - (boxes_xywh[:, 2] / 2)
        bboxes[:, 1] = boxes_xywh[:, 1] - (boxes_xywh[:, 3] / 2)
        bboxes[:, 2] = boxes_xywh[:, 2]
        bboxes[:, 3] = boxes_xywh[:, 3]
        
        # restore original class_id
        boxes_cls = (boxes.cls + 1).to('cpu').unsqueeze(1)
        boxes_conf = boxes.conf.to('cpu').unsqueeze(1)
        submits = torch.cat((bboxes, boxes_cls, boxes_conf), dim=1)
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
    
def main(fe=False, cl=False, sd=False, tm=False, gs=False):
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/kongminseok/aicity/datasets", help="")
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
        
    # STEP 3. split the dataset: sd
    if sd:
        if args.val_type=='kfold':
            custom_dataset.kfold_val_split(yaml_file='/home/kongminseok/aicity/cfg/yolov8.yaml')
        elif args.val_type=='holdout':
            custom_dataset.holdout_val_split()
        else:
            print("Splitting the data was not conducted. Please select either 'kfold' or 'holdout' for the 'val_type' argument.")
    
    # STEP 4. Train the model: tm
    if tm:
        train_yolov8(args)
    
    # STEP 5. Generate the submission: gs
    if gs: # 추가로 할 일: submission arguments args로 통합
        submission(dataset_path=args.dataset_path,
                   weight_path="/home/kongminseok/aicity/practices/accessible/weights/best.pt",
                   save_path='/home/kongminseok/aicity/submissions',
                   filename='submission.txt')
    
    end = time.time()
    print(f"Running Time: {int((end-start)//60)}m {int((end-start)%60)}s")
    
if __name__=="__main__":
    main(gs=True)
    
