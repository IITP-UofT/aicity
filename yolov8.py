import os
import torch
import time
import argparse

import shutil
import datetime
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from tqdm import tqdm

def train():
    dataset_path = Path('./datasets/raw_dataset')
    labels = sorted(dataset_path.rglob("*labels/*train/*.txt"))
    yaml_file = './cfg/yolov8.yaml'
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
    cls_idx = sorted(classes.keys())
    idx = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=idx)
    
    for label in labels:
        label_counter = Counter()
        with open(label, 'r') as label_file:
            lines = label_file.readlines()
        
        for line in lines:
            label_counter[int(line.split(' ')[0])] += 1
            
        labels_df.loc[label.stem] = label_counter
    labels_df = labels_df.fillna(0.0)
    
    # K-Fold Dataset Split
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=42)
    kfolds = list(kf.split(labels_df))
    folds = [f'split_{n}' for n in range(1, ksplit+1)]
    folds_df = pd.DataFrame(index=idx, columns=folds)
    
    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{i}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{i}'].loc[labels_df.iloc[val].index] = 'val'
    
    fold_label_distribution = pd.DataFrame(index=folds, columns=cls_idx)
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1E-7)
        fold_label_distribution.loc[f'split_{n}'] = ratio
        
    supported_extensions = ['.jpg', '.jpeg', '.png']
    
    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / 'images' / 'train').rglob(f"*{ext}")))
        
    print(images[0])

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': split_dir.as_posix(),
                'train': 'train',
                'val': 'val',
                'names': classes
                }, ds_y)
        
        for img, label in zip(images, labels):
            for split, k_split in folds_df.loc[img.stem].items():
                # Destination directory
                img_to_path = save_path / split / k_split / 'images'
                label_to_path = save_path / split / k_split / 'labels'

                # Copy image and label files to new directory (SamefileError if file already exists)
                shutil.copy(img, img_to_path / img.name)
                shutil.copy(label, label_to_path / label.name)
        
        folds_df.to_csv(save_path / "kfold_datasplit.csv")
        fold_label_distribution.to_csv(save_path / "kfold_label_distribution.csv")
        
    results = {}
    model = YOLO('yolov8x.pt', task='detect')
    for k in range(ksplit):
        dataset_yaml = ds_yamls[k]
        model.train(data=dataset_yaml, 
                epochs=100,
                batch=16, 
                imgsz=(1280, 720), 
                device=[0, 1],
                save=True,
                save_period=1,
                seed=0,
                project="practices",
                name="practice_1"
                )
        results[k] = model.metrics
        
    
    
    """
    model = YOLO('yolov8x.pt')
    # default imgsz(nHD): 640*260, original imgsz(FHD) = 1920 * 1080, input imgsz(HD) = 1280 * 720
    model.train(data='./cfg/yolov8.yaml', 
                epochs=100, 
                imgsz=(1280, 720), 
                device=[0, 1],
                save=True,
                save_period=1,
                seed=0,
                project="practices",
                name="practice_1"
                )
    """
                          
def submission(args):
    # submission format: 〈video_id〉, 〈frame〉, 〈bb_left〉, 〈bb_top〉, 〈bb_width〉, 〈bb_height〉, 〈class〉, 〈confidence〉
    model = YOLO(args.weight_path)
    test_dir = args.test_dir
    test_imgnames = os.listdir(test_dir)
    test_imgs = [f'{test_dir}/{imgname}' for imgname in test_imgnames]
    for i, (test_img, test_imgname) in enumerate((zip(test_imgs, test_imgnames))):
        if (i+1)%10==0:
            print(f"Complete {i+1}/{len(test_imgnames)} img")
        video_id, frame = os.path.splitext(test_imgname)[0].replace('v','').replace('f','').split('_')
        result = model(test_img, verbose=False)
        boxes = result[0].boxes
        boxes_xywh = boxes.xywh.to('cpu')
        boxes_cls = boxes.cls.to('cpu').unsqueeze(1)
        boxes_conf = boxes.conf.to('cpu').unsqueeze(1)
        submits = torch.cat((boxes_xywh, boxes_cls, boxes_conf), dim=1)
        submits_array = submits.numpy().astype(str)
        submit_path = os.path.join(args.submit_dir, args.submit_filename)
        with open(submit_path, 'a') as file:
            for row in submits_array:
                row = [video_id, frame] + list(row)
                row[-2] = str(int(float(row[-2])))
                file.write(' '.join(row) + '\n')
                
    # sorting
    submit_path = os.path.join(args.submit_dir, args.submit_filename)
    with open(submit_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        ann = line.split()
        video_id = int(ann[0].strip())
        frame = int(ann[1].strip())
        data.append((video_id, frame, line))
       
    data.sort(key=lambda x:(x[0], x[1]))
    new_submit_path = f'{os.path.splitext(submit_path)[0]}_sorted.txt'
    with open(new_submit_path, 'w') as file:
        for item in data:
            file.write(item[2])
            
    print(f"Complete generating {submit_path}")
        
def main(mode='s'):
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", type=str, default="./practices/practice_32/weights/best.pt", help="directory for loading trained model")
    parser.add_argument("--test_dir", type=str, default="./datasets/yolo_dataset/images/test", help="directory for loading test images")
    parser.add_argument("--submit_dir", type=str, default="", help="saving directory for submission txt file")
    parser.add_argument("--submit_filename", type=str, default="submission.txt", help="submission txt file name")
    args = parser.parse_args()
    if mode=='t':
        train()
    if mode=='s':
        submission(args)
    end = time.time()
    print(f"Running Time: {int((end-start)//60)}m {int((end-start)%60)}s")
    
if __name__=="__main__":
    main(mode='t')
    
