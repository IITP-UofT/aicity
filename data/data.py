import os
import random
import torch
import shutil
import subprocess
import datetime
import yaml
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import KFold
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import ffmpeg
from sahi.slicing import slice_image



class CustomDataset(Dataset):
    def __init__(self, args: dict, transform=None):
        self.dataset_path = args.dataset_path
        self.ksplit = args.ksplit
        self.seed = args.seed
        self.val_type = args.val_type
        
        self.annotations = self.read_txt_file()
        self.label_dir = Path(self.dataset_path) / 'labels'/ 'train'
        self.img_dir = Path(self.dataset_path) / 'images' / 'train'
        
        self.transform = transform
        
    def read_txt_file(self):
        txt_file = os.path.join(self.dataset_path, 'gt.txt')
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        annotations = [line.strip().split(',') for line in lines]
        return annotations
    
    def convert_data_to_yolo(self):
        # To generate label directory
        if os.path.exists(self.label_dir):
            shutil.rmtree(self.label_dir)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        
        for ann in tqdm(self.annotations):
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id = ann
            img_filename = f'v{video_id}_f{frame}.jpg'
            img_path = os.path.join(self.img_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"No such file or directory: {img_path}")
                continue
            
            img = Image.open(img_path).convert("RGB")
        
            # Convert bbox coordinates (YOLO format): xywh -> cx, cy, w, h (normalized)
            bboxes = [((float(bb_left) + float(bb_width) / 2) / img.width, 
                    (float(bb_top) + float(bb_height) / 2) / img.height, 
                    float(bb_width) / img.width, 
                    float(bb_height) / img.height)]
        
            # Class id ailgnment
            class_id = int(class_id) - 1
            # YOLO format: <class_id> <x_center> <y_center> <width> <height>
            yolo_format = f"{class_id} {bboxes[0][0]} {bboxes[0][1]} {bboxes[0][2]} {bboxes[0][3]}"
        
            # Save to txt file
            txt_filename = os.path.splitext(img_filename)[0] + ".txt"
            txt_path = os.path.join(self.label_dir, txt_filename)
            with open(txt_path, 'a') as file:
                file.write(yolo_format + '\n')

    def slicing_with_sahi(self):
        images=[]
        slice_size = 800
        for image_name in os.listdir(self.img_dir):
            image_path = os.path.join(self.img_dir, image_name)
            images.append(image_path)

            #slice each of the images -> 6 slices
            slice_image_result = slice_image(
                image= image_path ,
                output_file_name= image_name[:-4],
                output_dir='/user/datasets/slice_images/train',
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            #make the annotation for slice image version
            label_file = image_name.replace('.jpg', '.txt')
            txt_path = os.path.join(self.label_dir, label_file)
            
            if os.path.exists(txt_path):

                for i in range(len(slice_image_result)):
                    top_left_x = slice_image_result[i]['starting_pixel'][0]
                    top_left_y = slice_image_result[i]['starting_pixel'][1]
                    adjusted_annotations= []
                    with open(txt_path, 'r') as f:
                        for line in f.readlines():
                            class_id, x_center, y_center, width, height = map(float, line.split())
                            # Convert YOLO coordinates to absolute coordinates in the original image
                            original_width, original_height = 1920, 1080  # Update with your actual image dimensions
                            abs_x_center = x_center * original_width
                            abs_y_center = y_center * original_height
                            abs_width = width * original_width
                            abs_height = height * original_height

                            abs_x_min = abs_x_center - (abs_width / 2 )
                            abs_x_max = abs_x_center + (abs_width / 2)
                            abs_y_min = abs_y_center - (abs_height / 2)
                            abs_y_max = abs_y_center + (abs_height / 2)

                            clipped_x_min = max(abs_x_min, top_left_x)
                            clipped_x_max = min(abs_x_max, top_left_x + slice_size)
                            clipped_y_min = max(abs_y_min, top_left_y)
                            clipped_y_max = min(abs_y_max, top_left_y + slice_size)

                            if clipped_x_min < clipped_x_max and clipped_y_min < clipped_y_max:
                                # Convert clipped bounding box back to relative coordinates within the slice
                                clipped_x_center = ((clipped_x_min + clipped_x_max) / 2 - top_left_x) / slice_size
                                clipped_y_center = ((clipped_y_min + clipped_y_max) / 2 - top_left_y) / slice_size
                                clipped_width = (clipped_x_max - clipped_x_min) / slice_size
                                clipped_height = (clipped_y_max - clipped_y_min) / slice_size

                                adjusted_annotation = f"{class_id} {clipped_x_center} {clipped_y_center} {clipped_width} {clipped_height}\n"
                                adjusted_annotations.append(adjusted_annotation)
                    new_label_file = slice_image_result[i]['filename'][:-4] +'.txt'
                    adjusted_annotation_path=os.path.join('/user/datasets/slice_labels/train', new_label_file)
                    with open(adjusted_annotation_path, 'w') as f:
                        f.writelines(adjusted_annotations)
    
    # K-Fold cross-validation
    def kfold_val_split(self, yaml_file):
        dataset_path = Path(self.dataset_path)
        labels = sorted(dataset_path.glob("slice_labels/train/*.txt"))
        with open(yaml_file, 'r', encoding="utf8") as y:
            classes = yaml.safe_load(y)['names']
        cls_idx = sorted(classes.keys())
        idx = [label.stem for label in labels]
        labels_df = pd.DataFrame([], columns=cls_idx, index=idx)
    
        print("START STEP 1/2")
        for label in tqdm(labels):
            label_counter = Counter()
            with open(label, 'r') as label_file:
                lines = label_file.readlines()
        
            for line in lines:
                label_counter[float(line.split(' ')[0])] += 1
            
            labels_df.loc[label.stem] = label_counter
            
        labels_df = labels_df.fillna(0.0)
    
        # K-Fold Dataset Split
        kf = KFold(n_splits=self.ksplit, shuffle=True, random_state=self.seed)
        kfolds = list(kf.split(labels_df))
        folds = [f'split_{n}' for n in range(1, self.ksplit+1)]
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
        
        images = [(Path(f'{os.path.splitext(name)[0]}.png'.replace('slice_labels', 'slice_images'))) for name  in labels]

        #supported_extensions = ['.jpg', '.jpeg', '.png']
        ## Initialize an empty list to store image file paths  
        #images = []
        ## Loop through supported extensions and gather image files
        #for ext in supported_extensions:
        #    images.extend(sorted((dataset_path / 'images' / 'train').rglob(f"*{ext}")))
        
        # Create the necessary directories and dataset YAML files (unchanged)
        #save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{self.ksplit}-Fold_Cross-val')
        if self.val_type=='kfold':
            save_path = Path(dataset_path / self.val_type)
        else:
            print("Generating kfold directory was not conducted. Please select either 'kfold' for the 'val_type' argument.")
            return
        save_path.mkdir(parents=True, exist_ok=True)

        for split in (folds_df.columns):
            # Create directories
            split_dir = save_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

            # Create dataset YAML files
            yaml_dir = save_path / 'cfg'
            yaml_dir.mkdir(parents=True, exist_ok=True)
            dataset_yaml = yaml_dir / f'{split}_dataset.yaml'

            with open(dataset_yaml, 'w') as ds_y:
                yaml.safe_dump({
                    'path': split_dir.as_posix(),
                    'train': 'train',
                    'val': 'val',
                    'names': classes
                    }, ds_y)
        
        print("START STEP 2/2")
        for img, label in tqdm(zip(images, labels)):
            for split, k_split in folds_df.loc[img.stem].items():
                # Destination directory
                img_to_path = save_path / split / k_split / 'images'
                label_to_path = save_path / split / k_split / 'labels'

                # Copy image and label files to new directory (SamefileError if file already exists)
                shutil.copy(img, img_to_path / img.name)
                shutil.copy(label, label_to_path / label.name)
        
        folds_df.to_csv(save_path / "kfold_datasplit.csv")
        fold_label_distribution.to_csv(save_path / "kfold_label_distribution.csv")
    
    # Holdout validation
    def holdout_val_split(self): # 추가로 할 일: cfg/holdout.yaml 만들기
        #slice label로 경로 수정
        label_path = Path('/user/datasets/slice_labels') / 'train'
        img_path = Path('/user/datasets/slice_images') / 'train'
        ##
        txt_files = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
        random.seed(self.seed)
        random.shuffle(txt_files)
        train_ratio = 0.8
        split_idx = int(len(txt_files) * train_ratio)
        train_labels = txt_files[:split_idx]
        val_labels = txt_files[split_idx:]
        
        print("START STEP 1/6")
        train_imgs = list()
        val_imgs = list()
        for train_label in tqdm(train_labels):
            train_img = os.path.splitext(train_label)[0] + '.png'
            train_imgs.append(train_img)
            
        print("START STEP 2/6")
        for val_label in tqdm(val_labels):
            val_img = os.path.splitext(val_label)[0] + '.png'
            val_imgs.append(val_img)
        
        # Copy labels to each train and val folder
        if self.val_type=='holdout':
            save_path = "/user/datasets/holdout"
        else:
            print("Generating holdout directory was not conducted. Please select either 'holdout' for the 'val_type' argument.")
            return
        print("START STEP 3/6")
        copying_dir = os.path.join(save_path, 'labels', 'train')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for train_label in tqdm(train_labels):
            source_path = os.path.join(label_path, train_label)
            copying_path = os.path.join(copying_dir, train_label)
            shutil.copy(source_path, copying_path)
        
        print("START STEP 4/6")
        copying_dir = os.path.join(save_path, 'labels', 'val')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for val_label in tqdm(val_labels):
            source_path = os.path.join(label_path, val_label)
            copying_path = os.path.join(copying_dir, val_label)
            shutil.copy(source_path, copying_path)
            
        # Copy images to each train and val folder
        print("START STEP 5/6")
        copying_dir = os.path.join(save_path, 'images','train')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for train_img in tqdm(train_imgs):
            source_path = os.path.join(img_path, train_img)
            copying_path = os.path.join(copying_dir, train_img)
            shutil.copy(source_path, copying_path)
        
        print("START STEP 6/6")
        copying_dir = os.path.join(save_path, 'images', 'val')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for val_img in tqdm(val_imgs):
            source_path = os.path.join(img_path, val_img)
            copying_path = os.path.join(copying_dir, val_img)
            shutil.copy(source_path, copying_path)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id = self.annotations[idx]
        
        img_filename = f'v{video_id}_f{frame}.jpg'
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        
        # Convert bbox coordinates (YOLO format)
        bboxes = [((float(bb_left) + float(bb_width) / 2) / img.width, 
                  (float(bb_top) + float(bb_height) / 2) / img.height, 
                  float(bb_width) / img.width, 
                  float(bb_height) / img.height)]
        bboxes =  torch.tensor(bboxes)
        
        if self.transform:
            image = self.transform(img)
        
        return img, bboxes, torch.tensor([class_id])

def frame_extraction(dataset_path, fps='fps=10', data='train'):
    load_path = Path(dataset_path) / 'videos' / data
    save_path = Path(dataset_path) / 'images' / data
    load_data = sorted(load_path.glob("*"))
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    #save_path.mkdir(parents=True, exist_ok=True)
    for i, video in enumerate(load_data):
        input_file = os.path.join(load_path, video)
        #input_file = video
        output_filename = f'v{i+1}_f%d.jpg'
        output_file = os.path.join(save_path, output_filename)
        #output_file = save_path / output_filename
        command = ["ffmpeg",
                   "-i", input_file,
                   "-vf", fps,
                   output_file
                   ]
        subprocess.run(command)