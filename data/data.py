import torch
import argparse
import time
import shutil
import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, args: dict, transform=None):
        self.annotations = self.read_txt_file(args.txt_file)
        self.img_dir = args.img_dir
        self.label_dir = args.label_dir
        self.yolo_dir = args.yolo_dir
        self.num_classes = args.num_classes
        self.transform = transform
        
    def read_txt_file(self, txt_file):
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        annotations = [line.strip().split(',') for line in lines]
        return annotations
    
    def convert_data_to_yolo_v1(self):
        # To generate label directory
        if os.path.exists(self.label_dir):
            shutil.rmtree(self.label_dir)
            os.mkdir(self.label_dir)
        else:
            os.mkdir(self.label_dir)
        
        for ann in tqdm(self.annotations):
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id = ann
            img_filename = f'v{video_id}_f{frame}.jpg'
            img_path = os.path.join(self.img_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"No such file or directory: {img_path}")
                continue
            
            img = Image.open(img_path).convert("RGB")
        
            # Convert bbox coordinates (YOLO format)
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
    
    # To alleviate long-tail problem
    def convert_data_to_yolo_v2(self):
        # To generate label directory
        if os.path.exists(self.label_dir):
                shutil.rmtree(self.label_dir)
                os.mkdir(self.label_dir)
        else:
            os.mkdir(self.label_dir)
        
        img_duplicate = {}
        for ann in tqdm(self.annotations):
            video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id = ann
            img_filename = f'v{video_id}_f{frame}.jpg'
            img_path = os.path.join(self.img_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"No such file or directory: {img_path}")
                continue
            
            # One-to-one correspondence b/w label and image.
            img = Image.open(img_path).convert("RGB")
            if img_duplicate.get(img_path, False):
                img_filename = f'v{video_id}_f{frame}_{img_duplicate[img_path]}.jpg'
                new_img_path = os.path.join(self.img_dir, img_filename)
                img.save(new_img_path, "JPEG")
                img = Image.open(new_img_path).convert("RGB")
                img_duplicate[img_path] += 1
            else:
                img_duplicate[img_path] = 1
            
            # Convert bbox coordinates (YOLO format)
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
            with open(txt_path, 'w') as file:
                file.write(yolo_format + '\n')
    
    def train_val_split(self):
        dic = dict()
        for i in range(self.num_classes):
            dic[i] = list()
        
        txt_files = [f for f in os.listdir(self.label_dir) if os.path.isfile(os.path.join(self.label_dir, f))]
        print("START STEP 1/8")
        for txt_file in tqdm(txt_files):
            with open(os.path.join(self.label_dir, txt_file), 'r') as file:
                line = file.readline()
                class_id, _, _, _, _ = line.split()
            dic[int(class_id)].append(txt_file)
        
        print("START STEP 2/8")
        for i, (k, v) in enumerate(tqdm(dic.items())):
            random.seed(42)
            random.shuffle(v)
            train_ratio = 0.8
            split_idx = int(len(v) * train_ratio)
            if i==0:
                train_labels = v[:split_idx]
                val_labels =v[split_idx:]
            else:
                train_labels.extend(v[:split_idx])
                val_labels.extend(v[split_idx:])
        
        print("START STEP 3/8")
        train_imgs = list()
        val_imgs = list()
        for train_label in tqdm(train_labels):
            train_img = os.path.splitext(train_label)[0] + '.jpg'
            train_imgs.append(train_img)
            
        print("START STEP 4/8")
        for val_label in tqdm(val_labels):
            val_img = os.path.splitext(val_label)[0] + '.jpg'
            val_imgs.append(val_img)
        
        # Copy labels to each train and val folder
        print("START STEP 5/8")
        copying_dir = os.path.join(self.yolo_dir, 'labels', 'train')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for train_label in tqdm(train_labels):
            source_path = os.path.join(self.label_dir, train_label)
            copying_path = os.path.join(copying_dir, train_label)
            shutil.copy(source_path, copying_path)
        
        print("START STEP 6/8")
        copying_dir = os.path.join(self.yolo_dir, 'labels', 'val')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for val_label in tqdm(val_labels):
            source_path = os.path.join(self.label_dir, val_label)
            copying_path = os.path.join(copying_dir, val_label)
            shutil.copy(source_path, copying_path)
            
        # Copy images to each train and val folder
        print("START STEP 7/8")
        copying_dir = os.path.join(self.yolo_dir, 'images','train')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for train_img in tqdm(train_imgs):
            source_path = os.path.join(self.img_dir, train_img)
            copying_path = os.path.join(copying_dir, train_img)
            shutil.copy(source_path, copying_path)
        
        print("START STEP 8/8")
        copying_dir = os.path.join(self.yolo_dir, 'images', 'val')
        if os.path.exists(copying_dir):
            shutil.rmtree(copying_dir)
        os.makedirs(copying_dir)
        for val_img in tqdm(val_imgs):
            source_path = os.path.join(self.img_dir, val_img)
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


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_file", type=str, default="../datasets/raw_dataset/gt.txt", help="annotation txt file")
    parser.add_argument("--img_dir", type=str, default="../datasets/raw_dataset/images/train", help="directory for loading images")
    parser.add_argument("--label_dir", type=str, default="../datasets/raw_dataset/labels/train", help="directory for saving labels")
    parser.add_argument("--yolo_dir", type=str, default="../datasets/yolo_dataset", help="directory for saving converted data")
    parser.add_argument("--num_classes", type=int, default=9, help="the number of classes")
    args = parser.parse_args()
    custom_dataset = CustomDataset(args)
    custom_dataset.convert_data_to_yolo_v2()
    custom_dataset.train_val_split()
    end = time.time()
    print(f"Running Time: {int((end-start)//60)}m {int((end-start)%60)}s")

if __name__== "__main__" :
    main()
        