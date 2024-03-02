import os
import time
import shutil
import pandas as pd
import warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings('ignore')


# Extract (8, P0Helmet), (6, P2Helmet), (9, P0NoHelmet), (7, P2NoHelmet), (4, P1Helmet)
def helmet_extraction(dataset_path, save=True, seed=0):
    start = time.time()
    annotation_path = Path(dataset_path) / 'helmet' / 'annotation'
    annotations = sorted(annotation_path.glob("*.csv"))
    df = pd.DataFrame()
    for ann in annotations:
        video_id = ann.name.split('.')[0]
        #print(video_id)
        df_part = pd.read_csv(ann, encoding='utf-8')
        df_part.drop('track_id', axis=1, inplace=True)
        df_part['video_id'] = video_id
        df = pd.concat([df, df_part], axis=0, ignore_index=True)
        
    #print(f"P0Helmet Count: {len(df[df['label'].str.contains('P0Helmet')])}")
    #print(f"P2Helmet Count: {len(df[df['label'].str.contains('P2Helmet')])}")
    #print(f"P0NoHelmet Count: {len(df[df['label'].str.contains('P0NoHelmet')])}")
    #print(f"P2NoHelmet Count: {len(df[df['label'].str.contains('P2NoHelmet')])}")
    #print(f"P1Helmet Count: {len(df[df['label'].str.contains('P1Helmet')])}")
    
    # filtered_df = df[df['label'].str.contains('P2Helmet|P0Helmet|P0NoHelmet|P2NoHelmet|P1Helmet')]
    df_p0h = df[df['label'].str.contains('P0Helmet')]
    df_p2h = df[df['label'].str.contains('P2Helmet') & ~df.index.isin(df_p0h.index)]
    df_p0nh = df[df['label'].str.contains('P0NoHelmet') & ~df.index.isin(df_p0h.index) & ~df.index.isin(df_p2h.index)]
    df_p2nh = df[df['label'].str.contains('P2NoHelmet') & ~df.index.isin(df_p0h.index) & ~df.index.isin(df_p2h.index) & ~df.index.isin(df_p0nh.index)]
    df_p1h = df[df['label'].str.contains('P1Helmet') & ~df.index.isin(df_p0h.index) & ~df.index.isin(df_p2h.index) & ~df.index.isin(df_p0nh.index) & ~df.index.isin(df_p2nh.index)]
    
    # step 0. extract (8, P0Helmet)
    print(f"length of df_p0h: {len(df_p0h)}")
    
    # step 1. extract len(df_p0h) of (6, P2Helmet)
    new_df_p2h = pd.DataFrame()
    vid_uq = df_p2h['video_id'].value_counts().sort_values(ascending=True).index
    escape = False
    while not escape:
        for vid in vid_uq:
            if len(new_df_p2h) >= len(df_p0h):
                escape = True
                break
            if df_p2h[df_p2h['video_id']==vid].empty:
                continue
            row = df_p2h[df_p2h['video_id']==vid].sample(n=1, random_state=seed)
            new_df_p2h = pd.concat([new_df_p2h, row], axis=0)
            df_p2h.drop(row.index, axis=0, inplace=True)
    print(f"length of new_df_p2h: {len(new_df_p2h)}")
            
    # step 2. extract len(df_p0h) of (9, P0NoHelmet)
    new_df_p0nh = pd.DataFrame()
    vid_uq = df_p0nh['video_id'].value_counts().sort_values(ascending=True).index
    escape = False
    while not escape:
        for vid in vid_uq:
            if len(new_df_p0nh) >= len(df_p0h):
                escape = True
                break
            if df_p0nh[df_p0nh['video_id']==vid].empty:
                continue
            row = df_p0nh[df_p0nh['video_id']==vid].sample(n=1, random_state=seed)
            new_df_p0nh = pd.concat([new_df_p0nh, row], axis=0)
            df_p0nh.drop(row.index, axis=0, inplace=True)          
    print(f"length of new_df_p0nh: {len(new_df_p0nh)}")
    
    # step 3. extract len(df_p0h) of (7, P2NoHelmet)
    new_df_p2nh = pd.DataFrame()
    vid_uq = df_p2nh['video_id'].value_counts().sort_values(ascending=True).index
    escape = False
    while not escape:
        for vid in vid_uq:
            if len(new_df_p2nh) >= len(df_p0h):
                escape = True
                break
            if df_p2nh[df_p2nh['video_id']==vid].empty:
                continue
            row = df_p2nh[df_p2nh['video_id']==vid].sample(n=1, random_state=seed)
            new_df_p2nh = pd.concat([new_df_p2nh, row], axis=0)
            df_p2nh.drop(row.index, axis=0, inplace=True)          
    print(f"length of new_df_p2nh: {len(new_df_p2nh)}")
    
    # step 4. extract len(df_p0h) of (4, P1Helmet)
    new_df_p1h = pd.DataFrame()
    vid_uq = df_p1h['video_id'].value_counts().sort_values(ascending=True).index
    escape = False
    while not escape:
        for vid in vid_uq:
            if len(new_df_p1h) >= len(df_p0h):
                escape = True
                break
            if df_p1h[df_p1h['video_id']==vid].empty:
                continue
            row = df_p1h[df_p1h['video_id']==vid].sample(n=1, random_state=seed)
            new_df_p1h = pd.concat([new_df_p1h, row], axis=0)
            df_p1h.drop(row.index, axis=0, inplace=True)          
    print(f"length of new_df_p1h: {len(new_df_p1h)}")
    
    final_df = pd.concat([df_p0h, new_df_p2h, new_df_p0nh, new_df_p2nh, new_df_p1h], 
                         axis=0, 
                         ignore_index=True).sort_values(by=['video_id', 'frame_id'])
    final_df = pd.concat([final_df['video_id'], final_df.iloc[:, :-1]], axis=1)
    
    if final_df.duplicated().sum() != 0:
        print("Duplicated rows exist.")
        return
    
    if save:
        save_path = os.path.join(dataset_path, 'helmet' ,'additional_datalist.csv')
        final_df.to_csv(save_path, encoding='utf-8',index=False)
        print("The save has been completed.")
    
    img_path = Path(dataset_path) / 'helmet' / 'image'
    copying_dir = Path(dataset_path) / 'helmet' / 'selected_image'
    if os.path.exists(copying_dir):
        shutil.rmtree(copying_dir)
    copying_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in final_df.iterrows():
        video_id = row.video_id
        # Format 'frame_id' as a two-digit string.
        frame_id = str(row.frame_id).zfill(2)
        label = row.label
        source_img = img_path / video_id / f'{frame_id}.jpg'
        if not source_img.exists():
            print(f"{str(source_img)} does not exist.")
            continue
        copying_path = copying_dir / f'v{video_id}_f{frame_id}.jpg'
        if copying_path.exists():
            print(f"{str(copying_path)} already exist.")
        shutil.copy(source_img, copying_path)
    end = time.time()
    print(f"Running Time: {int((end-start)//60)}m {int((end-start)%60)}s")
        
if __name__=="__main__":
    helmet_extraction("/home/kongminseok/aicity/datasets")