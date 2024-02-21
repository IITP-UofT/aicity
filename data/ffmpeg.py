import os
import subprocess

def frame_extraction(load_dir, save_dir, fps='fps=10'):
    load_path = sorted(os.listdir(load_dir))
    os.makedirs(save_dir, exist_ok=True)
    for i, video in enumerate(load_path):
        input_file = os.path.join(load_dir, video)
        output_filename = f'v{i+1}_f%d.jpg'
        output_file = os.path.join(save_dir, output_filename)
        command = ["ffmpeg",
                   "-i", input_file,
                   "-vf", fps,
                   output_file
                   ]
        subprocess.run(command)

if __name__ == "__main__":
    frame_extraction(load_dir='../datasets/raw_dataset/videos/test', 
                     save_dir='../datasets/raw_dataset/images/test')