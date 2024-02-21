# [2024 AI CITY CHALLENGE](https://www.aicitychallenge.org/) 


[Minseok Kong](https://kongminseok.github.io/), Jeongae Lee, Jiho Park, Jinhee Park


## Intro
### Track 5: Detecting Violation of Helmet Rule for Motorcyclists

Motorcycles are one of the most popular modes of transportation, particularly in developing countries such as India. Due to lesser protection compared to cars and other standard vehicles, motorcycle riders are exposed to a greater risk of crashes. Therefore, wearing helmets for motorcycle riders is mandatory as per traffic rules and automatic detection of motorcyclists without helmets is one of the critical tasks to enforce strict regulatory traffic safety measures.

- **Data**

The training dataset contains 100 videos and groundtruth bounding boxes of motorcycle and motorcycle rider(s) with or without helmets. Each video is 20 seconds duration, recorded at 10 fps. The video resolution is 1920×1080.

Each motorcycle in the annotated frame has bounding box annotation of each rider with or without helmet information, for upto a maximum of 4 riders in a motorcycle. The class id (labels) of the object classes in this dataset is as follows:

```
1, motorbike: bounding box of motorcycle
2, DHelmet: bounding box of the motorcycle driver, if he/she is wearing a helmet
3, DNoHelmet: bounding box of the motorcycle driver, if he/she is not wearing a helmet
4, P1Helmet: bounding box of the passenger 1 of the motorcycle, if he/she is wearing a helmet
5, P1NoHelmet: bounding box of the passenger 1 of the motorcycle, if he/she is not wearing a helmet
6, P2Helmet: bounding box of the passenger 2 of the motorcycle, if he/she is wearing a helmet
7, P2NoHelmet: bounding box of the passenger 2 of the motorcycle, if he/she is not wearing a helmet
8, P0Helmet: bounding box of the child sitting in front of the Driver of the motorcycle, if he/she is wearing a helmet
9, P0NoHelmet: bounding box of the child sitting in front of the Driver of the motorcycle, if he/she is wearing not a helmet
```
## Training
### STEP 1 Frame Extraction

Submissions for track 5 require frame IDs for frames that contain information of interest. We use the [FFmpeg library](https://www.ffmpeg.org/) to extract/count frames to ensure frame IDs are consistent across teams.
```
cd data
python ffmpeg.py
```
### STEP 2 Data preprocessing
```
python data.py
```
### STEP 3 Training
```
cd ..
python yolov8.py
```
### Directory Layout
```
aicity
|── cfg
|   └── yolov8.yaml
├── data
|   |── data_analysis.ipynb
|   |── data.py
|   └── ffmpeg.py
├── datasets
|   |── raw_dataset
|   |   |── images
|   |   |   |── train
|   |   |   └── test
|   |   |── labels
|   |   |   └──  train
|   |   |── videos
|   |   |   |── train
|   |   |   └── test
|   |   |── gt.txt
|   |   └── labels.txt
|   └── yolo_dataset
|       |── images
|       |   |── train
|       |   |── test 
|       |   └── val
|       └── labels 
|           |── train
|           └── val    
└── yolov8.py
```
## Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT).
