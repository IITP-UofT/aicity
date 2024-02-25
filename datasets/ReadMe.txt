%********************************************************************************************************%
% The helmet violation detection dataset is captured by cameras in real-world traffic surveillance environments of an Indian city.             %
% The training dataset consists of 100 videos, 20 seconds duration each..                                                                    %
%********************************************************************************************************%

Content in the directory:
1. "videos/". This dir contains 100 videos for training. The files are named as 001.mp4, 002.mp4, …, 100.mp4
2. "gt.txt". This file contains ground truth bounding boxes for all training dataset videos. It contains one object instance per line and values are comma separated. The schema is as follows:
<video_id>, <frame>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <class>
3. "labels.txt". This file contains the object class labels used in the dataset.
4. “rider_position_encoding.jpg”: This contains the rider position in the motorcycle used as labels for the bounding box. “D” denotes driver, “P1” denotes passenger 1, “P2” denotes passenger 2, and "P0" denotes passenger sitting in front of the driver, if any.
