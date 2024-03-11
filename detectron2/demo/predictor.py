# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.structures.boxes import Boxes


class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, bboxes, real_classes):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            # added arg
            bboxes (list): a N x 4 list.
                Original bbox coordinates (x, y, w, h) of selected Helmet image.
            # added arg
            real_classes (list): a N x nc list.
                Class labels of selected Helmet image.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        #===========================================================================================================#
        #===========================================================================================================#
        
        success_cnt = 0
        # Set up ROI
        for i, (bbox, real_class) in enumerate(zip(bboxes, real_classes)):
            # Initiate mask 
            mask = np.zeros_like(image)
            
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            mask[y:y+h, x:x+w] = 255
            masked_image = cv2.bitwise_and(image, mask)
            predictions = self.predictor(masked_image)
            
            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            image = image[:, :, ::-1]
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                
                    custom_cls_indices = [idx for idx, cls in enumerate(instances.pred_classes) if cls==0 or cls==3]
                    custom_cls_indices = torch.tensor(custom_cls_indices).to(torch.int)
                
                    custom_predictions = dict()
                    custom_classes_part = torch.Tensor(real_class)
                    a = torch.index_select(instances.pred_classes, dim=0, index=custom_cls_indices)
                    # bbox coordinates: (x1, y1, x2, y2)==(top_left, bottom_right)
                    custom_pred_boxes_part = torch.index_select(instances.pred_boxes.value(), dim=0, index=custom_cls_indices)
                    custom_scores_part = torch.index_select(instances.scores, dim=0, index=custom_cls_indices)
                    #custom_pred_masks_part = torch.index_select(instances.pred_masks, dim=0, index=custom_cls_indices)
                    
                    # class-bbox matching
                    if len(custom_classes_part) != len(custom_pred_boxes_part):
                        print("The number of classes and bounding boxes does not match.")
                        continue
                    
                    success_cnt += 1
                    
                    # Step 1. motorbike matching
                    y1s = custom_pred_boxes_part[:, 1]
                    y2s = custom_pred_boxes_part[:, -1]
                    max_y2, motorbike_bbox_idx = torch.max(y1s + y2s, 0)
                    
                    if custom_classes_part[motorbike_bbox_idx] != 0:
                        current_idx = torch.where(custom_classes_part==0)
                        current_idx = current_idx[0].item()
                        tmp = custom_classes_part[motorbike_bbox_idx].clone()
                        custom_classes_part[motorbike_bbox_idx] = custom_classes_part[current_idx]
                        custom_classes_part[current_idx] = tmp
                        
                    if 'custom_classes' not in locals() and 'custom_classes' not in globals():
                        custom_classes = custom_classes_part
                        custom_pred_boxes = custom_pred_boxes_part
                        custom_scores = custom_scores_part
                    else:
                        custom_classes = torch.cat([custom_classes, custom_classes_part], dim=0)
                        custom_pred_boxes = torch.cat([custom_pred_boxes, custom_pred_boxes_part], dim=0)
                        custom_scores = torch.cat([custom_scores, custom_scores_part], dim=0)
        
        class customDict:
            def __init__(self, dictionary):
                for key in dictionary:
                    setattr(self, key, dictionary[key])
                    self.dictionary = dictionary
                            
            def has(self, name: str) -> bool:
                """
                Returns:
                bool: whether the field called `name` exists.
                """
                return name in self.dictionary.keys()
        
        if 'custom_classes' in locals() or 'custom_classes' in globals():
            custom_predictions = dict()
            custom_predictions["pred_classes"] = custom_classes.to(torch.int)
            custom_predictions["pred_boxes"] = Boxes(custom_pred_boxes)
            custom_predictions["scores"] = custom_scores
            custom_predictions = customDict(custom_predictions)
        
            vis_output = visualizer.draw_instance_predictions(predictions=custom_predictions)
        
            return predictions, vis_output, success_cnt
                
        #===========================================================================================================#
        #===========================================================================================================#
        '''
        #predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                #===========================================================================================================#
                #===========================================================================================================#
                custom_cls_indices = [idx for idx, cls in enumerate(instances.pred_classes) if cls==0 or cls==3]
                custom_cls_indices = torch.tensor(custom_cls_indices)
                
                class customDict:
                    def __init__(self, dictionary):
                        for key in dictionary:
                            setattr(self, key, dictionary[key])
                        self.dictionary = dictionary
                            
                    def has(self, name: str) -> bool:
                        """
                        Returns:
                        bool: whether the field called `name` exists.
                        """
                        return name in self.dictionary.keys()
                
                custom_predictions = dict()
                custom_pred_classes = torch.index_select(instances.pred_classes, dim=0, index=custom_cls_indices)
                # bbox coordinates: (x1, y1, x2, y2)==(top_left, bottom_right)
                custom_pred_boxes = Boxes(torch.index_select(instances.pred_boxes.value(), dim=0, index=custom_cls_indices))
                custom_scores = torch.index_select(instances.scores, dim=0, index=custom_cls_indices)
                #custom_pred_masks = torch.index_select(instances.pred_masks, dim=0, index=custom_cls_indices)
                
                custom_predictions["pred_classes"] = custom_pred_classes
                custom_predictions["pred_boxes"] = custom_pred_boxes
                custom_predictions["scores"] = custom_scores
                #custom_predictions["pred_pred_masks"] = custom_pred_masks
                custom_predictions = customDict(custom_predictions)
                
                vis_output = visualizer.draw_instance_predictions(predictions=custom_predictions)
                #===========================================================================================================#
                #===========================================================================================================#
                #vis_output = visualizer.draw_instance_predictions(predictions=instances)
        
        #print(instances.pred_classes.size())
        #a = Boxes(torch.index_select(instances.pred_boxes.value(), dim=0, index=custom_cls_indices))
        #print(torch.index_select(instances.pred_boxes.value(), dim=0, index=custom_cls_indices).size())
        #print(instances.scores.size())
        #print(instances.pred_masks.size())
        #print(instances.pred_boxes[0])
        #print(instances.pred_boxes.value().size())

        return predictions, vis_output
'''
    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
