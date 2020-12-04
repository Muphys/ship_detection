# import some common libraries
import os
import numpy as np
import cv2
import random
import time

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
local_time = time.localtime()
log_file = ''
for i in local_time[:5]: log_file += str(i)
setup_logger(output=f"/home/appuser/project/train/log/{log_file}.log")

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetMapper
import detectron2.data.transforms as T

# register Dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("airbus_2020_train", {}, 
    "/home/appuser/project/data/annotations/instances_train_v2.json", 
    "/home/appuser/project/data/train_v2")
register_coco_instances("airbus_2020_validation", {}, 
    "/home/appuser/project/data/annotations/instances_train_v2.json", 
    "/home/appuser/project/data/train_v2")

'''
# config
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("airbus_2020_train",)
cfg.DATASETS.TEST = ()
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") # Let training initialize from model zoo
cfg.MODEL.WEIGHTS = "/home/appuser/project/model/pretrained/mask_rcnn_X_101_32x8d_FPN_3x.pkl"

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 180000
cfg.SOLVER.STEPS = (0, 120000, 160000)
cfg.SOLVER.CHECKPOINT_PERIOD = 10000
cfg.SOLVER.GAMMA = 0.1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.NUM_GPUS = 1
cfg.TEST.EVAL_PERIOD = 500
cfg.OUTPUT_DIR = "/home/appuser/project/train/output"
'''

# config
cfg = get_cfg()

#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("airbus_2020_train",)
cfg.DATASETS.TEST = ()
#cfg.DATASETS.TEST = ("airbus_2020_train",)

#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = "/home/appuser/project/model/pretrained/mask_rcnn_X_101_32x8d_FPN_3x.pkl"
#cfg.MODEL.WEIGHTS = "/home/appuser/project/train/output/model_final.pth"
cfg.MODEL.WEIGHTS = "/home/appuser/project/model/pretrained/mask_rcnn_R_50_FPN_1x.pkl"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
#cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 200000
cfg.SOLVER.STEPS = (120000, 170000)
#cfg.SOLVER.MAX_ITER = 20000
#cfg.SOLVER.STEPS = (40000, 60000)
cfg.SOLVER.CHECKPOINT_PERIOD = 10000
cfg.SOLVER.GAMMA = 0.1

#cfg.NUM_GPUS = 1
cfg.TEST.EVAL_PERIOD = 5000
cfg.OUTPUT_DIR = "/home/appuser/project/train/output_r50_200000_ROI512_augment"

# train
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataloader = build_detection_train_loader(cfg,
            mapper=DatasetMapper(cfg, is_train=True, augmentations=[
                T.RandomApply(tfm_or_aug=T.RandomBrightness(0.8, 1.2), prob=0.2),
                T.RandomApply(tfm_or_aug=T.RandomContrast(0.8, 1.2), prob=0.1),
                T.RandomFlip(prob=0.25, horizontal=True, vertical=False),
                T.RandomRotation(angle=[0, 90, 180, 270], expand=True, center=None, sample_style='choice', interp=None)
            ]))
        return dataloader

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()