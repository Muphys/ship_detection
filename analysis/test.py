import argparse
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
'''
from pycocotools.coco import COCO

# COCO
coco=COCO("/home/appuser/project/data/annotations/instances_train_v2.json")
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
im = cv2.imread(f"/home/appuser/project/data/train_v2/{img['file_name']}")
'''

# register Dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("airbus_2020_train", {}, 
    "/home/appuser/project/data/annotations/instances_train_v2.json", 
    "/home/appuser/project/data/train_v2")

# config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("airbus_2020_train",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "/home/appuser/project/train/output_r50_200000_ROI512_augment/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

metaData = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
dataDict = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
predictor = DefaultPredictor(cfg)

for d in random.sample(dataDict, 200):
    im = cv2.imread(d["file_name"])
    imid = d["file_name"].split('/')[-1][:-4]
    outputs = predictor(im)

    print(outputs['instances'].pred_masks.size())

    vo = Visualizer(im[:, :, ::-1], metaData, scale=1.0)
    vt = Visualizer(im[:, :, ::-1], metaData, scale=1.0)
    truth = vt.draw_dataset_dict(d)
    cv2.imwrite(f"/home/appuser/project/analysis/figs/{imid}_truth.png", truth.get_image()[:, :, ::-1])
    out = vo.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2_imshow(out.get_image()[:, :, ::-1])
    cv2.imwrite(f"/home/appuser/project/analysis/figs/{imid}_output.png", out.get_image()[:, :, ::-1])