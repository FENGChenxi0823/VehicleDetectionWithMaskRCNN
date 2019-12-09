import numpy as np
from detectron2.structures import BoxMode
import itertools
from glob import glob

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer

import random
import os
import cv2


import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)

def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

# mapping labels (0~22) to (0~3)
def MappingLabels(label):
	if(label==0 or (label>=15 and label<=17) or label==22):
		label = 0
	elif(label>=1 and label<=8):
		label = 1
	elif((label>=11 and label<=13) or (label>=18 and label<=21)):
		label = 2
	elif(label==9 or label==10 or label==14):
		label = 3
	return label


# input height and width of image
def DataConvert(img_dir, height, width):
	dataset_dicts = []
	files = glob(os.path.join(img_dir, '*/*_image.jpg'))
	for idx in range(len(files)):
		snapshot = files[idx]

		record = {}
		record["file_name"] = snapshot
		record["image_id"] = idx
		record["height"] = height
		record["width"] = width

		# extract bbox and project bbox onto 2D world
		proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
		proj.resize([3, 4])
		try:
		    bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
		except FileNotFoundError:
		    print('[*] bbox not found.')
		    bbox = np.array([], dtype=np.float32)

		bbox = bbox.reshape([-1, 11])
		objs = []
		for k, b in enumerate(bbox):
		    R = rot(b[0:3])
		    t = b[3:6]

		    sz = b[6:9]
		    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
		    vert_3D = R @ vert_3D + t[:, np.newaxis]

		    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
		    vert_2D = vert_2D / vert_2D[2, :]

		    px = vert_2D[0]
		    py = vert_2D[1]
		    poly = [(x, y) for x, y in zip(px, py)]
		    poly = list(itertools.chain.from_iterable(poly))

		    obj = {
		        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
		        "bbox_mode": BoxMode.XYXY_ABS,
		        "segmentation": [poly],
		        "category_id": MappingLabels(b[9]),
		        "iscrowd": 0
		    }
		    objs.append(obj)

		record["annotations"] = objs
		dataset_dicts.append(record)
	return dataset_dicts


img_width = 1914;
img_height = 1052;
for d in ["trainval", "test"]:
    DatasetCatalog.register(d, lambda d=d: DataConvert("../" + d, img_height, img_width))
    MetadataCatalog.get(d).set(thing_classes=["unkown", "car", "truck", "person"])
vehicle_metadata = MetadataCatalog.get("trainval")

dataset_dicts = DataConvert("../trainval", img_height, img_width)
print(len(dataset_dicts))

count=0;
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    height, width, channels = img.shape
    print(height, width)
    visualizer = Visualizer(img[:, :, ::-1], metadata=vehicle_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # print(d)
    cv2.imwrite("predict"+ str(count)+ ".jpg",vis.get_image()[:, :, ::-1])
    count = count +1


cfg = get_cfg()
cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainval",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.MODEL.DEVICE = "cpu"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # we have four classes

cfg.OUTPUT_DIR = "/home/chenxif/Documents/self_driving_car/FinalProject/Perception/detectron-output/"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()