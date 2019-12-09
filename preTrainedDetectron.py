import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def project2D(file_path, classes, boxes, count):
	num_class = list(classes.shape)[0]
	if num_class == 0:
		return 0
	img = plt.imread(file_path)
	xyz = np.fromfile(file_path.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
	xyz = xyz.reshape([3, -1])
	print("xyzcloud shape: ", xyz.shape)
	proj = np.fromfile(file_path.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
	proj.resize([3, 4])
	uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
	uv = uv / uv[2, :]
	uv[2,:] = xyz[2,:]
	print("uv shape: ", uv.shape)
	# fig1 = plt.figure(1, figsize=(16, 9))
	fig, ax1 = plt.subplots()
	ax1.imshow(img)
	ax1.axis('scaled')
	# num_class = list(classes.shape)[0]
	print("classes ",num_class)
	depth = np.ones(num_class)*100;

	for b in range(num_class):
		projected = [];
		for i in range(uv.shape[1]):
			if (uv[0, i] > boxes.tensor[b, 0]) and (uv[0, i] < boxes.tensor[b,2]) and (uv[1, i] > boxes.tensor[b,1]) and (uv[1, i] < boxes.tensor[b,3]) :
				projected.append(uv[:,i]);

		projected = np.transpose(np.asarray(projected))
		print("projected_shape ",projected.shape)
		if projected.shape[0] != 0 :
			ax1.scatter(projected[0, :], projected[1, :], marker='.', s=1)
			depth[b] = np.mean(projected[2,:])

	plt.savefig("fig"+ str(count)+ ".png")
	print(depth)
	idx = np.argmin(depth)
	print(classes[idx])
	if depth[idx] < 50:
		return classes[idx]
	return 0




files = glob("../test/0213ada8-e776-4404-9312-c264153c57c1/*.jpg")

images = [cv2.imread(file) for file in files]


cfg = get_cfg()
cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.OUTPUT_DIR = "/tmp/detectron-output/"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

count = 0
for im in images:
	outputs = predictor(im)
	print(outputs["instances"].pred_classes)
	print(outputs["instances"].pred_boxes)
	pred_class = project2D(files[count], outputs["instances"].pred_classes, outputs["instances"].pred_boxes, count);
	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	cv2.imwrite("predict" + str(count) +".jpg",v.get_image()[:, :, ::-1])
	count  = count +1


