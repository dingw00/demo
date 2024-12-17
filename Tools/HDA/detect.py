#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import random
import numpy as np


import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from Tools.HDA.yolo_utils import non_max_suppression
from Tools.Models.utils import load_model
from Tools.Utils.dataloader import load_dataset

def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5, device="cpu"):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode

    # Configure input
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    input_img = input_img.to(device)

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()


def detect_batch(model, batch, img_size=416, conf_thres=0.5, nms_thres=0.5):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    model.eval()  # Set model to evaluation mode
    # Get detections
    with torch.no_grad():
        detections = model(batch)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        # detections = rescale_boxes(detections[0], img_size, batch[0].shape[:2])
    return detections

def detect_one(model, dataloader, conf_thres=0.5, nms_thres=0.5, device="cpu"):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # Create output directory, if missing
    # os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode
    model.device

    Tensor = torch.cuda.FloatTensor if (str(device) == "cuda") else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    # for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
    #     # Configure input
    #     input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
    with torch.no_grad():
        detections = model(dataloader)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    # Store image and detections
    # img_detections.extend(detections)
    # imgs.extend(img_paths)
    return img_detections, imgs



