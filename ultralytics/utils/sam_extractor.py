"""
This file contains utility functions and a class for extracting object-oriented bounding boxes (OBB) from images using the Ultralytics library.

Functions:
- get_enclosing_points(mask: torch.Tensor) -> np.ndarray: Accepts a mask tensor and returns the 4 points of the enclosing rectangle as a numpy array.
- get_point_prompts(lb_path, normalized): Accepts a label file path and a flag indicating whether the points are normalized, and returns the denormalized points and labels.
- xyxyxyxy_to_xyxy(box: list) -> list: Converts a bbox list from xyxyxyxy format to xyxy format.
- get_box_prompts(lb_path) -> tuple[Bboxes, list]: Reads the label file and returns the bounding boxes and labels.
- write_bboxes_to_file(bboxes: np.ndarray, labels: list, path: str, format: str = "xyxy"): Writes the bounding boxes and labels to a file.
- get_img_size(img_path: str) -> tuple[int, int]: Returns the size of an image.
- check_if_inside(point, box): Checks if a point is inside a box.
- associate_points_to_boxes(points, boxes, og_labels, no_match_class): Associates boxes with the first matching original label.
- get_dataset_info(): Retrieves information about the dataset.
- process(idxs=None): Processes the dataset and extracts OBBs.

Class:
- DatasetOBBExtractor: A class for extracting OBBs from a dataset using a model.

"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import yaml
from typing import Union
import numpy
from ultralytics import SAM
from ultralytics.utils.instance import Bboxes, Instances
from ultralytics.utils.ops import xyxyxyxy2xywhr
from ultralytics.data.split_dota import load_yolo_dota

import os
import shutil

def format_patches_for_image_classification(base_dir, output_dir, move=False):
    # Discover all files in the folder
    files = os.listdir(base_dir)
    assert(len(files)), f"No files to move in {base_dir}"
    # Extract class number from files
    for file in files:
        class_num = file.split("_")[-1].split(".")[0]
        class_dir = f"{output_dir}/class_{class_num}"
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        if move:
            shutil.move(f"{base_dir}/{file}", f"{class_dir}/{file}")
        else:
            shutil.copy(f"{base_dir}/{file}", f"{class_dir}/{file}")

class Points:
    def __init__(self, points: Union[list[list[int]],np.ndarray], normalized):
        "Accepts a list of list of xyxy points or an ndarray, and a boolean normalized flag"
        if isinstance(points, np.ndarray):
            assert points.ndim == 2, "The points must be a 2D array"
            assert points.shape[1] == 2, "The points must have 2 columns"
            # Reshape nx2 array to list of tuples
            self.points = points.tolist()
        else:
            assert(isinstance(points, list)), "The points must be a list"
            assert(all(len(point) == 2 for point in points)), "Each point must be a list of 2 elements"
            self.points = points
        self.normalized = normalized
    def denormalize(self, w,h):
        # self.points is a list of list
        " Denormalize the points"
        if self.normalized:
            self.points = [[int(x * w), int(y * h)] for x, y in self.points]
            self.normalized = False
    def normalize(self, w, h):
        if not self.normalized:
            self.points = [[x / w, y / h] for x, y in self.points]
            self.normalized = True
    def asarray(self):

        return np.array(self.points)

_formats = ["xyxyxyxy"]
_dims = {"xyxyxyxy": 8}

class OBBox:
    def __init__(self, points: np.ndarray, format, normalized) -> None:
        # Assert format in xyxyxyxy
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        _points = points[None, :] if points.ndim == 1 else points
        assert _points.ndim == 2
        assert _points.shape[1] == _dims[format], f"Invalid bounding box shape: {_points.shape}, must be (N, {_dims[format]})"
        if format == "xyxyxyxy":
            self._points = Points(_points.reshape(-1,2), normalized)
            self.points_callback = lambda x : x.asarray().reshape(-1,8)
        self.format = format
        self.normalized = normalized
    def normalize(self, w,h):
        if not self.normalized:
            self._points.normalize(w, h)
            self.normalized = True
    def denormalize(self, w,h):
        if self.normalized:
            self._points.denormalize(w, h)
            self.normalized = False

    @property
    def points(self):
        return self.points_callback(self._points)

def get_enclosing_points(mask: torch.Tensor) ->  Union[np.ndarray, bool]:
    "Accepts a mask tensor HW of type bool, and returns the 4 points of the enclosing rectangle as nd.array"
    # Convert the mask to a numpy array if it's a PyTorch tensor
    if isinstance(mask, torch.Tensor):
         mask = (mask.cpu().numpy() * 255).astype(np.uint8)
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the contour with the maximum area (in case there are multiple contours)
    if len(contours):
        max_contour = max(contours, key=cv2.contourArea)

        # Get a rotated rectangle that encloses the contour
        rect = cv2.minAreaRect(max_contour)

        # Get the 4 points of the rectangle
        points = cv2.boxPoints(rect)

        return points.reshape(1,-1)
    else:
        return False

def get_point_prompts(lb_path, normalized):
    "Accepts lb_path containing labels, each labels is a line with xyxy box coordinates"
    # Read the file
    with open(lb_path) as file:
        lines = file.readlines()
    # Parse the lines
    points = []
    labels = []
    for line in lines:
        components = line.split()
        class_idx = int(components[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, components[1:])
    # Calculate the center of the box
        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4

        points.append([center_x, center_y])
        labels.append(class_idx)
    return Points(points, normalized), labels



def xyxyxyxy_to_xyxy(box: list)-> list:
    " Convert 1x8 bbox list from xyxyxyxy to xyxy list"
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    y = [x_min, y_min, x_max, y_max]
    return y

def get_box_prompts(lb_path) -> tuple[Bboxes, list]:
    " Get the boxes and labels from the label file, label is in the YOLO DET format,i.e bottom left, top right: class_idx, x1, y1, x2, y2"
    with open(lb_path) as file:
        lines = file.readlines()
    # Open the image

    # Parse the lines
    boxes = []
    labels = []
    for line in lines:
        components = line.split()
        class_idx = int(components[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, components[1:])
        boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        labels.append(class_idx)
    xyxy_boxes = list(map(xyxyxyxy_to_xyxy, boxes))
    bboxes = Bboxes(
        bboxes=np.array(xyxy_boxes),
        format="xyxy",
    )
    return bboxes, labels

def write_bboxes_to_file(bboxes: np.ndarray, labels: list, path: str, format: str = "xyxy"):
    " Write the bboxes and labels to a file, bboxes is a N,X array depending on the format, labels is a list of class indices, path is the output file path to save the labels"
    assert len(bboxes) == len(labels), "The number of bboxes and labels must be the same"
    with open(path, "w") as file:
        for box, label in zip(bboxes, labels):
            match format:
                case "xyxy":
                    assert len(box) == 4, "The box must be in the format x1, y1, x2, y2"
                case "xywh":
                    assert len(box) == 4, "The box must be in the format x, y, w, h"
                case "xyxyxyxy":
                    assert len(box) == 8, "The box must be in the format x1, y1, x2, y2, x3, y3, x4, y4"
                case "xywhr":
                    assert len(box) == 5, "The box must be in the format x, y, w, h, r"
            formatted_coords = ["{:.6f}".format(float(coord)) for coord in box]
            file.write(f"{int(label)} {' '.join(formatted_coords)}\n")

def get_img_size(img_path: str) -> tuple[int, int]:
    img = Image.open(img_path)
    return img.size


def check_if_inside(point, box):
    " Box can be in xyxy format or xyxyxyxy format, point is a tuple of x,y coordinates"
    if len(box) == 4: # xyxy format
        x, y = point
        x1, y1, x2, y2 = box
        is_inside = x1 <= x <= x2 and y1 <= y <= y2
    elif len(box) == 8: # xyxyxyxy format
        x, y = point
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        is_inside = cv2.pointPolygonTest(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), (x, y), False) >= 0
    else:
        raise NotImplementedError
    return is_inside


def associate_points_to_boxes(points,boxes, og_labels, no_match_class):
                " Associate boxes to first matching OG label"
                labels_ = []
                for box in boxes:
                    for i,point in enumerate(points):
                        if check_if_inside(point, box):
                            labels_.append(og_labels[i])
                            break
                    else:
                        labels_.append(no_match_class)
                return labels_

def sort_points(points):
    """
    Sort a list of 4 points in the order: top-left, top-right, bottom-right, bottom-left.
    Each point is a tuple containing the x and y coordinates.
    """
    # Sort the points based on y-coordinate (ascending) and then x-coordinate (ascending)
    points.sort(key=lambda point: (point[1], point[0]))

    # The points are now in the order: top-left, bottom-left, top-right, bottom-right
    # Swap the middle two points to get the desired order
    points[1], points[2] = points[2], points[1]
    points[3], points[2] = points[2], points[3]

    return points


# function copy-pasted from https://stackoverflow.com/a/14178717/744230
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float32)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


def crop_quadrilateral(image, points):
    """
    Crop a quadrilateral defined by `points` from `image`.
    `points` is a list of 4 tuples, each containing the x and y coordinates of a point.
    The points are sorted automatically.
    """
    # Sort the points
    points = sort_points(points) #  top-left, top-right, bottom-right, bottom-left.

    # Width is euclidean distance between top-left and topright points
    width = numpy.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
    # Height is euclidean distance between top-left and bottom-left points
    height = numpy.sqrt((points[3][0] - points[0][0])**2 + (points[3][1] - points[0][1])**2)

    # output_size = (int(width), int(height))

    # output_points = [(0, 0), (width, 0), (width, height), (0, height)] #  top-left, top-right, bottom-right, bottom-left.
    # try:
    #     coeffs = find_coeffs(
    #     output_points,
    #     points)
    # except numpy.linalg.LinAlgError: # Singula rmatrix in coeffs calculation
    #     return None
    #transform = image.transform(output_size, Image.PERSPECTIVE, coeffs)

    # Crop image based on min max coordinates of points
    left, upper, right, lower = min(points, key=lambda x: x[0])[0], min(points, key=lambda x: x[1])[1], max(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[1])[1]
    transform = image.crop((left, upper, right, lower))

    return transform

def crop_image(im_path, lb_path, output_dir):
    """
    Crop the image based on the bounding box coordinates provided in the label file.

    Args:
        im_path (str): The path to the input image file.
        lb_path (str): The path to the label file containing bounding box coordinates (clsxyxyxyxy).
        output_dir (str): The directory to save the cropped images.

    Returns:
        None
    """
    output_imgs = []

    with open(lb_path) as file:
        lines = file.readlines()
        if not lines:
            print("Label file is empty.")
            return output_imgs

    for i, line in enumerate(lines):
        components = line.split()
        class_idx = int(components[0])
        boxes = np.asarray(list(map(float, components[1:])))
        boxes = boxes.reshape(-1, 2)
        # Get the points as pairs of two from box
        points = Points([box for box in boxes], normalized=True)
        # Open the image
        img = Image.open(im_path)
        w,h = img.size
        points.denormalize(w=w, h=h)
        # Crop the image
        crop = crop_quadrilateral(img, [tuple(_) for _ in points.points])
        if crop:
            output_filename = f"{output_dir}/{str(Path(im_path).name)}_crop{i}_class_{class_idx}.jpg"
            # Save the cropped image
            crop.save(output_filename)
            output_imgs.append(output_filename)

    return output_imgs
class DatasetOBBExtractor:
    def __init__(self, model, dataset_dir, output_dir, default_class=None, debug=False, yaml_cfg=None, keep_only_classes=False): # Ship classes for xView
        self.model = model
        self.keep_only_classes = keep_only_classes
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        if not yaml_cfg:
            #Read classes names from ultralytics/cfg/datasets/xView-patches.yaml as default
            cfg_path = Path("../ultralytics/cfg/datasets/xView-patches-ship-sam.yaml")
            with open(cfg_path) as file:
                cfg = yaml.safe_load(file)
        elif isinstance(yaml_cfg, dict):
            # Use provided object as cfg
            cfg = yaml_cfg
        elif isinstance(yaml_cfg, str):
            cfg_path = Path(yaml_cfg)
            with open(cfg_path) as file:
                cfg = yaml.safe_load(file)
        else:
            raise NotImplementedError(f"Something is wrong with the data configuration, not a str, not a dict {cfg}")
        self.debug = debug
        self.CLASS_NAMES = cfg["names"]
        self.IDX_NAMES = {v: k for k, v in self.CLASS_NAMES.items()}
        self.default_class = default_class
        if self.default_class:
            assert self.default_class in self.IDX_NAMES.keys(), f"The default class {self.default_class} is not in the class names"

        # Create output dirs
        if self.output_dir:
            (Path(output_dir) / Path('labels') / Path('train') ).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / Path('labels') / Path('val') ).mkdir(parents=True, exist_ok=True)
            if self.debug:
                (Path(output_dir) / Path('debug') / Path('train') ).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / Path('debug') / Path('val') ).mkdir(parents=True, exist_ok=True)
        # Check if dataset_dir exists
        assert Path(dataset_dir).exists(), f"The dataset_dir {dataset_dir} does not exist"

    def extract_patches(self, idxs, output_dir):
        # Create output_dir if does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        " Extract patches from the dataset"
        dataset = self.data_info
        crop_fpaths = []
        if idxs:
            # Grab only indexes of interest (might be not useful because empty labels)
            dataset = [dataset[i] for i in idxs]
        for data in tqdm(dataset):
            # labels = data["label"]
            lb_path = data["filepath"].replace("images", "labels").replace("jpg", "txt")
            im_path = data["filepath"]
            croped_image_segments = crop_image(im_path, lb_path, output_dir)
            crop_fpaths.append(croped_image_segments)
        return crop_fpaths
    def get_dataset_info(self,fast=False, splits=["train"]):
        " Get the dataset information, filter out empty labels"
        self.data_info = []
        for split in splits:
            self.data_info += load_yolo_dota(self.dataset_dir,split=split, fast=fast)
        self.data_info = list(filter(lambda x: len(x["label"]) > 0, self.data_info))
    def process(self, idxs=None):
        dataset = self.data_info
        if idxs:
            # Grab only indexes of interest (might be not useful because empty labels)
            dataset = [dataset[i] for i in idxs]
        for data in tqdm(dataset):
            lb_path = data["filepath"].replace("images", "labels").replace("jpg", "txt")
            points, labels = get_point_prompts(lb_path=lb_path, normalized=True)
            xyxyboxes, labels = get_box_prompts(lb_path)
            labels_arr = np.array(labels)
            # Keep only ship points
            if self.keep_only_classes:
                labels_reduced = [label for label in labels if label in self.keep_only_classes] # Hard coded ship classes
                points.points = [point for point, label in zip(points.points, labels) if label in self.keep_only_classes]
                xyxyboxes.bboxes = xyxyboxes.bboxes[np.where(np.isin(labels_arr, self.keep_only_classes))]
            else:
                labels_reduced = labels
            # Keep only ship boxes, ie  where label is in self.keep_only_classes list


            # If no objects of interest in this image, skip.
            if len(labels_reduced) < 1:
                print(f"Skipping {lb_path} because no objects of interest")
                continue

            w,h = data["ori_size"]
            # Px coordinates of point queries
            points.denormalize(w=w, h=h)
            # Normalized box queries
            instances = Instances(bboxes=xyxyboxes.bboxes, bbox_format="xyxy", normalized=True)
            # Px box queries
            instances.denormalize(w=w, h=h)
            # Probe SAM
            res=self.model(data["filepath"], bboxes=instances.bboxes, points=points.points, labels=labels,verbose=False)
            # Assert the length of results is 1
            assert len(res) == 1, "The model should return only one result"
            r = res[0]

            # Get boxes from masks XXX: Some masks are sometimes disjoint even though they correspond to the same object.
            rectangle_pts = [get_enclosing_points(mask) for mask in r.masks.data]
            rectangle_pts = [rect for rect in rectangle_pts if rect is not False]
            if len(rectangle_pts) < 1:
                print(f"Skipping {lb_path} because no masks found")
                continue
            obb_boxes = np.vstack(rectangle_pts)
            # Get original label path
            lb_relative_path = Path(data["filepath"]).relative_to(Path(data["filepath"]).parents[2])
            lb_abs_path = Path(self.output_dir) / lb_relative_path
            # New label path
            output_label = str(lb_abs_path.with_suffix('.txt')).replace("images", "labels")
            # Transfer old label classes to obb boxes by association.
            labels_ = associate_points_to_boxes(points.points, obb_boxes, labels_reduced, self.IDX_NAMES[self.default_class])
            # Write label file
            obb_boxes_normalized = obb_boxes.reshape(-1,2)/np.array([w,h])
            obb_boxes_normalized = obb_boxes_normalized.reshape(-1,8)
            write_bboxes_to_file(obb_boxes_normalized, labels_, output_label, format="xyxyxyxy")
            # If debug save images with OG and new labels, return generator object.
            if self.debug:
                og_det_path, obb_det_path = self.plot(r, instances.bboxes, obb_boxes, labels_, output_label)
                yield  og_det_path, obb_det_path

    def plot(self,res, axes_boxes, obb_boxes, labels, output_label):
        obb_boxes_xywhr = xyxyxyxy2xywhr(obb_boxes)
        lb = np.array([labels]).T
        res.names = self.CLASS_NAMES
        obb_labels_xywhrclass = np.hstack((obb_boxes_xywhr, np.zeros_like(lb), np.ones_like(lb), lb)) # [xywh, rotation,] track_id, conf, cls
        xyxy_labels = np.hstack((axes_boxes, np.zeros((axes_boxes.shape[0],1)), np.ones((axes_boxes.shape[0],1)), np.zeros((axes_boxes.shape[0],1)) ))
        res.update(boxes=xyxy_labels)
        og_det_path = str(Path(output_label).with_suffix(".jpg")).replace("labels","debug")
        obb_det_path = str(Path(output_label).with_suffix(".obb.jpg")).replace("labels","debug")
        res.plot(save=True, filename=og_det_path)  # plot a BGR numpy array of predictions
        res.update(obb=obb_labels_xywhrclass)
        res.plot(save=True, filename=obb_det_path)  # plot a BGR numpy array of predictions
        return og_det_path, obb_det_path


if __name__ == "__main__":
    model = SAM('sam_b.pt')
    # Display model information (optional)
    model.info()
    debug = True
    dat = DatasetOBBExtractor(model=model, dataset_dir="./datasets/xView-patches", output_dir="./datasets/xView-patches/obb-sam", default_class="Maritime Vessel", debug=debug)
    dat.get_dataset_info()
    processor_generator = dat.process(idxs=None)
    if debug:
        results = [next(processor_generator) for _ in range(300)]
