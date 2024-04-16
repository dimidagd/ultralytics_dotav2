# Transform xView from yolo det to yolo obb
from ultralytics.data.split_dota import load_yolo_dota
import shutil
from tqdm import tqdm
import numpy as np
from pathlib import Path
import tempfile
import subprocess


def process_item(item, output_dir, label_transform):
    # Create the output directory if it doesn't exist

    # Transform the labels
    transformed_labels = [label_transform(label) for label in item["label"]]

    # Write the transformed labels to the output file
    output_file = Path(output_dir) / Path(item["filepath"]).with_suffix(".txt").name
    with open(output_file, "w") as f:
        f.writelines(transformed_labels)


def yolo2obb_lb_inverse(label):
    # Extract the components
    class_index, x1, y1, x2, y2, x3, y3, x4, y4 = label

    # Calculate the center and dimensions
    x_center = (x1 + x3) / 2
    y_center = (y1 + y3) / 2
    width = x3 - x1
    height = y3 - y1

    # Create the original label
    original_label = f"{int(class_index)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    return original_label


def yolo2obb_lb(label):
    # Extract the components
    class_index, x_center, y_center, width, height = label

    # Calculate the coordinates of the four corners
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center - height / 2
    x3 = x_center + width / 2
    y3 = y_center + height / 2
    x4 = x_center - width / 2
    y4 = y_center + height / 2

    # Create the transformed label
    # Keep 6 decimals
    transformed_label = f"{int(class_index)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"

    return transformed_label


def transform_yolo_dota_to_obb(yolo_dota_path, output_dir, splits=["val", "train"], inverse=False):
    label_transform = yolo2obb_lb if not inverse else yolo2obb_lb_inverse

    print("Transforming dataset to OBB format...")
    # Asserts contents of splits are valid
    assert all([split in ["train", "val"] for split in splits])
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in splits:
            print(f"Processing {split} split...")
            split_dir = Path(yolo_dota_path) / Path("labels") / Path(split)
            output_dir_split = Path(output_dir) / Path("labels") / Path(split)
            split_dir_original = Path(yolo_dota_path) / Path("labels") / Path(split + "_original")
            tmp_output_dir_split = Path(tmpdir) / Path(split + "_obb")
            print(f"Removing {tmp_output_dir_split}...")
            subprocess.run(["rm", "-rf", str(tmp_output_dir_split), str(split_dir_original)])
            tmp_output_dir_split.mkdir(parents=True, exist_ok=True)
            print(f"Loading {split} split...")
            dataset = load_yolo_dota(yolo_dota_path, split=split)
            print("Split loaded. Processing items...")

            # Process each item in the dataset

            for item in tqdm(dataset, desc=f"Processing items in {tmp_output_dir_split} dir"):
                process_item(item, tmp_output_dir_split, label_transform)
            print(f"Finished processing items for {split_dir} into {tmp_output_dir_split}.")
            print(f"Moving {split_dir} to {split_dir_original}...")
            if yolo_dota_path == output_dir:
                print(f"Moving {split_dir} to {split_dir_original}...")
                subprocess.run(["mv", str(split_dir), str(split_dir_original)])
            print(f"Moving {tmp_output_dir_split} to {output_dir_split}...")
            subprocess.run(["mv", str(tmp_output_dir_split), str(output_dir_split)])
    return True
