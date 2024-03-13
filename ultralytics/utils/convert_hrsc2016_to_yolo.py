import numpy as np
import os
from pathlib import Path
from ultralytics.utils.sam_extractor import write_bboxes_to_file, OBBox
from tqdm import tqdm
import logging
import argparse

import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(filename)s - %(funcName)s - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_annotation(xml_path, mapping):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # size = root.find('Size')
        w = int(root.find('Img_SizeWidth').text)
        h = int(root.find('Img_SizeHeight').text)
        bboxes = []
        obboxes = []
        labels = []
        objects = root.find('HRSC_Objects')
        for obj in objects.findall('HRSC_Object'):
            # difficult = int(obj.find('difficult').text)
            # bnd_box = obj.find('bndbox')
            bbox = [
                float(obj.find('box_xmin').text),
                float(obj.find('box_ymin').text),
                float(obj.find('box_xmax').text),
                float(obj.find('box_ymax').text),
            ]

            bboxes.append(bbox)


            cx = float(obj.find('mbox_cx').text)
            cy = float(obj.find('mbox_cy').text)
            w = float(obj.find('mbox_w').text)
            h = float(obj.find('mbox_h').text)
            ang = float(obj.find('mbox_ang').text)

            # Calculate the coordinates of the four corners of the oriented bounding box
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            x1, y1 = np.dot(R, [-w/2, -h/2]) + [cx, cy]
            x2, y2 = np.dot(R, [w/2, -h/2]) + [cx, cy]
            x3, y3 = np.dot(R, [w/2, h/2]) + [cx, cy]
            x4, y4 = np.dot(R, [-w/2, h/2]) + [cx, cy]

            obbox = [x1, y1, x2, y2, x3, y3, x4, y4]
            obboxes.append(obbox)
            labels.append(mapping[int(obj.find('Class_ID').text)])
        bboxes = np.array(bboxes)
        obboxes = np.array(obboxes)
        annotation = {
            'filename': root.find('Img_FileName').text + '.bmp',
            'width': int(root.find('Img_SizeWidth').text),
            'height': int(root.find('Img_SizeHeight').text),
            'boxes': bboxes.astype(np.float32),
            'oriented_boxes': obboxes.astype(np.float32),
            'labels': labels,
                # 'bboxes_ignore': bboxes_ignore.astype(np.float32),
                # 'labels_ignore': labels_ignore.astype(np.int64)

        }
    except Exception:
        logger.exception(f"Error in processing {xml_path}")
    return annotation


def get_class_ids2str(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)

    # Get the root element
    root = tree.getroot()

    # Create a list to hold the Class_IDs
    class_ids2str = {}
    # Iterate over all HRSC_Class elements
    for class_element in root.findall(".//HRSC_Class"):
        # Get the Class_ID element
        class_id_element = class_element.find("Class_ID")
        class_name_element = class_element.find("Class_ShortName")
        class_layer = class_element.find("Class_Layer")

        # If the element is found, add to the list
        if class_id_element is not None and class_name_element is not None and int(class_layer.text) == 1 or int(class_layer.text) == 0:
            # Add to the dict if key does not exist
            class_ids2str[int(class_id_element.text)] = class_name_element.text

    for class_element in root.findall(".//HRSC_Class"):
        # Get the Class_ID element
        class_id_element = class_element.find("Class_ID")
        class_name_element = class_element.find("Class_ShortName")
        class_layer = class_element.find("Class_Layer")
        HRS_Class_ID = class_element.find("HRS_Class_ID")
        # If the element is found, add to the list

        if class_id_element is not None and class_name_element is not None and int(class_layer.text) == 2 and HRS_Class_ID is not None:
            # Add to the dict if key does not exist
            class_ids2str[int(class_id_element.text)] = class_ids2str[int(HRS_Class_ID.text)]

    return class_ids2str


if __name__ == "__main__":

    # import args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="./FullDataSet/Annotations/", required=True)
    parser.add_argument("--output-dir", type=str, help="./FullDataSet/yolo_labels/labels/", required=True)
    args = parser.parse_args()
    xml_path = args.input_dir
    sys_data_path = str(Path(xml_path) / ".." / "sysdata.xml")
    assert os.path.exists(sys_data_path), f"sysdata.xml not found at {sys_data_path}"
    logger.info("Reading sysdata.xml to get class mapping.")
    mapping= get_class_ids2str(sys_data_path)
    class2id = {"ship": 0, "aircraft_carrier":1, "warcraft":2, "merchant ship":3, "aircraft carrier":4, "submarine":5}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def is_valid_annotation_xml(xml):
        return "1000" in xml and xml.split(".")[-1] == "xml"

    # Set up the Python logger

    logger.info("Starting the conversion process")

    for xml in tqdm(os.listdir(xml_path)):
        path = os.path.join(xml_path,xml)
        if not is_valid_annotation_xml(xml):
            logger.info(f"Skipping {xml}")
            continue
        ann = preprocess_annotation(path, mapping)
        if np.size(ann['boxes'])!=0 :
            boxes = [OBBox(box_coords,format="xyxyxyxy",normalized=False) for box_coords in ann["oriented_boxes"]]
            [box.normalize(w=ann["width"],h=ann["height"]) for box in boxes]
            boxes_array = np.vstack([box.points for box in boxes])
            lb_basename = Path(ann["filename"]).with_suffix(".txt")
            label_path = str(output_dir / lb_basename)
            labels = [class2id[label] for label in ann["labels"]]
            write_bboxes_to_file(bboxes=boxes_array, labels=labels, path=label_path, format="xyxyxyxy")
