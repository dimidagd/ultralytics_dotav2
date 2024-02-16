import os
import subprocess
git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip()

os.chdir(git_root)
import ultralytics
print(os.path.dirname(ultralytics.__file__))
# https://docs.ultralytics.com/tasks/obb/#__tabbed_1_1
# https://docs.ultralytics.com/datasets/obb/ OBB Format
from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n-obb.yaml')  # build a new model from YAML
model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='DOTAv2.0-patches.yaml', epochs=20, imgsz=640, fraction=.01, batch=5)	 # data='dota8.yaml' multi_scale=True,mixup=1.0,
