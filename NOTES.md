### Build the image

```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t yolo-train-img .
```

### Training an obb model

```bash
yolo obb train \
    data=DOTAv2.0-patches-ship.yaml \
    pretrained=False \
    device=0 \
    imgsz=640 \
    batch=16 \
    epochs=100 \
    workers=8
```

or

```bash
docker run -it --rm --gpus all --shm-size=5G yolo-train-img -c \
"
yolo obb train \
    data=DOTAv2.0-patches-ship.yaml \
    pretrained=False \
    device=0 \
    imgsz=640 \
    batch=16 \
    epochs=100 \
    workers=8
"
```

#### Getting data and checkpoints outside

```bash
# Create the directories first otherwise the container user won't be able to write into them as they will be owned by root.
mkdir ~/runs ~/data
docker run -it --rm --gpus all --shm-size=5G \
-v ~/runs:/home/userCoE/workdev/runs \
-v ~/data:/home/userCoE/workdev/datasets \
yolo-train-img -c \
"
yolo obb train \
    data=DOTAv2.0-patches-ship.yaml \
    pretrained=False \
    device=0 \
    imgsz=640 \
    batch=16 \
    epochs=100 \
    workers=8
"
```

### Evaluating an obb model

```bash
yolo obb val \
data=xView-patches-ship-sam.yaml \
model=runs/obb/TESTMODEL/weights/best.pt \
device=0 \
imgsz=640 \
batch=16 \
plots=True
```

or

```bash
# Create the directories first otherwise the container user won't be able to write into them as they will be owned by root.
mkdir ~/runs ~/data
docker run -it --rm --gpus all --shm-size=5G \
-v ~/runs:/home/userCoE/workdev/runs \
-v ~/data:/home/userCoE/workdev/datasets \
yolo-train-img -c \
"
yolo obb val \
data=xView-patches-ship-sam.yaml \
model=/best.pt \
device=0 \
imgsz=640 \
batch=16 \
plots=True
"
```

### Training a classification model

#### First extract patches from xView

```python
import os
from ultralytics.utils.sam_extractor import DatasetOBBExtractor, format_patches_for_image_classification
from pathlib import Path
final_dir = Path("/work3/dimda/ultralytics_dotav2/examples/datasets/xView-patches-ship-sam")
output_dir = str(final_dir) + "-crops"
dat = DatasetOBBExtractor(model=None, yaml_cfg="/work3/dimda/ultralytics_dotav2/ultralytics/cfg/datasets/xView-patches-ship-sam.yaml", dataset_dir=final_dir, output_dir=None, default_class=None, debug=False)
dat.get_dataset_info()
patches = dat.extract_patches(idxs=None, output_dir=output_dir)
format_patches_for_image_classification(
    base_dir=output_dir,
    output_dir=output_dir,
    move=False)
```

or

```bash
docker run -it --rm --gpus all --shm-size=5G \
-v ~/runs:/home/userCoE/workdev/runs \
-v ~/data:/home/userCoE/workdev/datasets \
--entrypoint python3 \
yolo-train-img -c \
"
import os
from ultralytics.utils.sam_extractor import DatasetOBBExtractor, format_patches_for_image_classification
from pathlib import Path
final_dir = Path('/work3/dimda/ultralytics_dotav2/examples/datasets/xView-patches-ship-sam')
output_dir = str(final_dir) + '-crops'
dat = DatasetOBBExtractor(model=None, yaml_cfg='/work3/dimda/ultralytics_dotav2/ultralytics/cfg/datasets/xView-patches-ship-sam.yaml', dataset_dir=final_dir, output_dir=None, default_class=None, debug=False)
dat.get_dataset_info()
patches = dat.extract_patches(idxs=None, output_dir=output_dir)
format_patches_for_image_classification(
    base_dir=output_dir,
    output_dir=output_dir,
    move=False)
"
```

#### Train classifier

```python
python3 examples/train_classifier.py \
    --data-dir=/work3/dimda/ultralytics_dotav2/examples/datasets/xView-patches-ship-sam-crops \
    --project-name=test \
    --wandb-mode=offline
```

or

```bash
mkdir ~/runs ~/data
docker run -it --rm --gpus all --shm-size=5G \
-v ~/runs:/home/userCoE/workdev/runs \
-v ~/data:/home/userCoE/workdev/datasets \
yolo-train-img -c \
"
python3 examples/train_classifier.py \
    --data-dir=/home/userCoE/workdev/datasets/xView-patches-ship-sam-crops \
    --project-name=test \
    --wandb-mode=offline
"
```

### Releasing a dataset

1. Setup gh (use compiled binary)
2. Find the dataset dir

Follow the recipe below from

```bash
export DS_NAME=HRSC2016
export GITREPO=/work3/dimda/ultralytics_dotav2 DATASET_DIR=/work3/dimda/ultralytics_dotav2/examples/datasets/hsrc-ds/HRSC2016 OUTPUTFILE=$DS_NAME.zip TMPDIR=/tmp/splits
cd $DATASET_DIR && zip -r /tmp/$OUTPUTFILE ./ && \
rm -rf $TMPDIR && mkdir -p $TMPDIR
split -d -b 1G /tmp/$OUTPUTFILE $TMPDIR/$OUTPUTFILE. && \
cd $TMPDIR && \
md5sum ./* > md5list && \
cd $GITREPO && \
gh release create $DS_NAME $TMPDIR/* --title "$OUTPUTFILE dataset" --notes "This release includes files with sub 1gb parts and relates to the $DS_NAME dataset."
```
