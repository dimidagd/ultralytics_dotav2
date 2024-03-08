### Build the image

```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t yolo-train-img .
```

### Training an obb model

```bash
yolo obb train \
    data=xView-patches.yaml \
    pretrained=False \
    device=0 \
    imgsz=640 \
    batch=16 \
    epochs=100 \
    workers=8
```

or

```bash
docker run -it --rm --gpus all --shm-size=1G yolo-train-img -c \
"
yolo obb train \
    data=xView-patches-ship-sam.yaml \
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
docker run -it --rm --gpus all --shm-size=1G \
-v ~/runs:/workdev/runs \
-v ~/data:/workdev/datasets \
yolo-train-img -c \
"
yolo obb train \
    data=xView-patches.yaml \
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
docker run -it --rm --gpus all --shm-size=1G \
-v ~/runs:/workdev/runs \
-v ~/data:/workdev/datasets \
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
os.chdir('./examples')
from ultralytics.utils.sam_extractor import DatasetOBBExtractor
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
docker run -it --rm --gpus all --shm-size=1G \
-v ~/runs:/workdev/runs \
-v ~/data:/workdev/datasets \
yolo-train-img -c \
"
import os
os.chdir('./examples')
from ultralytics.utils.sam_extractor import DatasetOBBExtractor
from pathlib import Path
final_dir = Path("/workdev/datasets/xView-patches-ship-sam")
output_dir = final_dir + "-crops"
dat = DatasetOBBExtractor(model=None, dataset_dir=final_dir, output_dir=None, default_class=None, debug=False)
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
    --data-dir=/work1/dimda/xView-patches-ship-sam-crops \
    --project-name=test \
    --wandb-mode=offline
```

or

```bash
mkdir ~/runs ~/data
docker run -it --rm --gpus all --shm-size=1G \
-v ~/runs:/workdev/runs \
-v ~/data:/workdev/datasets \
yolo-train-img -c \
"
python3 examples/train_classifier.py \
    --data-dir=/workdev/datasets/xView-patches-ship-sam-crops \
    --project-name=test \
    --wandb-mode=offline
"
```

### Releasing a dataset

1. Setup gh (use compiled binary)
2. Find the dataset dir

Follow the recipe below from

```bash
export GITREPO=GITDIR DATASET_DIR=/mydatasetDIR OUTPUTFILE=xView.zip TMPDIR=/tmp/splits
zip -r /tmp/$OUTPUTFILE ./ && \
rm -rf $TMPDIR && mkdir -p $TMPDIR
split -d -b 1G /tmp/$OUTPUTFILE $TMPDIR/$OUTPUTFILE. && \
cd $TMPDIR && \
md5sum ./* > md5list && \
cd $GITREPO && \
gh release create xView $TMPDIR* --title "$OUTPUTFILE dataset" --notes "This release includes files with sub 1gb parts"
```
