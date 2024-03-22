### Build the image

```bash
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t yolo-train-img .
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
model=runs/obb/test-run/weights/best.pt \
device=0 \
imgsz=640 \
batch=16 \
plots=True
"
```

### Training a classification model

#### First extract patches from xView

```python
python3 ultralytics/utils/extract_patches.py \
    --input-dir datasets/xView-ships-obb \
    --yaml-cfg ultralytircs/cfg/datasets/xView-ships-obb.yaml \
    --move
```

or

```bash
docker run -it --rm --gpus all --shm-size=5G \
-v ~/runs:/home/userCoE/workdev/runs \
-v ~/data:/home/userCoE/workdev/datasets \
--entrypoint python3 \
yolo-train-img ultralytics/utils/extract_patches.py \
--input-dir datasets/xView-ships-obb \
--yaml-cfg ultralytics/cfg/datasets/xView-ships-obb.yaml \
--move
```

#### Train classifier

```python
python3 jobs/classifier/train_classifier.py \
    --data-dir=datasets/HRSC2016-crops \
    --project-name=HRSC2016-crops \
    --wandb-mode=offline
```

or

```bash
mkdir ~/runs ~/data
docker run -it --rm --gpus all --shm-size=5G \
-v ~/runs:/home/userCoE/workdev/runs \
-v ~/data:/home/userCoE/workdev/datasets \
--entrypoint python3
yolo-train-img jobs/classifier/train_classifier.py \
    --data-dir=datasets/xView-patches-ship-sam-crops \
    --project-name=test \
    --wandb-mode=offline
```

### Releasing a dataset

1. Setup gh (use compiled binary)
2. Find the dataset dir

Follow the recipe below from

```bash
export DS_NAME=DOTAv2.0-test
export GITREPO=/work3/dimda/ultralytics_dotav2 DATASET_DIR=/work3/dimda/ultralytics_dotav2/datasets/DOTA-v2.0 OUTPUTFILE=$DS_NAME.tar TMPDIR=/work1/dimda/splits
rm -f /tmp/$OUTPUTFILE && cd $DATASET_DIR && tar cvf /tmp/$OUTPUTFILE ./ && \
rm -rf $TMPDIR && mkdir -p $TMPDIR
split -d -b 1G /tmp/$OUTPUTFILE $TMPDIR/$OUTPUTFILE.
cd $TMPDIR
for file in $OUTPUTFILE.??; do
tar -zcvf "${file}.tgz" "$file" && rm -f "$file" &
done
wait
cd $TMPDIR && \
md5sum ./*.tgz > md5list && \
cd $GITREPO && \
gh release create $DS_NAME $TMPDIR/* --title "$OUTPUTFILE dataset" --notes "This release includes files with sub 1gb parts and relates to the $DS_NAME dataset."
```

### HF offline download

```bash
huggingface-cli download google/vit-base-patch32-224-in21k --local-dir /home/user/workdev/datasets/google-vit --local-dir-use-symlinks=False --force-download
```
