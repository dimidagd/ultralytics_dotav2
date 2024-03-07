### Training an obb model

```bash
yolo obb train \
    data=xView-patches-ship-sam.yaml \
    pretrained=False \
    device=0 \
    imgsz=640 \
    batch=16 \
    epochs=100
    workers=8 \
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

### Training a classification model

#### Extract patches

```python
from ultralytics.utils.sam_extractor import DatasetOBBExtractor
from pathlib import Path
final_dir = Path("/work3/dimda/ultralytics_dotav2/examples/datasets/xView-patches-ship-sam")
output_dir = final_dir + "-crops"
dat = DatasetOBBExtractor(model=None, dataset_dir=final_dir, output_dir=None, default_class=None, debug=False)
dat.get_dataset_info()
patches = dat.extract_patches(idxs=None, output_dir=output_dir)

```

#### Prepare dataset

```python
format_patches_for_image_classification(
    base_dir=output_dir,
    output_dir=output_dir,
    move=False)
```

#### Train classifier

```python
cd examples && python3 train_classifier.py
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
