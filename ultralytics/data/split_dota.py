# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools
from glob import glob
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from ultralytics.data.utils import exif_size, img2label_paths
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.logger import setup_logging

check_requirements("shapely")
from shapely.geometry import Polygon

LOGGER = setup_logging()


def bbox_iof(polygon1, bbox2, eps=1e-6):
    """
    Calculate iofs between bbox1 and bbox2.

    Args:
        polygon1 (np.ndarray): Polygon coordinates, (n, 8).
        bbox2 (np.ndarray): Bounding boxes, (n ,4).
    """
    polygon1 = polygon1.reshape(-1, 4, 2)
    lt_point = np.min(polygon1, axis=-2)
    rb_point = np.max(polygon1, axis=-2)
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    l, t, r, b = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([l, t, r, t, r, b, l, b], axis=-1).reshape(-1, 4, 2)

    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def process_file(im_file, lb_file):
    w, h = exif_size(Image.open(im_file))
    if Path(lb_file).exists():  # XXX: Mention in PR that this is needed for images without labels.
        with open(lb_file) as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
    else:  # empty label
        LOGGER.info(f"Empty label for {im_file}")
        lb = np.array([], dtype=np.float32)
    return dict(ori_size=(h, w), label=lb, filepath=im_file)


def load_yolo_dota(data_root, split="train", fast=False):
    """
    Load DOTA dataset.

    Args:
        data_root (str): Data root.
        split (str): The split data set, could be train or val.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    if fast:
        im_files = im_files[0:fast]
    lb_files = img2label_paths(im_files)
    annos = []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        annos = list(
            tqdm(executor.map(process_file, im_files, lb_files), total=len(im_files), desc="Loading image metadata")
        )

    return annos


def get_windows(im_size, crop_sizes=[1024], gaps=[200], im_rate_thr=0.6, eps=0.01, whole_image=False):
    """
    Get the coordinates of windows.

    Args:
        im_size (tuple): Original image size, (h, w).
        crop_sizes (List(int)): Crop size of windows.
        gaps (List(int)): Gap between crops.
        im_rate_thr (float): Threshold of windows areas divided by image ares.
    """
    h, w = im_size
    windows = []
    if whole_image:
        return np.array([[0, 0, w, h]], dtype=np.int64)
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]


def get_window_obj(anno, windows, iof_thr=0.7, whole_image=False):
    """Get objects for each window."""
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        if not whole_image:
            iofs = bbox_iof(label[:, 1:], windows)
            # Unnormalized and misaligned coordinates
            return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
        else:
            return [(label[:]) for i in range(len(windows))]
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns


def crop_and_save(anno, windows, window_objs, im_dir, lb_dir):
    """
    Crop images and save new labels.

    Args:
        anno (dict): Annotation dict, including `filepath`, `label`, `ori_size` as its keys.
        windows (list): A list of windows coordinates.
        window_objs (list): A list of labels inside each window.
        im_dir (str): The output directory path of images.
        lb_dir (str): The output directory path of labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        label = window_objs[i]
        if (
            len(label) != 0
        ):  # Mention in PR that this is needed for images without labels as it generated an empty .txt file.
            label[:, 1::2] -= x_start
            label[:, 2::2] -= y_start
            label[:, 1::2] /= pw
            label[:, 2::2] /= ph

        with open(Path(lb_dir) / f"{new_name}.txt", "w") as f:
            for lb in label:
                formatted_coords = ["{:.6g}".format(coord) for coord in lb[1:]]
                f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")


def process_anno(args):
    anno, crop_sizes, gaps, im_dir, lb_dir, whole_image = args
    if anno is None:
        return True
    windows = get_windows(anno["ori_size"], crop_sizes, gaps, whole_image=whole_image)
    window_objs = get_window_obj(anno, windows, whole_image=whole_image)
    crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))
    return True


def apply_mapping(annos, mapping):
    """
    Apply mapping to the labels.

    Args:
        annos (List[dict]): The list of annotations.
        mapping (dict): The mapping dict from original indexes to final indexes eg {0:1}.
    """
    new_annos = []
    for annot in annos:
        array = annot["label"]
        if array.ndim < 2:
            LOGGER.debug("Skipping empty label.")
            new_annos.append(None)
            continue
        mask = np.isin(array[:, 0], list(mapping.keys()))
        # Apply the mask to the array
        filtered_array = array[mask]
        # Apply the mapping to the first column of the array
        if filtered_array.size > 0:
            filtered_array[:, 0] = np.vectorize(mapping.get)(filtered_array[:, 0])
            annot["label"] = filtered_array
            new_annos.append(annot)
        else:
            LOGGER.debug("Filtering for specific classes resulted in an empty label.")
            # annot['label'] = filtered_array
            new_annos.append(None)
    return new_annos


def split_images_and_labels(
    data_root, save_dir, split="train", crop_sizes=[1024], gaps=[200], mapping=None, whole_image=False
):
    """
    Split both images and labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root, split=split)
    # Apply mapping of original labels to target labels, useful when
    if mapping:
        annos = apply_mapping(annos, mapping)
    empty_labels = len([1 for value in annos if value is None])
    empty_labels_pct = empty_labels / len(annos) * 100 if len(annos) > 0 else 0
    # log portion of empty labels
    LOGGER.info(f"Empty labels or invalid classes: {empty_labels_pct:.2f}%")
    args = [(anno, crop_sizes, gaps, im_dir, lb_dir, whole_image) for anno in annos]
    import multiprocessing

    pool = multiprocessing.Pool()

    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                process_anno,
                args,
            ),
            total=len(args),
        )
    )
    pool.close()
    # assert all mapped values are true
    assert all(mapped_values)
    # Check for exceptions raised in process_anno
    for value in mapped_values:
        if isinstance(value, Exception):
            raise value
    LOGGER.info(f"Done splitting {split} into patches!")
    # Count the number of empty labels


def split_trainval(data_root, save_dir, crop_size=1024, gap=200, rates=[1.0], mapping=None, whole_image=False):
    """
    Split train and val set of DOTA.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        and the output directory structure is:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    for split in ["train", "val"]:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps, mapping=mapping, whole_image=whole_image)


def crop_and_save_patches(args):
    im_file, save_dir, crop_sizes, gaps, whole_image = args
    w, h = exif_size(Image.open(im_file))
    windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps, whole_image=whole_image)
    im = cv2.imread(im_file)
    name = Path(im_file).stem
    for window in windows:
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)


def split_test(data_root, save_dir, crop_size=1024, gap=200, rates=[1.0], whole_image=False):
    """
    Split test set of DOTA, labels are not included within this set.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - test
        and the output directory structure is:
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / "*"))
    args = [(im_file, save_dir, crop_sizes, gaps, whole_image) for im_file in im_files]

    import multiprocessing

    pool = multiprocessing.Pool()
    mapped_values = list(
        tqdm(
            pool.imap_unordered(
                crop_and_save_patches,
                args,
            ),
            total=len(args),
        )
    )
    pool.close()
    for value in mapped_values:
        if isinstance(value, Exception):
            raise value
    LOGGER.info("Done splitting test set into patches!")


if __name__ == "__main__":
    split_trainval(data_root="DOTAv2", save_dir="DOTAv2-split")
    split_test(data_root="DOTAv2", save_dir="DOTAv2-split")
