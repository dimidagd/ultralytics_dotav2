# This file is modified from https://huggingface.co/docs/transformers/tasks/image_classification
import os
import torch
import logging
from datetime import datetime
import numpy as np
import wandb
from datasets import load_dataset, load_from_disk
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, Normalize
from transformers import DefaultDataCollator
import evaluate
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments
import argparse
from torchvision.transforms import RandomResizedCrop, ToTensor
import torchvision.transforms as transforms
from transformers import EarlyStoppingCallback  # https://stackoverflow.com/a/69087153

METRICS_PATH = "./evaluate/metrics"
metric_path_find = lambda x: METRICS_PATH + f"/{x}/{x}.py"
accuracy = evaluate.load(metric_path_find("accuracy"))
precision = evaluate.load(metric_path_find("precision"))
recall = evaluate.load(metric_path_find("recall"))
f1 = evaluate.load(metric_path_find("f1"))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(references=labels, predictions=predictions, average="macro")["precision"],
        "recall": recall.compute(references=labels, predictions=predictions, average="macro")["recall"],
        "f1": f1.compute(references=labels, predictions=predictions, average="macro")["f1"],
    }


def smart_load(dataset_dir, dataset_dir_cache):
    try:
        # Try to load from cache
        dataset = load_from_disk(dataset_dir_cache)
    except FileNotFoundError:
        # Try to load from imagefolder
        logger.warning(f"Dataset not found in cache {dataset_dir_cache}. Loading from {dataset_dir}")
        dataset = load_dataset("imagefolder", data_dir=dataset_dir)
        # Save to cache
        logger.info(f"Dataset loaded from {dataset_dir}")
        logger.info(f"Saving dataset to cache {dataset_dir_cache}")
        dataset.save_to_disk(dataset_dir_cache)
        logger.info(f"Dataset {dataset_dir} cached at {dataset_dir_cache}")
    logger.info("Loading dataset finished")
    return dataset


def create_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a console handler and set the log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


BATCHSZ_PER_MODEL = {
    "google/efficientnet-b0": 345,
    "google/efficientnet-b7": 345,
    "google/vit-base-patch32-224-in21k": 345,
}

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train classifier")

    # Add the data-dir flag
    parser.add_argument("--data-dir", type=str, help="Path to the data directory")
    parser.add_argument("--project-name", type=str, help="Project name for wandb")
    # Add the wandb mode flag
    parser.add_argument("--wandb-mode", type=str, default="offline", help="Wandb mode (online/offline/disabled)")
    parser.add_argument("--sharded-dataset", action="store_true", help="Enable sharded dataset")  # Defaults to False
    # Add the bbone-checkpoint flag
    parser.add_argument(
        "--bbone-checkpoint",
        type=str,
        default="google/efficientnet-b0",
        help="Backbone checkpoint for image processor \
                        eg [google/vit-base-patch32-224-in21k | google/efficientnet-b0 ]. \
                        Or could be any other checkpoint supporting AutoImageProcessor",
    )

    # Hyper params
    # Add augmentation parameters as command-line arguments
    parser.add_argument("--rotation-degrees", type=int, default=180, help="Rotation degrees for augmentation")
    parser.add_argument("--scale-min", type=float, default=0.9, help="Minimum scale factor for augmentation")
    parser.add_argument("--scale-max", type=float, default=1.0, help="Maximum scale factor for augmentation")
    parser.add_argument(
        "--flip", type=float, default=0.5, help="Probability of random horizontal and vertical flipping"
    )
    parser.add_argument("--shear", type=float, default=20, help="Shear factor for augmentation")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation factor for augmentation")
    parser.add_argument("--brightness", type=float, default=0.1, help="Brightness factor for augmentation")
    parser.add_argument("--contrast", type=float, default=0.1, help="Contrast factor for augmentation")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for training")
    args = parser.parse_args()

    # Define augmentation parameters based on CLI arguments
    rotation_degrees = args.rotation_degrees
    scale_min = args.scale_min
    scale_max = args.scale_max
    flip = args.flip
    shear = args.shear
    translate = args.translate
    brightness = args.brightness
    contrast = args.contrast
    grad_accum_steps = args.grad_accum_steps
    learning_rate = args.learning_rate
    batch_size_args = args.batch_size
    # Get the data directory from the command line arguments
    DATA_DIR = args.data_dir
    PROJNAME = args.project_name
    SHARDED_DATASET = os.environ.get("SHARDED", args.sharded_dataset)
    WANDB_MODE = args.wandb_mode
    DATASET_CACHE_DIR = DATA_DIR + "-hf-cache"
    BBONE_CHECKPOINT = args.bbone_checkpoint

    checkpoint = BBONE_CHECKPOINT
    if not batch_size_args:
        batch_size = BATCHSZ_PER_MODEL.get(checkpoint, 1)
    else:
        batch_size = batch_size_args

    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", WANDB_MODE)
    os.environ["WANDB_LOG_MODEL"] = (
        "end"  # end |checkpoint|false. "end" should be used with load_best_model_at_end=True in TrainingArgs
    )

    logger = create_logger()
    # Get the current date and time
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting experiment named {now_str} with project name {PROJNAME}")
    # Init wandb
    run = wandb.init(project=PROJNAME, name=now_str, settings=wandb.Settings(code_dir="."))
    # Log code excluding datasets and runs
    # run.log_code("./",exclude_fn=lambda path,root: os.path.relpath(path, root).startswith("examples/datasets") or os.path.relpath(path, root).startswith("runs"))
    run.config.update({"backbone": BBONE_CHECKPOINT})
    logger.info("Loading dataset")
    dataset = smart_load(DATA_DIR, DATASET_CACHE_DIR)

    # Keep only a small part of the dataset
    if SHARDED_DATASET:
        logger.info("Sharding dataset")
        sharded_dataset_dict = {split: dataset.shard(num_shards=100, index=0) for split, dataset in dataset.items()}
        dataset = sharded_dataset_dict

    labels = dataset["train"].features["label"].names
    logger.info("Defining label2idx and vice versa")
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Define a function that remaps the labels XXX: not working ATM
    # keep_labels_idx = None #list(range(23, 33))
    # if keep_labels_idx:
    #     keep_label_names = [id2label[str(i)] for i in keep_labels_idx]
    #     def remap_labels(example):
    #         if example['label'] in keep_labels_idx:
    #             pass
    #         else:
    #             example['label'] = 0
    #         return example

    #     # Remap the labels
    #     mapped_dataset_dict = {split: dataset.map(remap_labels) for split, dataset in dataset.items()}
    #     dataset = mapped_dataset_dict

    # Split test val set
    ship = dataset["train"].train_test_split(test_size=0.2)

    # Load the image processor backbone
    logger.info(f"Loading image processor backbone {checkpoint}")
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    # Define Transforms based on bbone specifications
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )

    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])  # Simple transform

    # More capable transform
    interpolation_mode = transforms.InterpolationMode.BILINEAR
    _transforms = Compose(
        [
            transforms.RandomAffine(degrees=0, scale=(scale_min, scale_max), interpolation=interpolation_mode),
            transforms.RandomRotation(degrees=rotation_degrees),  # Random rotation
            transforms.RandomResizedCrop(
                size=size, scale=(0.9, 1.0), ratio=(0.9, 1.1), interpolation=interpolation_mode
            ),  # Random cropping and resizing
            transforms.RandomHorizontalFlip(p=flip),
            transforms.RandomVerticalFlip(p=flip),  # Random horizontal flipping
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast
            ),  # Random brightness and contrast adjustment
            transforms.GaussianBlur(kernel_size=3),  # Random Gaussian blur
            transforms.RandomAffine(
                degrees=0, translate=(translate, translate), shear=shear, interpolation=interpolation_mode
            ),  # Random affine transformation (shearing)
            transforms.ToTensor(),
            normalize,
        ]
    )

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    # Apply transform to the entire dataset on the fly
    ship.set_transform(transforms)  # Opposed to ship = ship.with_transform(transforms) which applies transforms once.
    # Load pretrained model
    logger.info(f"Loading model {checkpoint}")
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    data_collator = DefaultDataCollator()
    steps_per_epoch = round(len(ship["train"]) / batch_size) or 1
    logger.info(f"Will use backbone {checkpoint} with batch size {batch_size}")

    training_args = TrainingArguments(
        output_dir=f"./runs/{PROJNAME}-{BBONE_CHECKPOINT}",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="wandb",
        auto_find_batch_size=True,
        dataloader_num_workers=4,
    )

    # Log every epoch
    # training_args = training_args.set_logging(strategy="epoch", report_to=["wandb"])

    # Log every n steps (equivalent to above)
    # training_args = training_args.set_logging(strategy="steps", steps=steps_per_epoch, report_to=["wandb"]) # Equivalent to above, but only works like this for early stopping
    training_args = training_args.set_logging(
        strategy="epoch", report_to=["wandb"]
    )  # Equivalent to above, but only works like this for early stopping

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ship["train"],
        eval_dataset=ship["test"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )
    # Log the number of GPUs
    num_gpus = torch.cuda.device_count()
    run.config.update({"num_gpus": num_gpus})
    run.config.update({"batch_size": batch_size})

    logging.info(f"Number of GPUs: {num_gpus}")

    logging.info("Starting model training")
    trainer.train()
    logging.info("Finished model training")

    logging.info("Calculating confusion matrix")
    val_pred = trainer.predict(ship["test"])
    top_pred_ids = val_pred.predictions.argmax(axis=1)
    target_labels = np.array(ship["test"][:]["label"])
    ground_truth_ids = target_labels
    all_labels = list(label2id.keys())
    logging.info("Logging confusion matrix")
    run.log(
        {
            "my_conf_mat_id": wandb.plot.confusion_matrix(
                preds=top_pred_ids, y_true=ground_truth_ids, class_names=all_labels
            )
        }
    )

    run.finish()
