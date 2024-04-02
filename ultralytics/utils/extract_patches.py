import argparse
from ultralytics.utils.sam_extractor import DatasetOBBExtractor, format_patches_for_image_classification
from pathlib import Path
from ultralytics.utils.logger import setup_logger
import os
import sys

if __name__ == "__main__":
    """
    This script extracts patches from images and formats them for image classification.

    Example script call:
    python3 extract_patches.py --input-dir datasets/HRSC2016 --yaml-cfg ultralytics/cfg/datasets/HRSC2016.yaml --move
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--yaml-cfg", type=str, required=True, help="YAML configuration file")
    parser.add_argument("--default-class", type=str, default=None, help="Default class")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--move", action="store_true", help="Move patches instead of copying")
    parser.add_argument("--splits", nargs="+", default=['train', 'val'], help="List of splits")
    args = parser.parse_args()

    # Set up logger
    logger = setup_logger()

    # Rest of the code
    input_dir = Path(args.input_dir)
    output_dir = args.output_dir if args.output_dir else str(input_dir) + '-crops'

    if os.path.exists(output_dir):
        logger.warning(f"{output_dir} already exists, this call will be a no-op")
        sys.exit(0)

    dat = DatasetOBBExtractor(model=None, yaml_cfg=args.yaml_cfg, dataset_dir=input_dir, output_dir=None, default_class=args.default_class, debug=args.debug)
    logger.info(f"Reading from {input_dir} and writing to {output_dir}")
    logger.info(f"Using YAML configuration file {args.yaml_cfg}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Default class: {args.default_class}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Move patches: {args.move}")

    logger.info(f"Reading dataset info for splits {args.splits} from {input_dir}.")
    dat.get_dataset_info(splits=args.splits)
    # Extract patches
    logger.info(f"Extracting patches to {output_dir}.")
    patches = dat.extract_patches(idxs=None, output_dir=output_dir)
    # Copy/Move patches around
    logger.info("Formatting patches file structure for image classification using HF.")
    format_patches_for_image_classification(base_dir=output_dir, output_dir=output_dir, move=args.move)
