IMG_NAME=yolo-train-img
TAG=latest
BASE_HOST_VOLUME=/tmp
DATA_FOLDER_HOST=${BASE_HOST_VOLUME}/data
RUNS_FOLDER_HOST=${BASE_HOST_VOLUME}/runs
PRETRAINED_HF_TAR=./vit-base-patch32-224-in21k.tar.gz
IMGSZ=640
DEVICE=0
BATCHSZ=16
WANDB_MODE=online
epochs=100
MODEL=yolov8n-obb.yaml
PRETRAINED=False

welcome:
	echo "Welcome to the YOLO detector build workflows"

build: welcome
	docker build --build-arg USER_UID=$(shell id -u) --build-arg USER_GID=$(shell id -g) -t ${IMG_NAME}:${TAG} .

make_dirs:
	mkdir -p ${DATA_FOLDER_HOST} ${RUNS_FOLDER_HOST}

train_dotav2: build make_dirs
	docker run -it --rm --gpus all --shm-size=5G \
	-e WANDB_MODE \
	-v ${RUNS_FOLDER_HOST}:/home/user/workdev/runs \
	-v ${DATA_FOLDER_HOST}:/home/user/workdev/datasets \
	${IMG_NAME}:${TAG} -c \
	" \
	yolo obb train \
		data=DOTAv2.0-patches-ship.yaml \
		pretrained=${PRETRAINED} \
		model=${MODEL} \
		device=${DEVICE} \
		imgsz=${IMGSZ} \
		epochs=${epochs} \
		batch=${BATCHSZ} \
		workers=8 \
		plots=True \
	"

evaluate_dotav2: build make_dirs
	docker run -it --rm --gpus all --shm-size=5G \
	-e WANDB_MODE \
	-v ${RUNS_FOLDER_HOST}:/home/user/workdev/runs \
	-v ${DATA_FOLDER_HOST}:/home/user/workdev/datasets \
	${IMG_NAME}:${TAG} -c \
	" \
	yolo obb val \
		data=DOTAv2.0-patches-ship.yaml \
		model=${MODEL} \
		device=${DEVICE} \
		imgsz=${IMGSZ} \
		batch=${BATCHSZ} \
	"

train_HRSC2016: build make_dirs
	docker run -it --rm --gpus all --shm-size=5G \
	-e WANDB_MODE \
	-v ${RUNS_FOLDER_HOST}:/home/user/workdev/runs \
	-v ${DATA_FOLDER_HOST}:/home/user/workdev/datasets \
	${IMG_NAME}:${TAG} -c \
	" \
	yolo obb train \
		data=HRSC2016.yaml \
		pretrained=False \
		device=${DEVICE} \
		imgsz=${IMGSZ} \
		batch=${BATCHSZ} \
		epochs=${epochs} \
		workers=8 \
	"

extract_patches_HRSC2016: train_HRSC2016
	docker run -it --rm --gpus all --shm-size=5G \
	-v ${RUNS_FOLDER_HOST}:/home/user/workdev/runs \
	-v ${DATA_FOLDER_HOST}:/home/user/workdev/datasets \
	--entrypoint python3 \
	${IMG_NAME}:${TAG} \
	ultralytics/utils/extract_patches.py \
	--input-dir datasets/HRSC2016 \
	--yaml-cfg ultralytics/cfg/datasets/HRSC2016.yaml \
	--move

untar_pretrained: make_dirs
	tar -xvf ${PRETRAINED_HF_TAR} -C ${DATA_FOLDER_HOST}

train_classifier_HRSC2016: build untar_pretrained extract_patches_HRSC2016
	docker run -it --rm --gpus all --shm-size=5G \
	-v ${RUNS_FOLDER_HOST}:/home/user/workdev/runs \
	-v ${DATA_FOLDER_HOST}:/home/user/workdev/datasets \
	--entrypoint python3  \
	-e HF_DATASETS_OFFLINE=1 --network=none \
	${IMG_NAME}:${TAG} \
	jobs/classifier/train_classifier.py \
	--data-dir=datasets/HRSC2016-crops \
	--project-name=HRSC2016-crops-multiclass-classifier \
	--wandb-mode=${WANDB_MODE} \
	--bbone-checkpoint="/home/user/workdev/datasets/vit-base-patch32-224-in21k"
	--batch-size=${BATCHSZ}

clean:
	@echo "You are about to delete the directories ${DATA_FOLDER_HOST} ${RUNS_FOLDER_HOST}"
	@read -p "Are you sure you want to proceed? [y/N] " response; \
	if [ "$$response" != "y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	rm -rf ${DATA_FOLDER_HOST} ${RUNS_FOLDER_HOST}
	docker rmi ${IMG_NAME}:${TAG} || echo "No such image found"

bundle:
	git tag -f last-bundle && git bundle create repo.bundle main

create_release:
	export DS_NAME=DOTAv2.0-test GITREPO=${PWD} DATASET_DIR=/work3/dimda/ultralytics_dotav2/datasets/DOTA-v2.0 OUTPUTFILE=${DS_NAME}.tar TMPDIR=/work1/dimda/splits
	rm -f /tmp/${OUTPUTFILE} && cd ${DATASET_DIR} && tar cvf /tmp/${OUTPUTFILE} ./ && \
	rm -rf ${TMPDIR} && mkdir -p ${TMPDIR}
	split -d -b 1G /tmp/${OUTPUTFILE} ${TMPDIR}/${OUTPUTFILE}.
	cd ${TMPDIR}
	for file in ${OUTPUTFILE}.??; do
	tar -zcvf "${file}.tgz" "${file}" && rm -f "${file}" &
	done
	wait
	cd ${TMPDIR} && \
	md5sum ./*.tgz > md5list && \
	cd ${GITREPO} && \
	gh release create $DS_NAME ${TMPDIR}/* --title "${OUTPUTFILE} dataset" --notes "This release includes files with sub 1gb parts and relates to the ${DS_NAME} dataset."
