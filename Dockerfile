FROM python:3.10

RUN apt-get update && \
    apt-get install -y unzip pv libgl1-mesa-glx



RUN pip install -U transformers \
datasets \
scikit-learn \
shapely \
pillow \
evaluate \
transformers[torch] \
torchvision

COPY . /workdev
RUN pip install -e /workdev
WORKDIR /workdev
RUN yolo settings && sed -i 's|datasets_dir: /datasets|datasets_dir: $PWD/datasets|' /root/.config/Ultralytics/settings.yaml


ENTRYPOINT [ "bash" ]