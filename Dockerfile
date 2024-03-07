FROM python:3.10

RUN apt-get update && \
    apt-get install -y unzip pv libgl1-mesa-glx

COPY . /workdev

RUN pip install -U transformers \
datasets \
scikit-learn \
shapely \
pillow \
evaluate \
transformers[torch] \
torchvision

RUN pip install -e /workdev

ENTRYPOINT [ "bash" ]