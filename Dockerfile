FROM python:3.10

RUN apt-get update && \
    apt-get install -y unzip pv libgl1-mesa-glx sudo

# Set build-time variables
ARG USER_ID
ARG GROUP_ID
ENV containerusername userCoE
ENV workdev /home/${containerusername}/workdev
# Create a new user with the user ID and group ID passed as build arguments
RUN groupadd -g $GROUP_ID CoEgroup && useradd -m -u $USER_ID -g CoEgroup ${containerusername}

RUN echo "${containerusername} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${containerusername}

RUN pip install --trusted-host pypi.org \
datasets \
scikit-learn \
shapely \
pillow \
evaluate \
transformers[torch] \
torchvision

COPY . $workdev
RUN sudo chown -R ${containerusername}:CoEgroup $workdev
RUN pip install --trusted-host pypi.org -e $workdev
ENV PATH="/home/${containerusername}/.local/bin:${PATH}"
WORKDIR $workdev
RUN yolo settings && sed -i 's|datasets_dir: /home/'"${containerusername}"'/datasets|datasets_dir: '"${workdev}"'/datasets|' /home/${containerusername}/.config/Ultralytics/settings.yaml

ENTRYPOINT [ "bash" ]
