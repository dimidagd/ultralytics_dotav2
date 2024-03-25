FROM python:3.10

RUN apt-get update && \
    apt-get install -y unzip pv libgl1-mesa-glx

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV workdev /home/${USERNAME}/workdev
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

COPY ./container_setup/ps1.sh /home/${USERNAME}/ps1.sh
RUN echo "source /home/${USERNAME}/ps1.sh" >> /home/${USERNAME}/.bashrc

USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
RUN pip install --user --trusted-host pypi.org \
datasets \
scikit-learn \
shapely \
pillow \
evaluate \
transformers[torch] \
wandb \
torchvision \
tensorboard

COPY . $workdev
RUN sudo chown -R ${USER_UID}:${USER_GID} $workdev
RUN pip install --user --trusted-host pypi.org -e $workdev
WORKDIR $workdev
RUN yolo settings && sed -i 's|datasets_dir: /home/'"${USERNAME}"'/datasets|datasets_dir: '"${workdev}"'/datasets|' /home/${USERNAME}/.config/Ultralytics/settings.yaml

ENTRYPOINT [ "bash" ]
