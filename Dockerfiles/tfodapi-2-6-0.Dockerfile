FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    #python3-opencv \
    libgl1 \
    protobuf-compiler \
    wget
RUN rm -rf /var/lib/apt/lists/*

# Trick pip into thinking that the 'tensorflow' package is installed.
# https://stackoverflow.com/questions/65098672/how-to-build-a-docker-image-with-tensorflow-nightly-and-the-tensorflow-object-de
# NOTE: Both TF and Python version needs to be updated when updating TF versions. Python 3.8 is used in TF Nightly builds (as of 2021/10/18)
WORKDIR /usr/local/lib/python3.6/dist-packages
RUN ln -s tensorflow-2.6.0.dist-info tensorflow-2.6.0.dist-info

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
#RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
#    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#    apt-get update -y && apt-get install google-cloud-sdk -y

# clone the TF Object Detection API codebase
RUN mkdir -p /tensorflow/models
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

# Compile protobuf configs
WORKDIR /tensorflow/models/research/
RUN ls -lh
RUN protoc object_detection/protos/*.proto --python_out=.


# Run the Object Detection API setup
RUN cp object_detection/packages/tf2/setup.py ./
ENV PATH="/tensorflow/.local/bin:${PATH}"

# Update pip, install Python dependencies
RUN python3 -m pip install -U pip
RUN python3 -m pip install --no-cache-dir .
#RUN python3 -m pip install opencv-python

# Set the log level
ENV TF_CPP_MIN_LOG_LEVEL 3