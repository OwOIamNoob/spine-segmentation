# Use the NVIDIA CUDA image as the base image
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

# Install base utilities
# RUN apt-get update \
#     && apt-get install -y build-essential \
#     && apt-get install -y wget \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# VOLUME ["/data/hpc/spine", "/work/hpc/spine_segmentation"]

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init 

RUN conda create -n spine python=3.10

RUN conda activate mri

# RUN pip install -r requirements.txt

# Optionally, you can run any additional setup or installation commands here
# For example:
# RUN apt-get update && apt-get install -y <package_name>

# Set the default command to run when the container starts
# Replace `<command>` with the command to start your application
CMD ["bin/bash"]
