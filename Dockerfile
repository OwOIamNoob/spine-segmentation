# Use the NVIDIA CUDA image as the base image
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# COPY NGC-DL-CONTAINER-LICENSE /

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64


MAINTAINER Matt McCormick <matt.mccormick@kitware.com>

RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y cmake-curses-gui
RUN apt-get install -y ninja-build
RUN apt-get install -y wget
RUN apt install -y cmake>=3.16
RUN useradd -m user
RUN echo 'root:root' | chpasswd
RUN echo 'user:user' | chpasswd
ENV HOME /opt
RUN mkdir /opt/src
RUN mkdir /opt/bin
RUN chown -R user:user /opt/src
RUN chown -R user:user /opt/bin
USER user
WORKDIR /opt

USER root
RUN apt-get update
RUN apt-get install -y libxcb1 libxcb1-dev
RUN apt-get install -y libx11-dev
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libxt-dev libxft-dev
RUN apt-get install -y python
USER user
WORKDIR /opt/src
RUN wget "https://download.qt.io/official_releases/qt/5.15/5.15.0/single/qt-everywhere-src-5.15.0.tar.xz"
RUN tar -xf qt-everywhere-src-5.15.0.tar.xz
WORKDIR /opt/src/qt-everywhere-src-5.15.0
RUN ./configure -skip qttools -skip qtmultimedia -opensource -confirm-license
RUN make -j3
USER root
RUN make install
USER user
WORKDIR /opt/src
RUN wget "http://www.vtk.org/files/release/6.1/VTK-6.1.0.tar.gz"
RUN tar -xvzf VTK-6.1.0.tar.gz
WORKDIR /opt/src/VTK-6.1.0
RUN sed -i 's/\/\/#define\ GLX_GLXEXT_LEGACY/#define\ GLX_GLXEXT_LEGACY/g' Rendering/OpenGL/vtkXOpenGLRenderWindow.cxx
WORKDIR /opt/bin
RUN mkdir VTK-6.1.0
WORKDIR /opt/bin/VTK-6.1.0
RUN cmake -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DBUILD_SHARED:BOOL=ON ~/src/VTK-6.1.0
RUN ninja
USER root
RUN ninja install
USER user
WORKDIR /opt/src
RUN wget "http://sourceforge.net/projects/itk/files/itk/4.6/InsightToolkit-4.6.1.tar.xz/download" -O InsightToolkit-4.6.1.tar.xz
RUN tar xvJf InsightToolkit-4.6.1.tar.xz
RUN mkdir -p /opt/bin/InsightToolkit-4.6.1
WORKDIR /opt/bin/InsightToolkit-4.6.1
RUN cmake -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_TESTING:BOOL=OFF -DBUILD_SHARED:BOOL=ON -DBUILD_EXAMPLES:BOOL=OFF -G Ninja ~/src/InsightToolkit-4.6.1
RUN ninja
USER root
RUN ninja install

RUN apt-get install -y libglu1-mesa-dev
USER user
WORKDIR /opt/src
RUN git clone "http://git.code.sf.net/p/itk-snap/src" itksnap
RUN mkdir /opt/bin/itksnap-Release
WORKDIR /opt/bin/itksnap-Release
RUN cmake -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_CXX_FLAGS:STRING=-fPIC -DCMAKE_PREFIX_PATH:PATH=/usr/local/Qt-5.3.2 ~/src/itksnap
RUN ninja

CMD ["/bin/bash"]