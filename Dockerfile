FROM ros:melodic-ros-core-bionic

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# r2live stuff
RUN apt-get update && apt-get install -y \
    ros-melodic-cv-bridge ros-melodic-tf ros-melodic-message-filters ros-melodic-image-transport

# build tools and libraries
RUN apt-get update && apt-get install -y \
    libgoogle-glog-dev libatlas-base-dev libeigen3-dev cmake \
    curl wget vim build-essential unzip

# ceres solver
WORKDIR /opt/ceres_build

RUN wget -O ceres.zip https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip \
    && unzip ceres.zip

RUN cd ceres-solver-1.14.0 && mkdir ceres-bin && cd ceres-bin \
    && cmake .. && make install -j4

# livox_ros_driver

WORKDIR /opt/livox_build

RUN git clone https://github.com/Livox-SDK/Livox-SDK.git && cd Livox-SDK && cd build && cmake .. && make && make install

RUN apt-get update && apt-get install -y ros-melodic-pcl-conversions ros-melodic-pcl-ros ros-melodic-perception ros-melodic-octomap-*

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; git clone https://github.com/Livox-SDK/livox_ros_driver.git ws_livox/src && cd ws_livox && catkin_make'

# r2live build
WORKDIR /opt/catkin_ws/src

RUN cat /opt/livox_build/ws_livox/devel/setup.sh >> /opt/ros/melodic/setup.bash

RUN    mv /usr/include/flann/ext/lz4.h /usr/include/flann/ext/lz4.h.bak \
    && mv /usr/include/flann/ext/lz4hc.h /usr/include/flann/ext/lz4.h.bak \
    && ln -s /usr/include/lz4.h /usr/include/flann/ext/lz4.h \
    && ln -s /usr/include/lz4hc.h /usr/include/flann/ext/lz4hc.h

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; git clone https://github.com/hku-mars/r2live.git && cd ../ && catkin_make'

RUN cat /opt/catkin_ws/devel/setup.bash >> /opt/ros/melodic/setup.bash
