#!/usr/bin/sh

# Raspberry Pi Facial Recognition Setup Script
# Copyright, Tom Alessi, 2021

# Usage: Execute this script (as pi user) with no arguments on a vanilla
# Raspberry Pi OS installation to setup a basic OpenCV and Dlib environment
#
# This script assumes a Raspberry Pi 4b (8GB).  If using a Pi 3/4
# with minimal RAM, modify the swap space before/after executing
# this script (increase to 2048MB).
#
# Execution takes approximately 1 hour on a Pi 4 (8GB)


# Ensure the pi user is executing this script
if [ `whoami` != 'pi' ]
  then
    echo "You must be pi to execute this script."
    exit
fi

# Move to the root of the pi home directory
cd ~

# Install build and make tools
sudo apt -y install cmake build-essential pkg-config git

# Install image/video libraries
sudo apt -y install libjpeg-dev libtiff-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

# Install GUI toolkit and Python bindings
sudo apt -y install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5

# Install Atlas and Fortran
sudo apt -y install libatlas-base-dev liblapacke-dev gfortran

# Install HDF5
sudo apt -y install libhdf5-dev libhdf5-103

# Install Numpy and Python bindings
sudo apt -y install python3-dev python3-pip python3-numpy

# Checkout opencv
# Note: using the 4.5.1 branch because we know it works
# Cleanup any existing directory first
rm -rf ~/opencv
git clone --branch 4.5.1 https://github.com/opencv/opencv.git

# Checkout opencv_contrib
# Note: using the 4.5.1 branch because we know it works.  For Pi Zero use 4.4.1
# Cleanup any existing directory first
rm -rf ~/opencv_contrib
git clone --branch 4.5.1 https://github.com/opencv/opencv_contrib.git

# Create a build directory and move into it
mkdir ~/opencv/build && cd ~/opencv/build

# Configure the build
# Note:for Pi Zero, turn off NEON and VFPv3
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D ENABLE_NEON=ON \
-D ENABLE_VFPV3=ON \
-D BUILD_TESTS=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D CMAKE_SHARED_LINKER_FLAGS=-latomic \
-D BUILD_EXAMPLES=OFF ..

# Build OpenCV
make -j$(nproc)

# Install OpenCV
sudo make install

# Update run-time bindings
sudo ldconfig

# Install Python face-recognition (Python interface to DLib)
# face-recognition: https://pypi.org/project/face-recognition/
# Dlib: http://dlib.net/
pip install face-recognition

# Install imutils
# IMutils contains convenience functions for working with OpenCV
pip install imutils




