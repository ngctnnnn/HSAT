#!/bin/sh
CUDA_VERSION="cu117"
# pip3 install -U torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
pip3 install -U torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+${CUDA_VERSION}.html
pip3 install -U torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+${CUDA_VERSION}.html
pip3 install -U torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.1+${CUDA_VERSION}.html
pip3 install -U torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1+${CUDA_VERSION}.html
pip3 install torch-geometric==1.5.0
