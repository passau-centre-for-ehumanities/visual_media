# Detectron-Installation

**Before you consider installing Detectron, it is important to keep in mind, that you will need a powerfull Nividia Graficcard with at least 8 gb RAM (better 12) to make it work.**

This instructions are for an Installation of Detectron under Ubuntu 18.04. It should also work for any other Linux-distribution. Windows and Mac-systems are not supported.

## First step:
Install Cuda and CuDNN.
For a proper installation of Cuda (I installed Cuda 10.0), go tu the [Nvidia Website](https://developer.nvidia.com/cuda-10.0-download-archive) and follow the instructions.

## Second step:
install Caffe2
Caffe2 is now included in pytorch. There is also an anaconda and pip package, yet we always installe pytorch from source. The source can be found [here](https://github.com/pytorch/pytorch#from-source).

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

## Third step:

Install the [COCO API](https://github.com/cocodataset/cocoapi):

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python setup.py install --user
```
## FoDetectron

Clone the Detectron repository:

```
git clone https://github.com/facebookresearch/detectron $DETECTRON
```

Install Python dependencies:

```
cd detectron
pip install -r requirements.txt
```

Set up Python modules:

```
make
```

Check that Detectron tests pass (e.g. for [`SpatialNarrowAsOp test`](detectron/tests/test_spatial_narrow_as_op.py)):

```
python $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py
```

make sure to set pythonpath on detectron before using it:
```
export PYTHONPATH=/home/administrator/Detectron/
```

