# ShuffleNet V1&V2

this code is mxnet implementation of ShuffleNetV1 and ShuffleNetv2, For details, please read the original paper:</br>
[shufflenetv1](https://arxiv.org/pdf/1707.01083.pdf)</br>
[shufflenetv2](https://arxiv.org/pdf/1807.11164.pdf)<br>
This code is based on farmingyard's implementation(https://github.com/farmingyard/ShuffleNet)

Code is test  on   MxNet 1.11.0 

## Installation

1. Clone this repository, and we'll call the directory that you cloned mxnet-shufflenet as ${SHUFFLENET_ROOT}.
```
git clone https://github.com/Tveek/mxnet-shufflenet.git
```

2. Install shuffle channel operator to MXNet:

	2.1 Clone MXNet and checkout to [MXNet](https://github.com/apache/incubator-mxnet.git) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git submodule update
	```
	2.2 Copy operators in `$(SHUFFLENET_ROOT)/source/shuffle_channel*.xxx`  by
	```
	cp -r $(SHUFFLENET_ROOT)/operator/* $(MXNET_ROOT)/src/operator/contrib/
	```
	2.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	2.4 Install the MXNet Python binding by
	```
	cd python
	sudo python setup.py install 
	```

3. Python operator function are also in symbol file, so you can use it without above

## Pretrained Models on ImageNet

- RGB mean and std are used(rgb_mean=**[123.68,116.779,103.939]**, rgb_std=**[58.393,57.12,57.375]**)

- The top-1/5 accuracy rates by using single random crop (crop size: 224x224, image size: 256xN)

Network|Top-1|Top-5|model size
ShuffleNet_V1| 63.94| 85.27| 7.1MB |

