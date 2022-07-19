# ExtrudeNet

(This repo is still under construction. If you face any problem, feel free to open an issue.)

The official implementation of ExtrudeNet: Unsupervised Inverse Sketch-and-Extrude for Shape Parsing

### Paper and Project Page are comming soon.

## Citation
If you find our work interesting and benifits your research, please consider citing:

	@inproceedings{ren2022extrude,
		title = {ExtrudeNet: Unsupervised Inverse Sketch-and-Extrude for Shape Parsing},
		author = {Ren, Daxuan and Zheng, Jianmin and Cai, Jianfei and Li, Jiatong and Zhang, Junzhe},
		booktitle = {ECCV},
		year = {2022}}

## Setup
### Install envoriment:
We recommand using Anaconda to set the envoriment, once Anacodna in installed, run the following command.

```
conda create --name ExtrudeNet python=3.7
conda activate ExtrudeNet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install -c open3d-admin open3d
conda install numpy
conda install pymcubes
conda install tensorboard
conda install scipy
pip install tqdm
```

## Dataset
As the processed dataset is extreamly large, we cannot provide a ready to download dataset.
To prepare the dataset:
1. Download ShapeNet dataset (remember to agree their EULA)

	```
	mkdir data
	cd data
	wget http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetCore.v1.zip
	```
2. unzip the desired category for example Plane (02691156)
	```
	unzip ShapeNetCore.v1.zip
	cd ShapeNetCore.v1
	unzip 02691156.zip
	```
3. Build "watertight" and add it to PATH
	```
	cd ../../
	git clone https://github.com/skanti/generate-watertight-meshes-and-sdf-grids.git
	cd generate-watertight-meshes-and-sdf-grids
	mkdir build
	cd build
	cmake ..
	make -j
	export PATH=$PATH:`pwd`
	```
4. Build triangle hash
	```
	cd datasets
	python setup.py install
	```
4. Edit the categories you want to run in the pre-processing script and run it
	```
	vim preprocess.py
	python preprocess.py
	```

### Train the model
```
python train.py --config_path ./configs/plane.json
```

## Eval the model
```
python eval.py --config_path ./configs/plane.json
```
### Evaluation
```
python metrics.py --config_path ./configs/plane.json
```

## License
This project is licensed under the terms of the MIT license (see LICENSE for details).


