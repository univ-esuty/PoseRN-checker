## Evaluate accuracy of  Multi-View-OpenPose comparing to Optical Motion Capture System with Simple ICP algorithm.

 [[Paper](https://arxiv.org/abs/2107.03000)] 

This repository is sub-project of the MV-OpenPose and PoseRN which estimates human fullbody pose in 3d. MV-OpenPose is the extension of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and estimates fullbody pose in 3d by using calibrated multiple RGB cameras. In the examples, we used 6 HD-RGB cameras.   If you don't know the details of the MV-OpenPose and PoseRN fullbody motion capture system, please check our paper in the link at the top. 

**Codes of MV-OpenPose and PoseRN will be opened soon**



## Description

These programs in this repository are used to verify how accurate the fullbody pose estimated by MV-OpenPose are. We captured the fullbody of human motions with Optical Motion Capture system and our MV-OpenPose system at the same time. Therefore, By comparing these two estimated fullbody pose and calculating the errors, it enables to find out how accurate the fullbody pose estimated by MV-OpenPose are. Note that, generally, Optical Motion Capture system provides fullbody pose estimation with high accuracy. Therefore, we used it as GT data.



## Setup

#### Setup Virtual Environment & Install Dependencies

We developed and tested on windows10. You can use `Anaconda3` to set up the virtual environment. Run the following commands in the `command prompt`.

```shell
## create some directories for saving results and setting input_data.
$ cd path/to/this-project-root-dir
$ mkdir input_data keyframes results
$ mkdir input_data\mv-openpose input_data\opt-mocap keyframes\mv-openpose keyframes\opt-mocap

## Setup New Virtualenv on Anaconda3
$ conda env create -n venv -f env_setup.yml
$ conda activate venv
```

#### Optional Dependencies

* ffmpeg : use for making animation gifs.

#### Set Data

You have to set the capture data of MV-OpenPose and Optical Motion Capture.

1. Copy MV-OpenPose data to `mv-openpose/3dpose`
2. Copy Optical Motion Capture data to `opt-mocap/mocap.trc`

Note: Optical Motion Capture data is a single file. File extension format is `*.trc` and MV-OpenPose data is the group of `*.ply` and `*.txt` files.

Note: You can set any names for files and directories but if you've changed the name, you'll have to change `configs` parameters in the programs.



## Visualize MV-OpenPose Capture data

Run `view-mvopenpose-3d.py` to visualize MV-OpenPose capture data.

```shell
$ python view-mvopenpose-3d.py 
```

Run `view-optmocap-3d.py` to visualize Optical Motion Capture data.

```shell
$ python view-optmocap-3d.py
```

Run `view-mvopenpose-and-optmocap-3d.py` to visualize same time.

```shell
$ python view-mvopenpose-and-optmocap-3d.py
```

Before you evaluate the MV-OpenPose capture data, you can quickly visualize your MV-OpenPose and Optical motion capture data. You can get `.gif` animation files and visualize it.  In `view-mvopenpose-3d.py`, You can set  path to MV-OpenPose data and `NUM_OF_FRAM` as argument parameters. In `view-optmocap-3d.py`, You can set path to Optical Motion Capture data and export gif animation file path. These arguments are optional so if you've not set these arguments, the default parameters will be used written in the programs.



## 1. Select Keyframes.

#### Select keyframes one at a time

Run `select-keyframes.py` 

```sh
$ python select-keyframes.py 
```

First, you have to select some keyframes from Optical Motion Capture data and MV-OpenPose capture data. In the examples, We selected each of the 8 keyframes. The control panel window and visualize window will be open after running the program. Move the slide bars in the control panel or write the frame number in the textboxes, you can change the frame. Press `visualize` button so the change will be reflected to the visualize window at the same time. if you find good keyframes, press `exportply` button. You can get `.ply` format data of MV-OpenPose and Optical motion capture corresponding to the frame number. 

#### Process with batch

Run `getPly.py`

```shell
$ python getPly.py 
```

If you have already selected all key frames, You can make point-cloud data with batch process. Before run the program, open file `getPly.py` and change the list of the frame numbers at `export_conf['FRAME_NUM']`  .



## 2. Create Keyframes sets.

Run `make-keyframes-plysets.py`

```shell
$ python make-keyframes-plysets.py
```

Second, you have to make a single point-cloud data which is group of selected keyframes data at `process 1.` .

Note: Before you run this code, Please confirm the keyframe point-cloud data are placed on the directories `keyframes/mv-openpose` and `keyframes/opt-mocap`. This program needs to access the keyframe point-cloud data. 



## 3. Quick Comparation between MV-OpenPose and Optical Motion Capture

Run `compare-by-icp.py`

```shell
$ python compare-by-icp.py  #optional:--scale_False
```

Finally, You can compare the capture data between MV-OpenPose and Optical Motion Capture and visualize it. You can select whether or not the icp algorithm will consider the scale parameters.  Set `--scale_False` argument when you run this program without scale fitting. By default, the icp algorithm consider the scale parameters. After running this program, you can get `transfrom paramters`. By default, these parameters are packed and saved in `results/transform.npy`.



## 4. Comparation between MV-OpenPose and Optical Motion Capture

Run `view-mvopenpose-and-optmocap-3d-transform.py`

```shell
$ python view-mvopenpose-and-optmocap-3d-transform.py
```

Note: In this program, it requires `transform parameters` . By default, it loads these parameters from `results/transform.npy`. Please check whether this file is placed correctly before running the program.



## Check the frame Synchronization

Run `check-framesync-by-icp.py`

```shell
$ python check-framesync-by-icp.py
```

if you want to check whether paired MV-OpenPose and Optical Motion Capture keyframe data  you selected are completely synchronized, Run this program so you can check the frame consistency with graph.



## Other

* You can change ***_conf parameters in the programs to run the programs in your custom environment settings.
* The parameter `SKIP_OPT_CAP_FRAME` is to match the Optical Motion Capture sampling rate and MV-OpenPose of that. Generally, Optical Motion Capture system captures data with high frequency.

* Tree of the repository (default)

```
|- input_data
	|- mv-openpose
		|- 3dpose
			|- *.ply
			|- *.txt
	|- opt-mocap
		|- optmocap.trc
|- keyframes
	|- mv-openpose
		|- *.ply
	|- opt-mocap
		|- *.ply
	|- mv-openpose.ply
	|- opt-mocap.ply
|- results
	|- transform.npy
|- python-scripts
|- env_setup.yml
|- README.md
```

