# Pseudo-LiDAR End to End

## Introduction

In this work, I implemented the Pseudo-LiDAR End to End which can directly generate accurate 3D bounding box **from binocular camera images in a end-to-end method**. This implementation solution includes three parts: stereo matching, depth map to Pseudo-LiDAR, and point cloud-based 3D object detection algorithm. First, I chose [GA-Net](https://arxiv.org/abs/1904.06587) as the stereo matching algorithm for this project because GA-Net is the top-ranked algorithm on the KITTI leaderboard and its aggregation strategy is more conducive to generating more detailed depth map. Second, according to the idea introduced in this paper on [Pseudo-LiDAR](https://arxiv.org/abs/1812.07179), the depth map can be transformed into a pseudo-point-cloud representation through the camera's intrinsics parameter matrix. Finally, the point cloud-based [PointRCNN](https://arxiv.org/abs/1812.04244) detection algorithm is used as a 3D object detector. PointRCNN can also be used to detect Pseudo-LiDAR data, because of foreground points classification strategy, refinement in the canonical coordinate by the proposed bin-based 3D box regression loss, and the set abstraction of its backbone network PointNet++.

### Supported features and ToDo list

- [x] PyTorch 1.2
- [x] KITTI Dataset support
- [x] TensorboardX
- [x] Multiple GPUs support
- [ ] Some bugs of inference and training will be fixed in some days
- [ ] Inference end to end will coming soon
- [ ] Training end to end will coming soon

## Installation

### Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 16.04)
- Python 3.6+
- PyTorch 1.2
- CUDA 10.0

### Install Pseudo-LiDAR End to End

a. Clone the Pseudo-LiDAR End to End repository.

```
git clone https://github.com/ustc-keyanjie/Pseudo-LiDAR-End-to-End.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX `etc.

c. Build and install the`GANet`, `sync_bn`, `pointnet2_lib`,  `iou3d`,  `roipool3d` libraries by executing the following command:

```
cd Pseudo-LiDAR-End-to-End/scripts
bash compile.sh
```

## Dataset preparation

Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

```
Pseudo-LiDAR-End-to-End 
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & label_2 & image_2 & image_3
│   │   │   ├──testing
│   │   │      ├──calib & image_2 & image_3
├── lib
├── pointnet2_lib
├── tools
```

You can organize files manually or run shell script.

a. change `public_dataset_path` in  scripts/dataset_preparation.sh to your own kitti dataset path.

b. run `public_dataset_path.sh`

```
bash dataset_preparation.sh
```

## Pretrained model

```
Car AP@0.70, 0.70, 0.70:
bbox AP:88.5676, 74.2350, 67.0732
bev  AP:76.3447, 55.5241, 48.7155
3d   AP:64.4004, 45.3808, 39.2332
aos  AP:87.57, 72.14, 64.92
```

## Inference

Coming soon.

## Training

Coming soon.

## Reference

- [GA-Net](https://github.com/feihuzhang/GANet)
- [Pseudo-LiDAR](https://github.com/mileyan/pseudo_lidar)
- [PointRCNN](https://github.com/sshaoshuai/PointRCNN)

