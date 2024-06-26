# LMIINet: Long-range and Multi-scale Information Interaction Network for 3D Object Detection

<p align="center">
  <img src="docs/LMIINet.png" width="85%">
</p>

## Abstract

LiDAR-based 3D object detection is crucial for perception systems for autonomous driving. Limited by the sparse and uneven distribution of point clouds within the 3D scene, LiDAR-based detectors perform poorly in detecting small objects and encountering occlusions. Therefore, we propose LMIINet, a long-range and multi-scale information interaction network for 3D object detection. LMIINet adopts voxel-based methods and improves on the representative SECOND algorithm. First, the proposed spatial feature pyramid (SFP) module is applied in the 3D backbone to capture the long-range dependencies of spatial objects and accelerate the diffusion of sparse voxel features. Then, the geospatial multi-scale features are adaptively fused at the neck and the fine-grained information of the object is recovered by additional up-sampling. Finally, by fully utilizing the correlations between bounding box parameters, the proposed RCIoU loss is employed for the classification and regression supervision of bounding boxes to enhance the one-stage point cloud detectors. LMIINet achieves competitive performance on the KITTI benchmark. Compared to the benchmark network SECOND, LMIINet increases mAP_3D/mAP_BEV by 1.98\%/2.09\%, 12.25\%/11.14\%, and 6.44\%/6.45\% for the car, pedestrian, and cyclist classes, respectively.

## Overview

- [Installation](docs/INSTALL.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [License](#license)
- [Acknowledgement](#acknowledgement)

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## License

`LMIINet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

We would like to thank the authors of [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) for their open source release of their codebase.




