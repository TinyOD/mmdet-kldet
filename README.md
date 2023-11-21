# KLDet: Detecting Tiny Objects in Remote Sensing Images via Kullback-Leibler Divergence

## Introduction
**Abstract**: Remote sensing images (RSIs) frequently contain quite a few tiny objects with a finite number of pixels to study. The limited spatial information poses a challenge in extracting discriminative features for representing the characteristics of tiny objects. Existing solutions mainly focus on aggregating contextual information at different levels, while rarely touching the step that is crucial for training, i.e., label assignment.  Tiny instances occupy fairly small regions of images and have limited overlaps to priors (anchors or dots), which is a dilemma for traditional label assignment strategies. Despite being simple and effective, the mainstream IoU-based (Intersection over Union) label assignment strategy fails to accurately measure the localization of tiny bounding boxes. In contrast, the Kullback-Leibler Divergence (KLD) localization metric accurately reflects minor offsets of small bounding boxes. What's more, KLD is able to measure non-overlapping bounding boxes, providing it with an advantage in mining more potential positive samples of tiny objects. In this paper, from a cost-efficient point of view, we detect tiny objects via Kullback-Leibler Divergence in the form of single-stage. Specifically, in view of the difficulty of IoU to accurately measure the offset of tiny objects, we model the parameterized Bounding box as a two-dimensional (2D) Gaussian distribution (Bbox2Gaussian) in order to use KLD as a localization metric. Then, we propose an adaptive online training sample mining strategy (Ali-TSM) based on inter-distribution similarity, which selects high-quality positive samples by considering localization and classification rather than just centroid distance or IoU. Finally, task-level attention (TlA) is introduced to guide the model in freely selecting the appropriate features for the classification or regression task. Extensive experiments are conducted on the AI-TOD (Tiny Object Detection in Aerial Images) and DIOR (Object Detection in Optical Remote Sensing Images) datasets. Compared to the baseline, KLDet improves detection accuracy on AI-TOD and DIOR by 4.1 mAP and 6.7 mAP, respectively. Additionally, our best model achieves a state-of-the-art performance of 73.5 mAP on DIOR.

![fig](https://i.postimg.cc/q7Kr64Nn/1.png)

## Installation
### Required environments:
-  Linux
-  Python **3.7** (Python 2 is not supported)
-   PyTorch **1.5**  or higher
-   CUDA **10.1** or higher
-   GCC(G++)  **5.4**  or higher
-   [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
-   [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod)==**12.0.3**

### Install:
This project is implemented based on the [MMDetection](https://github.com/open-mmlab/mmdetection) toolkit. Once your environment has met the above requirements, follow the steps below to install.
```c
git clone https://github.com/TinyOD/KLDet.git
cd mmdet-kldet
pip install -r requirements/build.txt
python setup.py develop
```

## Get Started

### Prepare datasets
Please refer to [AI-TOD](https://github.com/jwwangchn/AI-TOD) for the AI-TOD dataset.
If your folder structure is different, you may need to change the corresponding paths in config files.
```
├── mmdet-kldet
├── tools
├── configs
├── data
│   ├── AI-TOD
│   │   ├── annotations
│   │   │    │─── aitod_training_v1.json
│   │   │    │─── aitod_validation_v1.json
│   │   ├── trainval
│   │   │    │─── ***.png
│   │   │    │─── ***.png
│   │   ├── test
│   │   │    │─── ***.png
│   │   │    │─── ***.png
```

### Training
To train a detector with pre-trained models, run:
```c
#single-gpu training:
python tools/train.py configs/kldet/fcos_r50_kldet_1x_aitod.py

#multi-gpu training:
bash ./tools/dist_train.sh configs/kldet/fcos_r50_kldet_1x_aitod.py ${gpu_num}$
```

### Inference
```c
python tools/test.py configs/kldet/fcos_r50_kldet_1x_aitod.py work_dirs/fcos_r50_kldet_1x_aitod/epoch_12.pth --eval bbox
```

## Visualization
Some representative detection results of KLDet on the AI-TOD dataset:
![](https://i.postimg.cc/tRrqshpG/1.png)