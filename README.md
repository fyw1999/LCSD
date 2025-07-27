# Learning Crowd Scale and Distribution for Weakly Supervised Crowd Counting and Localization (TCSVT)
## Introduction
This is the official PyTorch implementation of paper: [Learning Crowd Scale and Distribution for Weakly Supervised Crowd Counting and Localization](https://ieeexplore.ieee.org/abstract/document/10680129) (extended from paper [Weakly-supervised scene-specific crowd counting using real-synthetic hybrid data](https://ieeexplore.ieee.org/abstract/document/10095275)). This paper proposes a weakly supervised crowd counting and localization method  based on scene-specific synthetic data for surveillance scenarios, which can accurately predict the number and location of person without any manually labeled point-wise or count-wise annotations.

![pipeline](figures/pipeline.jpg)

# Catalog
✅ MovingDroneCrowd

✅ Training and Testing Code for SDNet

✅ Pretrained models for MovingDroneCrowd

# MovingDroneCrowd
To promote practical crowd counting, we introduce MovingDroneCrowd — a video-level dataset specifically designed for dense pedestrian scenes captured by moving drones under complex conditions. **Notably, our dataset provides precise bounding box and ID labels for each person across frames, making it suitable for multiple pedestrian tracking from drone perspective in complex scenarios.**

![dataset_example](figures/dataset_example.jpg)

The folder organization of MovingDroneCrowd is illustrated below:
```bibtex
$MovingDroneCrowd/
├── frames
│   ├── scene_1
│   │   ├── 1
│   │   │   ├── 1.jpg 
│   │   │   ├── 2.jpg
│   │   │   ├── ...
│   │   │   └── n.jpg
│   │   ├── 2
│   │   ├── ...
│   │   └── m
│   ├── scene_2
│   ├── ...
│   └── scene_k
├── annotations
│   ├── scene_1
│   │   ├── 1.csv
│   │   ├── 2.csv
│   │   ├── ...
│   │   └── m.csv
│   ├── scene_2
│   ├── ...
│   └── scene_k
├── scene_label.txt
├── train.txt
├── test.txt
└── val.txt
```
Each scene folder contains several clips captured within that scene, and each clip has a corresponding CSV annotation file. Each annotation file consists of several rows, with each row in the following format:
`0,0,1380,2137,27,23,-1,-1,-1,-1`.

The first column indicates the frame index, the second column represents the pedestrian ID, and the third to sixth columns specify the bounding box of the pedestrian's head — including the top-left corner coordinates (x, y), width (w), and height (h). Note that image files are named starting from 1.jpg, while both frame indices and pedestrian IDs start from 0. The last four -1 values are meaningless. MovingDroneCrowd are available at the [Google Drive](https://drive.google.com/file/d/1RUGncEVEi3cUtqEWJLFejt8CF8BNbuxv/view?usp=drive_link).

# Getting started

## preparatoin
* Clone this repo in the directory 

* Install dependencies. We use python 3.11 and pytorch == 2.4.1 : http://pytorch.org.

    ```bibtex
    conda create -n MovingDroneCrowd python=3.11
    conda activate MovingDroneCrowd
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    cd ${MovingDroneCrowd}
    pip install -r requirements.txt
    ```
* Datasets

    ◦ **MovingDroneCrowd**: Download MovingDroneCrowd dataset from this [link](https://drive.google.com/file/d/1RUGncEVEi3cUtqEWJLFejt8CF8BNbuxv/view?usp=drive_link). Unzip `MovingDroneCrowd.zip` and place `MovingDroneCrowd` into your datasets folder.

    ◦ **UAVVIC**: Please refer to their code repository [CGNet](https://github.com/streamer-AP/CGNet).

## Training

Check some parameters in `config.py` before training:

* Use `__C.DATASET = 'MovingDroneCrowd'` to set the dataset (default: `MovingDroneCrowd`).
* Use `__C.NAME = xxx` to set the name of the training, which will be a part of the save directory.
* Use `__C.PRE_TRAIN_COUNTER` to set the pre-trained counter to accelerate the training process. The pre-trained counter can be download from this [link](https://drive.google.com/file/d/1ILLLMM3vDIm773XNOerj8rQH-DCQYzRA/view?usp=drive_link).
* Use `__C.GPU_ID = '0'` to set the GPU. You can set `__C.GPU_ID = '0, 1, 2, 3'` if you have multiple GUPs.
* Use `__C.MAX_EPOCH = 100` to set the number of the training epochs (default:100). 
* Set dataset related parameters (`DATA_PATH`, `TRAIN_BATCH_SIZE`, `TRAIN_SIZE` etc.) in the `datasets/setting`.
* run `python train.py` for one GPU, or run `torchrun --master_port 29515 --nproc_per_node=4 train.py`for multiple GPUs. (for example, 4 GPUs)

Tips: The training process takes ~12 hours on `MovingDroneCrowd` dataset with two A800 (80GB Memory).

## Test

<!--To reproduce the performance, download the pre-trained models from [Google Drive]() and then place pretrained_model files to `SDNet/pre_train_model/`. -->
Check some parameters in `test.py` before test:

* Use `DATASET = MovingDroneCrowd` to set the dataset used for test.
* Use `test_name = xxx` to set a test name, which will be a part of the save director of test reults.
* Use `test_intervals = 4` to set frame interval for test (default `4` for `MovingDroneCrowd`). 
* Use `model_path = xxx` to set the pre-trained model file.
* Use `GPU_ID = 0` to set the GPU used for test.
* run `test.py`

# Citation
If you find this project is useful for your research, please cite:

```bibtex
@ARTICLE{LCSD,
  author={Fan, Yaowu and Wan, Jia and Ma, Andy J.},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Learning Crowd Scale and Distribution for Weakly Supervised Crowd Counting and Localization}, 
  year={2025},
  volume={35},
  number={1},
  pages={713-727}
  }

  @INPROCEEDINGS{ICASSP_2023_FAN
  author={Fan, Yaowu and Wan, Jia and Yuan, Yuan and Wang, Qi},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Weakly-Supervised Scene-Specific Crowd Counting Using Real-Synthetic Hybrid Data}, 
  year={2023},
  pages={1-5}
  }


 ```

# Acknowledgement

The released PyTorch training script borrows some codes from the [DRNet](https://github.com/taohan10200/DRNet). If you think this repo is helpful for your research, please consider cite them.