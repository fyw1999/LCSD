# Learning Crowd Scale and Distribution for Weakly Supervised Crowd Counting and Localization (TCSVT)
## Introduction
This is the official PyTorch implementation of paper: [Learning Crowd Scale and Distribution for Weakly Supervised Crowd Counting and Localization](https://ieeexplore.ieee.org/abstract/document/10680129) (extended from paper [Weakly-supervised scene-specific crowd counting using real-synthetic hybrid data](https://ieeexplore.ieee.org/abstract/document/10095275)). This paper proposes a weakly supervised crowd counting and localization method  based on scene-specific synthetic data for surveillance scenarios, which can accurately predict the number and location of person without any manually labeled point-wise or count-wise annotations.

![pipeline](figures/pipeline.jpg)


# Getting started

## preparatoin
* Clone this repo in the directory 

* Install dependencies. We use python 3.7 and pytorch == 1.10.0 : http://pytorch.org.

    ```
    conda create -n LCSD python=3.7
    conda activate LCSD
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    cd ${MovingDroneCrowd}
    pip install -r requirements.txt
    ```
* Download the resource files used by the code from this [link](https://drive.google.com/file/d/1cljjDA50K1MTQhHKKfx2gCHsjNk7YIoY/view?usp=drive_link), including datasets, pre-trained models, pedestrian gallery, and negetive samples. Unzip `resources.zip` to `resources`. The `resources` folder is organized as follows:
    ```
    $resources/
    ├── CityUHK-X  # dataset name
    │   ├── scene_001  # scene name
    │   │   ├── CityUHK-X_scene_001_20_40  # specific dataset for this scene
    │   │   │   ├── train_data
    │   │   │   │   ├── images
    │   │   │   │   │   └── xx.jpg
    │   │   │   │   ├── ground_truth_txt
    │   │   │   │   │   └── xx.txt
    │   │   │   ├── test_data
    │   │   │   ├── train_data.txt
    │   │   │   └── test_data.txt
    │   │   └── scene.jpg
    │   ├── scene_002
    │   ├── ...
    │   └── scene_k
    ├── Mall
    │   ├── scene_001  # only one scene for Mall
    │   │   ├── mall_800_1200
    │   │   │   ├── train_data
    │   │   │   │   ├── images
    │   │   │   │   │   └── xx.jpg
    │   │   │   │   ├── ground_truth_txt
    │   │   │   │   │   └── xx.txt
    │   │   │   ├── test_data
    │   │   │   ├── train_data.txt
    │   │   │   └── test_data.txt
    │   │   └── scene.jpg
    ├── UCSD
    │   ├── scene_001
    │   │   ├── ucsd_800_1200
    │   │   │   ├── train_data
    │   │   │   │   ├── images
    │   │   │   │   │   └── xx.jpg
    │   │   │   │   ├── ground_truth_txt
    │   │   │   │   │   └── xx.txt
    │   │   │   ├── test_data
    │   │   │   ├── train_data.txt
    │   │   │   └── test_data.txt
    │   │   └── scene.jpg
    ├── pedestrians  #  pedestrian gallery
    │   ├── GCC #  default
    │   │   └── xx.png
    │   ├── SHHB
    │   └── LSTN
    ├── indoor_negetive_samples 
    │   └── xx.jpg
    ├── outdoor_negetive_samples
    │   └── xx.jpg
    ├── darknet53.conv.74  #  pre-trained model for detection
    └── net_G_last.pth.txt  #  pre-trained model for image harmonization
    ```
<!-- 
* Download datasets:

    ◦ **Mall**: Download Mall dataset from this [link](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html). You can randomly select 800 images for predicting pseudo labels and 1200 images for test.

    ◦ **UCSD**: Download frames data from [link](http://visal.cs.cityu.edu.hk/static/downloads/ucsdpeds_vidf.zip) and annotations from [link](http://www.svcl.ucsd.edu/projects/peoplecnt/db/vidf-cvpr.zip). Frames in folder `vidf1_33_000.y` – `vidf1_33_009.y` in total 2000 frames are used (only this part has coordinates labels). In out settings, `vidf1_33_003.y` – `vidf1_33_006.y` are used for predicting pseudo labels, and `vidf1_33_00.y` – `vidf1_33_002.y` and `vidf1_33_009.y` – `vidf1_33_009.y`are used for test. 

    ◦ **CityUHK-X**: Download CityUHK-X dataset from this[link](http://visal.cs.cityu.edu.hk/static/downloads/CityUHK-X.zip). Each surveillance scene in CityUHK-X has 60 images, with 20 used for predicting pseudo labels and 40 for test.

    Place images in the `images` folder, and extracte the corresponding labels into the `ground_truth_txt` folder. Ensure that the image and label filenames are identical. In each label `.txt` file, each line contains the coordinates `x y` representing the position of a pedestrian. -->

## Training

Check some parameters in `train.py` before training:

* Use `dataset = Mall` to set the dataset.
* Use `scene = scene_001` to set the scene of the dataset. `Mall` and `UCSD` only have one scene, so set `scene` as `scene_001`.
* Use `source-path = {$resources}` to set the path of the resource folder downloaded above.
* Use `real-data-dir = mall_800_1200` to set the specific dataset of the scene.
* Use `device = 0` to set the gpu id for training. 
* run `python train.py`.

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
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Weakly-Supervised Scene-Specific Crowd Counting Using Real-Synthetic Hybrid Data}, 
  year={2023},
  pages={1-5}
}


 ```

# Acknowledgement

The released PyTorch training script borrows some codes from the [DRNet](https://github.com/taohan10200/DRNet). If you think this repo is helpful for your research, please consider cite them.