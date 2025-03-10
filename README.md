# _Multi-Camera Person Tracking and Re-Identification_ (using video)
[![HitCount](http://hits.dwyl.com/samihormi/https://githubcom/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification.svg)](http://hits.dwyl.com/samihormi/https://githubcom/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification)
Simple model to _"Detect/Track"_ and _"Re-identify"_ individuals in different cameras/videos.

<img align="right" img src="assets/2.gif" width="380" />  <img align="left" img src="assets/1.gif" width="380" />
<p align="center">
  <img src="assets/arrow.png" width="400"/>
</p>
<p align="center">
  <img src="assets/3.gif" width="500"/>
</p>


# # Introduction
This repository is forked from https://githubcom/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification and adapted to the specific needs of a project that needs Multi-Person-ReID.
The following ReadMe stands for itself and is still valid. However whats added to the initial coding base is stated here:

- Select the people you want to track:
    - After all videos are processed all IDs which were found are displayed to select if it should be tracked with a "y" or "no" in the console.
    - Everything except the IDs bounding box selected with "y" is blacked out in the video - This ensures that in later processing stages only the correct person is subject to further evaluations.
- Speed up the process:
    - Include the option to look only at every n-th frame and apply the bounding-box to every following (n-1)th frame. 


This project aims to track people in different videos accounting for different angles.


The framework used to accomplish this task relies on MOT and ReID to track and re-identify ID's of humans, respectively.
The tracking can be completed using YOLO_v3 or YOLO_v4 and ReID relies on KaiyangZhou's Torchreid library.

# # Installation
 - Download [Anaconda](https://www.anaconda.com/products/individual) if it is not installed on your machine



 - Clone the repository
```python
git clone https://github.com/danielkrauss2/Multi-Camera-Person-Tracking-and-Re-Identification
```
- Create a project environment
```python
cd Multi-Camera-Person-Tracking-and-Re-Identification
```
- If not already happened, install miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
- Activate the environment
```
source ~/miniconda3/bin/activate
```
- Create a new environment
```
conda init --all
conda create -n py37 python=3.7 anaconda
```

- Install dependencies
```python
pip install -r requirements.txt
```
- Install torch and torchvision based on the cuda version of your machine
```python
conda install pytorch torchvision cudatoolkit -c pytorch
```
# # Convert models
- Download the YOLO models for [YOLO_v3](https://drive.google.com/file/d/18fmQMegNsAzPte7tJeCxwf1iE8JUTQhQ/view?usp=sharing) and [YOLO_v4](https://drive.google.com/file/d/1w9furPagm3KytRW2uNooLcBoiYWDwbop/view?usp=sharing) and add them to /model_data/weights/
* YOLO_v3
```python
python convert_y3.py model_data\weights\yolov3.weights model_data\models\yolov3.h5
```
* YOLO_v4
```python
python convert_y4.py model_data\weights\yolov4.weights model_data\models\yolov4.h5
```

# Pre-trained models (.h5) (If you want to start right away)
- Download the Keras models for [YOLO_v3](https://drive.google.com/file/d/1a7JI-A920lrdt6OKya-qCXx-5ZUWvkMg/view?usp=sharing) and [YOLO_v4](https://drive.google.com/file/d/1pwFo4aHKPi0ztpL5tEYaXIr8RltYYQeY/view?usp=sharing) and add them to \model_data\models\

- Download either one of the following Torchreid models [1](https://drive.google.com/file/d/1EtkBARD398UW93HwiVO9x3mByr0AeWMg/view?usp=sharing),[2](https://drive.google.com/open?id=15Ayri_sHtrctJ1Zb8qERjvdi66y6QaI4) and add them to \model_data\models\ (you might have to change the path in reid.py)

# # Demo

You can try out your own videos by running tracking_and_reid.py.
You should specify the path of the videos and the version of YOLO you would like to use (v3 or v4)

```python
python tracking_and_reid.py --videos videos\init\Double1.mp4 videos\init\Single1.mp4 --version v3
```

# # Acknowledgement
This model is build on top of the incredible work done in the following projects:
  * https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification
  * https://github.com/nwojke/cosine_metric_learning
  * https://github.com/KaiyangZhou/deep-person-reid
  * https://github.com/Qidian213/deep_sort_yolov3
  * https://github.com/Ma-Dan/keras-yolo4
  * https://github.com/lyrgwlr/Human-tracking-multicam
