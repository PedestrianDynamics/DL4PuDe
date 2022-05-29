# `DL4PuDe:` A hybrid framework of deep learning and visualization for pushing behavior detection in pedestrian dynamics

[![DOI](https://zenodo.org/badge/447236287.svg)](https://zenodo.org/badge/latestdoi/447236287) [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](./LICENSE)  ![Python 3.7 | 3.8](https://img.shields.io/badge/Python-3.7|3.8-blue.svg)  ![GPU](https://img.shields.io/badge/GPU-No-yellow) ![RAM16GB](https://img.shields.io/badge/RAM-16GB-red)  

This repository is for the [paper](https://www.mdpi.com/1424-8220/22/11/4040/htm):
```
Alia, Ahmed, Mohammed Maree, and Mohcine Chraibi. 2022. "A Hybrid Deep Learning and Visualization Framework for Pushing Behavior Detection in Pedestrian Dynamics" Sensors 22, no. 11: 4040. 
```
#### Content
1. Source code of the framework.
2. How to install and use the framework?
3. Source code for building and training CNN-based classifiers.
4. Source code with patch-based RAFT-MIM test sets for evaluating the trained CNN-based classifiers. 
5. Trained CNN-based classifiers.
6. Video experiments.

#### Goal
`Dl4PuDe` aims to automatically detect pushing behavior at the patch level in videos. In particular,  it exploits video recordings of crowded entrances captured by a top-view static camera.
#### Pushing Behavior Defention
<table border="0" width="100%" align="center">
<tr>
    <td align="center"> <img src="./files/snake.jpeg" height="200"/>  </td>
    <td align="center"> <img src="./files/snakemotion.jpg" height="200"/>  </td>
    <td align="center"> <img src="./files/pb.gif" height="200"/>  </td>
</tr>
</table>

Pushing behavior in this article is defined as an unfair strategy that some pedestrians use to move quickly and enter an event faster.
#### The Architecture of `DL4PuDe`

`DL4PuDe` mainly relied on the power of EfficientNet-B0-based classifier, RAFT and wheel visualization methods.

<img src="./files/framework1.png"/>
Kindly note that we use the [RAFT repository](https://github.com/princeton-vl/RAFT) for optical flow estimation in our project.

**Example**
<table border="0" width="100%" align="center">
<tr>
   <th align="cenetr"> Input video </th>
   <th align="cenetr"> Output video * </th>
   
</tr>
<tr>
   <td align="center"> <img src="./files/input150-distorted.gif" width="300"/> </td>
   <td align="center"> <img src="./files/output150-distorted.gif" width="200"/> </td>
</tr>


<tr>
   <td colspan="2"> * The framework detects pushing patches every 12 frames (12/25 s), the red boxes refer to the pushing patches. </td>
</tr>
</table>

#### Installation
1. Clone the repository in your directory.
```
git clone https://github.com/PedestrianDynamics/PushingBehaviorDetectionFramework.git
```
2. Install the required libraries.
```
cd PushingBehaviorDetectionFramework
pip install -r libraries.txt
```
3. Run the framework. 
```
python3 run.py --video [input video path]  
               --roi ["x coordinate of left-top ROI corner" "y coordinate of left-top ROI corner"
               "x coordinate of  right-bottom ROI corner" "y coordinate of right-bottom ROI corner" ] 
               --patch [rows cols]    
               --ratio [scale of video]   
               --angle [angle in degrees for rotating the input video to make crowd flow direction
               from left to right ---> ]
```   
#### Demo 
<br/>

>Run the following command

```   
python3 run.py --video ./videos/150.mp4  --roi 380 128 1356 1294 --patch 3 3 --ratio 0.5  --angle 0
```  
> Then, you will see the following details.

<img src="./files/run.png"/>

> When the progress of the framework is complete, it will generate the annotated video in the framework directory. Please note that the "150 annotated video" is available on the directory root under the "150-demo.mp4" name.

#### Experiments Videos

The original experiments videos that are used in this work are available through the [Pedestrian Dynamics Data Archive hosted](http://ped.fz-juelich.de/da/2018crowdqueue) by the Forschungszentrum Juelich. Also, the undistorted videos are available by [this link.](https://drive.google.com/drive/folders/16eZhC9mnUQUXxUeIUXd6xwBU2fSf3qCz?usp=sharing) 

#### CNN-based Classifiers

We use four CNN-based classifiers for building and evaluating our classifier, including EfficientNet-B0, MobileNet, InceptionV3, and ResNet50. The source code for building, training and evaluating the CNN-based classifiers, as well as the trained classifiers are available in the below links.
1. Source code for building and training the CNN-based classifiers.
   * [EfficientNet-B0-based classifier.](./CNN/CNN-Architectures/efficientNetB0.ipynb)
   * [MobileNet-based classifier.](./CNN/CNN-Architectures/InceptionV3.ipynb)
   * [InceptionV3-based classifier.](./CNN/CNN-Architectures/InceptionV3.ipynb)
   * [ResNet50-based classifier.](./CNN/CNN-Architectures/ResNet50.ipynb)
2. [Trained CNN-based classifiers.](https://drive.google.com/drive/folders/1vmgYufnt4_NNQUE9PGYZLkrn5DmErENu?usp=sharing)
3. CNN-based classifiers Evaluation.
   * [Patch-based medium RAFT-MIM12 dataset.](./CNN/Classifiers-evaluation/patch-based-medium-RAFT-MIM12/)
   * [Patch-based medium RAFT-MIM25 dataset.](./CNN/Classifiers-evaluation/patch-based-medium-RAFT-MIM25/)
   * [Patch-based small RAFT-MIM12 dataset.](./CNN/Classifiers-evaluation/patch-based-small-RAFT-MIM12/)
   * [Patch-based small RAFT-MIM25 dataset.](./CNN/Classifiers-evaluation/patch-based-small-RAFT-MIM25/)
   * [Patch-based medium FB-MIM12 dataset.](./CNN/Classifiers-evaluation/patch-based-medium-FB-MM12/)
   * [Frame-based RAFT-MIM12 dataset.](./CNN/Classifiers-evaluation/frame-based-RAFT-MIM12/)
   * [Frame-based RAFT-MIM25 dataset.](./CNN/Classifiers-evaluation/frame-based-RAFT-MIM25/)
4. [Patch-based MIM test sets.](./CNN/Classifiers-evaluation/test-sets/)
5. MIM Training and validation sets are available from the corresponding authors upon request.
   
#### Citation
To cite the paper, kindly use the following BibTex entry:
```
Alia, Ahmed, Mohammed Maree, and Mohcine Chraibi. 2022. "A Hybrid Deep Learning and Visualization Framework for Pushing Behavior Detection in Pedestrian Dynamics" Sensors 22, no. 11: 4040. 
```
  
