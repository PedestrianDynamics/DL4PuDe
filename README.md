# Automatic Pushing Behavior Detection Framework
This repository is for the automatic pushing behavior detection framework. 
#### Content
1. Souce code of the framework.
2. Source code of building and training supervised CNN architictures.
3. Source code with patch-based test sets for evaluating the CNN-based classifiers. 
4. Generated CNN-based classifiers.
5. Experiments videos.
6. Demos.

#### Goal
The framework aims to automatically detect pushing behavior at the patch level in videos. It focuses on videos of crowded event entrances that captured by static top-view cameras.

We would like to draw your attention that our pushing behavior differs from the known aggressive pushing behavior (physical aggressive human behavior ). Our pushing behavior is defined as a  set of unfair strategies (e.g., increasing the speed with changing the direction)  for faster access toward the event.


#### The architecture of the framework
<img src="./files/framework1.png"/>
Kindly note, we use the [RAFT repo](https://github.com/princeton-vl/RAFT) for optical flow estimation in our project.

<video src='./files/150-undistorted.mp4' width=180/> | <video src='./files/labeled-150-undistorted.mp4' width=180/>
