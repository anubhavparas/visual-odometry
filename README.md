# Visual Odometry - Camera motion estimation
Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the robot (the camera on the robot to be precise).

This projects aims at implementing different steps to estimate the 3D motion of the camera, and provides as output a plot of the trajectory of the camera.

Frames of a driving sequence taken by a camera in a car, and the scripts to extract the intrinsic parameters are given [here](https://drive.google.com/drive/folders/1hAds4iwjSulc-3T88m9UDRsc6tBFih8a?usp=sharing).

### Approach and implementation:
- To estimate the 3D motion (translation and rotation) between successive frames in the sequence:
    - Point correspondences between successive frames were found using [SIFT](https://medium.com/data-breach/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40) (Scale Invariant Feature Transform) algorithm. (refer to _extract_features()_ function in [commonutils.py](./Code/commonutils.py))
    - [Fundamental matrix](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf) (F) was estimated using 8-point algorithm within RANSAC (refer to _fundmntl_mat_from_8_point_ransac()_ function in [commonutils.py](./Code/commonutils.py))
    - [Essential matrix](https://www2.cs.duke.edu/courses/fall15/compsci527/notes/epipolar-geometry.pdf) (E) was estimated from the fundamental matrix using the camera calibration parameters given. (refer to _calc_essential_matrix()_ function in [commonutils.py](./Code/commonutils.py))
    - E matrix was decomposed into to translation(T) and rotation(R) matrices to get four possible combinations.
    - Correct R and T were found from testing the depth positivity, i.e. for each of the four solutions depth of all the points was linearly estimated using the cheirality equations. The R and T that gave the maximum number of positive depth values was chosen. 
    - For each frame, the position of the camera center was plotted based on the rotation and translation parameters between successive frames.
- The rotation and translation parameters that were calculated and the plot were compared against the ones calculated using opencv's _cv2.findEssentialMat()_ and _cv2.recoverPose()_ functions.
- To have a better estimate of the R and T parameters, [code](./Code/nonlinear_methods.py) was enhanced to solve for depth and the 3D motion non-linearly using non-linear triangulation (for estimating depth) and non-linear PnP (for estimating R and T).
- Refer to this [page](https://cmsc733.github.io/2019/proj/p3/) (section 3.1-3.5.1) for more information about the steps involved.
- Elaborate explanation about the approach, concepts and the pipeline can be found in the [report](Report.pdf).

### Output:
Comparison of the plots calculated using inbuilt opencv functions (in blue) and by estimating F and E matrix without using the inbuilt functions (in red):

![alt text](./output/comparison.PNG?raw=true "Comparinf plots")

**Output video can be found [here](https://drive.google.com/file/d/1K8uQAhiYeXw4GkELiAmqYnMzW-g0c_2F/view?usp=sharing)**



### Instructions to run the code:

[Input dataset](https://drive.google.com/drive/folders/1hAds4iwjSulc-3T88m9UDRsc6tBFih8a?usp=sharing)

- Go to directory:  _cd Code/_
- To pre-process the images: 
    - _$ python imagepreprocessor.py_

- To the camera motion estimation task (implementation without OpenCV functions):
    - _$ python motionestimation.py_

- You need to add the processed frames in the 'processed_data/frames' directory. Or, you can add the raw frames to './Oxford_dataset/stereo/centre/' dir

- To estimate the camera motion using inbuilt functions:
    - _$python motionestimator_inbuilt.py_

- The accuracy of the motion depended upon the number of iteration for the RANSAC algo and the approximated recovery of the pose.

- The algorithm sometimes plotted differnt tracks with same tuning parameters.

- We experimented with the different combinations of parameters and implementations. 

- To check whether our implementation calculated appropriate essential matrix we used OpenCV's pose recovery method. We got fair results in this case.

- Non-linear methods were consuming more time.

### References:
- [Lecture on Fundamental Matrix](https://www.youtube.com/watch?v=K-j704F6F7Q)
- [Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)
- [Eight point algorithm](http://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf)
- [Camera Calibration and Fundamental Matrix Estimation with RANSAC](https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html)
- [Structure from Motion](https://cmsc426.github.io/sfm/)
