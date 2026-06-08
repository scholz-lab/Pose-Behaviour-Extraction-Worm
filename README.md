# Pose-Behaviour-Extraction-Worm(C-elegans)
## Overview
This project provides an automated pipeline for the kinematic analysis of worm locomotion. It processes raw tracking data (e.g., from DeepLabCut), performs data interpolation for missing coordinates, applies a power-smoothing filter, and generates synchronized video visualizations of postural and bending dynamics.

### Step-1: Pose Estimation (DeepLabCut)
 The initial stage involved training a convolutional neural network (ResNet50) using the DeepLabCut (DLC) framework to track the worm’s body part segments. We manually labeled key anatomical points (Head, Mid-body, Tail) across a representative subset of frames around(200 frames among 8000) that extracted randomly from a video file. Model predicted bodyparts for all frames and Exported frame-by-frame (x, y) coordinates in .csv format.

 ### Data Refinement & Interpolation
 The DeepLabCut model often fails to track the 3rd point during tight bends or in curvature posture causing data gaps.
 #####  Directory structure
    /|__Corrected_Trajectories
       /|__DLC_interp_video.py
The script uses "good" tracking data to fill in those gaps via linear interpolation. This creates a smooth, continuous path for your analysis.

### Synchronized Visualization
It uses vector geometry to calculate Postural Orientation and Bending Magnitude.
#####  Directory structure
   /|__Kinematic_Visualizations
      /|___plots_posture_bending.py
      
/It generates a dual-panel video: the left panel shows the original video with a real-time behavioral data overlay, and the right panel displays dynamically updating "YY-plots" (showing angle and velocity simultaneously).
 
