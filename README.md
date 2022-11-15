# drosophila-census
Exploring ML models to perform drosophila census estimates. The objective is to (1) crop the image to a region of interest and (2) estimate the number of flies in that region.

### Method #1: Manual Thresholding

This program provides a simple GUI to estimate drosphila melanogaster fruit flies from already cropped images using basic user-supervised image processing techniques, including thresholding and connected component identification. No ML is used.

### Method #2: K-Means + Viola-Jones

This model uses traditional computer vision techniques, such as k-means clustering for segmentation and the Viola-Jones algorithm for object detection.

### Method #3: YOLOv5 Object Detection

This model uses deep learning to detect flies from already cropped images.

This project is being developed for a study that tracks Drosophila adaptation to seasonal insecticide pressure being conducted in the Petrov Lab at Stanford Univeristy.

I'm new to machine learning and computer vision. I would appreciate any mentorship and advice.
