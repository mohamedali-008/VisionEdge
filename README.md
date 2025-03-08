Overview

This project implements various computer vision techniques for image processing, including edge detection, line detection, circle detection, and active contour segmentation. The goal is to extract meaningful structures from images, making it useful in applications such as medical imaging, object tracking, and autonomous navigation.

Features

Edge Detection: Implements Gaussian blurring, Sobel filtering, non-maximum suppression, and double thresholding.

Line Detection: Uses Canny edge detection followed by the Hough Transform.

Circle Detection: Implements the Hough Circle Transform with adjustable parameters.

Active Contour Segmentation: Uses the Snake algorithm for object boundary detection with energy minimization.

Image Processing Techniques

1. Edge Detection

Noise Removal: Gaussian blurring is applied with a kernel size of 3 and standard deviation 1.

Gradient Computation: Uses the Sobel operator (kernel size = 5) to detect intensity changes.

Non-Maximum Suppression: Retains only the strongest edge pixels.

Double Thresholding: Classifies pixels into strong, weak, and non-edges.

Edge Tracking by Hysteresis: Connects weak edges to strong edges for continuity.

Example Output:



2. Line Detection (Hough Transform)

Converts the image to grayscale.

Applies Canny edge detection.

Uses Hough Transform to detect lines based on a threshold.

Converts polar coordinates (rho, theta) to Cartesian coordinates for visualization.

Example Output:



3. Circle Detection (Hough Circle Transform)

Uses Canny edge detection.

Defines a Hough space accumulator.

Iterates over potential circle centers and radii to find the best matches.

Draws detected circles on the original image.

Example Output:



4. Active Contour Segmentation (Snake Algorithm)

Allows user to define an initial contour (center, radius, and number of points).

Iterates over a given number of steps, adjusting contour position.

Uses internal and external energy minimization for optimal boundary detection.

User can adjust alpha, beta, gamma, and iterations for accuracy control.

Example Output:



Installation

To set up and run the project, install the required dependencies:

pip install opencv-python numpy matplotlib

Usage

Run the image processing scripts:

python edge_detection.py --input images/sample.png --output images/edge_output.png

python hough_transform.py --input images/sample.png --output images/hough_output.png

python active_contour.py --input images/sample.png --output images/contour_output.png

Future Work

Optimize performance for real-time processing.

Implement deep learning-based contour detection.

Enhance accuracy for noisy or low-contrast images.

License

This project is licensed under the MIT License. See LICENSE for details.
