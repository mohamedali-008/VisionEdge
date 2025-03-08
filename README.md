# VisionEdge: Advanced Computer Vision for Edge and Contour Detection

## Overview
VisionEdge is a powerful **computer vision-based** system designed for detecting edges, lines, and contours in images. It integrates multiple image processing techniques such as **edge detection, Hough transform for line and circle detection, and active contour segmentation (Snake algorithm)**. This project is useful in applications like **medical imaging, object tracking, and autonomous navigation**.

## Features

- **Edge Detection**: Implements **Gaussian blurring, Sobel filtering, non-maximum suppression, and double thresholding**.
- **Line Detection**: Uses **Canny edge detection** followed by the **Hough Transform**.
- **Circle Detection**: Implements the **Hough Circle Transform** with adjustable parameters.
- **Active Contour Segmentation**: Uses the **Snake algorithm** for object boundary detection with energy minimization.

## Image Processing Techniques

### 1. Edge Detection
- **Noise Removal**: Gaussian blurring is applied with a kernel size of **3** and standard deviation **1**.
- **Gradient Computation**: Uses the **Sobel operator (kernel size = 5)** to detect intensity changes.
- **Non-Maximum Suppression**: Retains only the strongest edge pixels.
- **Double Thresholding**: Classifies pixels into strong, weak, and non-edges.
- **Edge Tracking by Hysteresis**: Connects weak edges to strong edges for continuity.

**Example Output:**

![Edge Detection](images/edge_detection.png)

### 2. Line Detection (Hough Transform)
- Converts the image to grayscale.
- Applies **Canny edge detection**.
- Uses **Hough Transform** to detect lines based on a threshold.
- Converts **polar coordinates (rho, theta) to Cartesian coordinates** for visualization.

**Example Output:**

![Line Detection](images/line_detection.png)

### 3. Circle Detection (Hough Circle Transform)
- Uses **Canny edge detection**.
- Defines a **Hough space accumulator**.
- Iterates over potential circle centers and radii to find the best matches.
- Draws detected circles on the original image.

**Example Output:**

![Circle Detection](images/circle_detection.png)

### 4. Active Contour Segmentation (Snake Algorithm)
- Allows user to define an **initial contour** (center, radius, and number of points).
- Iterates over a given number of steps, adjusting contour position.
- Uses **internal and external energy minimization** for optimal boundary detection.
- User can adjust **alpha, beta, gamma, and iterations** for accuracy control.

**Example Output:**

![Active Contour](images/active_contour.png)

## Installation
To set up and run the project, install the required dependencies:

```bash
pip install opencv-python numpy matplotlib
```

## Usage
Run the image processing scripts:

```bash
python edge_detection.py --input images/sample.png --output images/edge_output.png
```

```bash
python hough_transform.py --input images/sample.png --output images/hough_output.png
```

```bash
python active_contour.py --input images/sample.png --output images/contour_output.png
```

## Future Work
- **Optimize performance** for real-time processing.
- **Implement deep learning-based contour detection**.
- **Enhance accuracy** for noisy or low-contrast images.

## License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

