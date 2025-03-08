import cv2 
import math
import numpy as np

from itertools import combinations
import sys
from PyQt5.QtGui import QPixmap, QImage

class app(object):
    def __init__(self):
        
        self.gray_image = None
        self.image = None
    
    def rgb_to_grayscale(self, rgb_image):
    # Convert RGB image to grayscale using luminance method
      self.gray_image = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # Ensure that the grayscale image has a valid shape (height, width)
      if len(self.gray_image.shape) == 3:  # Check if the grayscale image has a third dimension
          gray_image =self.gray_image.squeeze()  # Remove the third dimension if it exists

      height, width = self.gray_image.shape

    # Convert grayscale NumPy array to bytes
      gray_image_bytes = self.gray_image.tobytes()

    # Create QImage from bytes
      qimage = QImage(gray_image_bytes, width, height, width, QImage.Format_Grayscale8)
      pixmap = QPixmap.fromImage(qimage)
      return pixmap
      
    def cany_edg_detection(self, image):
        # Convert image to grayscale
        grayscale_image = self.gray_image
        
        # Apply Gaussian blur to the grayscale image
        blurred_img = self.gaussian_blur(grayscale_image, kernel_size=3, std=1)
        
        # Find gradient magnitude and direction using Sobel operator
        gradient_magnitude, gradient_direction = self.find_graident(blurred_img, 5)
        
        # Quantize gradient directions into 8 categories
        direction_categories = np.round((gradient_direction + np.pi) / (np.pi / 4)) % 8
        
        # Perform non-maximum suppression to thin out edges
        suppress_edge = self.non_max_suppression(gradient_magnitude, direction_categories)
        
        # Define high and low thresholds for edge detection
        T_high = np.max(suppress_edge) * 0.18
        T_low = np.max(suppress_edge) * 0.08
        
        # Categorize edges into strong, weak, and non-edges
        strong_edges, weak_edges, magnitude = self.categorize_edges(suppress_edge, T_low, T_high)
        
        # Perform edge tracking to link weak edges to strong ones
        cany_edg_detection = self.Edg_Tracking(magnitude, weak_edges, strong_edges)
        
        # Get height and width of the resulting edge-detected image
        height, width = cany_edg_detection.shape
        
        # Convert the image array to uint8 format
        cany_edg_detection = cany_edg_detection.astype(np.uint8)
        
        # Convert the numpy array to QImage format
        qimage = QImage(cany_edg_detection.data, width, height, width, QImage.Format_Grayscale8)
        
        # Convert QImage to QPixmap for display purposes
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale the QPixmap to fit the width
        pixmap = pixmap.scaledToWidth(width)
        
        # Return the scaled QPixmap
        return pixmap

        
        
    def find_gradient(self, image, ksize):
    # Define Sobel kernels for x and y directions
      sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

      sobel_y_kernel = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])

    # Convolve image with Sobel kernels
      sobel_x = self.convoltion(image, sobel_x_kernel)
      sobel_y = self.convoltion(image, sobel_y_kernel)

    # Compute gradient magnitude
      gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Compute gradient direction (in radians)
      direction_categories = np.arctan2(sobel_y, sobel_x)

      return gradient_magnitude, direction_categories  
    def convoltion(self, image, kernel):
    # Convert the image and kernel to numpy arrays
      image = np.asarray(image)
      kernel = np.asarray(kernel)
    
    # Get the dimensions of the image and kernel
      image_height, image_width = image.shape
      kernel_height, kernel_width = kernel.shape
    
    # Compute the padding needed for valid convolution
      pad_height = kernel_height // 2
      pad_width = kernel_width // 2
    
    # Initialize the output image
      output = np.zeros_like(image, dtype=np.float64)
    
     # Pad the image
      padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Flip the kernel
      flipped_kernel = np.flipud(np.fliplr(kernel))
    
    # Perform convolution
      for y in range(image_height):
          for x in range(image_width):
              output[y, x] = np.sum(flipped_kernel * padded_image[y:y+kernel_height, x:x+kernel_width])
    
      return output      
         
    def gaussian_blur(self, image, kernel_size=5, std=10):
   
      self.kernel = self.gaussian_kernel(kernel_size, std)
    
    # Convolve the image with the kernel
      blurred_image = self.convolve(image,self.kernel)
    
      return blurred_image

    def gaussian_kernel(self, size, sigma):
   
      kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
      return kernel / np.sum(kernel)

    def convolve(self, image, kernel):
      image = np.asarray(image)
      kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel

    # Get dimensions
      kernel_height, kernel_width = kernel.shape
      image_height, image_width = image.shape

    # Initialize output image
      output = np.zeros_like(image, dtype=np.float64)  # Set dtype to float64

    # Apply kernel to image
      for y in range(image_height - kernel_height + 1):
          for x in range(image_width - kernel_width + 1):
            # Perform convolution
              output[y:y+kernel_height, x:x+kernel_width] += kernel * image[y:y+kernel_height, x:x+kernel_width]

    # Convert output to uint8
      output = np.uint8(output)

      return output

    def non_max_suppression(self,gradient_magnitude, direction_categories):
      suppressed_edges = np.zeros_like(gradient_magnitude)
      height, width = gradient_magnitude.shape
    
      for y in range(1, height - 1):
          for x in range(1, width - 1):
              direction = direction_categories[y, x]  # Direction in radians
            
            # Convert direction to degrees
              direction_deg = np.degrees(direction) % 180
            
              if (0 <= direction_deg < 22.5) or (157.5 <= direction_deg <= 180):  # East-West direction
                neighbors = [gradient_magnitude[y, x-1], gradient_magnitude[y, x+1]]
              elif (22.5 <= direction_deg < 67.5):  # Northeast-Southwest direction
                  neighbors = [gradient_magnitude[y-1, x-1], gradient_magnitude[y+1, x+1]]
              elif (67.5 <= direction_deg < 112.5):  # North-South direction
                  neighbors = [gradient_magnitude[y-1, x], gradient_magnitude[y+1, x]]
              else:  # Northwest-Southeast direction
                neighbors = [gradient_magnitude[y-1, x+1], gradient_magnitude[y+1, x-1]]
            
              if gradient_magnitude[y, x] >= max(neighbors):
                  suppressed_edges[y, x] = gradient_magnitude[y, x]
              
      return suppressed_edges


    def categorize_edges(self,gradient_magnitude, T_low, T_high):
    # Create strong edges and weak edges based on thresholds
       strong_edges = gradient_magnitude > T_high
       weak_edges = (gradient_magnitude >= T_low) & (gradient_magnitude <= T_high)
       gradient_magnitude[gradient_magnitude < T_low] = 0
    
       return strong_edges, weak_edges,gradient_magnitude


    def Edg_Tracking(self,gradient_magnitude, weak_edges, strong_edges):
       for i in range(1, gradient_magnitude.shape[0] - 1):
           for j in range(1, gradient_magnitude.shape[1] - 1):
               if weak_edges[i, j]:
                # Create a window around the current pixel in gradient_magnitude
                  window = gradient_magnitude[i-1:i+2, j-1:j+2]
                # Check if at least one of the neighboring pixels in window is associated with a strong edge
                  if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    # At least one neighbor is associated with a strong edge, keep the value unchanged
                      pass
                  else:
                    # Suppress weak edge if none of the neighbors are associated with strong edges
                      gradient_magnitude[i, j] = 0

       return gradient_magnitude
  

     
    
    def detect_line(self, image):
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      lines = self.hough_lines(gray, threshold=150)
    
    # Draw lines on the image
      self.draw_lines(image, lines)
    
    # Convert the modified image to bytes
      modified_image_bytes = image.astype(np.uint8).tobytes()

    # Get image dimensions
      height, width = image.shape[:2]

    # Create QImage from bytes
      qimage = QImage(modified_image_bytes, width, height, QImage.Format_RGB888)
      pixmap = QPixmap.fromImage(qimage)
      return pixmap

    def hough_lines(self, img, threshold):
        # Define the Hough space accumulator
        height, width = img.shape[:2]
        max_dist = int(np.ceil(np.sqrt(height**2 + width**2)))
        theta_range = np.deg2rad(np.arange(-90, 90))

        # Initialize accumulator array
        accumulator = np.zeros((2 * max_dist, len(theta_range)), dtype=np.uint8)

        # Find edge pixels (using Canny edge detection)
        edges = cv2.Canny(img, 75, 150)

        # Voting process
        edge_pixels = np.nonzero(edges)
        for y, x in zip(*edge_pixels):
            for theta_idx, theta in enumerate(theta_range):
                rho = int(x * np.cos(theta) + y * np.sin(theta))
                accumulator[rho + max_dist, theta_idx] += 1

        # Extract lines from the accumulator based on the threshold
        lines = []
        for rho_idx in range(accumulator.shape[0]):
            for theta_idx in range(accumulator.shape[1]):
                if accumulator[rho_idx, theta_idx] > threshold:
                    rho = rho_idx - max_dist
                    theta = theta_range[theta_idx]
                    lines.append((rho, theta))

        return lines

  # problem with finding equation of endpoint of lines and draw them 
    def draw_lines(self, img, lines):
      for rho, theta in lines:
        # Convert polar coordinates to Cartesian coordinates for two points on the line
        x1 = int(rho * np.cos(theta) - 1000 * np.sin(theta))
        y1 = int(rho * np.sin(theta) + 1000 * np.cos(theta))
        x2 = int(rho * np.cos(theta) + 1000 * np.sin(theta))
        y2 = int(rho * np.sin(theta) - 1000 * np.cos(theta))
        
        # Draw the line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
   
          
    def Hough_Circle_Transform(self,image, threshold, min_radius=10, max_radius=200, canny_min_threshold=100, canny_max_threshold=200):
    # Convert the image to grayscale
      gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection
      edges = cv2.Canny(gray_image, canny_min_threshold, canny_max_threshold)

    # Get image dimensions
      rows, cols = gray_image.shape[:2]

    # Define Hough parameters
      num_thetas = 100
      dtheta = int(360 / num_thetas)
      theta_bins = 360

    # Initialize 3D Accumulator
      accumulator = np.zeros((cols + 1, rows + 1, max_radius - min_radius + 1), dtype=np.uint8)

    # Loop over edge points
      for y in range(rows):
          for x in range(cols):
              # Check if edge point
              if edges[y, x] > 0:
                  # Loop over r values
                  for r in range(min_radius, max_radius):
                      # Loop over theta values
                      for theta in range(0, 360, dtheta):
                        # Calculate x_center and y_center
                          x_center = x - r * np.cos(np.deg2rad(theta))
                          y_center = y - r * np.sin(np.deg2rad(theta))
                        # check that xcenter and ycenter is in valid range
                          if 0 <= int(x_center) < cols and 0 <= int(y_center) < rows:
                            # Increment accumulator
                              accumulator[int(x_center), int(y_center), r - min_radius] += 1

    # Create circles image
      circles_img = image.copy()

    # Loop over accumulator to find circles
      for y in range(rows):
          for x in range(cols):
              for r in range(min_radius, max_radius):
                # Check for accumulator threshold
                  if accumulator[x, y, r - min_radius] > threshold:
                    # Draw circle
                      cv2.circle(circles_img, (x, y), r - min_radius, (255, 0, 0), 1, cv2.LINE_AA)

      return circles_img
        
    def Hough_Ellipse_Transform(self, image):
        # Apply Gaussian blur to the input image
        blurred = self.gaussian_blur(image)
        
        # Perform Canny edge detection to detect edges in the blurred image
        edges = cv2.Canny(blurred, 50, 150)
        
        # Extract pixel coordinates of edge points
        edge_pixels = np.column_stack(np.where(edges > 0))
        
        # Generate combinations of edge pixel pairs
        combinations_list = np.array(list(combinations(edge_pixels, 2)))
        edge_pairs = combinations_list
        
        # Set minimum values for ellipse parameters and relative vote
        alpha_min = 10  # Minimum value for alpha
        relative_vote_min = 0.2  # Minimum relative vote threshold
        
        # Detect ellipses using the provided edge pairs and minimum parameters
        ellipses = self.detect_ellipses(edge_pairs, alpha_min, relative_vote_min)
        
        return ellipses


    def calculate_parameters(self, t, u):
        # Calculate parameters of the ellipse given two edge pixels
        tx, ty = t
        ux, uy = u
        
        # Calculate center of the ellipse
        ox = (tx + ux) / 2
        oy = (ty + uy) / 2
        
        # Calculate semi-major axis length
        alpha = np.sqrt((ux - tx) ** 2 + (uy - ty) ** 2) / 2
        
        # Calculate orientation angle of the ellipse
        theta = np.arctan2(uy - ty, ux - tx)
        
        return ox, oy, alpha, theta


    def calculate_distance_to_major_axis(self, pixel, ox, oy, alpha):
        # Calculate the distance of a pixel to the major axis of the ellipse
        px, py = pixel
        return np.sqrt((px - ox) ** 2 + (py - oy) ** 2)


    def calculate_beta(self, pixel, ox, oy, alpha, theta):
        # Calculate beta parameter of the ellipse given a pixel and ellipse parameters
        px, py = pixel
        
        # Calculate distance from the pixel to the center of the ellipse
        delta = np.sqrt((py - oy) ** 2 + (px - ox) ** 2)
        
        # Calculate the projection of the pixel onto the major axis of the ellipse
        gamma = np.sin(np.abs(theta)) * (py - oy) + np.cos(np.abs(theta)) * (px - ox)
        
        # Calculate beta parameter of the ellipse
        beta = (alpha ** 2 * delta ** 2 - alpha ** 2 * gamma ** 2) / (alpha ** 2 - gamma ** 2)
        
        return beta


    def compute_circumference(self, alpha, beta):
        # Compute the circumference of the ellipse using semi-major axis and beta parameter
        circumference = 2 * np.pi * np.sqrt((alpha ** 2 + beta ** 2) / 2)
        
        return circumference

    def detect_ellipses(self, edge_pairs, alpha_min, relative_vote_min):
        # List to store detected ellipses
        detected_ellipses = []

        # Iterate over pairs of edge pixels
        for pair in edge_pairs:
            t, u = pair
            
            # Calculate parameters of the assumed ellipse
            ox, oy, alpha, theta = self.calculate_parameters(t, u)

            # Reshape edge pairs for easy iteration
            k_pixels = edge_pairs.reshape(-1, edge_pairs.shape[-1])

            # Initialize accumulator and circumference arrays
            accumulator = np.zeros(k_pixels.shape[0])
            circumference_of_assumed_ellipse = np.zeros(k_pixels.shape[0])

            # Iterate over each pixel in the edge pairs
            for pixel in k_pixels:
                # Skip if pixel is one of the edge pixels used to define the ellipse
                if (pixel == t).all() or (pixel == u).all():
                    continue

                # Calculate distance from pixel to major axis
                dk = self.calculate_distance_to_major_axis(pixel, ox, oy, alpha)
                
                # Skip if distance is greater than semi-major axis length
                if dk > alpha:
                    continue

                # Calculate beta parameter for the pixel
                beta = self.calculate_beta(pixel, ox, oy, alpha, theta)
                
                # Increment accumulator and update circumference for the assumed ellipse
                accumulator[beta] += 1
                circumference_of_assumed_ellipse[beta] = self.compute_circumference(alpha, beta)
            
            # Adjust accumulator by subtracting relative vote
            accumulator -= circumference_of_assumed_ellipse * relative_vote_min
            
            # Find indices where accumulator is greater than 0
            beta_i = np.where(accumulator > 0)[0]

            # If any valid beta indices are found, choose the one with maximum accumulator value
            if len(beta_i) > 0:
                max_vote_beta = beta_i[np.argmax(accumulator[beta_i])]
                detected_ellipses.append((ox, oy, alpha, theta, max_vote_beta))

        return detected_ellipses


        
