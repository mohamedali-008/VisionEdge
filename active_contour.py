import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView,QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys
from time import sleep



class ActiveContour(object):

   

    def __init__(self, graphics_view):

       
        self.graphics_view = graphics_view
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
       
        self.image_path = None
        self.center = None
        self.radius = None
        self.npoints = None
        self.contour = None
        self.image = None
    
    def _update_graphics_view(self, cv_image):
        # Convert the OpenCV image to a QImage
        qimage = self._cv_to_qimage(cv_image)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # Clear the current scene
        self.scene.clear()

        # Add the pixmap to the scene
        self.scene.addPixmap(pixmap)

        # Fit the QGraphicsView to the scene's bounding rectangle, maintaining aspect ratio
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        
        

    def load_image(self, filename):
    # Store the path of the loaded image
        self.image_path = filename

        # Read the image using OpenCV and convert it to grayscale
        self.image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)

        # Update the graphics view with the loaded image
        self._update_graphics_view(self.image)


   

    def draw_contour(self, center, radius, npoints):
    # Store the center, radius, and number of points for the contour
        self.center = center
        self.radius = radius
        self.npoints = npoints
        
        # Generate points for the contour using polar coordinates
        # and convert them to Cartesian coordinates
        self.contour = np.array([[int(self.radius * np.cos(x) + self.center[0]),
                                int(self.radius * np.sin(x) + self.center[1])] 
                                for x in np.linspace(0, 2 * np.pi, self.npoints)])
    
    # Update the contour
        self._update_contour()


    def _cv_to_qimage(self, cv_image):
    # Get the height and width of the OpenCV image
        height, width = cv_image.shape
        
        # Calculate the number of bytes per line
        bytes_per_line = width
        
        # Convert the OpenCV image data to a QImage
        # Format_Grayscale8 represents an 8-bit grayscale image
        return QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        
        
    def _update_contour(self):
        # Read the original grayscale image
        self.image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)
        
        # Draw the contour on the image
        cv.polylines(self.image, [self.contour.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Update the graphics view with the image containing the contour
        self._update_graphics_view(self.image)

    def _compute_external_energy(self, image, contour):
        # Initialize external energy
        external_energy = 0
        
        # Iterate over each point in the contour
        for i in range(len(contour)):
            p = contour[i]
            p_next = contour[(i + 1) % len(contour)]
            
            # Compute gradient in x-direction and y-direction
            gradient_x = int(image[int(p_next[0]), int(p[1])]) - int(image[int(p[0]), int(p[1])])
            gradient_y = int(image[int(p[0]), int(p_next[1])]) - int(image[int(p[0]), int(p[1])])
            
            # Calculate gradient magnitude
            gradient_magnitude = gradient_x ** 2 + gradient_y ** 2
            
            # Add gradient magnitude to external energy
            external_energy += gradient_magnitude
        
        return external_energy

        

    def _compute_internal_energy(self, contour, alpha, beta):
        # Initialize internal energy
        internal_energy = 0
        
        # Iterate over each point in the contour
        for i in range(len(contour)):
            p = contour[i]
            p_prev = contour[i - 1]
            p_next = contour[(i + 1) % len(contour)]
            
            # Compute distance between consecutive points
            distance_next = np.linalg.norm(p_next - p) ** 2
            distance_prev = np.linalg.norm(p_next - 2 * p + p_prev) ** 2
            
            # Update internal energy
            internal_energy += alpha * distance_next + beta * distance_prev
        
        return internal_energy
    
    def start(self, alpha=1, beta=1, gamma=1, iterations=50):
        # Iterate over the specified number of iterations
        for _ in range(iterations):
            # Move the contour to minimize the energy
            for i in range(len(self.contour)):
                p = self.contour[i]
                neighbor_points = np.zeros((3, 3))
                x_center = neighbor_points.shape[0] // 2
                y_center = neighbor_points.shape[1] // 2
                
                # Iterate over neighbor points
                for j in range(-x_center, x_center + 1):
                    for k in range(-y_center, y_center + 1):
                        # Move the contour to the neighbor point
                        self.contour[i] = np.array([p[0] + j, p[1] + k])
                        
                        # Compute total energy
                        total_energy = self._compute_internal_energy(self.contour, alpha, beta) - gamma * self._compute_external_energy(self.image, self.contour)
                        neighbor_points[x_center + j, y_center + k] = total_energy
                        
                # Find the best neighbor location with minimum energy
                best_neighbor_loc = np.unravel_index(neighbor_points.argmin(), neighbor_points.shape)
                self.contour[i] = best_neighbor_loc + p - [x_center, y_center]
        
        # Update the contour after all iterations
        self._update_contour()


######################################################################################################################## test

