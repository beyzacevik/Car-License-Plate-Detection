import numpy as np
import cv2
from matplotlib import pyplot as plt


class EdgeDetector(object):

    def __init__(self, low_threshold=50, high_threshold=150, kernel_size=3):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size

    def canny(self, image):  # Returns edge map

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to gray scale
        d, sigma_color, sigma_space = 11, 17, 17
        filtered_img = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)  # Remove noise while protecting the edges
        edges = cv2.Canny(filtered_img, self.low_threshold, self.high_threshold, self.kernel_size)

        return edges



    def contour(self, image):

        # Extract the edge map
        edged = self.canny(image)
        # Calculate the contours
        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Contours are sorted to most probable candidates
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # Traverse contours and find n probably rectangle candidates
        possible_coordinates =[]
        for c in contours:

            n = 0
            peri = cv2.arcLength(c, True)
            epsilon = 0.025 * peri
            approx = cv2.approxPolyDP(c, epsilon, True) # Approximate to the closed shapes

            if len(approx) == 4:  # Choose quadrilateral shapes
                possible_coordinates.append(approx)
                n +=1
                break

        return edged, possible_coordinates

    def reduce_noise(self, image, edged_image, interest_area_coordinates):

        interest_area = image.copy()
        # Create a masking array for the edge map
        mask = np.zeros((image.shape[0], image.shape[1]))

        # Scale a little bit bigger to preserve and strengthen the edges in the edge map
        for c in interest_area_coordinates:
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            c_norm = c - [cx, cy]
            c_scaled = c_norm * 1.2
            c_scaled = c_scaled + [cx, cy]
            c_scaled = c_scaled.astype(np.int32)

            cv2.drawContours(mask, [c_scaled], -1, (255),-1)
            cv2.drawContours(interest_area, [c_scaled], 0, (255,0,0), 2)

        # Activate only the interest areas and deactive the rest
        mask[np.where(mask == 255)] = 1
        # Masked the edge map by multiplying masking array
        masked_edge_map = np.multiply(edged_image, mask)

        return masked_edge_map, interest_area