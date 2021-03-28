import numpy as np
import cv2 as cv
import math


class Hough(object):
    def __init__(self, image, theta_min=0, theta_max=180, threshold=15):
        self.image = image
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.threshold = threshold
        self.x = image.shape[1]  # image [y,x] şeklinde okunur
        self.y = image.shape[0]

        self.rho_max = int(round(math.hypot(self.x, self.y)))
        self.thetas = np.deg2rad(np.arange(theta_min, theta_max))

    def hough_space(self):

        num_rhos = 2 * self.rho_max + 1  # -rho max to rho max
        num_thetas = len(self.thetas)
        accumulator = np.zeros((num_thetas, num_rhos))

        for x in range(self.x):
            for y in range(self.y):
                if self.image[y, x] > self.threshold:
                    for ind, theta in enumerate(self.thetas):
                        cos_theta = math.cos(theta)
                        sin_theta = math.sin(theta)
                        rho = int(round(x * cos_theta + y * sin_theta))
                        accumulator[ind, rho + self.rho_max] += 1

        return accumulator


    def find_local_maximas(self, accumulator, no_of_lines):

        accumulator = np.asarray(accumulator)
        flat_accum = accumulator.flatten()
        indices = np.argpartition(-flat_accum, no_of_lines)[:no_of_lines]
        indices = np.unravel_index(indices, accumulator.shape)
        return indices

    def extract_hough_lines(self, local_maximas):

        lines = []
        for theta_ind, rho_ind in zip(local_maximas[0], local_maximas[1]):
            theta = np.deg2rad(theta_ind)
            rho = rho_ind - self.rho_max
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            x1, y1 = int(x0 + 1000 * (-sin_t)), int(y0 + 1000 * (cos_t))
            x2, y2 = int(x0 - 1000 * (-sin_t)), int(y0 - 1000 * (cos_t))
            lines.append((theta, rho, (x1, y1), (x2, y2)))

        return lines

    def intersection(self, line1, line2):
        l1_x1, l1_y1 = line1[2]
        l1_x2, l1_y2 = line1[3]
        l2_x1, l2_y1 = line2[2]
        l2_x2, l2_y2 = line2[3]

        d = (l2_y2 - l2_y1) * (l1_x2 - l1_x1) - (l2_x2 - l2_x1) * (l1_y2 - l1_y1)  # if not parallel
        if d:
            l1 = ((l2_x2 - l2_x1) * (l1_y1 - l2_y1) - (l2_y2 - l2_y1) * (l1_x1 - l2_x1)) / d
            l2 = ((l1_x2 - l1_x1) * (l1_y1 - l2_y1) - (l1_y2 - l1_y1) * (l1_x1 - l2_x1)) / d
        else:
            return
        if not (0 <= l1 <= 1 and 0 <= l2 <= 1):
            return
        x = l1_x1 + l1 * (l1_x2 - l1_x1)
        y = l1_y1 + l1 * (l1_y2 - l1_y1)

        return [int(x), int(y)]

    def segmented_intersections(self, vertical_lines, horizontal_lines):
        intersections = []
        for v_line in vertical_lines:
            for h_line in horizontal_lines:
                if self.intersection(v_line, h_line) != None:
                    intersections.append(self.intersection(v_line, h_line))
        return intersections

    def mark_intersections(self, intersections, image):

        for point in intersections:
            cv.drawMarker(image, (point[0], point[1]), (255, 165, 0), markerType=cv.MARKER_SQUARE, markerSize=10,
                          thickness=3, line_type=cv.LINE_AA)

        return image

    def extract_vertical_horizontal(self, hough_space):

        # Points with 0<theta<10 and 170<theta<180 are vertical lines
        vertical_first = hough_space[0:10, :]
        vertical_second = hough_space[170:180, :]
        vertical = np.vstack((vertical_first, vertical_second))

        # Points with highest votes are horizontal
        horizontal = hough_space[:, :]

        vertical_maximas = self.find_local_maximas(vertical, 16)
        horizontal_maximas = self.find_local_maximas(horizontal, 16)

        vertical_lines = self.extract_hough_lines(vertical_maximas)
        horizontal_lines = self.extract_hough_lines(horizontal_maximas)

        return vertical_lines, horizontal_lines


    def choose_plate_corners(self, image, hough_space, vertical_lines, horizontal_lines):

        only_marked_img = image.copy()
        intersection_points = self.segmented_intersections(vertical_lines, horizontal_lines)
        # Get only positive values, there are neg. values as
        # -+1000 used to find points on lines
        intersection_points_pos = [p for p in intersection_points if p[0] > 0 and p[1] > 0]

        only_marked_img = self.mark_intersections(intersection_points_pos, only_marked_img)

        sorted_intersections = sorted(intersection_points_pos , key = lambda tup: (tup[0]), reverse = True )


        max_diff1 = -999999
        max_diff2 = -999999
        bottom_left = -1
        top_right = -1
        for point in sorted_intersections:
            diff1 = point[1] - point[0]
            if diff1 > max_diff1:
                bottom_left = point
                max_diff1 =  diff1
        # There might not be bottom left corner if no area is masked
        if bottom_left == -1:
            return image, only_marked_img, None, None

        for point in sorted_intersections:
            diff2 = point[0] - point[1]
            if diff2 > max_diff2:
                top_right = point
                max_diff2 = diff2

        # There might not be top right corner if no area is masked
        if top_right == -1:
            return image, only_marked_img, None, None

        # Draw the bounding box
        cv.rectangle(image, (bottom_left[0], bottom_left[1]),(top_right[0], top_right[1]), (0, 255, 0), 2)
        image = self.mark_intersections([bottom_left, top_right], image)

        return image, only_marked_img, bottom_left, top_right

    def calculate_iou(self,image, gt_coordinates, bottom_left, top_right):
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_coordinates
        pred_xmin, pred_ymax, pred_xmax, pred_ymin = bottom_left[0], bottom_left[1],top_right[0], top_right[1]

        inter_xmin = max(pred_xmin, gt_xmin)
        inter_xmax = min(pred_xmax, gt_xmax)
        inter_ymin = max(pred_ymin, gt_ymin)
        inter_ymax = min(pred_ymax, gt_ymax)

        iw = np.maximum(inter_xmax - inter_xmin + 1., 0.)
        ih = np.maximum(inter_ymax - inter_ymin + 1., 0.)

        inters = iw * ih

        union = ((pred_xmax - pred_xmin + 1.) * (pred_ymax - pred_ymin + 1.) +
               (gt_xmax - gt_xmin + 1.) * (gt_ymax - gt_ymin + 1.) -
               inters)

        iou = inters / union

        return iou

    def draw(self, img, lines):
        for line in lines:
            cv.line(img, line[2], line[3], (0, 0, 255), 1, cv.LINE_AA)
        return img
