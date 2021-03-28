from EdgeDetector import *
from Hough import *
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2


class visualize(object):

    def __init__(self, dataset_path='/Users/beyzacevik/Downloads/LicensePlateDataset', original_path='/Users/beyzacevik/Downloads/LicensePlateDataset/images', annotations_path='/Users/beyzacevik/Downloads/LicensePlateDataset/annotations'):

        self.dataset_path = dataset_path
        self.original_path = original_path
        self.annotations_path = annotations_path
        self.edge_detector = EdgeDetector()

    def execute(self):

        gt_xy = self.extract_metadata()
        ious = list()
        for filename in os.listdir(self.original_path):
            all_images = []
            car_name = filename.split('/')[-1].replace('.png', '')
            gt_coordinates = gt_xy[car_name]
            original_image = cv2.imread(os.path.join(self.original_path, filename))
            raw_image = cv2.imread(os.path.join(self.original_path, filename))
            edge_map, interest_area_coordinates = self.edge_detector.contour(original_image)
            masked_edge_map, interest_area_image = self.edge_detector.reduce_noise(raw_image, edge_map, interest_area_coordinates)

            hough = Hough(image=masked_edge_map)
            hough_space = hough.hough_space()
            vertical_hough_lines, horizontal_hough_lines = hough.extract_vertical_horizontal(hough_space)
            marked_image_bbox, marked_image, bottom_left, top_right = hough.choose_plate_corners(original_image, hough_space, vertical_hough_lines, horizontal_hough_lines)
            iou = 0
            if bottom_left != None and top_right != None:
                iou = hough.calculate_iou(original_image, gt_coordinates, bottom_left, top_right)
            ious.append(iou)
            iou = "{:.2f}".format(iou)

            lined_image = hough.draw(raw_image.copy(), vertical_hough_lines)
            lined_image = hough.draw(lined_image, horizontal_hough_lines)


            f, axs = plt.subplots(2,3)

            axs[1,1].imshow(raw_image)
            axs[1,1].imshow(marked_image)
            axs[1,1].axis("off")
            axs[1,1].set_title('Intersection Points')


            axs[0,0].imshow(edge_map, cmap=plt.cm.gray, interpolation='nearest')
            axs[0,0].axis("off")

            axs[0,0].set_title("Edge Map with noise")

            axs[0,1].imshow(interest_area_image)
            axs[0,1].axis("off")
            axs[0,1].set_title("Region of Interest")

            axs[0,2].imshow(masked_edge_map, cmap=plt.cm.gray, interpolation='nearest')
            axs[0,2].axis('off')
            axs[0,2].set_title("Masked Edge Map")

            axs[1,0].imshow(lined_image)
            axs[1,0].axis("off")
            axs[1,0].set_title("Hough Lines")


            axs[1,2].imshow(marked_image_bbox)
            axs[1, 2].set_title("Result IOU: "+ iou)
            axs[1,2].axis("off")


            # f.savefig('/Users/beyzacevik/Desktop/houghCarLicense/'+car_name+'.png')
            plt.show()
            plt.pause(0.001)

    def extract_metadata(self):

        gt_xy = dict()
        for filename in os.listdir(self.annotations_path):
            file = os.path.join( self.annotations_path, filename)
            car_name = filename.split('/')[-1].replace('.xml', '')
            try:
                root = ET.parse(file)
                root = list(root.iter())
                root = root[-4::]
            except:
                print(file)
                continue

            coordinates = list(map(int, [child.text for child in root]))
            gt_xy[car_name] = coordinates

        return gt_xy




