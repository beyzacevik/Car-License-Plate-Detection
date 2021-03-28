## Hough Space Transformation for Car License Plate Detection

Hough Space Transformation implemented from scratch.

- EdgeDetector.py includes
    * canny(): edge detection method
    * contour(): extracts possible coordinates for candidates areas
    * reduce_noise(): removes unnecessary edges
    
- Hough.py includes the methods related to creating the accumulator matrix, finding local maximas, finding intersections and drawing the lines.
    * hough_space(): creates hough space
    * find_local_maximas(): extracts local extremums on hough space
    * extract_hough_lines(): de-houghes the polar coordinates and creates lines
    * intersection(): finds intersection point of lines
    * segmented_intersections(): searches possibility for intersections between vertical and horizontal lines
    * mark_intersections(): marks intersection points
    * extract_vertical_horizontal(): groups vertical and horizontal lines in hough space according to theta
    * choose_plate_corners(): eliminates candidate intersection points and find bottom left and top right
    * calculate_iou(): calculate iou between prediction and ground truth
    * draw(): draws identified lines on the image
    
- visualize.py includes operations for reading images and visualizing the results. 
    * extract_metadata(): extracts ground truth metadata from .xml files
    * execute(): calls required commands and plots the results

**Execution**

    python run.py  'path_to_root_folder' 'path_to_image_folder' 'path_to_annotation_folder'

