import cv2
import numpy as np

## MAKE ALL THESE VALUES INTERACTIVE
lower_bgr_values = np.array([192,  193,  193])
upper_bgr_values = np.array([255, 255, 255])
MIN_AREA = 10000
MIN_AREA_TRACK = 30000
MAX_CONTOUR_VERTICES = 30

def get_contour_data(mask, out):
    """
    Return the centroid of the largest contour in
    the binary image 'mask' (the line)
    and draw all contours on 'out' image
    """

    # erode image (filter excessive brightness noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    # get a list of contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mark = {}
    line = {}
    over = False
    tried_once = False

    possible_tracks = []
    for contour in contours:
        M = cv2.moments(contour)
        # Search more about Image Moments on Wikipedia :)

        contour_vertices = len(cv2.approxPolyDP(contour, 1.0, True))
        # print("vertices: ", contour_vertices)

        if (M['m00'] < MIN_AREA):
            continue

        if (contour_vertices < MAX_CONTOUR_VERTICES) and (M['m00'] > MIN_AREA_TRACK):
            # Contour is part of the track
            line['x'] = crop_w_start + int(M["m10"]/M["m00"])
            line['y'] = int(M["m01"]/M["m00"])

            possible_tracks.append(line)

            # plot the amount of vertices in light blue
            cv2.drawContours(out, contour, -1, (255,255,0), 1)

            cv2.putText(out, str(contour_vertices), (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])),
                cv2.FONT_HERSHEY_PLAIN, 3, (100,200,150), 1)

        else:
            # plot the area in pink
            cv2.drawContours(out, contour, -1, (255,0,255), 1)
            cv2.putText(out, f"({contour_vertices}){M['m00']}", (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])),
                cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)

    return possible_tracks


image_path = "glare.jpeg"
image = cv2.imread(image_path)
mask = cv2.inRange(image, lower_bgr_values, upper_bgr_values)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.erode(mask, kernel)

data = get_contour_data(mask, image)

cv2.imshow("image", image)
# cv2.imshow("mask", mask)

cv2.waitKey(0)

# cv2.imwrite("output.png", image)
# cv2.imwrite("mask.png", mask)
