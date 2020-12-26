import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag



def canny(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5,5), 0) 
    image_canny = cv2.Canny(image_blur, 50, 150)
    return image_canny


def region_of_intrest(image):
    height =image.shape[0]
    triangular_shape_polygon = np.array([[(220, height), (1050, height), (570, 270)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangular_shape_polygon, (255, 255, 255))
    region_of_intrest_in_image = cv2.bitwise_and(image, mask)
    return region_of_intrest_in_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image         


def make_coordinates(image, line_parameters):
    """
    docstring
    """
    try:
        m, b = line_parameters
        print(m, b)
    except:
        m, b = -1, 3
        if m==-1 and b == 3:
            print("********************************************************************************")    
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1-b)/m)
    x2 = int((y2-b)/m)
    return np.array([x1, y1, x2, y2])

def average_lines_intercept(image, lines):
    left_side_lines = []
    right_side_lines =[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        [m, b] = np.polyfit((x1, x2), (y1, y2), 1)
        if m > 0:
            right_side_lines.append((m, b))
        if m < 0:
            left_side_lines.append((m, b))
    left_lines_average = np.float32(np.average(left_side_lines, axis=0))
    right_lines_average = np.float32(np.average(right_side_lines, axis=0))
    averaged_left_line = make_coordinates(image, left_lines_average)
    averaged_right_line = make_coordinates(image, right_lines_average)           
    return np.array([averaged_left_line, averaged_right_line])

#image = cv2.imread("data\test_image.jpg")
#image_copy = np.copy(image)


video = cv2.VideoCapture("data\test2.mp4")

while(video.isOpened()):
    success, image_copy = video.read()
    canny_image = canny(image_copy)
    roi = region_of_intrest(canny_image)
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(image_copy, average_lines_intercept(image_copy, lines))
    combo_image = cv2.addWeighted(image_copy, 0.8, line_image, 1, 1)
    cv2.imshow("image",combo_image)
    if cv2.waitKey(1) & 0xFF == ord('e'): 
        break   
video.release()
cv2.destroyAllWindows()

