import cv2
import numpy as np

#gray scale image for canny edge detection
image = cv2.imread('R-img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#external edges countours detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

#maximum width 
aligned_widths = []
for contour in contours:
    if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 4:
        x, y, width, height = cv2.boundingRect(contour)
        aligned_widths.append(width)
aligned_width = max(aligned_widths)

#max height 
aligned_heights = []
for contour in contours:
    if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 4:
        x, y, width, height = cv2.boundingRect(contour)
        aligned_heights.append(height)
aligned_height = max(aligned_heights)


#spacing and padding between rectangles
spacing = 20  
padding = 50  

#for grid
num_rows = len(contours)
num_cols = 2  

#total width and height of the aligned image
total_width = num_cols * aligned_width + (num_cols - 1) * spacing + 2 * padding
total_height = num_rows * aligned_height + (num_rows - 1) * spacing + 2 * padding

#blank canvas to draw rectangles
aligned_image = np.zeros((total_height, total_width, 3), dtype=np.uint8) 


x_offset = padding  
y_offset = padding  

for contour in contours:
    #rectangle approximation
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        #sort
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        #find the longer side of the rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        if width < height:
            width, height = height, width
            #rotate by 90 degrees
            box = np.roll(box, 1, axis=0)

        src_pts = box.astype("float32")

        #final plotting points
        dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

        #perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aligned_rect = cv2.warpPerspective(image, M, (width, height))
        aligned_image[y_offset:y_offset + height, x_offset:x_offset + width] = aligned_rect
        x_offset += aligned_width + spacing  

        #move to the next row if needed
        if x_offset + aligned_width + padding > total_width:
            x_offset = padding
            y_offset += aligned_height + spacing

cv2.imshow('Aligned Rectangles', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
