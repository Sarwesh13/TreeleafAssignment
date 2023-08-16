import cv2
import numpy as np

#canny edge detection is best on gray-scaled image
img = cv2.imread('R-img.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#probabilistic hough line transform gives better result than standard.
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 23, minLineLength=5, maxLineGap=25)

#contour detection to find rectangles, using external retrieve to only take outer edges.
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangle_data = {}
#for each detected lines
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    #length using distance formula
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    #center point of the line using mid-point formula
    line_center_x = (x1 + x2) / 2
    line_center_y = (y1 + y2) / 2
    
    #for each detected rectangles
    #point polygon test for determining a point(center of the line) is inside the polygon or not
    for contour in contours:
        is_center_inside = cv2.pointPolygonTest(contour, (float(line_center_x), float(line_center_y)), False) >= 0
        if is_center_inside:
            index = tuple(contour.ravel())  #convert contour to a 1D tuple of coordinates using numpy's ravel function.
            
            if index not in rectangle_data: 
                rectangle_data[index] = line_length
            else:
                rectangle_data[index] += line_length

#assign numbers to rectangles based on line lengths
order_of_rectangles = sorted(rectangle_data.keys(), key=lambda key: rectangle_data[key])

#corresponding-assigned numbers for each rectangles
for i, index in enumerate(order_of_rectangles):
    contour = np.array(index).reshape((-1, 1, 2))  #convert index back to array using numpy's reshape function.
    x, y, _, _ = cv2.boundingRect(contour)
    cv2.putText(img, str(i + 1), (x, y), 1, 1, (0, 0, 255), 2) #red text

cv2.imshow('numbering', img)
cv2.waitKey(0)
cv2.destroyAllWindows()