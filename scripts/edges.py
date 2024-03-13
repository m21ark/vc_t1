import cv2
import numpy as np
import math

# Use Edge Detection to identify the boundaries of the lego pieces
def applyCanny(image, minThreshold=50, maxThreshold=200):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny Edge Detection
    imgCanny = cv2.Canny(blurred, minThreshold, maxThreshold)

    return imgCanny

def applyHoughLines(imgOriginal, imgCanny, numVotes=60, minLineLength=50, maxLineGap=0):
    # Create a copy of the original image
    imgLines = np.copy(imgOriginal)
    
    # Detect lines using HoughLines
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, numVotes, minLineLength, maxLineGap, 0)

    # Draw the lines on the image
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(imgLines, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
    return imgLines, lines