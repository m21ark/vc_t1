import cv2
import numpy as np
import os
import math
from sklearn.cluster import KMeans
from scripts.utils import *
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# 1. K-means clustering to get color regions
def kmeansBlur(img, blurQuantity = 3, clusterSize = 10):
    blurred = cv2.GaussianBlur(img, (blurQuantity, blurQuantity), 0)
    
    pixels = blurred.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
    num_clusters = clusterSize
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img.shape))
    
    return segmented_image    

# 2. Get the edges of the image with Canny edge detection
def getEdges(img, cannyThreshold1 = 100, cannyThreshold2 = 200, showGray = False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if(showGray):
        render(gray)
    edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
    edge_mask = np.zeros_like(img)
    edge_mask[edges > 0] = (255, 255, 255)
    edge_image = cv2.bitwise_and(img, edge_mask)
    return edges, edge_image

# 3. Get the contours of the edges in the image
def getContours(img, edges, connectSize = 3):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (connectSize, connectSize))
    dilated_edges = cv2.dilate(edges, kernel)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = img.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image, contours

# 4. Get the bounding boxes for the legos based on the contours edges
def getBoundingBoxes(img, contours, minPossibleArea = 100, intersectionThreshold = 0.2):
    
    num_legos = 0
    bounding_box_image = img.copy()
    bounding_rectangles = []

    for contour in contours:
        
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Ignore small area rectangles because they are likely noise
        if area < minPossibleArea:
            continue
        
        # Check if this bounding rectangle overlaps with any previously drawn ones
        overlap = False
        contained = False
        for (bx, by, bw, bh) in bounding_rectangles:
            intersection_area = max(0, min(x+w, bx+bw) - max(x, bx)) * max(0, min(y+h, by+bh) - max(y, by))
            union_area = area + bw*bh - intersection_area
            if intersection_area / union_area > intersectionThreshold:
                overlap = True
                # print("OVERLAP")
                break
            if x >= bx and y >= by and x + w <= bx + bw and y + h <= by + bh:
                contained = True
                # print("CONTAINED")
                break
        
        # If not fully contained and no significant overlap, draw the bounding box and increment lego count
        if not contained and not overlap:
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            bounding_rectangles.append((x, y, w, h))
            num_legos += 1
            
    return num_legos, bounding_box_image, bounding_rectangles


# 5.1. Get the main frequency colors of the original image
def getMainColors(img, blurQuantity = 3, numColors = 200, showHistogram = False):
    blurred = cv2.GaussianBlur(img, (blurQuantity, blurQuantity), 0)
    bgr = cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
    pixels = bgr.reshape(-1, 3)
    pixel_tuples = [tuple(pixel) for pixel in pixels]

    # Get the most common colors
    color_counter = Counter(pixel_tuples)
    most_common_colors = color_counter.most_common(numColors)
    colors, counts = zip(*most_common_colors)
    colors = np.array(colors)
    
    if showHistogram:
        plt.bar(range(len(colors)), counts, color=colors/255)
        plt.show()
        
    return colors

# 5.2 Get the image contained in a bounding box
def getBoundingBoxImage(img, box):
    x, y, w, h = box
    cropped_image = img[y:y+h, x:x+w]
    return cropped_image
   

# 5.3. Remove the main colors of the image from the bounding boxes
def remove_similar_colors(img, most_common_colors, blurQuantity = 3, threshold_distance = 100):
    
    replacement_color = (254, 254, 254)
    new_img = img.copy()
    new_img = cv2.GaussianBlur(new_img, (blurQuantity, blurQuantity), 0)

    # Iterate over each pixel in the image and replace the pixel color if it is similar to one of the most common colors
    for y in range(new_img.shape[0]):
        for x in range(new_img.shape[1]):
            distances = [np.linalg.norm(np.array(new_img[y, x]) - np.array(color)) for color in most_common_colors]
            if min(distances) < threshold_distance:
                new_img[y, x] = replacement_color

    return new_img
    
# 5.4 Display a square of a specific color in RGB format
def display_color_square(color):
    color = (color[0] / 255, color[1] / 255, color[2] / 255)
    _, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rectangle = plt.Rectangle((0, 0), 1, 1, color=color)
    ax.add_patch(rectangle)
    name = f"({color[0] * 255:.0f}, {color[1] * 255:.0f}, {color[2] * 255:.0f})"
    ax.text(0.5, 0.5, name, ha='center', va='center', color='black', bbox=dict(facecolor='white', edgecolor='white'))
    ax.axis('off')
    plt.show()   
    
def getLegoColor(lego_img, colorSimilarityThreshold = 1, numClusters = 5):

    most_frequent_colors = []
    most_frequent_colors.append((254, 254, 254))

    def color_distance(color1, color2):
        return np.sqrt(np.sum((color1 - color2) ** 2))
    
    bgr = cv2.cvtColor(lego_img, cv2.COLOR_RGB2BGR)

    pixels = bgr.reshape(-1, 3)
    kmeans = KMeans(n_clusters = numClusters, n_init = 10, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    color_counter = Counter(kmeans.labels_)
    sorted_colors = sorted(color_counter.items(), key=lambda x: x[1], reverse=True)
    most_freq_color = None
    
    # Find the most frequent color that isn't one of the previously found common colors
    for color_index, _ in sorted_colors:
        color = colors[color_index]
        if not any(color_distance(color, common_color) < colorSimilarityThreshold for common_color in most_frequent_colors):
            most_freq_color = color
            most_frequent_colors.append(color)
            break 
        
    return most_freq_color

# 6. Evaluate results
def guessPieceCount(imgID, legoNum, showResults = False):
    # Load the csv file as pd
    df = pd.read_csv("scripts/lego_sets.csv")
    
    # compare legoNum to column piece_count in id row
    piece_count = df.loc[df['id'] == imgID, 'piece_count'].values[0]
    
    if(showResults):
        if(legoNum == piece_count):
            print("Guessed correct number!")
        else:
            (f"Guessed: {legoNum} | Actual: {piece_count} legos") 
        
    return abs(legoNum - piece_count), piece_count 


# =========================== MODEL ===========================

def model(image_id):
    og_img = loadImage(image_id)
    segmented_img = kmeansBlur(og_img, 3, 15)
    edges, edge_img = getEdges(segmented_img, 100, 200)
    contour_image, contours = getContours(og_img, edges, 3)
    num_legos, bounding_box_image, boxes = getBoundingBoxes(og_img, contours, 200, 0.25)
    delta, piece_count = guessPieceCount(image_id, num_legos)
    return delta, piece_count, edge_img, bounding_box_image
