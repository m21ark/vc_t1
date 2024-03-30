# Computer Vision - Project 1

## Description

From a sample of images containing lego pieces, we want to be able to create a program that can detect:
- The number of lego pieces
- The position of each lego piece in the image
- The number of different colors

This should be done using manual image processing techniques, such as thresholding, edge detection, etc. As such, no deep learning techniques should be used.

## Example Results

The following images represent the multiple stages of the program running. In the second sequence, there is one more image, due to an extra iteration with a larger blur to reduce the hitbox count.

![image](https://github.com/m21ark/vc/assets/72521279/82a79453-fdd7-43a1-9125-9a9ada8471d4)

### Color Extraction

![image](https://github.com/m21ark/vc/assets/72521279/b4a9e6cf-9c44-4f4c-9300-8796e364b0ee)

## Developed Model Algorithm

### Step 1: K-means and Canny Edge Detection

- Apply a small Gaussian Blur
- Multiple iterations of K-means with different K values and seeds to reduce color count
- Apply Canny Edge Detection Algorithm for each one of them
- Taking the edges for each iteration and choosing the overlapping ones across iterations according to a certain threshold for more precision on edge definition

### Step 2: Contours

- Get contours based on the edges, connecting nearby ones within a certain distance

### Step 3: Bounding Boxes

- Taking each contour, the algorithm tries to find the respective bounding box
- If there are too many boxes, it restarts from step 1 but with a bigger Gaussian Blur
- Bounding box culling to filter false positives: too small/big and overlapping boxes

### Step 4: Color Detection

- For each bounding box found earlier in Step 3: 
    - Apply K-means clustering to reduce the number of colors in the bounding box
    - Apply GrabCut to remove background and leave only the lego
    - Apply a big Gaussian Blur to ensure the lego color is very homogeneous
    - Analyze pixels to get the most common color values and store it on an colors array
- With an array of lego colors contained in each bounding box, leverage the LAB color space to compare all colors pairs more accurately than RGB and remove pairs that are too close (Euclidean Distance) according to a threshold  

## Known Issues

- The results can take a while (up to 40s) to compute mainly due to the iterations of Step 1
- There are some detection problems related with false positives in shadows and corners or others where the Canny edges for shadows overlap with the legos and big bounding boxes are defined
- Some light colors (gray, pink) or that are very similar to background (gray, white, transparent) are sometimes not found as pieces when if a bigger blur needs to be applied
- Pieces that are too close are sometimes considered a single piece (like on the 3rd image) due to their shadows connecting the contours

![image](https://github.com/m21ark/vc/assets/72521279/c568e582-cd2d-4ad6-aa46-53959ba4f5b9)

## Group Members

|      Name      | Student Number |
| -------------- | -------------- |
| João Alves     |   202007614    |
| Marco André    |   202004891    |
| Rúben Monteiro |   202006478    |



