import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import os

def wait(windowName):
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Check if a key is pressed (excluding special cases)
            break

        # Check if the window is closed
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            break

def loadImage(num, dataDir='imgs'):
    img = cv2.imread(os.path.join(dataDir, f'{num}.jpg'))
    resize_ratio = 0.1
    img = cv2.resize(img, (0, 0), fx = resize_ratio, fy = resize_ratio)
    return img

def resizeImage(image, screen_height, screen_width):
    # Calculate the scaling factor to fit the image within the screen
    scaling_factor = min(screen_width / image.shape[1], screen_height / image.shape[0])

    # Resize the image
    resized_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor)
    return resized_image

def render(image, convertRGB=False):
    if convertRGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    showImages([image], ['Image'])

def showImages(images, titles, inWindow=False):
    if inWindow:
        for i in range(len(images)):
            # Get the dimensions of the screen
            screen_height, screen_width = 1080, 1920

            # Resize the image
            resized_image = resizeImage(images[i], screen_height, screen_width)

            cv2.imshow(titles[i], resized_image)
        
        for i in range(len(images)):
            wait(titles[i])
            
        cv2.destroyAllWindows()
    else:
        for i in range(len(images)):
            # Convert image to RGB format
            if images[i].dtype == np.float64:
                image = cv2.convertScaleAbs(images[i])
            else:
                image = images[i]
            if len(image.shape) == 3 and image.shape[2] == 3: # BGR or RGB
                # Check if the image is in BGR format
                if np.array_equal(image[:, :, 0], image[:, :, 2]):
                    print("Converting BGR to RGB")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert numpy array to bytes
            img_bytes = cv2.imencode('.png', image)[1].tobytes()
            # Display the image in the notebook
            display(Image(data=img_bytes))

