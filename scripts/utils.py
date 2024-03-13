import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image

def wait(windowName):
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Check if a key is pressed (excluding special cases)
            break

        # Check if the window is closed
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            break



def showImages(images, titles, inWindow=False):
    if inWindow:
        for i in range(len(images)):
            # Get the dimensions of the screen
            screen_height, screen_width = 1080, 1920

            # Calculate the scaling factor to fit the image within the screen
            scaling_factor = min(screen_width / images[i].shape[1], screen_height / images[i].shape[0])

            # Resize the image
            resized_image = cv2.resize(images[i], None, fx=scaling_factor, fy=scaling_factor)

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


# def showImages(images, titles, inWindow=False):
#     for i in range(len(images)):
#         if inWindow:
#             # Get the dimensions of the screen
#             screen_height, screen_width = 1080, 1920 # Put your actual screen dimensions here

#             # Calculate the scaling factor to fit the image within the screen
#             scaling_factor = min(screen_width / images[i].shape[1], screen_height / images[i].shape[0])

#             # Resize the image
#             resized_image = cv2.resize(images[i], None, fx=scaling_factor, fy=scaling_factor)

#             cv2.imshow(titles[i], resized_image)
#             wait(titles[i])
#         else:
#             plt.subplot(1, len(images), i+1)
#             plt.title(titles[i])
#             plt.xticks([]), plt.yticks([])
#             # Convert image to 8-bit unsigned integer for visualization
#             if images[i].dtype == np.float64:
#                 image = cv2.convertScaleAbs(images[i])
#             else:
#                 image = images[i]
#             # Detect if the image is BGR, RGB, or grayscale
#             if len(image.shape) == 3:
#                 if image.shape[2] == 3: # BGR or RGB
#                     # If BGR, convert to RGB
#                     if image[0, 0, 0] == image[0, 0, 2]:
#                         print("Converting BGR to RGB")
#                         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                         plt.imshow(image)
#                     else:
#                         print("Image is already in RGB")
#                         plt.imshow(image)
                    
#                 else:
#                     plt.imshow(image, cmap='gray')
            
#             # Make sure the plot is big enough
#             # plt.gcf().set_size_inches(20, 20)
#     if not inWindow:
#         plt.show()