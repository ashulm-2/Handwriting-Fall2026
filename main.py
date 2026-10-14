from PIL import Image
import cv2
import numpy as np

from preprocess import preprocess_image, save_steps_grid

# Testing P_thick

thick = preprocess_image("custom_writting/P/P_thick.png")
thick.show()


# Testing P_thin

thin = preprocess_image("custom_writting/P/P_thin.png")
thin.show()



# =========================================
# Trying dilation on the thin image
# To see if making the stroke thicker helps it look closer to EMNIST

thin_array = np.array(thin)

kernel = np.ones((2, 2), np.uint8)

thin_dilated = cv2.dilate(
    thin_array,
    kernel,
    iterations=1
)

Image.fromarray(thin_dilated).show()


# =========================================
# Trying erosion on the thick image
# To see if making the stroke thinner helps it look closer to EMNIST

thick_array = np.array(thick)

thick_eroded = cv2.erode(
    thick_array,
    kernel,
    iterations=1
)

Image.fromarray(thick_eroded).show()


# =========================================
# Saving every preprocessing step side by side for my P and O test images

save_steps_grid(
    ["custom_writting/P/P_thin.png", "custom_writting/P/P_medium.png", "custom_writting/P/P_thick.png",
     "custom_writting/O/O_Thin.png", "custom_writting/O/O_Medium.png", "custom_writting/O/O_Thick.png"],
    "preprocess_steps.png"
)
"""
Tried both a thin and a thick version of P and used dilation and erosion to see how the stroke width 
changed after resizing
The really thick version looked a little too bold and the eroded one I thought was too thin
Medium strokes looked closer to the EMNIST examples in my eyes 
"""