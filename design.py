"""
design.py
User editable configuration file for CNN training.
Change variables in this file only.
CNNEMNIST.py should not need modification.
"""
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from emnist_style import make_binary, crop_to_char, fix_thickness, resize_char, center_char

##############################################################
# DATASET SETTINGS
##############################################################
# Options:
# "EMNIST"
# "MNIST"
# "CUSTOM"
DATASET_TYPE = "MNIST"

##############################################################
# EMNIST SETTINGS
##############################################################
EMNISTDir = "C:\\Users\\anuja\\OneDrive\\Documents\\GitHub\\emnist"
# Options:
# balanced, byclass, bymerge, letters, digits, mnist
EMNIST_SPLIT = "bymerge"

MNISTDir = "C:\\Users\\anuja\\OneDrive\\Documents\\GitHub\\mnist"

##############################################################
# CUSTOM DATASET SETTINGS
##############################################################
# Expected format:
#
# CustomDataset/
#   0/
#     image1.png
#     image2.png
#   1/
#   A/
#     image1.png
#
CUSTOM_DATASET_DIRECTORY = r"C:\Users\anuja\Documents\MyDataset"

##############################################################
# CHARACTER MAPPING
##############################################################
def CToC(c):
    """
    Converts CNN class number to ASCII/Unicode character.
    Modify this if your custom dataset uses
    a different ordering.
    """
    mapping = {
        # digits
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        # uppercase
        10: 'A',
        11: 'B',
        12: 'C',
        13: 'D',
        14: 'E',
        15: 'F',
        16: 'G',
        17: 'H',
        18: 'I',
        19: 'J',
        20: 'K',
        21: 'L',
        22: 'M',
        23: 'N',
        24: 'O',
        25: 'P',
        26: 'Q',
        27: 'R',
        28: 'S',
        29: 'T',
        30: 'U',
        31: 'V',
        32: 'W',
        33: 'X',
        34: 'Y',
        35: 'Z',
        # lowercase
        36: 'a',
        37: 'b',
        38: 'c',
        39: 'd',
        40: 'e',
        41: 'f',
        42: 'g',
        43: 'h',
        44: 'i',
        45: 'j',
        46: 'k',
    }
    return mapping[c]


##############################################################
# IMAGE TRANSFORMS
##############################################################

# turns the image into pure black and white pixels (same as make_binary)
class BinarizeTransform:
    def __call__(self, img):
        arr = np.array(img)
        mask = make_binary(arr)
        out = mask.astype(np.uint8) * 255
        return Image.fromarray(out)


# crops the image down to just the character (same as crop_to_char)
class CropTransform:
    def __call__(self, img):
        arr = np.array(img)
        mask = arr > 0
        mask = crop_to_char(mask)
        out = mask.astype(np.uint8) * 255
        return Image.fromarray(out)


# redraws the strokes at one even thickness (same as fix_thickness)
class ThicknessTransform:
    def __init__(self, thickness=3, box_size=20):
        self.thickness = thickness
        self.box_size = box_size

    def __call__(self, img):
        arr = np.array(img)
        mask = arr > 0

        # scale the thickness up to match this image's current size
        scale = max(mask.shape) / self.box_size
        real_thickness = self.thickness * scale

        mask = fix_thickness(mask, real_thickness)
        out = mask.astype(np.uint8) * 255
        return Image.fromarray(out)


# shrinks the character down to fit a small box (same as resize_char)
class ResizeTransform:
    def __init__(self, box_size=20):
        self.box_size = box_size

    def __call__(self, img):
        arr = np.array(img)
        mask = arr > 0
        small = resize_char(mask, self.box_size)
        return Image.fromarray(small)


# centers the character on the final canvas (same as center_char)
class CenterTransform:
    def __init__(self, canvas_size=28):
        self.canvas_size = canvas_size

    def __call__(self, img):
        arr = np.array(img)
        final = center_char(arr, self.canvas_size)
        return Image.fromarray(final)


class TransposeTransform:
    def __call__(self, img):
        return torch.transpose(img, 1, 2)


# User can change this
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    BinarizeTransform(),
    CropTransform(),
    ThicknessTransform(thickness=3, box_size=20),
    CropTransform(),
    ResizeTransform(box_size=20),
    CenterTransform(canvas_size=28),
    transforms.ToTensor(),
    TransposeTransform()
])


def get_num_classes():
    if DATASET_TYPE == "EMNIST":
        if EMNIST_SPLIT == "bymerge":
            return 47
        if EMNIST_SPLIT == "balanced":
            return 47
        if EMNIST_SPLIT == "byclass":
            return 62
    elif DATASET_TYPE == "MNIST":
        return 10
    elif DATASET_TYPE == "CUSTOM":
        folders = [
            f for f in os.listdir(CUSTOM_DATASET_DIRECTORY)
            if os.path.isdir(
                os.path.join(CUSTOM_DATASET_DIRECTORY, f)
            )
        ]
        return len(folders)
