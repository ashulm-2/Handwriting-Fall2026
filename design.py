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

EMNISTDir = "C:\\Users\\ashul\\OneDrive\\Documents\\GitHub\\emnist"

# Options:
# balanced, byclass, bymerge, letters, digits, mnist
EMNIST_SPLIT = "bymerge"


MNISTDir = "C:\\Users\\ashul\\OneDrive\\Documents\\GitHub\\mnist"



##############################################################
# CUSTOM DATASET SETTINGS
##############################################################

# Expected format:
#
# CustomDataset/
#    0/
#     image1.png
#     image2.png
#    1/
#     image1.png
#    A/
#     image1.png
#

CUSTOM_DATASET_DIRECTORY = r"C:\Users\ashul\Documents\MyDataset"



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
    0:'0',
    1:'1',
    2:'2',
    3:'3',
    4:'4',
    5:'5',
    6:'6',
    7:'7',
    8:'8',
    9:'9',

    # uppercase
    10:'A',
    11:'B',
    12:'C',
    13:'D',
    14:'E',
    15:'F',
    16:'G',
    17:'H',
    18:'I',
    19:'J',
    20:'K',
    21:'L',
    22:'M',
    23:'N',
    24:'O',
    25:'P',
    26:'Q',
    27:'R',
    28:'S',
    29:'T',
    30:'U',
    31:'V',
    32:'W',
    33:'X',
    34:'Y',
    35:'Z',

    # lowercase
    36:'a',
    37:'b',
    38:'c',
    39:'d',
    40:'e',
    41:'f',
    42:'g',
    43:'h',
    44:'i',
    45:'j',
    46:'k',
  }


  return mapping[c]



##############################################################
# IMAGE TRANSFORMS
##############################################################


class BinaryTransform:

  def __init__(self, threshold=128):
    self.threshold = threshold


  def __call__(self,img):

    img = np.array(img)

    img = (img > self.threshold).astype(np.uint8)*255

    return Image.fromarray(img)



class TransposeTransform:
  def __call__(self,img):
    return torch.transpose(img,1,2)



# User can change this

IMAGE_TRANSFORM = transforms.Compose([
  transforms.Grayscale(num_output_channels=1),
  BinaryTransform(
    threshold=100
  ),
  transforms.ToTensor(),
  TransposeTransform()
])


def get_num_classes():
  if DATASET_TYPE=="EMNIST":
    if EMNIST_SPLIT=="bymerge":
      return 47
    if EMNIST_SPLIT=="balanced":
      return 47
    if EMNIST_SPLIT=="byclass":
      return 62
  
  elif DATASET_TYPE == "MNIST":
    return 10

  elif DATASET_TYPE=="CUSTOM":
    folders=[
      f for f in os.listdir(CUSTOM_DATASET_DIRECTORY)
      if os.path.isdir(
        os.path.join(CUSTOM_DATASET_DIRECTORY,f)
      )
    ]
    
  return len(folders)