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
from PIL import ImageFilter

from preprocessing_final import (
    EMNIST_AVERAGE_STROKE_WIDTH,
    PreprocessingError,
    _binary_mask,
    _crop_mask,
    _horizontal_stroke_width,
    _load_grayscale,
)


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

EMNISTDir = os.path.join(os.path.expanduser("~"), "datasets", "EMNIST")

# Options:
# balanced, byclass, bymerge, letters, digits, mnist
EMNIST_SPLIT = "bymerge"


MNISTDir = os.path.join(os.path.expanduser("~"), "datasets", "MNIST")



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

CUSTOM_DATASET_DIRECTORY = r"D:\7. Lectures\MURL FA26\Handwriting-Fall2026\MyDataset"



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


class GrayscaleTransform:
  def __call__(self, img):
    return _load_grayscale(img)


class ForegroundMaskTransform:
  def __call__(self, img):
    mask = _binary_mask(img)
    return Image.fromarray(mask.astype(np.uint8) * 255)


class CropCharacterTransform:
  def __call__(self, img):
    mask = np.asarray(img.convert("L")) > 127
    cropped = _crop_mask(mask)
    return Image.fromarray(cropped.astype(np.uint8) * 255)


class PadToSquareTransform:
  def __call__(self, img):
    side = max(img.size)
    square = Image.new("L", (side, side), 0)
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return square


class ResizeCharacterTransform:
  def __init__(self, character_size=20):
    self.character_size = character_size

  def __call__(self, img):
    return img.resize(
      (self.character_size, self.character_size),
      Image.Resampling.LANCZOS
    )


class CenterOnCanvasTransform:
  def __init__(self, output_size=28, center_of_mass=True):
    self.output_size = output_size
    self.center_of_mass = center_of_mass

  def __call__(self, img):
    canvas = Image.new("L", (self.output_size, self.output_size), 0)
    offset = (self.output_size - img.width) // 2
    canvas.paste(img, (offset, offset))

    if self.center_of_mass:
      pixels = np.asarray(canvas, dtype=np.float32)
      total = pixels.sum()
      if total <= 0:
        raise PreprocessingError("character became blank after resizing")

      rows, columns = np.indices(pixels.shape)
      shift_row = round(self.output_size / 2 - (rows * pixels).sum() / total)
      shift_column = round(self.output_size / 2 - (columns * pixels).sum() / total)
      canvas = canvas.transform(
        canvas.size,
        Image.Transform.AFFINE,
        (1, 0, -shift_column, 0, 1, -shift_row),
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
      )

    return canvas


class ThickenToEMNISTWidthTransform:
  def __init__(self, target_width=EMNIST_AVERAGE_STROKE_WIDTH):
    self.target_width = target_width

  def __call__(self, img):
    stroke_width = _horizontal_stroke_width(img)
    while stroke_width is not None and stroke_width < self.target_width:
      img = img.filter(ImageFilter.MaxFilter(3))
      stroke_width = _horizontal_stroke_width(img)
    return img


class ValidateCharacterTransform:
  def __init__(self, output_size=28):
    self.output_size = output_size

  def __call__(self, img):
    extrema = img.getextrema()
    fraction = np.count_nonzero(np.asarray(img)) / (self.output_size * self.output_size)
    if img.size != (self.output_size, self.output_size) or extrema[1] == 0:
      raise PreprocessingError("output is blank or has the wrong format")
    if not 0.002 <= fraction <= 0.75:
      raise PreprocessingError(f"white-pixel fraction is out of range: {fraction:.3f}")
    return img



class TransposeTransform:
  def __call__(self,img):
    return torch.transpose(img,1,2)



# User can change this

IMAGE_TRANSFORM = transforms.Compose([
  GrayscaleTransform(),
  ForegroundMaskTransform(),
  CropCharacterTransform(),
  PadToSquareTransform(),
  ResizeCharacterTransform(character_size=20),
  CenterOnCanvasTransform(output_size=28, center_of_mass=True),
  ThickenToEMNISTWidthTransform(),
  ValidateCharacterTransform(output_size=28),
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