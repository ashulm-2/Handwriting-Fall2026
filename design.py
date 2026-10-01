"""
design.py

User editable configuration file for CNN training.

Change variables in this file only.
CNNEMNIST.py should not need modification.
"""

import math
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import cv2


##############################################################
# DATASET SETTINGS
##############################################################

# Options:
# "EMNIST"
# "MNIST"
# "CUSTOM"

DATASET_TYPE = "EMNIST"


##############################################################
# EMNIST SETTINGS
##############################################################

EMNISTDir = "EMNIST/raw/"

# Options:
# balanced, byclass, bymerge, letters, digits, mnist
EMNIST_SPLIT = "bymerge"


MNISTDir = "MNIST/raw/"



##############################################################
# CUSTOM DATASET SETTINGS
##############################################################

# Expected format:
#
# CustomDataset/
#		0/
#		 image1.png
#		 image2.png
#		1/
#		 image1.png
#		A/
#		 image1.png
#

CUSTOM_DATASET_DIRECTORY = "custom_characters/"



##############################################################
# CHARACTER MAPPING
##############################################################

def CToC(c):

	"""
	Converts CNN class number to ASCII/Unicode character.

	Modify this if your custom dataset uses
	a different ordering.
	"""
	if DATASET_TYPE == "CUSTOM" or (DATASET_TYPE == "EMNIST" and EMNIST_SPLIT == "byclass"):
		characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
		return characters[c]

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

		# lowercase classes retained by EMNIST bymerge and balanced
		36:'a',
		37:'b',
		38:'d',
		39:'e',
		40:'f',
		41:'g',
		42:'h',
		43:'n',
		44:'q',
		45:'r',
		46:'t',
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



class EnsureDarkBackgroundTransform:

		def __init__(self, threshold=128):
				self.threshold = threshold

		def __call__(self, img):

			width, height = img.size

			corners = [img.getpixel((0,0)), img.getpixel((width - 1,0)),
					img.getpixel((0, height-1)), img.getpixel((width-1, height - 1)),]

			background = sum(corners) / len(corners)

			if (background > self.threshold):
				img = ImageOps.invert(img)

			return img



class TransposeTransform:

	def __call__(self, img):
		return torch.transpose(img,1,2)



class ResizeAndPadTransform:

	def __init__(self, max_width, max_height):
		self.max_width = max_width
		self.max_height = max_height

	def __call__(self, img):

		return ImageOps.pad(img, (self.max_width, self.max_height),  method=Image.Resampling.LANCZOS, color=0)


class CropToBoundingSquareTransform:
	def __init__(self, threshold=128, padding_percent=8):
		self.threshold = threshold
		self.padding_percent = padding_percent

	def __call__(self, img):
		img_np = np.array(img)

		_, thresh = cv2.threshold(img_np, self.threshold, 255, cv2.THRESH_BINARY)

		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if not contours:
				return img

		foreground_points = np.concatenate(contours, axis=0)
		x, y, w, h = cv2.boundingRect(foreground_points)
		
		size = max(w, h)
		size += math.floor(size * (self.padding_percent/50))
		
		center_x = x + w // 2
		center_y = y + h // 2
		x1 = center_x - size // 2
		y1 = center_y - size // 2
		
		x2 = x1 + size
		y2 = y1 + size
		
		img_h, img_w = img_np.shape[:2]
		
		if x1 < 0:
			x2 -= x1
			x1 = 0
		
		if y1 < 0:
			y2 -= y1
			y1 = 0
		
		if x2 > img_w:
			x1 -= x2 - img_w
			x2 = img_w
		
		if y2 > img_h:
			y1 -= y2 - img_h
			y2 = img_h
		
		x1 = max(0, x1)
		y1 = max(0, y1)
		
		cropped = img_np[y1:y2, x1:x2]
		
		return Image.fromarray(cropped)


def estimate_stroke_thickness(img):
	img_np = np.array(img)

	dist = cv2.distanceTransform(img_np, cv2.DIST_L2, 5)

	foreground_distances = dist[dist > 0]

	if len(foreground_distances) == 0:
		return 0

	stroke_width = 2 * np.median(foreground_distances)

	return stroke_width



class NormalizeStrokeThicknessTransform:
	def __init__(self, lowerbound=2.5, upperbound=5.5, max_iterations=3):
		self.lb = lowerbound
		self.ub = upperbound
		self.max_iterations = max_iterations

	def __call__(self, img):
		img_np = np.array(img)

		kernel = cv2.getStructuringElement(
			cv2.MORPH_ELLIPSE,
			(3, 3)
		)

		thickness = estimate_stroke_thickness(img_np)
		for _ in range(self.max_iterations):
			if thickness < self.lb:
				img_np = cv2.dilate(img_np, kernel, iterations=1)
			elif thickness > self.ub:
				img_np = cv2.erode(img_np, kernel, iterations=1)
			else:
				break
			thickness = estimate_stroke_thickness(img_np)

		return Image.fromarray(img_np)


class AddBlackBorderTransform:

	def __init__(self, thickness=10):
		self.thickness=thickness

	def __call__(self, img):
		img_np = np.array(img)

		img_np = cv2.copyMakeBorder(img_np, self.thickness, self.thickness, self.thickness, self.thickness, cv2.BORDER_CONSTANT, value=0)

		return Image.fromarray(img_np)

# User can change this

BINARY_THRESHOLD=80;
BORDER=5
PADDING_PERCENT=8
STROKE_THICKNESS_LB = 2.5
STROKE_THICKNESS_UB = 5.5


transforms_list = [
	transforms.Grayscale(num_output_channels=1), # Grayscale

	EnsureDarkBackgroundTransform(threshold=BINARY_THRESHOLD), # Black background, white character

	AddBlackBorderTransform(thickness=BORDER), # Add a black border around the image
	NormalizeStrokeThicknessTransform(STROKE_THICKNESS_LB, STROKE_THICKNESS_UB), # Dilate/errode stroke thickness based on estimation of current stroke thickness

	CropToBoundingSquareTransform(threshold=BINARY_THRESHOLD, padding_percent=PADDING_PERCENT), # Crop to a square around the character
	ResizeAndPadTransform(28, 28), # Resize the image to 28x28

	BinaryTransform(threshold=BINARY_THRESHOLD), # Black/White

	transforms.ToTensor(),
]

if (DATASET_TYPE == "EMNIST"):
	transforms_list.append(TransposeTransform())

IMAGE_TRANSFORM = transforms.Compose(transforms_list)

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
