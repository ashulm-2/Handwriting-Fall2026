

import numpy as np
from PIL import Image

PATH_TO_O = "EMNIST/raw/"

SPLIT = "bymerge"
O_LABEL = 24  # 'O' in the bymerge split

BLACK_THRESHOLD = 32
WHITE_THRESHOLD = 128


def get_O_images():
	images = np.fromfile(f"{PATH_TO_O}emnist-{SPLIT}-train-images-idx3-ubyte", dtype=np.uint8, offset=16).reshape(-1, 28, 28)
	labels = np.fromfile(f"{PATH_TO_O}emnist-{SPLIT}-train-labels-idx1-ubyte", dtype=np.uint8, offset=8)

	return [Image.fromarray(img) for img in images[labels == O_LABEL]]


def calculate_stroke_width(images, black_threshold=BLACK_THRESHOLD, white_threshold=WHITE_THRESHOLD):
	# images =  list of 28x28 PIL images of 0
	W = 28
	cx, cy = 13, 13
	total = cnt =  0

	for image in images:
		px = image.load()

		# Only measure image for which the center is black
		if (px[cx, cy] > black_threshold):
			continue

		x = cx + 1
		while (x < W and px[x, cy] <= black_threshold):
			x += 1
		if (x >= W):
			continue

		sw = 0
		while (x < W and px[x, cy] >= white_threshold):
			x += 1
			sw += 1
		if (sw == 0):
			continue

		cnt += 1
		total += sw 

	return total / cnt if cnt else 0.0


if __name__ == "__main__":
	images = get_O_images()
	print("Average stroke width is:", calculate_stroke_width(images), ".\n")
