import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FF
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
#from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST, EMNIST, ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import random

import design  # this is the external file with the variables

from PIL import Image

import time
import sys


NETWORK_OUTPUT_FILENAME = "cnn_mnist_1.pt"


transform = design.IMAGE_TRANSFORM


#num_classes = sum(1 for entry in os.scandir("cleandata") if entry.is_dir()) #this will count how many folders are in cleandata which is the number of classes of the CNN


class CNN(nn.Module):
	def __init__(self, lr=1e-3, batch_size=64):
		super(CNN, self).__init__()
		self.lr = lr
		self.batch_size = batch_size
		self.num_classes = design.get_num_classes()
		print("Number of classes is:", self.num_classes)
		self.loss_history = []
		self.acc_history = []
		if torch.backends.mps.is_available():
			self.device = torch.device('mps')
		elif torch.cuda.is_available():
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')

		self.conv_block = nn.Sequential(  # input is size 1x28x28
			# block 1
			nn.Conv2d(1, 32, 3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			#nn.GELU(),
			nn.Conv2d(32, 32, 3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			#nn.GELU(),
			nn.MaxPool2d(2),  # 14x14
			#nn.Dropout(0.25),

			# block 2
			nn.Conv2d(32, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			#nn.GELU(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			#nn.GELU(),
			nn.MaxPool2d(2),  # 7x7
			#nn.Dropout(0.25),

			# block 3
			nn.Conv2d(64, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			#nn.GELU(),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			#nn.GELU(),

			nn.AdaptiveAvgPool2d((1, 1))
		)

		self.fc_block = nn.Sequential(
			nn.Flatten(),
			#nn.Linear(128, 256),
			#nn.GELU(),
			#nn.Dropout(0.40),
			nn.Linear(128, self.num_classes)
		)

		self.to(self.device)
		#self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

		#self.loss = nn.CrossEntropyLoss()
		self.loss = nn.CrossEntropyLoss(label_smoothing=0.05)
		#self.get_data()

	def forward(self, batch_data):
		batch_data = self.conv_block(batch_data)
		batch_data = self.fc_block(batch_data)
		return batch_data

	def get_data(self):
		if design.DATASET_TYPE == "EMNIST":
			train_dataset = EMNIST(design.EMNISTDir, split=design.EMNIST_SPLIT, train=True,
								   download=True, transform=transform)
			test_dataset = EMNIST(design.EMNISTDir, split=design.EMNIST_SPLIT, train=False,
								  download=True, transform=transform)

		elif design.DATASET_TYPE == "MNIST":
			train_dataset = MNIST(design.MNISTDir, train=True, download=True, transform=transform)
			test_dataset = MNIST(design.MNISTDir, train=False, download=True, transform=transform)

		elif design.DATASET_TYPE == "CUSTOM":
			full_dataset = ImageFolder(
				design.CUSTOM_DATASET_DIRECTORY,
				transform=design.IMAGE_TRANSFORM
			)

			train_size = int(
				0.8 * len(full_dataset)
			)

			test_size = len(full_dataset) - train_size

			train_dataset, test_dataset = random_split(
				full_dataset,
				[train_size, test_size]
			)

		else:
			raise ValueError("Unknown dataset type")

		self.train_data_loader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=8
		)

		self.test_data_loader = torch.utils.data.DataLoader(
			test_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=8
		)

	def set_test_data_loader(self, dataset):
		self.test_data_loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=8
		)

	def _train(self, epochs):
		self.get_data()
		self.train()
		for i in range(epochs):
			ep_loss = 0
			ep_correct = 0
			ep_samples = 0
			for j, (input, label) in enumerate(self.train_data_loader):
				self.optimizer.zero_grad()
				input = input.to(self.device)
				label = label.to(self.device)
				prediction = self.forward(input)
				loss = self.loss(prediction, label)
				classes = torch.argmax(prediction, dim=1)
				correct = (classes == label).sum().item()
				batch_samples = label.size(0)

				ep_correct += correct
				ep_samples += batch_samples
				self.acc_history.append(correct / batch_samples)
				ep_loss += loss.item() * batch_samples
				loss.backward()
				self.optimizer.step()
			print(f"Finish epoch {i + 1} average loss {ep_loss / ep_samples:.3f} accuracy {ep_correct / ep_samples:.3f}")
			#self._test()
			self.loss_history.append(ep_loss / ep_samples)

	@torch.inference_mode()
	def _test(self):
		self.eval()

		ep_loss = 0
		ep_correct = 0
		ep_samples = 0
		for j, (input, label) in enumerate(self.test_data_loader):
			input = input.to(self.device)
			label = label.to(self.device)
			prediction = self.forward(input)
			loss = self.loss(prediction, label)
			classes = torch.argmax(prediction, dim=1)
			batch_samples = label.size(0)

			ep_correct += (classes == label).sum().item()
			ep_samples += batch_samples

			ep_loss += loss.item() * batch_samples

		print(f"Testing: average loss {ep_loss / ep_samples:.3f} accuracy {ep_correct / ep_samples:.3f}")


if __name__ == '__main__':
	start = time.time()
	network = CNN()  # creates the CNN

	network._train(epochs=1)
	end = time.time()

	print(f"It took {(end - start) / 60:.1f} minutes to train the network.")
	torch.save(network.state_dict(), NETWORK_OUTPUT_FILENAME)
	network._test()
