import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn.functional as F
import os
import torch
import sys
import importlib
import math
import matplotlib.pyplot as plt

import design
import CNN

def LoadImage(FileLocation):
  img = Image.open(FileLocation).convert("L")   # EMNIST images are grayscale
  return design.IMAGE_TRANSFORM(img)
  




def MakePrediction(FileLocation):
  Img = LoadImage(FileLocation)
  Img = torch.transpose(Img,1,2)
  plt.imshow(Img.squeeze(), cmap="gray")   # squeeze removes the channel dim -> (H, W)
  plt.title("Input to model")
  plt.axis("off")
  plt.show()
  ImgTensor = Img.unsqueeze(0)
 
  Model = CNN.CNN()
  Model.load_state_dict(torch.load("cnn_mnist_1.pt"))
  Model.eval()
 

  Guess = F.softmax(Model(ImgTensor), dim=1)
  Prediction = Guess.argmax(dim=1, keepdim=True)
  top3_values, top3_indices = torch.topk(Guess, k=3, dim=1)
  top3_values = top3_values.squeeze(0) #remove one layer
  Percentages = top3_values * 100
  PercentagesList = Percentages.tolist()
  FPer = [f"{p:.1f}%" for p in PercentagesList]
  #print("guess is",WhichNN.CToC(Prediction.item()))
  print("Top 3 values:", FPer)
  CHRs = [(int(j),design.CToC(int(j))) for j in top3_indices.squeeze(0)]
  print("Top 3 indices:", CHRs)
  return design.CToC(Prediction.item())
 


MakePrediction(r"C:\Users\ashul\OneDrive\Documents\GitHub\MURLFall26\enoh\test1.png")