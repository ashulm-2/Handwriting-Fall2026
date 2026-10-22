from PIL import Image
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
import imgresize

def LoadImage(FileLocation):
  # Convert the photo to an upright EMNIST-style 28x28 (white ink on black) first
  img = Image.fromarray(imgresize.toEmnist(FileLocation))
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
  Model.load_state_dict(torch.load("CNNEMNIST.pt", map_location=Model.device))
  Model.eval()
 

  Guess = F.softmax(Model(ImgTensor.to(Model.device)), dim=1)
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
 


if __name__ == "__main__":
  # usage: python predict.py image1.png [image2.png ...]
  for path in sys.argv[1:]:
    print(path, "->", MakePrediction(path))