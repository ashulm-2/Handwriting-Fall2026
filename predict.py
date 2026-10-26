import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn.functional as F
import os
import sys
import importlib
import math
import matplotlib.pyplot as plt

import design
import CNN


# Select the device: NVIDIA GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def LoadImage(FileLocation):
    img = Image.open(FileLocation).convert("L")  # EMNIST grayscale
    return design.IMAGE_TRANSFORM(img)


def MakePrediction(FileLocation):
    # Load and preprocess the image
    Img = LoadImage(FileLocation)
    Img = torch.transpose(Img, 1, 2)

    plt.imshow(Img.squeeze(), cmap="gray")
    plt.title("Input to model")
    plt.axis("off")
    plt.show()

    # Add batch dimension and move image to the selected device
    ImgTensor = Img.unsqueeze(0).to(device)

    # Create model and load trained weights
    Model = CNN.CNN()

    state_dict = torch.load(
        "cnn_mnist_1.pt",
        map_location=device,
        weights_only=True
    )

    Model.load_state_dict(state_dict)
    Model = Model.to(device)
    Model.eval()

    print("Model running on:", device)

    # Predict without calculating gradients
    with torch.no_grad():
        Guess = F.softmax(Model(ImgTensor), dim=1)

        Prediction = Guess.argmax(dim=1, keepdim=True)
        top3_values, top3_indices = torch.topk(Guess, k=3, dim=1)

    # Format top-3 probabilities
    top3_values = top3_values.squeeze(0)
    Percentages = top3_values * 100
    PercentagesList = Percentages.tolist()
    FPer = [f"{p:.1f}%" for p in PercentagesList]

    print("Top 3 values:", FPer)

    CHRs = [
        (int(j), design.CToC(int(j)))
        for j in top3_indices.squeeze(0).tolist()
    ]

    print("Top 3 indices:", CHRs)

    return design.CToC(Prediction.item())


MakePrediction(
    r"D:\7. Lectures\MURL FA26\Handwriting-Fall2026\charset\Screenshot 2026-10-19 104511.png"
)