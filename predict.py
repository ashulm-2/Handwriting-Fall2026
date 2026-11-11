import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

import design
from CNN import CNN
from design import TransposeTransform


_steps = list(design.IMAGE_TRANSFORM.transforms)
if isinstance(_steps[-1], TransposeTransform):
    _steps = _steps[:-1]
CUSTOM_IMAGE_TRANSFORM = transforms.Compose(_steps)

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
MODEL_PATH = "CNNEMNIST.pt"
THUMB_SIZE = 120


def guess_true_label(filename):
    # turns filename into char that it represents
    name = os.path.splitext(filename)[0]
    if name.startswith("small_"):
        return name.replace("small_", "")
    return name


def char_to_class(char):
    #character -> class number
    for class_num in range(design.get_num_classes()):
        if design.CToC(class_num) == char:
            return class_num
    return None


def load_model():
    network = CNN()
    state_dict = torch.load(MODEL_PATH, map_location=network.device)
    network.load_state_dict(state_dict)
    network.eval()
    return network


def predict_one(network, image_path):
    img = Image.open(image_path)
    tensor = CUSTOM_IMAGE_TRANSFORM(img)
    # batch dimension
    tensor = tensor.unsqueeze(0).to(network.device)

    with torch.no_grad():
        output = network.forward(tensor)
        probabilities = F.softmax(output, dim=1)

        # top 3 guesses
        top3_values, top3_indices = torch.topk(probabilities, k=3, dim=1)
        top3_values = top3_values.squeeze(0)
        top3_indices = top3_indices.squeeze(0)

    predicted_class = top3_indices[0].item()
    predicted_char = design.CToC(predicted_class)

    top3 = []
    for value, index in zip(top3_values, top3_indices):
        char = design.CToC(int(index))
        percent = value.item() * 100
        top3.append((char, percent))

    return predicted_char, top3


def build_result_image(results, output_path):
    cols = len(results)
    label_h = 90
    grid = Image.new("L", (THUMB_SIZE * cols, THUMB_SIZE + label_h), color=40)
    draw = ImageDraw.Draw(grid)

    for i, (image_path, true_char, predicted_char, top3) in enumerate(results):
        thumb = Image.open(image_path).convert("L").resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
        grid.paste(thumb, (i * THUMB_SIZE, label_h))

        x = i * THUMB_SIZE + 4
        draw.text((x, 2), f"true:{true_char} pred:{predicted_char}", fill=255)

        y = 18
        for char, percent in top3:
            draw.text((x, y), f"{char}: {percent:.1f}%", fill=255)
            y = y + 16

    grid.save(output_path)
    print("saved result image:", output_path)


def main():
    input_folder = sys.argv[1]
    output_image = sys.argv[2]

    network = load_model()

    names = os.listdir(input_folder)
    names.sort()

    results = []
    correct = 0
    total = 0

    for name in names:
        if not name.lower().endswith(EXTENSIONS):
            continue

        path = os.path.join(input_folder, name)
        true_char = guess_true_label(name)
        predicted_char, top3 = predict_one(network, path)

        # ignore upper/lower case for char like x/X
        is_correct = (true_char.lower() == predicted_char.lower())
        correct += int(is_correct)
        total += 1

        top3_text = ", ".join(f"{char}:{percent:.1f}%" for char, percent in top3)
        print(f"{name}: true={true_char} predicted={predicted_char} {'CORRECT' if is_correct else 'WRONG'} | top3: {top3_text}")
        results.append((path, true_char, predicted_char, top3))

    if total > 0:
        print(f"\n{correct}/{total} correct ({correct/total:.1%})")

    build_result_image(results, output_image)


main()