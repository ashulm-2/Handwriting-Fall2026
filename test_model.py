import torch
from torchvision import datasets, transforms

import design
from CNN import CNN

MODEL_FILE = "cnn_emnist_1.pt"

TEST_DATASET = "CUSTOM"
CUSTOM_DIRECTORY = "custom_characters"
DATASET_ROOT = "."

MODEL_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

def get_test_transform():
	transforms_list = [
		transform for transform in design.IMAGE_TRANSFORM.transforms
		if not isinstance(transform, design.TransposeTransform)
	]
	
	if (TEST_DATASET == "EMNIST"):
		transforms_list.append(design.TransposeTransform())

	return transforms.Compose(transforms_list)

def map_custom_labels(dataset):
	character_to_index = { character: index for index, character in enumerate(MODEL_CHARACTERS) }

	folder_index_to_model_index = {}

	for folder, folder_index in dataset.class_to_idx.items():
		character = folder.removesuffix("_upper").removesuffix("_lower")

		if len(character) != 1:
			raise ValueError(f"Unrecognized character folder: {folder!r}")

		if character not in character_to_index:
			character = character.upper()

		if character not in character_to_index:
			raise ValueError(f"No model class for folder: {folder!r}")

		folder_index_to_model_index[folder_index] = (character_to_index[character])

	dataset.target_transform = folder_index_to_model_index.__getitem__
	return dataset

def get_test_dataset():
	transform = get_test_transform()

	if TEST_DATASET == "CUSTOM":
		dataset = datasets.ImageFolder(CUSTOM_DIRECTORY, transform=transform)
		return map_custom_labels(dataset)

	if TEST_DATASET == "EMNIST":
		return datasets.EMNIST(root=DATASET_ROOT, split="bymerge", train=False, download=False, transform=transform)

	if TEST_DATASET == "MNIST":
		return datasets.MNIST(root=DATASET_ROOT, train=False, download=False, transform=transform)

	raise ValueError(f"Unknown test dataset: {TEST_DATASET!r}")

def main():
	if (design.DATASET_TYPE != "EMNIST" or design.EMNIST_SPLIT != "bymerge"):
		raise ValueError("Expected the model configuration in design.py to be EMNIST/bymerge.")

	model = CNN()

	weights = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
	model.load_state_dict(weights)

	dataset = get_test_dataset()

	if len(dataset) == 0:
		raise ValueError("The test dataset is empty.")

	print(f"Model: {MODEL_FILE}")
	print(f"Dataset: {TEST_DATASET}, {len(dataset)} images")

	model.set_test_data_loader(dataset)
	model._test()


if __name__ == "__main__":
	main()
