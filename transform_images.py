"""
Apply design.IMAGE_TRANSFORM and save the resulting images.

Examples:
	python transform_images.py custom_numbers -o transformed_numbers
	python transform_images.py image.png -o transformed_images

"""

import argparse
from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import to_pil_image

import design


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def transform_images(input_path, output_directory):
	input_path = Path(input_path).expanduser()
	output_directory = Path(output_directory).expanduser().resolve()

	if not input_path.exists():
		raise ValueError(f"Input does not exist: {input_path}")

	if input_path.is_dir():
		images = [path for path in sorted(input_path.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and output_directory not in path.resolve().parents]
	else:
		images = [input_path]

	if not images:
		raise ValueError(f"No png or jpg images found in: {input_path}")

	for image_path in images:
		with Image.open(image_path) as image:
			# Match the RGB input used by torchvision.datasets.ImageFolder.
			result = design.IMAGE_TRANSFORM(image.convert("RGB"))

		if not isinstance(result, Image.Image):
			result = to_pil_image(result)

		relative_path = image_path.relative_to(input_path) if input_path.is_dir() else Path(image_path.name)
		destination = output_directory / relative_path
		destination.parent.mkdir(parents=True, exist_ok=True)
		result.save(destination)

	return len(images)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("input", help="Image file or folder containing images")
	parser.add_argument("-o", "--output", default="transformed_images", help="Output folder (default: transformed_images)")
	args = parser.parse_args()

	try:
		count = transform_images(args.input, args.output)
	except (ValueError, OSError) as error:
		parser.exit(1, f"Error: {error}\n")
	print(f"Saved {count} transformed image(s) to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
	main()
