from PIL import Image, ImageOps, ImageDraw
import cv2
import numpy as np

# =========================================
# Make an image look like EMNIST
# Shared by main.py and stroke_width.py
#
# Each step is its own small function so it can
# later become a transform class
# =========================================

# EMNIST stroke width divided by the longest side of the character's bounding box.
# Measured with measure_stroke_width_dt (the same method match_stroke_width uses)
# on 5000 random EMNIST bymerge training images, all 47 classes,
# transposed upright binarized at 100 and cropped to the character.
# Mean ratio 0.1247 (median 0.1304). On the 28x28 images themselves
# this method gives a mean width of 2.796 px (longest side about 22.7 px)
# My 4.368 row-14 width in stroke_width.py is a different measurement
EMNIST_STROKE_RATIO = 0.1247
EMNIST_DT_WIDTH_28 = 2.796

# Same threshold BinaryTransform in design.py uses on the training data
BINARY_THRESHOLD = 100


def blank_image():
    return Image.new("L", (28, 28), 0)


# =========================================
# Step 1: clean the background
# =========================================

def to_grayscale(image):
    return ImageOps.grayscale(image)


def background_is_light(gray_array):

    # The border of the photo is the background
    border = np.concatenate([
        gray_array[0, :], gray_array[-1, :],
        gray_array[:, 0], gray_array[:, -1]
    ])

    return np.median(border) > 127


def flatten_lighting(gray):

    # Dividing by a heavily blurred copy removes shadows and uneven lighting.
    # The blur is about 1/4 of the image size so it only keeps the lighting,
    # not the character
    gray_array = np.array(gray).astype(np.float32)

    # The division works on a light background, so a dark one is
    # flipped first and flipped back after (auto_invert decides later)
    light = background_is_light(gray_array)
    if not light:
        gray_array = 255 - gray_array

    blur_size = max(gray_array.shape) / 4
    background = cv2.GaussianBlur(gray_array, (0, 0), sigmaX=blur_size / 6)

    # Background becomes about 1, ink is below 1
    flat = gray_array / (background + 1e-6)

    # A blank page has almost no contrast left
    # stretching it to 0-255 would only blow up the noise
    if 1 - flat.min() < 0.15:
        flat = np.ones_like(flat)

    # Pixels right next to the ink come out a little above 1 (a bright halo)
    # so I clip at 1 to keep the background flat, then rescale to 0-255
    flat = np.clip(flat, flat.min(), 1)
    flat = (flat - flat.min()) / (1 - flat.min() + 1e-6) * 255

    if not light:
        flat = 255 - flat

    return Image.fromarray(flat.astype(np.uint8))


def auto_invert(gray):

    # EMNIST uses white characters on a black background
    # I only invert if the background is light
    if background_is_light(np.array(gray)):
        return ImageOps.invert(gray)

    return gray


# =========================================
# Step 2: threshold and remove specks
# =========================================

def otsu_threshold(inverted):

    # Otsu picks a threshold between background and stroke.
    # Everything above it becomes 255, so the character is binary
    inverted_array = np.array(inverted)

    _, binary = cv2.threshold(
        inverted_array, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return Image.fromarray(binary)


def remove_ruled_lines(binary, length_fraction=1/3, max_thickness_fraction=0.02):

    # Lined paper leaves long horizontal lines after thresholding.
    # An opening with a long, thin horizontal kernel keeps only those lines
    # so I subtract just them from the image
    binary_array = np.array(binary)
    rows, cols = binary_array.shape

    length = max(1, int(cols * length_fraction))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    lines = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, kernel)

    # A flat part of the character (like the bottom of a thick O) also passes
    # the opening but it is taller than a ruled line
    # Ink that is taller than a line is kept also keeps the stroke
    # where a line crosses the character
    thickness = max(3, int(rows * max_thickness_fraction))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, thickness + 1))
    tall_ink = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, vertical)
    lines = cv2.subtract(lines, tall_ink)

    return Image.fromarray(cv2.subtract(binary_array, lines))


def remove_specks(binary, min_size_fraction=0.15, max_gap_fraction=0.25,
                  max_line_aspect=8, max_line_height_fraction=0.1):

    # Keep the largest piece (the character),
    # plus any piece that is reasonably large or close to it
    # (so the dots on i and j survive). Small, far away specks are removed
    binary_array = np.array(binary)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_array, connectivity=8
    )

    # Label 0 is the background
    if count <= 1:
        return Image.fromarray(np.zeros_like(binary_array))

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + np.argmax(areas)
    largest_area = stats[largest, cv2.CC_STAT_AREA]

    # If even the biggest piece is tiny, the image is nearly blank
    if largest_area < max(10, 0.0005 * binary_array.size):
        return Image.fromarray(np.zeros_like(binary_array))

    lx, ly, lw, lh = stats[largest, :4]
    max_gap = max_gap_fraction * max(lw, lh)

    keep = np.zeros_like(binary_array)

    for label in range(1, count):

        x, y, w, h, area = stats[label]

        # Gap between this piece's box and the largest piece's box
        gap_x = max(0, lx - (x + w), x - (lx + lw))
        gap_y = max(0, ly - (y + h), y - (ly + lh))
        gap = np.hypot(gap_x, gap_y)

        big = area >= min_size_fraction * largest_area

        # Close pieces still need to be more than a pixel or two of noise
        close = gap <= max_gap and area >= 0.005 * largest_area

        # Long, flat pieces are leftover bits of ruled lines,
        # so they go even when they are close to the character
        line_piece = (
            w > max_line_aspect * h
            and h <= max_line_height_fraction * max(lw, lh)
        )

        if label == largest or ((big or close) and not line_piece):
            keep[labels == label] = 255

    return Image.fromarray(keep)


# =========================================
# Step 3: crop to the character
# =========================================

def crop_to_character(image):

    bbox = image.getbbox()

    if bbox:
        return image.crop(bbox)

    return None


# =========================================
# Step 4: match the EMNIST stroke width
# Done at the large size, before resizing
# =========================================

def measure_stroke_width_dt(binary):

    # The distance transform gives each ink pixel its distance to the background.
    # Along the middle of a stroke that distance is about half the width.
    # 2 * d - 1 matched known widths within about 1 px when I tested it
    # Padding by 1 so the crop edge counts as background
    # (otherwise ink touching the edge, or a crop with no black at all, measures far too wide)
    binary_array = np.pad(np.array(binary), 1)

    distance = cv2.distanceTransform(binary_array, cv2.DIST_L2, 5)

    # Stroke centers are where the distance is a local maximum
    centers = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))

    if not centers.any():
        return 0

    return 2 * float(np.median(distance[centers])) - 1


def round_kernel_morph(binary_array, radius, grow):

    # Same result as cv2.dilate / cv2.erode with a round kernel of this radius,
    # but done with a distance transform so big radii stay fast
    # (cv2 with a 100+ px kernel takes minutes on my photos)
    if grow:
        distance = cv2.distanceTransform(255 - binary_array, cv2.DIST_L2, 5)
        return ((distance <= radius) * 255).astype(np.uint8)

    distance = cv2.distanceTransform(binary_array, cv2.DIST_L2, 5)
    return ((distance > radius) * 255).astype(np.uint8)


def main_pieces(binary_array, min_size_fraction=0.15):

    # Only the big pieces of the character (same rule as remove_specks),
    # so small leftovers like dots or line bits don't change the measurement
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_array, connectivity=8
    )

    if count <= 1:
        return binary_array

    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + np.flatnonzero(areas >= min_size_fraction * areas.max())

    return (np.isin(labels, big) * 255).astype(np.uint8)


def find_holes(binary_array, min_hole_fraction=0.01):

    # Holes are background pieces that don't touch the edge,
    # like the inside of an O or P. Tiny ones are just noise in the ink
    background = (binary_array == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=4
    )

    rows, cols = binary_array.shape
    min_area = min_hole_fraction * rows * cols
    holes = []

    for label in range(1, count):

        x, y, w, h, area = stats[label]
        touches_edge = x == 0 or y == 0 or x + w == cols or y + h == rows

        if not touches_edge and area >= min_area:
            holes.append(labels == label)

    return holes


def safe_dilation_radius(padded, radius, min_hole_kept=0.3):

    # Dilating by r fills every background pixel within r of the ink.
    # I lower r until every hole keeps at least min_hole_kept of its area,
    # so an O or P never gets closed up
    holes = find_holes(padded)

    if not holes:
        return radius

    distance = cv2.distanceTransform(255 - padded, cv2.DIST_L2, 5)

    while radius > 0:

        kept = [np.count_nonzero(distance[hole] > radius) / np.count_nonzero(hole)
                for hole in holes]

        if min(kept) >= min_hole_kept:
            break

        radius -= 1

    return radius


def match_stroke_width(cropped, min_ink_kept=0.3, max_change=2.0):

    w, h = cropped.size

    # Scale the EMNIST width up to the size of my cropped character
    target_width = EMNIST_STROKE_RATIO * max(w, h)

    # Measure only the main character pieces
    current_width = measure_stroke_width_dt(main_pieces(np.array(cropped)))

    if current_width <= 0:
        return cropped

    # Never make the stroke more than max_change times thicker or thinner
    target_width = np.clip(target_width, current_width / max_change,
                           current_width * max_change)

    # Dilating or eroding by r changes the width by 2r
    radius = int(round(abs(target_width - current_width) / 2))

    if radius == 0:
        return cropped

    # Padding so dilation doesn't get cut off at the crop edge
    padded = np.pad(np.array(cropped), radius + 1)
    ink_before = np.count_nonzero(padded)

    if current_width < target_width:
        radius = safe_dilation_radius(padded, radius)

        if radius == 0:
            return cropped

        adjusted = round_kernel_morph(padded, radius, grow=True)

    else:
        # Erosion must never delete the character,
        # so I shrink the amount until enough ink is left
        adjusted = padded

        while radius > 0:
            eroded = round_kernel_morph(padded, radius, grow=False)

            if np.count_nonzero(eroded) >= min_ink_kept * ink_before:
                adjusted = eroded
                break

            radius -= 1

    # Re-crop after adjusting
    return crop_to_character(Image.fromarray(adjusted))


# =========================================
# Step 5: resize and center like EMNIST
# =========================================

def resize_and_center(cropped, size=24):

    # Resizing without stretching the character
    # 24 pixels for the longest side, like EMNIST characters
    w, h = cropped.size
    scale = size / max(w, h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cropped.resize(
        (new_w, new_h),
        Image.Resampling.BICUBIC
    )

    # Making a 28x28 black background
    final = Image.new("L", (28, 28), 0)

    # Centering the character on the 28x28 image
    x = (28 - resized.width) // 2
    y = (28 - resized.height) // 2

    final.paste(resized, (x, y))

    return final


# =========================================
# Step 6: match the training format
# =========================================

def binarize(image, threshold=BINARY_THRESHOLD):

    # Same as BinaryTransform in design.py.
    # No transpose: my images are already upright
    image_array = np.array(image)
    image_array = (image_array > threshold).astype(np.uint8) * 255

    return Image.fromarray(image_array)


# =========================================
# Full pipeline
# =========================================

def preprocess_image(source, return_steps=False):

    steps = []

    def done(final):
        steps.append(("6. binarize", final))
        return (final, steps) if return_steps else final

    # Opening the image 
    image = source if isinstance(source, Image.Image) else Image.open(source)

    # Phone photos store rotation as metadata now this applies it
    image = ImageOps.exif_transpose(image)
    steps.append(("original", image.convert("L")))

    # 1. Clean the background
    gray = to_grayscale(image)
    flat = flatten_lighting(gray)
    steps.append(("1. flatten lighting", flat))

    inverted = auto_invert(flat)
    steps.append(("1. auto-invert", inverted))

    # 2. Threshold and remove specks
    binary = otsu_threshold(inverted)
    steps.append(("2. otsu", binary))

    no_lines = remove_ruled_lines(binary)
    steps.append(("2. remove lines", no_lines))

    cleaned = remove_specks(no_lines)
    steps.append(("2. remove specks", cleaned))

    # 3. Crop (a blank image has nothing to crop)
    cropped = crop_to_character(cleaned)

    if cropped is None:
        return done(blank_image())

    steps.append(("3. crop", cropped))

    # 4. Match stroke width
    matched = match_stroke_width(cropped)

    if matched is None:
        return done(blank_image())

    steps.append(("4. stroke width", matched))

    # 5. Resize and center
    centered = resize_and_center(matched)
    steps.append(("5. resize 24 + center", centered))

    # 6. Match the training format
    return done(binarize(centered))


# =========================================
# Side-by-side picture of every step
# =========================================

def save_steps_grid(filenames, output_path, cell=120):

    rows = [preprocess_image(f, return_steps=True)[1] for f in filenames]
    columns = max(len(r) for r in rows)

    label_h = 16
    grid = Image.new("L", (columns * cell, len(rows) * (cell + label_h)), 128)
    draw = ImageDraw.Draw(grid)

    for r, (filename, steps) in enumerate(zip(filenames, rows)):
        for c, (name, step) in enumerate(steps):

            # Small images are scaled up with NEAREST so the pixels stay visible
            thumb = step.copy()
            if max(thumb.size) <= 28:
                thumb = thumb.resize((cell - 8, cell - 8), Image.Resampling.NEAREST)
            else:
                thumb.thumbnail((cell - 8, cell - 8))

            x = c * cell + (cell - thumb.width) // 2
            y = r * (cell + label_h) + label_h + (cell - thumb.height) // 2
            grid.paste(thumb, (x, y))

            label = filename if c == 0 else name
            draw.text((c * cell + 4, r * (cell + label_h) + 2), label, fill=255)

    grid.save(output_path)
