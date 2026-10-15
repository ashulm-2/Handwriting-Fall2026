# Converts an image of a handwritten character into an EMNIST-style 28x28,
# following the conversion process in the EMNIST paper (arXiv:1702.05373, II-A).

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


# Measured over 3000 random EMNIST "bymerge" test samples, upright, threshold 100:
# stroke 3.740 +/- 0.947 px, extent 22.725 +/- 1.292 px.
EMNIST_STROKE_WIDTH = 3.74
EMNIST_EXTENT = 22.7
BINARY_THRESHOLD = 100

# Padding is applied before the downsample, so side S padded by m*S per side
# scales by 28/(S(1+2m)); solving that for the measured extent gives m.
MARGIN_FRAC = (28.0 - EMNIST_EXTENT) / (2.0 * EMNIST_EXTENT)

WORKING_SIZE = 128
STROKE_TOLERANCE = 0.3
MAX_PASSES = 5


# Read an image from disk as a 2-D uint8 array, or raise if it cannot be decoded.
def loadGrayscale(path) -> np.ndarray:
    img = cv2.imread(path, 0)
    if img is None: raise FileNotFoundError(f"Image not found at {path}")
    return img

# Display any number of grayscale images side by side.
def show(*images, titles=None):
    fig, axes = plt.subplots(1, len(images), squeeze=False)
    for ax, img in zip(axes[0], images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    if titles:
        for ax, t in zip(axes[0], titles):
            ax.set_title(t)
    plt.tight_layout()
    plt.show()

# Flatten to 0/255 at the level design.BinaryTransform uses, so measurements match.
def threshold(img, level=BINARY_THRESHOLD):
    return (img > level).astype(np.uint8) * 255


# True if the character is dark ink on light paper, judged from the border.
def isDarkOnLight(gray) -> bool:
    k = max(1, round(0.05 * min(gray.shape)))
    strip = np.concatenate([gray[:k, :].ravel(), gray[-k:, :].ravel(),
                            gray[:, :k].ravel(), gray[:, -k:].ravel()])
    return bool(np.median(strip) > 127)  # median survives a shadowed edge; a mean does not

# True if the image is already two-valued, so it needs no adaptive thresholding.
def isAlreadyClean(gray) -> bool:
    return bool(((gray <= 25) | (gray >= 230)).mean() >= 0.90) or min(gray.shape) < 64

# Reduce to a white-on-black binary image, tolerating uneven lighting in photos.
def binarize(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if isDarkOnLight(gray) else cv2.THRESH_BINARY

    # Adaptive thresholding hollows an already two-valued image into an outline.
    if isAlreadyClean(gray):
        return cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)[1]

    block = max(3, int(min(gray.shape) / 8) | 1)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block, 15)

# Delete dust and noise: any blob smaller than keepRatio of the largest one.
def removeSpecks(binary, keepRatio=0.05):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n == 1: return binary

    areas = stats[1:, cv2.CC_STAT_AREA]  # position j is label j+1
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas > keepRatio * areas.max()
    return (keep[labels] * 255).astype(np.uint8)


# Crop to the tightest box holding every white pixel.
def cropToInk(binary):
    rows = np.where(np.any(binary, axis=1))[0]
    cols = np.where(np.any(binary, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("no ink found")
    return binary[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

# Centre in a square of the longer side, preserving aspect ratio, then add the margin.
def makeSquare(img):
    h, w = img.shape
    side = max(h, w)
    padV, padH = side - h, side - w
    top, left = padV // 2, padH // 2
    sq = cv2.copyMakeBorder(img, top, padV - top, left, padH - left, cv2.BORDER_CONSTANT, value=0)
    m = round(MARGIN_FRAC * side)
    return cv2.copyMakeBorder(sq, m, m, m, m, cv2.BORDER_CONSTANT, value=0)

# Downsample to the final size, averaging into the soft grey edges EMNIST has.
def shrinkTo28(square):
    return cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)


# Stroke thickness in pixels, taken from the ridge of the distance transform.
def estimateStrokeWidth(binary) -> float:
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    localMax = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = (dist >= localMax) & (binary > 0)  # background is trivially a local max
    return float(2.0 * dist[ridge].mean()) if ridge.any() else 0.0

# Longer side of the character's bounding box, in pixels.
def measureExtent(binary) -> int:
    return int(max(cropToInk(binary).shape))

# Scale up to the resolution where integer-pixel morphology is precise enough.
def resizeToWorking(binary):
    h, w = binary.shape
    scale = WORKING_SIZE / max(h, w)
    resized = cv2.resize(binary, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return threshold(resized, 128)

# Dilate or erode until the stroke is target pixels thick.
def matchStrokeWidth(binary, target, verbose=False):
    current = estimateStrokeWidth(binary)
    if current <= 0: return binary

    delta = target - current
    if verbose: print(f"    stroke {current:.1f} -> {target:.1f} ({delta:+.1f})")
    if abs(delta) < 0.5: return binary

    delta = max(delta, -0.4 * current)  # over-eroding snaps thin joins
    k = (int(round(abs(delta))) + 1) | 1  # a k x k kernel changes the width by k-1
    if k < 3: return binary

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary, kernel) if delta > 0 else cv2.erode(binary, kernel)


# Convert an image of one handwritten character into an EMNIST-style 28x28 uint8.
def toEmnist(path, *, transposed=False, verbose=False) -> np.ndarray:
    binary = cropToInk(removeSpecks(binarize(loadGrayscale(path))))

    work = resizeToWorking(binary)
    pad = WORKING_SIZE // 4  # room to dilate into
    work = cv2.copyMakeBorder(work, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # Two resizes and two thresholds each nudge the apparent stroke width, so aim
    # at the measured output rather than trusting the arithmetic.
    scale = 28.0 / (WORKING_SIZE * (1.0 + 2.0 * MARGIN_FRAC))
    target = EMNIST_STROKE_WIDTH / scale
    best = None

    for i in range(MAX_PASSES):
        if verbose: print(f"  pass {i+1}")
        out = shrinkTo28(makeSquare(cropToInk(matchStrokeWidth(work, target, verbose))))
        error = EMNIST_STROKE_WIDTH - estimateStrokeWidth(threshold(out))
        if best is None or abs(error) < abs(best[1]):
            best = (out, error)
        if abs(error) <= STROKE_TOLERANCE:
            break
        target += error / scale

    if verbose: print(f"  final error {best[1]:+.2f} px")
    # transposed=True gives EMNIST's stored orientation, for design.IMAGE_TRANSFORM
    return np.ascontiguousarray(best[0].T) if transposed else best[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image of a handwritten character to EMNIST format.")
    parser.add_argument("image")
    parser.add_argument("--out", default="out.png")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--transposed", action="store_true", help="EMNIST's raw stored orientation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = toEmnist(args.image, transposed=args.transposed, verbose=args.verbose)
    binary = threshold(result)

    print(f"shape  : {result.shape}")
    print(f"dtype  : {result.dtype}")
    print(f"range  : {result.min()}-{result.max()}")
    print(f"stroke : {estimateStrokeWidth(binary):.2f} px (EMNIST {EMNIST_STROKE_WIDTH})")
    print(f"extent : {measureExtent(binary)} px (EMNIST {EMNIST_EXTENT:.1f})")

    cv2.imwrite(args.out, result)
    print(f"wrote  : {args.out}")

    if args.show:
        show(loadGrayscale(args.image), result,
             cv2.resize(result, (280, 280), interpolation=cv2.INTER_NEAREST),
             titles=["input", "28x28", "28x28 zoomed"])
