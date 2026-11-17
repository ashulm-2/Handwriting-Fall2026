# HASYv2 Baseline Evaluation Summary

## Objective

Evaluate the existing EMNIST-bymerge CNN externally on HASYv2 at the image level, without retraining.

## Population and method

- **Population:** 3,772 unique images pooled across ten test partitions, selected with the strict 46-character whitelist `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqr`.
- **Preprocessing:** The active `design.IMAGE_TRANSFORM` pipeline, using the project's existing preprocessing helpers without changes.
- **Orientation:** HASYv2 path B, visually checked on F, 7, R, and q: `IMAGE_TRANSFORM` followed by the existing inference transpose.
- **Outputs:** All 47 CNN outputs enabled. Lowercase `t` is not a target class; predictions of `t` count as incorrect.
- **Checkpoint provenance:** Based on project source/configuration; checkpoint metadata was not verified.

## Preprocessing approach

The evaluation calls `design.IMAGE_TRANSFORM` on each source image. Its stages normalize the image, isolate and standardize the handwritten character, validate the result, and convert it to the CNN's tensor layout. The active sequence is defined in [`design.py`](./design.py); its grayscale, mask, crop, and stroke-width helpers come from [`preprocessing_final.py`](./preprocessing_final.py). The evaluation does not call the separate file/directory-saving utilities in `preprocessing_final.py`.

1. **Load and grayscale.** `GrayscaleTransform` calls `_load_grayscale`. It copies a PIL image, composites transparency onto white when present, and converts it to grayscale. The helper also supports loading from a path with EXIF orientation correction; in this evaluation, images are opened and passed as PIL images.
2. **Estimate foreground and polarity.** `ForegroundMaskTransform` calls `_binary_mask`. It autocontrasts the grayscale values, selects a threshold with Otsu's method, and creates a binary mask. If the preliminary mask is nearly all foreground (`> 0.65`) or almost empty (`< 0.002`), it estimates uneven background with a Gaussian blur, corrects the image, and thresholds again. If more than half of the mask is foreground, it inverts the mask. The result is represented as white character pixels on black.
3. **Remove border noise and crop.** `CropCharacterTransform` converts pixels above 127 to foreground and calls `_crop_mask`. That helper clears a border of at least one pixel (about 2% of the smaller image dimension), removes connected components smaller than `max(2, image_pixel_count // 10000)`, and crops to the remaining foreground bounding box. It raises `PreprocessingError` if no character pixels remain or the bounding box covers more than 90% of the image.
4. **Make the character square.** `PadToSquareTransform` places the cropped character at the center of a black square whose side is the longer crop dimension. This preserves the crop's aspect ratio before resizing.
5. **Resize to the character canvas.** `ResizeCharacterTransform` resizes the square to 20×20 pixels using Lanczos resampling.
6. **Place and center on 28×28.** `CenterOnCanvasTransform` places the resized image on a 28×28 black canvas. It computes the intensity-weighted center of mass, shifts the image toward the canvas center using an affine transform with nearest-neighbor resampling, and raises `PreprocessingError` if the image has become blank.
7. **Adjust stroke thickness.** `ThickenToEMNISTWidthTransform` measures the image with `_horizontal_stroke_width` and, while that result is not `None` and below the configured target of 4.345 pixels, applies a 3×3 `MaxFilter` and measures again. The helper's default measurement is on row 14 at threshold 127: it measures from the first pixel above threshold to the next pixel at or below threshold. It returns `None` if the row has no foreground pixel or no following background pixel; in that case, the loop does not thicken the image. This is a one-row proxy measurement, not a measurement of every stroke in the character.
8. **Validate the character image.** `ValidateCharacterTransform` requires a 28×28 nonblank image and checks that the fraction of nonzero pixels is between 0.002 and 0.75. Invalid samples raise `PreprocessingError`.
9. **Convert to a tensor and orient for inference.** `ToTensor` converts the grayscale image to a float32 tensor with a channel dimension and scales byte intensities to `[0, 1]`. `design.IMAGE_TRANSFORM` then transposes the two spatial dimensions. For HASYv2 evaluation, path B applies the existing second transpose before batching; the two transposes cancel, leaving the model input in the visually checked orientation.

This sequence is the preprocessing method whose behavior is evaluated; none of these steps were changed for the baseline. All 3,772 eligible images completed preprocessing successfully.

## Results

- **Correct:** 2,775
- **Incorrect:** 997
- **Accuracy:** 73.5684% (`2775/3772`)
- **Macro-F1:** 0.713150 across the 46 target classes
- **Preprocessing failures:** 0
- **Predictions of `t`:** 27, all counted as errors

### Per-class F1 and eligible support

```csv
character,F1,support
0,0.609836,133
1,0.335484,118
2,0.609137,124
3,0.957983,120
4,0.737500,61
5,0.687179,78
6,0.728324,100
7,0.552239,75
8,0.882812,121
9,0.160494,90
A,0.958466,159
B,0.914729,61
C,0.914062,134
D,0.732824,58
E,0.658824,54
F,0.622754,56
G,0.901288,118
H,0.833333,64
I,0.529730,100
J,0.756477,104
K,0.964824,96
L,0.787879,111
M,0.913793,120
N,0.919192,105
O,0.000000,84
P,0.934211,72
Q,0.926471,67
R,0.969325,83
S,0.441860,63
T,0.729730,56
U,0.728477,59
V,0.940171,56
W,0.941176,62
X,1.000000,54
Y,0.473214,56
Z,0.730159,65
a,0.619048,86
b,0.842105,57
d,0.926829,58
e,0.680412,63
f,0.307692,66
g,0.330435,56
h,0.897638,58
n,0.796296,61
q,0.172840,69
r,0.747664,61
```

### Confusion matrix

Rows are true labels in the 46-class target order; columns are predicted labels in the explicit 47-output order, including `t`.

```csv
row\column,0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,a,b,d,e,f,g,h,n,q,r,t
0,93,0,0,0,0,0,0,0,0,0,1,0,2,18,0,0,0,0,0,0,0,0,0,0,3,0,1,0,0,0,14,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
1,0,26,0,0,8,0,0,37,0,21,1,0,0,0,0,0,0,0,8,3,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,1,1,0,0
2,0,0,60,0,0,1,0,53,0,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0
3,0,0,0,114,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0
4,0,0,0,0,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
5,0,0,0,1,0,67,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0
6,1,0,0,0,0,5,63,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,0,0,0,0,10,0,0,0,0
7,0,0,0,0,0,0,0,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
8,0,0,0,0,0,0,0,0,113,0,0,3,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0
9,0,0,0,1,7,0,0,1,0,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,0,0,0,0,0,0,26,0,0,0,0,0
A,0,0,0,0,5,0,0,1,0,0,150,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
B,0,0,0,0,0,0,0,0,1,0,0,59,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
C,0,3,0,0,0,0,3,0,4,0,0,0,117,0,1,0,0,0,1,1,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
D,0,0,0,0,0,0,0,0,0,0,0,1,0,48,0,0,0,0,7,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
E,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2
F,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,52,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3
G,0,0,0,0,0,2,6,0,0,0,0,0,0,0,0,0,105,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,0,0
H,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
I,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,49,2,0,17,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
J,0,0,3,2,0,0,0,2,3,0,0,0,0,1,0,0,0,0,1,73,0,0,0,0,0,0,0,0,0,0,1,0,0,0,4,6,0,0,7,0,0,1,0,0,0,0,0
K,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
L,0,4,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,13,2,0,91,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
M,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,4,0,0,0,0,106,2,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,3,0,0,0
N,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,91,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
O,75,0,0,0,0,0,0,0,0,0,0,1,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
P,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Q,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,63,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0
R,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0
S,0,0,0,0,0,40,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,19,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0
T,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
U,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,55,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
V,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
W,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,1,1,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0
X,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,0,0,0,0,0,0,0,0,0,0,0,0,0
Y,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,0,0,0,0,0,0,0,0,0,0,0,0
Z,0,0,1,0,0,0,0,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,46,0,0,0,0,0,0,0,0,0,0,0
a,3,0,7,0,2,0,0,2,0,8,1,0,0,2,0,0,0,0,1,0,0,0,2,0,0,0,4,0,0,0,12,0,0,0,0,0,39,0,1,0,0,1,0,0,1,0,0
b,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,0,0,0,0,0,0,0,0,0
d,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,57,0,0,0,0,0,0,0,0
e,0,0,1,0,0,0,0,0,0,0,0,1,1,0,2,3,1,0,0,0,4,0,0,0,0,7,0,0,0,0,1,0,0,0,6,0,0,0,0,33,0,0,0,0,0,0,3
f,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,17
g,0,0,0,0,0,0,1,0,3,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,0,0,0,0,0,0,19,0,0,0,0,0
h,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,57,0,0,0,0
n,0,1,0,0,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0,0,0,0,4,0,0,0,0,1,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,43,0,6,0
q,0,0,0,0,12,0,0,0,0,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,19,0,0,0,0,0,0,1,0,0,7,0,0
r,0,1,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,7,0,0,0,0,0,0,0,0,0,1,40,0
```

## Independent audit

The independent audit passed all checks: 46 true-label rows and 47 prediction columns; nonnegative integer entries; each row sum equal to its reported eligible support; matrix total 3,772; diagonal total 2,775; `t` column total 27; accuracy `2775/3772`; and recomputed per-class F1 and macro-F1 matching the printed values within six-decimal rounding tolerance.

## Exploratory observations

The largest observed off-diagonal counts include `O→0` (75), `2→7` (53), `9→Y` (42), `S→5` (40), and `f→F` (32). These patterns motivate hypotheses about visual similarity or variation in handwriting and stroke appearance. They do not establish that stroke thickness, orientation, or writing style caused any errors; the aggregate matrix cannot show causation.

## Limitations

- This result covers only the strict 46-character subset; it does not describe excluded labels or the full HASYv2 label population.
- The evaluation does not establish an unseen-writer generalization claim.
- Checkpoint training provenance rests on project source/configuration; checkpoint metadata was not verified.
- The deterministic examples selected for inspection are illustrative, not representative estimates of error rates.

## Optional future work

Inspect the active preprocessing stroke-width measurements for the eight selected `9` and `f` examples. This diagnostic has not been run.
