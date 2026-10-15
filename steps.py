# Shows the image at every stage of toEmnist, as a labelled strip and as raw arrays.

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

import imgresize as M


# Run the toEmnist pipeline, keeping the image after each stage.
def collectSteps(path):
    steps = []
    def record(name, img): steps.append((name, img.copy()))

    gray = M.loadGrayscale(path)
    record("1. loadGrayscale", gray)

    binary = M.binarize(gray)
    record("2. binarize", binary)

    binary = M.removeSpecks(binary)
    record("3. removeSpecks", binary)

    binary = M.cropToInk(binary)
    record("4. cropToInk", binary)

    work = M.resizeToWorking(binary)
    record("5. resizeToWorking", work)

    pad = M.WORKING_SIZE // 4
    work = cv2.copyMakeBorder(work, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    record("6. pad", work)

    scale = 28.0 / (M.WORKING_SIZE * (1.0 + 2.0 * M.MARGIN_FRAC))
    target = M.EMNIST_STROKE_WIDTH / scale
    stroked = M.cropToInk(M.matchStrokeWidth(work, target))
    record("7. matchStrokeWidth", stroked)

    square = M.makeSquare(stroked)
    record("8. makeSquare", square)

    record("9. shrinkTo28 (1 pass)", M.shrinkTo28(square))
    record("10. toEmnist (best pass)", M.toEmnist(path))
    return steps


# Print shape, dtype and value range for each stage.
def printSteps(steps):
    print(f"{'stage':20s} {'shape':>12s} {'dtype':>8s} {'range':>10s} {'values':>7s}")
    for name, img in steps:
        print(f"{name:20s} {str(img.shape):>12s} {str(img.dtype):>8s} "
              f"{str(img.min()) + '-' + str(img.max()):>10s} {len(np.unique(img)):>7d}")


# Draw every stage side by side, scaled to a common height.
def plotSteps(steps, out):
    fig, axes = plt.subplots(1, len(steps), figsize=(2.0 * len(steps), 3.0))
    for ax, (name, img) in zip(axes, steps):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name.replace(". ", ".\n"), fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show every stage of the imgresize pipeline.")
    parser.add_argument("image")
    parser.add_argument("--out", default="steps.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    steps = collectSteps(args.image)
    printSteps(steps)
    plotSteps(steps, args.out)
    if args.show:
        plt.show()
