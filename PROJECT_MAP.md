# Handwriting project map

This chart follows the files and data paths implemented in the repository. Mermaid diagrams render in GitHub and in Markdown previews that support Mermaid.

## End-to-end file and data flow

```mermaid
flowchart LR
    subgraph Inputs["Inputs and configuration"]
        RAW["charset/<br/>raw images for preprocessing"]
        BATCH["charset_2/<br/>images for batch prediction"]
        DATA["EMNIST / MNIST / custom folders<br/>selected in design.py"]
        CONFIG["design.py<br/>dataset settings, class labels,<br/>image transforms"]
    end

    subgraph Preprocess["Standalone preprocessing — python preprocessing_final.py"]
        PF["preprocessing_final.py"]
        STEPS["Load + grayscale → foreground mask<br/>→ crop → square pad → resize to 20×20<br/>→ center on 28×28 → thicken → validate"]
        OUT["output/<br/>processed PNGs"]
        LOG["output/preprocessing_log.csv<br/>accepted / rejected per input"]
        CHECK["secondary results/<br/>P01, P03, P05–P11 stage images"]
        RAW --> PF
        PF --> STEPS
        STEPS --> OUT
        STEPS --> LOG
        STEPS --> CHECK
    end

    subgraph Training["Training — python CNN.py"]
        CNN["CNN.py<br/>CNN model + training loop"]
        ARCH["CNN architecture<br/>Conv/BN/ReLU blocks: 32 → 64 → 128<br/>adaptive average pool → class logits"]
        LOAD["Dataset loader<br/>EMNIST / MNIST / ImageFolder"]
        TRANSFORM["design.IMAGE_TRANSFORM<br/>grayscale → mask → crop → pad<br/>→ resize → center → thicken → validate<br/>→ tensor → transpose"]
        FIT["Train / test batches<br/>cross-entropy + AdamW"]
        WEIGHTS["cnn_mnist_1.pt<br/>weights consumed by prediction"]
        DATA --> LOAD
        CONFIG --> LOAD
        CONFIG --> TRANSFORM
        LOAD --> TRANSFORM
        TRANSFORM --> FIT
        CNN --> ARCH
        ARCH --> FIT
        CNN --> LOAD
        FIT -. "intended model artifact;<br/>saving is not enabled in CNN.py" .-> WEIGHTS
    end

    subgraph Prediction["Batch inference — python predict.py"]
        PRED["predict.py"]
        IMG["Open each image as grayscale"]
        PTRANS["design.IMAGE_TRANSFORM<br/>then explicit spatial transpose in predict.py"]
        RESTORE["Create CNN.CNN()<br/>load cnn_mnist_1.pt"]
        TOP["Top 3 class probabilities<br/>map indices with design.CToC"]
        CSV["batch_prediction_output.csv<br/>filename + 3 labels/confidences"]
        BATCH --> PRED
        PRED --> IMG
        IMG --> PTRANS
        CONFIG --> PTRANS
        PTRANS --> RESTORE
        CNN --> RESTORE
        WEIGHTS --> RESTORE
        RESTORE --> TOP
        CONFIG --> TOP
        TOP --> CSV
    end

    subgraph Measurement["Stroke-width measurement — python \"Stroke Measurement/measure_stroke_width.py\""]
        MS["measure_stroke_width.py"]
        EMNIST["EMNIST training split<br/>(configured in design.py)"]
        MEASURE["Sample class O; measure<br/>horizontal stroke on row 14"]
        WIDTHCSV["stroke_width_measurements.csv<br/>per-image widths"]
        SUMMARY["stroke_width_summary.txt<br/>summary statistics"]
        VALUE["Average 4.345 px<br/>EMNIST_AVERAGE_STROKE_WIDTH"]
        EMNIST --> MS
        CONFIG --> MS
        MS --> MEASURE
        MEASURE --> WIDTHCSV
        MEASURE --> SUMMARY
        SUMMARY -. "reported average is recorded as a constant" .-> VALUE
        VALUE -. "used by design.py's thickening transform<br/>and preprocessing_final.py" .-> TRANSFORM
        VALUE -.-> STEPS
    end
```

The dotted training-to-weights edge indicates the intended relationship, not an active save operation: `CNN.py` currently has its `torch.save(...)` call commented out. The checked-in `cnn_mnist_1.pt` is loaded by `predict.py`.

## Repository inventory

| File or folder | Role |
|---|---|
| `CNN.py` | Defines the convolutional network, selects and transforms training data through `design.py`, and runs training/testing when executed directly. |
| `design.py` | Central runtime configuration: dataset selection and paths, class-index-to-character mapping, preprocessing transforms, and number of output classes. It reuses helper functions from `preprocessing_final.py`. |
| `preprocessing_final.py` | Standalone image preprocessing pipeline. Running it directly reads `charset/`, writes accepted PNGs and a processing log to `output/`, and saves intermediate checkpoint images to `secondary results/`. |
| `predict.py` | Reads supported image files from `charset_2/`, applies preprocessing, loads the CNN weights, and writes top-three predictions to `batch_prediction_output.csv`. |
| `cnn_mnist_1.pt` | Saved model state dictionary loaded for prediction. |
| `batch_prediction_output.csv` | Batch prediction results: source filename and three predicted characters with confidence percentages. |
| `charset/` | Raw image inputs for the standalone preprocessing script. |
| `charset_2/` | Image inputs enumerated by the batch prediction script. |
| `output/` | Processed character PNGs, a `rejected/` subfolder created by the script, and `preprocessing_log.csv` with accepted/rejected status and failure reasons. Failed source images are not moved into `rejected/`. |
| `secondary results/` | Per-image intermediate images produced by `preprocessing_final.py`; suffixes identify processing checkpoints. |
| `Stroke Measurement/measure_stroke_width.py` | Selects a repeatable sample of EMNIST `O` images and measures their horizontal stroke widths. |
| `Stroke Measurement/stroke_width_measurements.csv` | Per-image stroke-width measurements. |
| `Stroke Measurement/stroke_width_summary.txt` | Measurement settings and aggregate statistics, including the average used as the thickening target. |
| `PROJECT_MAP.md` | This overview of the repository files and processing flows. |

## Important distinction between image paths

- The standalone preprocessing command is `python preprocessing_final.py`; its default path is `charset/` → `output/`.
- Training uses the dataset configured in `design.py` and the transform assembled there. It does not train from `output/` by default.
- Prediction reads from `charset_2/` and writes `batch_prediction_output.csv`; it does not read the standalone preprocessing outputs.
- Both the standalone pipeline and the training/prediction transform use the same helper routines from `preprocessing_final.py`, but the standalone script writes checkpoint images and logs while the transform is applied in memory.
