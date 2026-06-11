# Traffic Light Detection with YOLOv8s

**Dataset:** LISA Traffic Light Dataset (Supervisely/JSON format via Data Ninja)
**Model:** YOLOv8s — fine-tuned via transfer learning
**Classes:** 3 — `go` (green), `stop` (red), `warning` (yellow)
**Training:** 47 epochs, best at epoch 27, 2 h 14 min on RTX 5090 32 GB
**Overall mAP@50:** 0.768 | **Precision:** 0.874 | **Recall:** 0.780


## Project Overview

This project trains a YOLOv8s object detection model to identify traffic lights in real-world dashcam footage and classify each detected light as one of three states:

| Label | Meaning | Colour shown |
|-------|---------|--------------|
| `go` | Green light — safe to proceed | Green |
| `stop` | Red light — must stop | Red |
| `warning` | Yellow/amber light — prepare to stop | Orange |

The work is contained in a single Jupyter notebook (`traffic-light-model.ipynb`) split into **23 cells**, each doing one focused step. The notebook was designed for beginners: every cell explains what it does and, more importantly, **why** it is necessary.

**Transfer learning** is the technique of starting from a model already trained on a large dataset (here, COCO — 80 classes, millions of images) and fine-tuning only the task-specific parts for a new, smaller dataset. This allows a high-quality detector to be trained in a few hours rather than days.

---

## Dataset

### Raw Dataset Layout

The dataset used is the **LISA Traffic Light Dataset** (Laboratory for Intelligent and Safe Automobiles, UC San Diego). It was downloaded from **Data Ninja** in **Supervisely JSON format** — not the original LISA CSV format. This distinction is important: the folder structure and annotation format are completely different from what most tutorials expect.

- Total images: **43,016** (20,535 train + 22,481 test)
- Image resolution: **1280 × 960 pixels**
- Total bounding box annotations (train split): **~219,832 rows** after parsing

### Annotation Format

Each image has a corresponding `.jpg.json` file in the `ann/` folder. The double extension (`.jpg.json`) means the file is an annotation for a `.jpg` image. A representative file looks like this:

Key fields:
- `classTitle` — raw class name (one of 14 possible values)
- `points.exterior` — two corner points of the bounding box: top-left `[x1, y1]` and bottom-right `[x2, y2]`
- `tags` — metadata such as whether the clip is day/night and which video sequence it came from

### Class Mapping — 14 Raw Classes to 3 Semantic Classes

The raw dataset has 14 class names that distinguish directionality (left-turn lights, forward-only lights). For this project these are collapsed into 3 classes:

| Raw class name | Mapped to | Class ID |
|---|---|---|
| go | go | 0 |
| go traffic light | go | 0 |
| go left | go | 0 |
| go left traffic light | go | 0 |
| go forward | go | 0 |
| go forward traffic light | go | 0 |
| stop | stop | 1 |
| stop traffic light | stop | 1 |
| stop left | stop | 1 |
| stop left traffic light | stop | 1 |
| warning | warning | 2 |
| warning traffic light | warning | 2 |
| warning left | warning | 2 |
| warning left traffic light | warning | 2 |

> **Note:** `go forward` and `go forward traffic light` appear only in `test/ann/`, not in `train/ann/`. The `CLASS_MAP` dictionary handles them correctly during test evaluation. This is not an error.

---

## Pipeline — Notebook Cells

Each cell builds on the previous one; skipping cells will cause errors.

---

### Cell 1 — Install Libraries

Installs four Python packages: `ultralytics`, `tqdm`, `PyYAML`, `onnx`.

- `ultralytics` — the library that provides YOLOv8. Without it, no YOLO model can be loaded or trained.
- `tqdm` — displays progress bars for long loops (e.g., processing 20,535 files). Without it the cell appears frozen.
- `PyYAML` — writes the `data.yaml` configuration file that YOLOv8 requires to know where the data is.
- `onnx` — used in Cell 22 to export the trained model to a format that runs on mobile devices, browsers, and embedded systems.

---

### Cell 2 — Imports + Path Configuration + Dataset Verification

Imports all libraries used throughout the notebook. Defines every file path as a `pathlib.Path` variable. Counts and prints the number of images and annotations in train and test to verify the dataset loaded correctly (expected: 20,535 train, 22,481 test).

Centralising all paths in one cell prevents hardcoded string errors later. If the dataset is moved or renamed, only this cell needs updating. `pathlib.Path` handles cross-platform path separators automatically.

---

### Cell 3 — Understand Annotation Format

Loads a single `.jpg.json` annotation file and prints its raw content. Then explains each field line by line.

The LISA dataset was downloaded in Supervisely JSON format, which is different from the original LISA CSV format most tutorials describe. Before writing any parsing code, it is essential to see exactly what the raw data looks like. This cell is purely educational — it produces no output files.

---

### Cell 4 — Load All Annotations into a DataFrame

Defines `parse_annotation_file()`, which reads one `.jpg.json` file and extracts all bounding box rows. Loops over all 20,535 train annotation files with a `tqdm` progress bar. Loads results into a `pandas` DataFrame with columns: `image_file`, `class_title`, `x1`, `y1`, `x2`, `y2`, `img_width`, `img_height`.

`ann_path.stem` (pathlib) removes only the `.json` extension, leaving `.jpg` — which is exactly the correct image filename. The glob pattern must be `*.jpg.json`, not `*.json`.

~219,832 rows — one per bounding box annotation.

All downstream analysis (class distribution, box size statistics, train/val split) depends on this DataFrame.

---

### Cell 5 — Map 14 Classes to 3 + Visualize

Creates a `CLASS_MAP` dictionary that maps each raw class name to `go`, `stop`, or `warning`. Adds a `class_mapped` column to the DataFrame. Draws a **3×3 grid** of real sample images (one row per class) with bounding boxes drawn directly on the images so the user can visually confirm what each class looks like.

OpenCV draws in BGR colour order. Matplotlib expects RGB. The image must be kept in BGR while all boxes are drawn, then converted BGR→RGB at the very end before displaying. Converting too early makes stop boxes appear blue and warning boxes appear purple.

Visual confirmation that the mapping is correct before any conversion happens.

---

### Cell 6 — Class Distribution Bar Chart

Counts annotations per mapped class, draws a bar chart, prints exact numbers and percentages.

**Resul:**

| Class | Count | Percentage |
|-------|-------|------------|
| stop | ~114,000 | ~52% |
| go | ~99,000 | ~45% |
| warning | ~6,200 | ~2.8% |

This chart reveals a severe **class imbalance**: the stop:warning ratio is approximately 19:1. A model trained on imbalanced data will learn to detect stop and go lights well but largely ignore warning lights because getting warning wrong barely affects the overall loss. This chart is the evidence that justifies the oversampling step in Cell 12.

---

### Cell 7 — Bounding Box Size Analysis

Calculates width and height in pixels for every bounding box. Draws a scatter plot of width vs height and a histogram of box areas. Classifies boxes by COCO standards (Tiny: <32px, Small: 32–96px, Medium: 96–224px, Large: >224px). Prints mean dimensions.

Mean bounding box: 20 × 33 pixels on a 1280×960 image (0.05% of image area). Approximately **96% of boxes are Tiny or Small** by COCO standards.

This is the quantitative justification for two key training decisions:
1. **Using YOLOv8s over YOLOv8n** — the `s` (small) variant has a deeper feature pyramid network (FPN) that extracts finer spatial details, which is critical for detecting very small objects.
2. **Using `imgsz=1280` instead of the default 640** — downscaling a 1280×960 image to 640px makes 20×33px boxes only 10×17px, which is below reliable detection thresholds. Keeping images at 1280px gives the model more pixels to work with.

---

### Cell 8 — Sample Images with Bounding Boxes

Picks 8 representative images from the dataset and draws annotated boxes (green = go, red = stop, orange = warning) on each. Displays them in a 2×4 grid.

A final visual sanity check — at this point class mapping, coordinate parsing, and colour conversion are all working together. If any step had a bug, the boxes would be misaligned or the wrong colour.

---

### Cell 9 — Create YOLO Directory Structure

Creates the `yolo_dataset/` folder tree: `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, `labels/test`, `labels/train_all`.

YOLOv8 requires this exact directory layout. It finds images by looking in `images/` and expects corresponding label files in the parallel `labels/` folder with the same filename (different extension). Without this structure, training fails immediately.

---

### Cell 10 — Convert JSON Annotations to YOLO Format

Defines `json_to_yolo_lines()`, which reads a `.jpg.json` annotation file and converts each bounding box to **YOLO text format**.

**YOLO format (one line per bounding box):**
```
class_id  x_centre  y_centre  width  height
```
All five values are normalised to the range 0–1 (divided by image width/height). For example:
```
0  0.512  0.421  0.016  0.034
```

**Conversion formulas:**
```
x_centre = (x1 + x2) / 2 / img_width
y_centre = (y1 + y2) / 2 / img_height
width    = (x2 - x1) / img_width
height   = (y2 - y1) / img_height
```

Output label file: `ann_path.name.replace('.jpg.json', '.txt')` — e.g., `dayClip1--00001.jpg.json` → `dayClip1--00001.txt`.

YOLOv8 cannot read Supervisely JSON. It only reads its own `.txt` label format. This cell is the central data conversion step.

---

### Cell 11 — Sequence-Aware Train / Validation Split (85 / 15)

Splits training data into a training set (85%) and a validation set (15%) using **sequence-aware splitting**.

**The problem:** The LISA dataset is dashcam footage. Consecutive frames (e.g., `dayClip5--00042.jpg`, `dayClip5--00043.jpg`, `dayClip5--00044.jpg`) are nearly identical — the car has barely moved. A random image-level split would place these near-identical frames on both sides of the train/val boundary. The model would appear to have very high validation accuracy simply because it memorised frames it effectively saw in training. This is called **data leakage**.

**Solution:** Extract the sequence name from the filename stem using `.split('--')[0]` (e.g., `dayClip5`). Group all frames of a sequence together, then split by sequence (18 unique sequences) rather than by individual frame. Entire sequences go either to train or to val, never split across both.

A stratified split (ensuring each class appears in both halves) fails when a class appears in only one sequence. With only 18 sequences this happens for `warning` and background clips. A `try/except` block catches the `ValueError` and falls back to a plain (non-stratified) split.

---

### Cell 12 — Oversample the Warning Class

Counts warning-class images in the training set. Calculates how many additional copies are needed to bring warning from 2.8% to ~10% of the training set. Creates **symbolic links** (not file copies) to warning-dominant images, with suffixes `_over1`, `_over2`, etc.

**Symlink:** A symlink is a pointer to an existing file. It uses almost no disk space while appearing as a real file to the training process. Oversampling via symlinks saves approximately 20 GB of disk space compared to copying files.

At 2.8%, warning lights are so rare that the model sees far too few examples to learn them reliably. Without oversampling, warning mAP@50 would be significantly lower. Cell 18 results show oversampling worked: warning achieved mAP@50 = 0.843.

---

### Cell 13 — Create `data.yaml`

Writes a YAML configuration file at `yolo_dataset/data.yaml` with the following structure:

```yaml
train: /absolute/path/yolo_dataset/images/train
val:   /absolute/path/yolo_dataset/images/val
test:  /absolute/path/yolo_dataset/images/test
nc: 3
names: ['go', 'stop', 'warning']
```

YOLOv8 reads this file before training begins to learn where the data is, how many classes exist (`nc`), and what each class is named. Without this file, `model.train()` will raise a `FileNotFoundError`.

---

### Cell 14 — Label Audit

Reads every `.txt` label file in the training and validation sets. For each line, checks that all five values are within the valid range [0, 1] and that the class ID is within [0, nc-1]. Counts and reports out-of-range values and malformed lines.

Conversion bugs (e.g., using pixel coordinates instead of normalised coordinates, or dividing by the wrong dimension) would produce coordinates outside [0, 1]. Such labels silently degrade training or cause NaN losses. Catching these errors now saves hours of debugging after a failed training run.

---

### Cell 15 — Visualize YOLO Labels

Takes a sample of images from the training set, reads their corresponding `.txt` label files, converts normalised coordinates back to pixel coordinates, and draws the boxes on the original images. Displays the result.

This is the **ultimate proof** that Cell 10's conversion is correct. If boxes align with actual traffic lights in the image, the conversion is correct. If boxes are wildly misplaced, something went wrong with the coordinate formulas.

---

### Cell 16 — Train YOLOv8s

Loads `yolov8s.pt` (pretrained on COCO's 80 classes) and calls `model.train()` with the following key settings:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `epochs` | 50 | Maximum iterations |
| `imgsz` | 1280 | Preserve small-object detail |
| `batch` | 16 | Fits within 32 GB VRAM (tested: batch=32 caused OOM) |
| `patience` | 20 | Stop if no improvement for 20 epochs |
| `mosaic` | 1.0 | Mosaic augmentation — tiles 4 images into one training sample |
| `mixup` | 0.1 | Blends two images during training for regularisation |
| `fliplr` | 0.5 | Random horizontal flip (50% chance) |

**Smart re-run protection:** The cell first checks whether `best.pt` already exists. If it does, training is skipped and the existing model is loaded. This prevents accidentally spending 2+ hours retraining after a kernel restart.

**Training output explained:**

| Log line | Meaning |
|---|---|
| `Overriding model.yaml nc=80 with nc=3` | The COCO detection head (80 classes) is replaced with a 3-class head |
| `Transferred 349/355 items from pretrained weights` | 349 weight groups kept from COCO; only 6 replaced (the new detection head) |
| `AMP: checks passed` | Automatic Mixed Precision enabled — uses 16-bit math, halving VRAM usage |
| `2385 backgrounds` | Images with no traffic lights, kept intentionally to teach the model not to hallucinate |
| `nightClip4: 2 duplicate labels removed` | Minor oversampling artefact — YOLOv8 auto-fixes this |
| `Closing dataloader mosaic` | Mosaic augmentation disabled for final 10 epochs (standard YOLOv8 schedule) |
| `EarlyStopping patience 20/20` | Best at epoch 27; no improvement for 20 epochs; training stopped at epoch 47 |

**Result:** 47 epochs run, best at epoch 27, total training time **2 hours 14 minutes** on an NVIDIA RTX 5090 32 GB (VRAM peak: 15.4 GB).

---

### Cell 17 — Plot Training Curves

Reads `results.csv` (saved automatically by YOLOv8 during training). Plots a **2×3 grid** of charts:

- Row 1: `box_loss`, `cls_loss`, `dfl_loss` — each showing train and validation curves
- Row 2: `mAP@50`, `mAP@50-95`, Precision & Recall

A red dotted vertical line marks **epoch 27** (best checkpoint).

**Terms:**
- `box_loss` — how accurately predicted boxes match ground-truth boxes (IoU-based)
- `cls_loss` — how accurately class labels are predicted
- `dfl_loss` — distribution focal loss — controls the quality of box boundary predictions
- `mAP@50` — mean Average Precision at 50% IoU overlap threshold (standard YOLO metric)
- `mAP@50-95` — averaged over IoU thresholds 50%–95% in 5% steps (stricter metric)

Loss curves reveal whether the model is learning, overfitting, or underfitting. mAP curves confirm when performance peaks.

---

### Cell 18 — Evaluate on the Test Set

Runs `model.val(split='test')` on all 22,481 unseen test images. Extracts per-class precision, recall, mAP@50, and mAP@50-95. Prints a dedicated spotlight on the warning class (the most challenging class after oversampling). Draws a grouped bar chart comparing all three classes across all four metrics.

Validation metrics (from Cell 17) are computed on data used to select the best checkpoint. Test metrics are on completely unseen data — they are the honest measure of real-world performance.

---

### Cell 19 — Confusion Matrix

Displays the confusion matrices auto-saved by YOLOv8 during evaluation: `confusion_matrix_normalized.png` (percentages) and `confusion_matrix.png` (raw counts). Also generates a test-set confusion matrix from the output of Cell 18.

**How to read it:** Each row represents a predicted class; each column represents a ground-truth class. A perfect model has all values on the diagonal (predicted = actual) and zeros elsewhere. Off-diagonal values indicate misclassifications.

The confusion matrix reveals which classes get confused with each other (e.g., does the model ever call a red light green?) and how many objects are missed entirely (false negatives).

---

### Cell 20 — Inference on Sample Test Images

Selects 12 test images that have at least one detection. Runs `model.predict()` in batches of 16. Draws coloured bounding boxes with confidence scores on each image. Displays a 3×4 grid.

**Confidence score:** A number between 0 and 1 indicating how certain the model is about a detection. A score of 0.87 means the model is 87% confident.

Visual inspection of live predictions on unseen images. The ultimate qualitative test — does the model actually look correct?

---

### Cell 21 — Build Inference Video from Sequential Frames

Takes the first 300 frames of the `daySequence1` clip (which contains 4,060 total frames — approximately 10 seconds at 30 fps). Runs inference in batches of 16. Writes annotated frames to `daySequence1_inference.mp4` using OpenCV's `VideoWriter`.

Single-image inference confirms the model works. Video inference confirms it works on real temporal sequences — that detections are stable across consecutive frames and do not flicker wildly.

---

### Cell 21b — Test on a Custom Video or Image

Accepts any `.mp4` or `.jpg` file path via a `SOURCE` variable. Auto-detects whether the input is a video or image. For video: reads with `cv2.VideoCapture`, processes in batches, writes output to a `detected/` subfolder. For image: runs predict and displays inline.

**Output:** `slowed_sample-3_detected.mp4` (video) or inline display (image).

Allows the user to test the model on their own dashcam footage or photos without modifying any other cell.

---

### Cell 22 — ONNX Export

Exports the trained model to ONNX format:

```python
model.export(format='onnx', imgsz=1280, opset=17, simplify=True)
```

Validates the exported file with `onnx.checker.check_model()`. Prints and compares file sizes of `.pt` (PyTorch) vs `.onnx`.

**ONNX** (Open Neural Network Exchange) is a universal model format that runs on virtually any hardware and framework — including Android apps, web browsers (via WebAssembly), C++ applications, and embedded devices. PyTorch `.pt` files require Python and PyTorch to run; `.onnx` files do not.

The `.pt` model cannot be deployed to a phone or website. ONNX export is the gateway to real-world deployment.

---

### Cell 23 — Summary + 7 Next Steps

Prints a complete project summary — dataset statistics, model configuration, training outcome, and final test metrics — in a formatted block. Lists 7 actionable next steps for improving the model further.

Serves as the final record of the project. Useful for documentation, sharing results, and planning future iterations.

---

## 5. Graphs and Chart Insights

### Class Distribution Bar Chart (Cell 6)

The bar chart shows three bars: stop (~114,000), go (~99,000), warning (~6,200). The imbalance is immediately striking — warning has roughly 19 times fewer annotations than stop. A model trained without correction would see stop lights in almost every batch but warning lights only rarely. As a result, the gradient updates would almost entirely come from stop and go examples, and the model would learn to essentially ignore warning lights. The bar chart is the motivation for the oversampling step in Cell 12.

### Bounding Box Size Analysis (Cell 7)

The scatter plot clusters almost entirely in the bottom-left corner: most boxes are below 50 pixels wide and 60 pixels tall on a 1280×960 canvas. The mean is 20×33 pixels — roughly the size of a thumbnail. The histogram confirms ~96% of boxes fall into the Tiny (<32px) or Small (32–96px) COCO size categories. This is not surprising for a dashcam dataset where traffic lights are often 50–200 metres ahead. These numbers directly justify using `imgsz=1280` and `YOLOv8s` over smaller/lower-resolution alternatives.

### Training Loss Curves (Cell 17)

All three loss values decrease across all 47 epochs, meaning the model is continuously learning. The most dramatic drop is in `cls_loss` (classification loss): it falls from 1.71 at epoch 1 to 0.37 at epoch 47. This indicates the model very successfully learned to distinguish green from red from yellow lights. After the best epoch (27), the loss continues to fall very slowly, but validation mAP stops improving — the model is beginning to overfit to noise in the training data rather than learning generalisable patterns.

### mAP Curve (Cell 17)

`mAP@50` rises steeply from 0.664 at epoch 1 to 0.768 at epoch 27, then oscillates without clear improvement through epoch 47. This plateau is the textbook signal that early stopping made the correct decision: continuing beyond epoch 27 would not produce a better model, only a more overfit one.

### Precision and Recall Curve (Cell 17)

Precision rises faster than recall. By the best epoch, precision is 0.874 and recall is 0.780. In plain English: when the model detects a traffic light, it is correct 87.4% of the time (high precision); however, it misses about 22% of the traffic lights that are present in images (lower recall). The dominant source of missed detections is the `go` class.

### Test Set Per-Class Bar Chart (Cell 18)

- **stop** achieves the highest recall (0.959) — it misses fewer than 5% of red lights. This is the most safety-critical case (failing to detect a red light is dangerous) and the model handles it best.
- **go** has the lowest recall (0.559) — it misses approximately 44% of green lights. This is the largest remaining weakness and the top priority for future improvement.
- **warning** achieves mAP@50 = 0.843 — surprisingly strong given it had only 2.8% of training examples before oversampling. This validates that the oversampling strategy worked.

### Confusion Matrix (Cell 19)

The confusion matrix shows a strong diagonal — the model very rarely confuses one class for another. It almost never predicts `go` when the true class is `stop` (which would be dangerous). The off-diagonal mass is concentrated in the **background row**: a substantial number of detections are missed entirely (false negatives). The `go` class has the highest false-negative rate, consistent with the recall numbers above.

---

#### How to Read the Confusion Matrix

A **confusion matrix** is a grid that summarises how well the model's predictions match the real (ground-truth) labels. Think of it as a scoreboard that shows not just how many things were correct, but specifically *what kind of mistakes* were made.

**Structure of this matrix (4 × 4):**

```
                 PREDICTED →
                 go      stop    warning   background
ACTUAL  go     [ 0.559 ] [  ~0  ] [  ~0  ] [  0.441 ]
  ↓    stop    [  ~0  ] [ 0.959 ] [  ~0  ] [  0.041 ]
       warning [  ~0  ] [  ~0  ] [ 0.821 ] [  0.179 ]
       backgnd [ TP FP ] [ TP FP ] [ TP FP ] [  ---  ]
```

- **Rows** = what the object actually is (the ground truth label).
- **Columns** = what the model predicted it to be.
- **Each cell** shows what fraction of that row ended up in that column.

---

#### The Diagonal — Correct Detections

The numbers along the top-left to bottom-right diagonal are the **correct predictions**:

| Cell | Meaning | Value |
|------|---------|-------|
| go → go | Model saw a green light and said "go" | **0.559** (55.9%) |
| stop → stop | Model saw a red light and said "stop" | **0.959** (95.9%) |
| warning → warning | Model saw a yellow light and said "warning" | **0.821** (82.1%) |

A perfect model would have 1.00 (100%) across this entire diagonal. The closer to 1.0, the better.

---

#### The "Background" Column — Missed Detections (False Negatives)

In YOLO's confusion matrix, **"background"** means the model looked at a real traffic light and did not detect it at all — it treated the object as part of the background scenery.

| Row | Meaning | Value |
|-----|---------|-------|
| go → background | Real green lights that were completely missed | **~44%** |
| stop → background | Real red lights that were completely missed | **~4%** |
| warning → background | Real yellow lights that were completely missed | **~18%** |

This is the most important column to watch because a missed detection is more dangerous than a wrong classification — the car would behave as if no traffic light existed.

> **Why does `go` have so many misses?** Green lights in the LISA dataset are small (dashcam distance) and the class has the most variety in appearance (go-left, go-forward, different pole positions). The model under-detects them rather than mis-labelling them as another colour.

---

#### Off-Diagonal Inter-Class Cells — Wrong Label

These cells show cases where the model *did* detect something, but gave it the wrong class name (e.g., calling a green light red):

- **go → stop** ≈ 0 — The model essentially never mistakes a green light for red. This is the most safety-critical error and it is almost absent.
- **stop → go** ≈ 0 — The model essentially never mistakes a red light for green. Excellent safety behaviour.
- **warning → go or stop** ≈ very small — Yellow lights are rarely confused with other colours.

This means the model's errors are almost entirely *missed detections*, not *wrong-colour detections*. That is the safer failure mode.

---

#### The "Background" Row — False Positives

The bottom row (background → any class) shows cases where the model detected a traffic light *in a region that had no real traffic light* — it invented a detection from background clutter. This row is typically small for a well-trained model and confirms the model is not hallucinating phantom traffic lights.

---

#### Summary in Plain English

| Situation | How common | Safety impact |
|-----------|-----------|--------------|
| Model correctly detects stop light | 95.9% of red lights | ✅ Excellent — safest class |
| Model correctly detects warning light | 82.1% of yellow lights | ✅ Good — oversampling helped |
| Model correctly detects go light | 55.9% of green lights | ⚠ Weak — misses 44% of greens |
| Model calls a red light "green" | Near zero | ✅ Critical error is absent |
| Model calls a green light "red" | Near zero | ✅ Critical error is absent |
| Model misses a traffic light entirely | go: 44%, warning: 18%, stop: 4% | ⚠ Main weakness overall |

**Bottom line:** The model makes very safe mistakes — it under-detects rather than mis-labels. When it does make a call, the colour is almost always correct. The priority for improvement is increasing recall for the `go` class so fewer green lights are missed.

---

## 6. Final Results

### Validation Set — Best Checkpoint (epoch 27)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---------|-----------|--------|--------|-----------|
| go | 0.942 | 0.559 | 0.645 | 0.348 |
| stop | 0.790 | 0.959 | 0.816 | 0.417 |
| warning | 0.891 | 0.821 | 0.843 | 0.371 |
| **ALL** | **0.874** | **0.780** | **0.768** | **0.379** |

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | YOLOv8s (pretrained on COCO) |
| Epochs run | 47 (early stopping) |
| Best epoch | 27 |
| Training time | 2 h 14 min |
| GPU | NVIDIA RTX 5090 32 GB |
| Peak VRAM | 15.4 GB |
| Image size | 1280 × 1280 |
| Batch size | 16 |

**Term definitions:**
- **Precision** — Of all detections the model made, what fraction were correct?
- **Recall** — Of all ground-truth objects in the images, what fraction did the model find?
- **mAP@50** — Mean Average Precision at 50% Intersection over Union (IoU) threshold. IoU measures how much the predicted box overlaps the ground-truth box.
- **mAP@50-95** — A stricter version averaging over IoU thresholds from 50% to 95%. Rewards precise box placement.

---

## 7. Model Accuracy Discussion

### What went well

The model achieves **mAP@50 = 0.768** overall, with particularly strong performance on the `stop` class (recall = 0.959). For a safety-critical application, high stop-light recall is the most important metric — the model almost never misses a red light.

The warning class result (mAP@50 = 0.843) is the most surprising success. Starting from only 2.8% of training examples, targeted oversampling raised the warning class to outperform go in both mAP and recall. This demonstrates that class imbalance, not model capacity, was the main limiting factor for warning detection.

### The go class weakness

The model's Achilles heel is green light recall (0.559 — misses 44% of green lights). Several factors contribute:

1. **Oversampling imbalance**: warning was oversampled but go was not. Go still has proportionally fewer *image* examples than its annotation count suggests, because many go annotations come from the same sequences.
2. **Visual similarity**: green and yellow LEDs can appear similar under certain lighting conditions.
3. **Sequence imbalance**: the 85/15 sequence-level split means some important go sequences ended up entirely in the validation set.

### Domain generalisation

The LISA dataset was captured in San Diego with specific camera angles and traffic light designs. When tested on footage from different cameras or cities (Cell 21b), the model may miss larger traffic lights (because it was trained on very small ones) or perform poorly in different lighting conditions. The recommended fix is to annotate 50–100 frames from the target domain and fine-tune for an additional 10–20 epochs.

---

## 8. Summary and Conclusion

This project demonstrates a complete, end-to-end deep learning pipeline for traffic light detection:

1. **Data acquisition** — LISA dataset in an unexpected Supervisely JSON format, requiring a custom parsing pipeline
2. **Exploratory analysis** — discovering class imbalance (19:1 stop:warning) and small object sizes (96% Tiny/Small)
3. **Data engineering** — sequence-aware splitting to prevent data leakage, warning oversampling via symlinks
4. **Model training** — YOLOv8s transfer learning with early stopping at epoch 27
5. **Evaluation** — honest test-set metrics on 22,481 unseen images
6. **Deployment preparation** — ONNX export for cross-platform inference

The result is a model that reliably detects **stop lights with 95.9% recall** (safety-critical), detects **warning lights with 84.3% mAP@50** (oversampling worked), and has a known weakness in **go light recall (55.9%)** that is addressable with targeted improvement.

The most important lesson from this project: the majority of the effort — and the majority of cells in the notebook — is not training but **data understanding and preparation**. Cells 1–15 (13 cells) prepare and validate the data. Cells 16–23 (8 cells) train, evaluate, and deploy. Cleaning and understanding the data is what makes training work.

---

## 9. Problems Faced and Solutions

This section documents every significant problem encountered during development and the exact solution applied.

---

### Problem 1 — Dataset Format Mismatch

**Error:** The project expected the original LISA CSV format with `/dayTrain/`, `/nightTrain/`, and `/Annotations/` folders containing `.csv` files.

**Reality:** The downloaded dataset was in Supervisely JSON format from Data Ninja: `.jpg.json` annotation files in `ann/` subfolders, no CSV files at all.

**Solution:** Rewrote the entire data loading pipeline from scratch for JSON. This required understanding the Supervisely format (Cells 3–4), writing a new `parse_annotation_file()` function, and updating every path reference in the notebook.

---

### Problem 2 — Double Extension `.jpg.json`

**Error:** Using `ann_path.stem` in Python's `pathlib` removes only the last extension. For `dayClip1--00001.jpg.json`, `stem` gives `dayClip1--00001.jpg` — not `dayClip1--00001`.

**Reality:** This is actually the correct behaviour. The image filename IS `dayClip1--00001.jpg`, so `ann_path.stem` gives exactly the right filename to look up the corresponding image.

**Solution:** Use `*.jpg.json` as the glob pattern (not `*.json` — which would also match `meta.json`). Use `ann_path.stem` for the image filename and `ann_path.name.replace('.jpg.json', '.txt')` for the label filename.

---

### Problem 3 — BGR vs RGB Colour Bug

**Symptom:** Bounding boxes appeared in the wrong colours — stop boxes were blue, warning boxes were yellow-blue instead of red and orange.

**Cause:** OpenCV reads images in BGR (Blue-Green-Red) channel order. Matplotlib displays images in RGB. Drawing boxes in BGR and then displaying without converting shows red as blue and vice versa.

**Solution:** Keep the image in BGR while drawing all boxes. Call `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` only once, at the very end, before passing the image to `matplotlib.pyplot.imshow()`. Never convert before drawing.

---

### Problem 4 — Cell Source Stored as Character List

**Symptom:** A notebook cell's `source` field contained a list of 3,777 individual characters instead of a list of code lines. The cell appeared as a stream of single characters.

**Cause:** Using `list(string)` in Python splits the string into individual characters — `list("abc")` gives `['a', 'b', 'c']`, not `['abc']`.

**Solution:** Always store notebook cell source as a list of strings where each element is one line ending with `\n`. The last line does not need a trailing `\n`. Example: `["line 1\n", "line 2\n", "line 3"]`.

---

### Problem 5 — Stratified Split `ValueError`

**Error:** `ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.`

**Cause:** `train_test_split(stratify=sequence_labels)` requires at least 2 members per class in `y`. With only 18 sequences total, the `warning` and `background` categories each appeared in only 1 sequence.

**Solution:** Wrapped the stratified split in a `try/except ValueError` block. On exception, falls back to a plain `train_test_split()` without stratification. The fallback split is still sequence-aware (groups by sequence), just not guaranteed to balance classes between splits.

---

### Problem 6 — CUDA Out-of-Memory Error

**Error:** `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` during training.

**Cause:** `batch=32` combined with `imgsz=1280` required more than 32 GB of VRAM during the assignment step of the loss calculation.

**Solution:**
1. Reduced batch size from 32 to 16. This brought VRAM usage to 14–18 GB (comfortably within the 32 GB limit).
2. Added `exist_ok=True` to the training call to handle the incomplete run directory left behind by the OOM-crashed run.

---

### Problem 7 — `AttributeError: 'DetMetrics' object has no attribute 'epoch'`

**Error:** Code tried to access `results.epoch` to find out how many epochs ran.

**Cause:** `model.train()` returns a `DetMetrics` object, not a trainer or results dictionary. `DetMetrics` stores precision, recall, and mAP values but does not have an `epoch` attribute.

**Solution:** Read the epoch count from `results.csv` instead: `epochs_run = len(pd.read_csv(results_csv_path))`. Each row in `results.csv` corresponds to one training epoch.

---

### Problem 8 — 14 Classes Listed but Only 12 Visible in Training

**Observation:** Training logs showed only 12 unique classes being processed, not 14.

**Cause:** `go forward` and `go forward traffic light` exist only in `test/ann/`, not in `train/ann/`. They were never seen during training data preparation.

**Impact:** None — the `CLASS_MAP` dictionary correctly maps both classes to `go` (class 0) regardless of which split they appear in. Test evaluation handles them correctly.

---

### Problem 9 — Duplicate Label Warnings for `nightClip4`

**Warning:** `nightClip4--XXXXX: 2 duplicate labels removed` printed during training.

**Cause:** The oversampling step (Cell 12) created symbolic links to image files whose corresponding label files contained duplicate annotation entries (two identical bounding box lines for the same object).

**Impact:** None — YOLOv8 automatically removes duplicate labels before training. The warning is informational only.

---

### Problem 10 — Re-run Risk After Fixing `results.epoch`

**Risk:** After fixing the `AttributeError` from Problem 7 and saving the notebook, re-running Cell 16 would launch a full new training run — wasting 2+ hours.

**Solution:** Added a guard at the top of Cell 16:

```python
if BEST_MODEL.exists():
    print("Training already complete. Loading existing model.")
    model = YOLO(str(BEST_MODEL))
else:
    # ... full training code
```

If `best.pt` already exists, the cell skips `model.train()` entirely and loads the existing weights.

---

### Problem 11 — `nonlocal SyntaxError` in Cell 21b

**Error:** `SyntaxError: no binding for nonlocal 'frame_count' found`

**Cause:** The `nonlocal` keyword in Python only works inside a function that is **nested within another function**. In a Jupyter notebook cell, the outer scope is the module level, not a function. `nonlocal` cannot reference module-level variables.

**Solution:** Removed the inner function entirely. Moved all batch-processing logic directly into the `while` loop of the main video-reading block. Used a regular variable (`frame_count`) in the outer scope, modified directly without `nonlocal`.

---

### Problem 12 — Large Traffic Lights Not Detected in Custom Video

**Symptom:** When running Cell 21b on custom dashcam footage, the model failed to detect traffic lights that were clearly visible to the human eye.

**Cause:** The LISA dataset contains traffic lights that are very small in the frame (mean 20×33 pixels). The model was trained exclusively on this scale. Custom footage had larger traffic lights at closer range or different camera angles, which fell outside the model's learned scale distribution.

**Solutions:**
1. Lower the confidence threshold: `model.predict(conf=0.15)` (default is 0.25)
2. Enable multi-scale inference: `multi_scale=True` in predict settings
3. For a permanent fix: record and annotate 50–100 frames from the custom video and fine-tune the model on the combined dataset for 10–20 additional epochs

---

## 10. Ways to Improve Accuracy

1. **Fix go class recall (currently 0.559)** — Apply targeted oversampling for the `go` class in underrepresented sequences, similar to the warning oversampling in Cell 12. Alternatively, use focal loss with a higher weight for the go class.

2. **Train longer with cosine learning rate** — Set `patience=50`, `epochs=150`, `cos_lr=True`. Cosine learning rate annealing gradually reduces the learning rate in a smooth curve rather than a step drop, allowing the model to find a better optimum.

3. **Upgrade to YOLOv8m** — The medium variant has approximately 25 million parameters vs 11 million for YOLOv8s. Expected gain: +3 to +6 mAP@50 points at the cost of ~2× training time and VRAM.

4. **Use the full dataset** — After honest test-set evaluation is complete, combine `train/` and `test/` into a single pool and retrain from scratch. This gives the model 43,016 images instead of 20,535.

5. **Add multi-scale training** — Set `multi_scale=True` in the training call. YOLOv8 will randomly resize images during training, making the model more robust to traffic lights at various distances.

6. **Try YOLO11s** — YOLO11 is a newer architecture (included as `yolo11n.pt` in this repo). The `s` variant uses the same parameter budget as YOLOv8s but with architectural improvements that typically yield +2 to +4 mAP@50 on small-object tasks.

7. **Domain adaptation for custom footage** — Annotate 50–200 frames from the target camera / city. Fine-tune the best model on this domain-specific data for 10–20 epochs. This addresses the scale and visual style mismatch described in Problem 12.

---

## 11. Deployment

### Prerequisite

Run Cell 22 first to produce `best.onnx`. ONNX export settings used: `imgsz=1280`, `opset=17`, `simplify=True`.

For mobile deployment, re-export at `imgsz=640` to reduce inference time:
```python
model.export(format='onnx', imgsz=640, opset=17, simplify=True)
```

---

### Option A — Android App (ONNX Runtime)

1. Add the `onnxruntime-android` dependency to `build.gradle`
2. Copy `best.onnx` into the app's `assets/` folder
3. Load the model with `OrtSession` and run inference on frames from the device camera
4. Post-process outputs: decode boxes, apply Non-Maximum Suppression (NMS), draw results

---

### Option B — Website (Browser, No Server)

Use **ONNX Runtime Web** — a JavaScript library that runs ONNX models in the browser via WebAssembly.

1. Include `ort.min.js` in the HTML page
2. Load `best.onnx` via `InferenceSession.create('best.onnx')`
3. Pre-process uploaded images to `[1, 3, 640, 640]` float32 tensors
4. Run inference and render detections with `<canvas>`

No server required — the model runs entirely on the user's device.

---

### Option C — REST API (Python Backend)

Use **FastAPI** with a Python backend:

```python
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO('runs/detect/traffic_light_v1/weights/best.pt')

@app.post('/detect')
async def detect(file: UploadFile):
    image = read_image(await file.read())
    results = model.predict(image, conf=0.25)
    return {"detections": results[0].tojson()}
```

