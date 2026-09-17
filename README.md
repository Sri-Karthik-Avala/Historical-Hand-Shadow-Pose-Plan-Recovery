# Historical Hand-Shadow Pose Plan Recovery

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 3 |
| Last submission | 2026-07-23 |

## Problem statement

### Overview

Plain language objective: infer the hand-and-arm arrangement that would cast a provided hand-shadow silhouette.

Hand-shadow performers can make a rabbit, bird, camel, or character appear on a wall by arranging their hands between a small light and a screen. This challenge turns that visual inverse problem into a structured computer-vision task: recover the expert hand-and-arm ink mask implied by the target shadow.

Each row contains a `128 x 128` target-shadow image. The answer is not a caption or object category. You must submit a structured pose-plan surrogate: a binary `128 x 128` mask of the historical hand/arm drawing in run-length encoding, a normalized bounding box for that hand ink, and a confidence value.

This is not hand-shadow classification, captioning, or visible-hand segmentation. The hand is not visible in the test input. A valid answer must reconstruct a spatial hand-pose proxy from the target silhouette: where the palm/arm mass should sit, which parts of the outline require extended fingers or compact hand mass, and how much uncertainty remains. The mask, bounding box, and confidence should describe one consistent visual construction rather than three independent guesses.

This is a CPU-only challenge. Solutions must run within 1.5 hours on 10 CPU cores and 62 GB RAM. Reasonable approaches include classical contour features, shape retrieval trained only on public train rows, lightweight image encoders, small autoencoders, and structured mask decoders. No GPU is required or assumed.

### Task

For each test row, read the target-shadow image and predict these three outputs:

- `hand_ink_rle`: row-major RLE for a `128 x 128` binary hand/arm ink mask, where `1` means predicted hand/arm ink.
- `hand_bbox_json`: a normalized box around the predicted hand/arm ink with exactly `x`, `y`, `w`, and `h`.
- `confidence`: your calibrated estimate of how accurate the row prediction is.

The central object is the mask. A strong submission should learn the visual relationship between shadow contours and hand construction from the public training pairs, then derive a compatible box and confidence from the predicted mask.

### Intended Approach And Validation

A practical CPU solution is to decode the training `hand_ink_rle` masks, featurize each target shadow, and learn a compact shadow-to-hand-mask predictor. Good starting points include contour descriptors plus train-only retrieval, k-nearest template alignment, random forests or gradient-boosted patch features, small image-to-mask encoder-decoders, PCA/autoencoder mask models, and lightweight refinement of training hand-pose templates.

Another reasonable route is a two-stage pipeline: first predict a coarse hand mask or hand-mask embedding from shadow shape, area, contour curvature, holes, endpoints, and aspect-ratio features; then postprocess the predicted mask with connected-component cleanup, derive the bounding box from that mask, and calibrate confidence on held-out training folds. This is still a visual modeling task: the target image gives the evidence, but its file path, row id, size, and prompt do not contain the answer.

Use only the released public training labels for model selection. Make your own validation folds from `train.csv`, for example by grouping visually similar shadow footprints, aspect-ratio bins, or contour-complexity bins that you compute from the training images. Check both mask quality and downstream bbox/confidence behavior. Open-source local CV libraries, classical features, and generic offline vision backbones are allowed if they run under the CPU limit and do not use external hand-shadow plate lookup or hidden test information.

**What Not To Use / Do**:

- Do not submit object captions, animal names, or generic class labels instead of the required hand-pose mask.
- Do not use external source lookup, original book filenames, page order, captions, or historical plate matching to recover held-out answers.
- Do not use metadata-only rules, row order, file size, mtime, prepared id, or filesystem side channels.
- Do not use hosted or closed-source APIs for training, inference, distillation, pseudo-labeling, or mask generation.
- Do not submit a generic segmentation model that ignores the target-shadow evidence and emits one fixed hand template.
- Do not reduce the task to one bounding box, one average mask, or one object-family decision; the spatial hand mask is required.
- Do not inspect private answer files, hidden platform state, grader internals, or filesystem artifacts outside the released public files.
- Do not use runtime internet access, source-image search, reverse image search, perceptual hashing against outside collections, or downloaded task-specific weights.
- Do not use malformed RLE/JSON, duplicate ids, extra columns, missing rows, NaN/inf values, or grader-probing attempts.

**Enforcement on invalid approaches:** source-lookup solutions, metadata-only predictors, fixed-template submissions, caption-only methods, hosted-API approaches, and submissions that do not attempt the required hand-mask reconstruction may be rejected before payout regardless of leaderboard score.

### Evaluation

Each row receives a score in `[0, 1]`. Higher is better.

The binary mask head combines exact and tolerant shape agreement:

```
IoU      = intersection / union
Dice     = 2 * intersection / (pred_pixels + true_pixels)
TolF1    = F1 after radius-2 binary dilation
S_mask   = 0.34*IoU + 0.38*Dice + 0.28*TolF1
```

The bounding-box head compares normalized `x`, `y`, `w`, and `h`:

```
S_bbox = exp(-L1_bbox_error / 0.55)
core   = 0.72*S_mask + 0.18*S_bbox
calib  = 1 - abs(confidence - core / 0.90)
row    = core + 0.10*calib
```

The final score blends mean row quality with worst hidden subgroup quality:

```
raw_final = 0.78*mean(row)
          + 0.12*worst_object_family(row)
          + 0.10*worst_shadow_footprint(row)

Final = raw_final ** 1.45
```

The hidden subgroup labels are private metadata and are not columns in `train.csv` or `test.csv`. The theoretical minimum is `0.0` and the theoretical maximum is `1.0`. Perfect labels with confidence `1.0` score exactly `1.0`.

Worst subgroup means the lowest subgroup mean across the private group buckets used by the grader.

The grader returns `0.0` for missing, extra, or reordered submission columns; duplicate ids; missing or extra ids; row-set mismatch; invalid id values; NaN/inf confidence; confidence outside `[0,1]`; or invalid answer files. Malformed or overlong row-local RLE/JSON receives zero credit for the affected row without leaking private labels.

### Dataset

The prepared dataset ships as a `public/` directory with train/test CSV files, target-shadow PNG images, and a sample submission template. Images are `128 x 128` grayscale PNGs derived from historical hand-shadow instruction plates. Public ids and paths are opaque and do not expose original source filenames, captions, or page order.

`sample_submission.csv` is a weak placeholder template with the required submission columns and test ids.

| Item | Description |
| --- | --- |
| `train/shadows/*.png` | Train shadows |
| `test/shadows/*.png` | Test shadows |
| `train.csv` | Inputs plus labels |
| `test.csv` | Inputs only |
| `sample_submission.csv` | Weak template |

### train.csv columns

The train-only label columns are `hand_ink_rle` and `hand_bbox_json`. The train input columns are `id`, `target_shadow_path`, `canvas_size`, and `prompt`.

| Column | Type | Description |
| --- | --- | --- |
| `id` | int | Unique row id |
| `target_shadow_path` | string | PNG path |
| `canvas_size` | int | Always 128 |
| `prompt` | string | Task instruction |
| `hand_ink_rle` | string | Target mask RLE |
| `hand_bbox_json` | string | Target bbox JSON |

### test.csv columns

The test input columns are `id`, `target_shadow_path`, `canvas_size`, and `prompt`. Test rows do not include `hand_ink_rle` or `hand_bbox_json`.

| Column | Type | Description |
| --- | --- | --- |
| `id` | int | Unique row id |
| `target_shadow_path` | string | PNG path |
| `canvas_size` | int | Always 128 |
| `prompt` | string | Task instruction |

The RLE is row-major over the `128 x 128` mask. Runs are written as `start:length` pairs separated by spaces, using zero-based flattened pixel indices. An empty string represents an all-zero mask.

### Submission

Write your prediction file to `./working/submission.csv`. Submit a CSV with exactly these four columns in this order: `id`, `hand_ink_rle`, `hand_bbox_json`, `confidence`.

| Column | Type | Constraint |
| --- | --- | --- |
| `id` | int | Same ids as test |
| `hand_ink_rle` | string | RLE, max length |
| `hand_bbox_json` | string | x/y/w/h JSON |
| `confidence` | float | In `[0,1]` |

Train-only label columns are `hand_ink_rle` and `hand_bbox_json`. Test input columns are `id`, `target_shadow_path`, `canvas_size`, and `prompt`. `hand_bbox_json` must be a JSON object with exactly the keys `x`, `y`, `w`, and `h`, such as `{"x":0.12,"y":0.08,"w":0.64,"h":0.77}`. Values must be finite and normalized. `x` and `y` must be at least `0`, `w` and `h` must be positive, all four values must be at most `1`, and the box may not extend outside the canvas.

Example rows in the same placeholder style as `sample_submission.csv`:

```
id,hand_ink_rle,hand_bbox_json,confidence
101,"24:5 151:7 278:9","{""x"":0.08,""y"":0.02,""w"":0.76,""h"":0.91}",0.28
102,"31:4 158:6 285:8","{""x"":0.10,""y"":0.04,""w"":0.72,""h"":0.88}",0.28
103,"","{""x"":0.20,""y"":0.20,""w"":0.60,""h"":0.60}",0.10
```
